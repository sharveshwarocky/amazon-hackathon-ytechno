#!/usr/bin/env python3
"""
FINAL PREPROCESSING PIPELINE V3
================================

Optimized + parallel version of the project's single-file preprocessing pipeline.

Project structure
-----------------
amazon-hackathon-ytechno/
├── preprocess_pipeline_v3.py
├── data/
│   └── train/
│       ├── train_source1.tsv
│       ├── train_source2.tsv
│       ├── train_source3.tsv
│       ├── train_ground_truth.tsv
│       ├── preprocessed/
│       └── blocking_polars/
└── src/
    └── ...

Pipeline
--------
RAW S1/S2/S3
    -> unique-value transliteration
    -> country encoding
    -> cached name/address feature extraction
    -> multilingual-aware main/native/translit features
    -> per-source preprocessed CSVs
    -> combined preprocessed CSV

Optimizations in V3
-------------------
1. No DataFrame.iterrows() for feature construction.
2. Feature extraction is performed on unique string values, then mapped back.
3. Native/main duplicate extraction is avoided for Latin-only values.
4. Transliteration uses per-source unique-value caches.
5. Combined CSV is assembled after workers finish, without reparsing the
   completed S1/S2/S3 CSVs through pandas.
6. S1/S2/S3 are processed independently in chunks.
7. S1/S2/S3 can run in parallel using multiprocessing.
8. Combined CSV is assembled at the end without reparsing CSV rows.
9. Chunk size and worker count are configurable.

Country encoding
----------------
India -> 0
US / USA / United States -> 1
France -> 2
Existing numeric 0/1/2 -> unchanged.

No country_* one-hot columns are created.

Dependency
----------
pip install indic-transliteration

Run
---
python preprocess_pipeline_v3.py

Optional:
python preprocess_pipeline_v3.py --chunk-size 100000
python preprocess_pipeline_v3.py --sample-rows 100000
python preprocess_pipeline_v3.py --data-dir "C:\\...\\data\\train"
python preprocess_pipeline_v3.py --output-dir "C:\\...\\data\\train\\preprocessed"
"""

from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from indic_transliteration import sanscript
except ImportError as exc:
    raise SystemExit(
        "\nMissing dependency: indic-transliteration\n\n"
        "Install it with:\n"
        "    pip install indic-transliteration\n"
    ) from exc


# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_FILES = {
    "S1": "train_source1.tsv",
    "S2": "train_source2.tsv",
    "S3": "train_source3.tsv",
}

DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_WORKERS = 3

COUNTRY_MAP = {
    "india": 0,
    "us": 1,
    "usa": 1,
    "united states": 1,
    "france": 2,
}

SUPPORTED_SCRIPT_RANGES = [
    ("DEVANAGARI", 0x0900, 0x097F, sanscript.DEVANAGARI),
    ("BENGALI", 0x0980, 0x09FF, sanscript.BENGALI),
    ("GURMUKHI", 0x0A00, 0x0A7F, sanscript.GURMUKHI),
    ("GUJARATI", 0x0A80, 0x0AFF, sanscript.GUJARATI),
    ("ORIYA", 0x0B00, 0x0B7F, sanscript.ORIYA),
    ("TAMIL", 0x0B80, 0x0BFF, sanscript.TAMIL),
    ("TELUGU", 0x0C00, 0x0C7F, sanscript.TELUGU),
    ("KANNADA", 0x0C80, 0x0CFF, sanscript.KANNADA),
    ("MALAYALAM", 0x0D00, 0x0D7F, sanscript.MALAYALAM),
]

# ============================================================================
# TRANSLITERATION
# ============================================================================

def detect_script_char(ch: str) -> Optional[Tuple[str, str]]:
    cp = ord(ch)
    for script_name, start, end, scheme in SUPPORTED_SCRIPT_RANGES:
        if start <= cp <= end:
            return script_name, scheme
    return None


def has_supported_nonlatin(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value)
    return any(detect_script_char(ch) is not None for ch in text)


def normalize_roman_output(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def transliterate_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)

    if not has_supported_nonlatin(text):
        return text

    result: List[str] = []
    current: List[str] = []
    current_scheme: Optional[str] = None

    def flush() -> None:
        nonlocal current, current_scheme

        if not current:
            return

        chunk = "".join(current)

        if current_scheme is None:
            result.append(chunk)
        else:
            try:
                roman = sanscript.transliterate(
                    chunk,
                    current_scheme,
                    sanscript.ITRANS,
                )
                result.append(normalize_roman_output(roman))
            except Exception:
                # Never lose the original source value.
                result.append(chunk)

        current = []
        current_scheme = None

    for ch in text:
        detected = detect_script_char(ch)

        if detected is not None:
            _, scheme = detected

            if current_scheme is None:
                current_scheme = scheme
                current.append(ch)
            elif scheme == current_scheme:
                current.append(ch)
            else:
                flush()
                current_scheme = scheme
                current.append(ch)
        else:
            if current_scheme is not None:
                flush()
            current.append(ch)

    flush()
    return "".join(result)


class TransliterationCache:
    def __init__(self, maxsize: int = 100_000) -> None:
        self.maxsize = maxsize
        self.cache: "OrderedDict[str, str]" = OrderedDict()

    def transform(self, value: Any) -> str:
        key = "" if value is None or pd.isna(value) else str(value)

        if key in self.cache:
            value_out = self.cache.pop(key)
            self.cache[key] = value_out
            return value_out

        value_out = transliterate_text(key)
        self.cache[key] = value_out

        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

        return value_out


def transliterate_unique_series(
    series: pd.Series,
    cache: TransliterationCache,
) -> pd.Series:
    """
    Transliterate only unique values in the current chunk, then map results
    back to all rows.
    """
    values = series.fillna("").astype(str)
    unique_values = pd.Index(values.unique())

    mapping = {
        value: cache.transform(value)
        for value in unique_values
    }

    return values.map(mapping)


def nonlatin_unique_series(series: pd.Series) -> pd.Series:
    """
    Detect non-Latin only once per unique value in the current chunk.
    """
    values = series.fillna("").astype(str)
    unique_values = pd.Index(values.unique())

    mapping = {
        value: int(has_supported_nonlatin(value))
        for value in unique_values
    }

    return values.map(mapping).astype("int8")


# ============================================================================
# COUNTRY
# ============================================================================

def encode_country(value: Any) -> Any:
    if pd.isna(value):
        return value

    normalized = str(value).strip().casefold()

    if normalized in {"0", "1", "2"}:
        return int(normalized)

    return COUNTRY_MAP.get(normalized, value)


# ============================================================================
# FEATURE ENGINE / PREPROCESSOR
# ============================================================================

class BusinessPreprocessor:
    """
    Same feature semantics as the existing preprocessing code, but optimized
    to operate on unique input strings and map the resulting feature dicts
    back to the chunk.
    """

    def __init__(self) -> None:
        self.LEGAL_SUFFIXES = {
            "co": "company",
            "company": "company",
            "corp": "corporation",
            "corporation": "corporation",
            "inc": "incorporated",
            "inc.": "incorporated",
            "incorporated": "incorporated",
            "ltd": "limited",
            "ltd.": "limited",
            "limited": "limited",
            "llc": "llc",
            "l.l.c": "llc",
            "llp": "llp",
            "llp.": "llp",
            "plc": "plc",
            "pllc": "pllc",
            "pc": "pc",
            # Professional credentials: preserve.
            "o.d.": "o.d.",
            "dds": "dds",
            "md": "md",
            "phd": "phd",
            "esq": "esq",
        }

        self.STATE_ABBREVIATIONS = {
            "al": "alabama",
            "ak": "alaska",
            "az": "arizona",
            "ar": "arkansas",
            "ca": "california",
            "co": "colorado",
            "ct": "connecticut",
            "de": "delaware",
            "fl": "florida",
            "ga": "georgia",
            "hi": "hawaii",
            "id": "idaho",
            "il": "illinois",
            "in": "indiana",
            "ia": "iowa",
            "ks": "kansas",
            "ky": "kentucky",
            "la": "louisiana",
            "me": "maine",
            "md": "maryland",
            "ma": "massachusetts",
            "mi": "michigan",
            "mn": "minnesota",
            "ms": "mississippi",
            "mo": "missouri",
            "mt": "montana",
            "ne": "nebraska",
            "nv": "nevada",
            "nh": "new hampshire",
            "nj": "new jersey",
            "nm": "new mexico",
            "ny": "new york",
            "nc": "north carolina",
            "nd": "north dakota",
            "oh": "ohio",
            "ok": "oklahoma",
            "or": "oregon",
            "pa": "pennsylvania",
            "ri": "rhode island",
            "sc": "south carolina",
            "sd": "south dakota",
            "tn": "tennessee",
            "tx": "texas",
            "ut": "utah",
            "vt": "vermont",
            "va": "virginia",
            "wa": "washington",
            "wv": "west virginia",
            "wi": "wisconsin",
            "wy": "wyoming",
            "dc": "district of columbia",
            "ka": "karnataka",
            "karnataka": "karnataka",
        }

        self.multiple_spaces = re.compile(r"\s+")
        self.number_pattern = re.compile(r"\b\d+(?:[-/]\d+)*\b")
        self.postal_pattern = re.compile(
            r"\b\d{5}(?:[-\s]\d{4})?\b"
            r"|\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b",
            re.IGNORECASE,
        )
        self.unit_pattern = re.compile(
            r"\b(?:unit|apt|apartment|flat|fl|suite|ste|#|"
            r"h\.?no\.?|s\.?no\.?)\s*[\w\d-]+\b",
            re.IGNORECASE,
        )
        self.domain_pattern = re.compile(
            r"\b[\w-]+\.[a-z]{2,}(?:\.[a-z]{2,})?\b",
            re.IGNORECASE,
        )
        self.legal_suffix_pattern = re.compile(
            r"\b("
            + "|".join(
                re.escape(x)
                for x in self.LEGAL_SUFFIXES.keys()
            )
            + r")\s*$",
            re.IGNORECASE,
        )

        # These are stable for the lifetime of one source.
        # They are useful because the same values often repeat across chunks.
        self._name_features_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._address_features_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_maxsize = 250_000

    # ------------------------------------------------------------------
    # General normalization
    # ------------------------------------------------------------------

    def normalize_text(self, text: Any) -> str:
        if text is None or pd.isna(text):
            return ""

        value = unicodedata.normalize(
            "NFKC",
            str(text),
        )
        value = value.replace("&", "and")
        value = value.casefold().strip()
        return self.multiple_spaces.sub(
            " ",
            value,
        )

    # ------------------------------------------------------------------
    # Name features
    # ------------------------------------------------------------------

    def extract_legal_suffix(
        self,
        name: Any,
    ) -> Tuple[str, str, str]:

        if name is None or pd.isna(name):
            return "", "", ""

        name_clean = self.normalize_text(name)

        match = self.legal_suffix_pattern.search(
            name_clean
        )

        legal_suffix = ""
        name_core = name_clean

        if match:
            suffix_raw = match.group(1).lower().strip(".")
            legal_suffix = self.LEGAL_SUFFIXES.get(
                suffix_raw,
                suffix_raw,
            )

            name_core = self.legal_suffix_pattern.sub(
                "",
                name_clean,
            ).strip()

            name_core = self.multiple_spaces.sub(
                " ",
                name_core,
            ).strip()

        return (
            name_clean,
            legal_suffix,
            name_core,
        )

    def extract_domain(
        self,
        text: Any,
    ) -> Tuple[str, str, bool]:

        if text is None or pd.isna(text):
            return "", "", False

        matches = self.domain_pattern.findall(
            str(text)
        )

        if not matches:
            return "", "", False

        domain = matches[0].lower()
        domain_core = domain.rsplit(".", 1)[0]

        return (
            domain,
            domain_core,
            True,
        )

    def extract_dba_patterns(
        self,
        name: Any,
    ) -> Dict[str, int]:

        if name is None or pd.isna(name):
            return {
                "has_dba": 0,
                "has_fka": 0,
                "has_aka": 0,
            }

        text = str(name).casefold()

        return {
            "has_dba": int(
                bool(
                    re.search(
                        r"\b(?:d/b/a|doing business as)\b",
                        text,
                    )
                )
            ),
            "has_fka": int(
                bool(
                    re.search(
                        r"\bf/k/a\b|formerly known as",
                        text,
                    )
                )
            ),
            "has_aka": int(
                bool(
                    re.search(
                        r"\ba/k/a\b|also known as",
                        text,
                    )
                )
            ),
        }

    def extract_name_features_uncached(
        self,
        name: Any,
    ) -> Dict[str, Any]:

        if name is None or pd.isna(name):
            return {
                "name_clean": "",
                "name_tokens": [],
                "name_core": "",
                "legal_suffix": "",
                "domain": "",
                "domain_core": "",
                "has_dba": 0,
                "has_fka": 0,
                "has_aka": 0,
                "name_numbers": [],
                "is_domain_like": False,
            }

        (
            name_clean,
            legal_suffix,
            name_core,
        ) = self.extract_legal_suffix(name)

        (
            domain,
            domain_core,
            is_domain_like,
        ) = self.extract_domain(name)

        dba_flags = self.extract_dba_patterns(name)

        tokens = (
            re.findall(
                r"\b\w+\b",
                name_clean,
            )
            if name_clean
            else []
        )

        name_numbers = self.number_pattern.findall(
            str(name)
        )

        return {
            "name_clean": name_clean,
            "name_tokens": tokens,
            "name_core": name_core,
            "legal_suffix": legal_suffix,
            "domain": domain,
            "domain_core": domain_core,
            "has_dba": dba_flags["has_dba"],
            "has_fka": dba_flags["has_fka"],
            "has_aka": dba_flags["has_aka"],
            "name_numbers": name_numbers,
            "is_domain_like": is_domain_like,
        }

    def extract_name_features(
        self,
        name: Any,
    ) -> Dict[str, Any]:

        key = "" if name is None or pd.isna(name) else str(name)

        cached = self._name_features_cache.get(key)
        if cached is not None:
            return cached

        features = self.extract_name_features_uncached(
            key
        )

        self._name_features_cache[key] = features

        if len(self._name_features_cache) > self._cache_maxsize:
            self._name_features_cache.popitem(
                last=False
            )

        return features

    # ------------------------------------------------------------------
    # Address features
    # ------------------------------------------------------------------

    def extract_address_components_uncached(
        self,
        address: Any,
    ) -> Dict[str, Any]:

        if address is None or pd.isna(address):
            return {
                "address_clean": "",
                "address_numbers": [],
                "street_number": "",
                "unit_number": "",
                "postal_code": "",
                "state_code": "",
                "city_tokens": [],
            }

        original = str(address)

        raw_numbers = self.number_pattern.findall(
            original
        )

        numbers: List[str] = []

        for num in raw_numbers:
            if re.search(
                r"\d+[-/]\d+",
                num,
            ):
                numbers.extend(
                    re.split(
                        r"[-/]",
                        num,
                    )
                )
            else:
                numbers.append(num)

        postal_matches = self.postal_pattern.findall(
            original
        )

        postal_code = (
            postal_matches[0]
            if postal_matches
            else ""
        )

        if postal_code:
            postal_code = re.sub(
                r"[-\s]",
                "",
                postal_code.upper(),
            )

        unit_matches = self.unit_pattern.findall(
            original
        )

        unit_number = ""

        if unit_matches:
            unit_number = re.sub(
                r"^(?:unit|apt|apartment|flat|fl|suite|ste|#|"
                r"h\.?no\.?|s\.?no\.?)\s*",
                "",
                unit_matches[0],
                flags=re.IGNORECASE,
            ).strip()

        state_code = ""
        lower = original.casefold()

        for abbr in self.STATE_ABBREVIATIONS:

            if re.search(
                r"\b" + re.escape(abbr) + r"\b",
                lower,
            ):
                state_code = abbr.upper()
                break

        city_text = original

        city_text = self.number_pattern.sub(
            "",
            city_text,
        )

        city_text = self.postal_pattern.sub(
            "",
            city_text,
        )

        for abbr in self.STATE_ABBREVIATIONS:

            city_text = re.sub(
                r"\b" + re.escape(abbr) + r"\b",
                "",
                city_text,
                flags=re.IGNORECASE,
            )

        city_text = self.unit_pattern.sub(
            "",
            city_text,
        )

        city_text = self.normalize_text(
            city_text
        )

        city_tokens: List[str] = []

        for token in re.findall(
            r"\S+",
            city_text,
        ):

            cleaned = re.sub(
                r"^[^\w]+|[^\w]+$",
                "",
                token,
                flags=re.UNICODE,
            )

            if cleaned and len(cleaned) > 1:
                city_tokens.append(cleaned)

        street_number = ""

        if numbers:

            for num in numbers:

                clean_num = re.sub(
                    r"[^\d]",
                    "",
                    num,
                )

                if 1 <= len(clean_num) <= 5:

                    if not (
                        len(clean_num) == 5
                        and clean_num.isdigit()
                    ):
                        street_number = num
                        break

            if not street_number:
                street_number = numbers[0]

        return {
            "address_clean": self.normalize_text(address),
            "address_numbers": numbers,
            "street_number": street_number,
            "unit_number": unit_number,
            "postal_code": postal_code,
            "state_code": state_code,
            "city_tokens": city_tokens,
        }

    def extract_address_components(
        self,
        address: Any,
    ) -> Dict[str, Any]:

        key = "" if address is None or pd.isna(address) else str(address)

        cached = self._address_features_cache.get(key)

        if cached is not None:
            return cached

        features = (
            self.extract_address_components_uncached(
                key
            )
        )

        self._address_features_cache[key] = features

        if len(self._address_features_cache) > self._cache_maxsize:
            self._address_features_cache.popitem(
                last=False
            )

        return features

    # ------------------------------------------------------------------
    # Unique-value feature table builders
    # ------------------------------------------------------------------

    def build_name_feature_table(
        self,
        values: pd.Series,
        base_table: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:

        unique_values = pd.Index(
            values.fillna("").astype(str).unique()
        )

        table: Dict[str, Dict[str, Any]] = {}

        for value in unique_values:
            if base_table is not None and value in base_table:
                table[value] = base_table[value]
            else:
                table[value] = self.extract_name_features(value)

        return table

    def build_address_feature_table(
        self,
        values: pd.Series,
        base_table: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:

        unique_values = pd.Index(
            values.fillna("").astype(str).unique()
        )

        table: Dict[str, Dict[str, Any]] = {}

        for value in unique_values:
            if base_table is not None and value in base_table:
                table[value] = base_table[value]
            else:
                table[value] = self.extract_address_components(value)

        return table

    # ------------------------------------------------------------------
    # Optimized chunk preprocessing
    # ------------------------------------------------------------------

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = df.copy()

        # Ensure preparation columns exist.
        if "name_has_nonlatin" not in result.columns:
            result["name_has_nonlatin"] = (
                nonlatin_unique_series(
                    result["business_name"]
                )
            )

        if "address_has_nonlatin" not in result.columns:
            result["address_has_nonlatin"] = (
                nonlatin_unique_series(
                    result["business_address"]
                )
            )

        # --------------------------------------------------------------
        # Main selected representations
        # --------------------------------------------------------------

        name_original = (
            result["business_name"]
            .fillna("")
            .astype(str)
        )

        address_original = (
            result["business_address"]
            .fillna("")
            .astype(str)
        )

        name_translit = (
            result["business_name_translit"]
            .fillna("")
            .astype(str)
        )

        address_translit = (
            result["business_address_translit"]
            .fillna("")
            .astype(str)
        )

        use_name_translit = (
            result["name_has_nonlatin"].astype(bool)
            & name_translit.str.strip().ne("")
        )

        use_address_translit = (
            result["address_has_nonlatin"].astype(bool)
            & address_translit.str.strip().ne("")
        )

        selected_name = name_original.where(
            ~use_name_translit,
            name_translit,
        )

        selected_address = address_original.where(
            ~use_address_translit,
            address_translit,
        )

        # --------------------------------------------------------------
        # Unique name/address feature tables.
        # --------------------------------------------------------------

        # Native table is needed for all records because we retain native
        # features. Main selected table is needed only when it differs.
        native_name_table = (
            self.build_name_feature_table(
                name_original
            )
        )

        native_address_table = (
            self.build_address_feature_table(
                address_original
            )
        )

        if bool(use_name_translit.any()):
            selected_name_table = self.build_name_feature_table(
                selected_name,
                base_table=native_name_table,
            )
        else:
            selected_name_table = native_name_table

        if bool(use_address_translit.any()):
            selected_address_table = self.build_address_feature_table(
                selected_address,
                base_table=native_address_table,
            )
        else:
            selected_address_table = native_address_table

        # --------------------------------------------------------------
        # Map feature dictionaries back to rows.
        # --------------------------------------------------------------

        def map_feature(
            series: pd.Series,
            table: Dict[str, Dict[str, Any]],
            key: str,
        ) -> pd.Series:

            values = (
                series.fillna("")
                .astype(str)
                .map(
                    lambda value: table[value][key]
                )
            )

            return values

        # Main name features.
        for key in (
            "name_clean",
            "name_tokens",
            "name_core",
            "legal_suffix",
            "domain",
            "domain_core",
            "has_dba",
            "has_fka",
            "has_aka",
            "name_numbers",
            "is_domain_like",
        ):
            result[key] = map_feature(
                selected_name,
                selected_name_table,
                key,
            )

        # Main address features.
        for key in (
            "address_clean",
            "address_numbers",
            "street_number",
            "unit_number",
            "postal_code",
            "state_code",
            "city_tokens",
        ):
            result[key] = map_feature(
                selected_address,
                selected_address_table,
                key,
            )

        # Native name features.
        for key in (
            "name_clean",
            "name_tokens",
            "name_core",
            "legal_suffix",
            "domain",
            "domain_core",
            "name_numbers",
        ):
            result[f"{key}_native"] = map_feature(
                name_original,
                native_name_table,
                key,
            )

        # Native address features.
        for key in (
            "address_clean",
            "address_numbers",
            "street_number",
            "unit_number",
            "postal_code",
            "state_code",
            "city_tokens",
        ):
            result[f"{key}_native"] = map_feature(
                address_original,
                native_address_table,
                key,
            )

        # Explicit transliterated name features.
        # For Latin-only rows these are kept empty, matching V1.
        for key in (
            "name_clean",
            "name_tokens",
            "name_core",
            "legal_suffix",
            "domain",
            "domain_core",
            "name_numbers",
        ):
            mapped_selected = map_feature(
                selected_name,
                selected_name_table,
                key,
            )
            result[f"{key}_translit"] = mapped_selected.where(
                use_name_translit,
                "" if key not in {"name_tokens", "name_numbers"} else None,
            )

        # Explicit transliterated address features.
        for key in (
            "address_clean",
            "address_numbers",
            "street_number",
            "unit_number",
            "postal_code",
            "state_code",
            "city_tokens",
        ):
            mapped_selected = map_feature(
                selected_address,
                selected_address_table,
                "address_clean" if key == "address_clean" else key,
            )

            if key in {"address_numbers", "city_tokens"}:
                # Lists need an explicit object Series so we don't accidentally
                # coerce to numeric/object inconsistently.
                result[f"{key}_translit"] = pd.Series(
                    [
                        mapped_selected.iloc[i]
                        if bool(use_address_translit.iloc[i])
                        else []
                        for i in range(len(result))
                    ],
                    index=result.index,
                    dtype=object,
                )
            else:
                result[f"{key}_translit"] = mapped_selected.where(
                    use_address_translit,
                    "",
                )

        # Keep the original explicit transliteration source fields already
        # created before preprocessing.
        result["business_name_translit"] = name_translit
        result["business_address_translit"] = address_translit

        return result


# ============================================================================
# CHUNK PROCESSING
# ============================================================================

def process_source(
    source_name: str,
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    sample_rows: Optional[int] = None,
) -> Dict[str, int]:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing source file: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    stats = {
        "rows": 0,
        "nonlatin_names": 0,
        "nonlatin_addresses": 0,
        "transliterated_names": 0,
        "transliterated_addresses": 0,
    }

    name_cache = TransliterationCache()
    address_cache = TransliterationCache()

    first_source_chunk = True
    import time
    source_start_time = time.perf_counter()

    preprocessor = BusinessPreprocessor()

    print()
    print("=" * 78)
    print(f"PROCESSING {source_name}")
    print("=" * 78)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    # One reader. For the full run this streams chunk-by-chunk.
    reader = pd.read_csv(
        input_path,
        sep="\t",
        chunksize=chunk_size,
        low_memory=False,
    )

    rows_remaining = sample_rows

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        if rows_remaining is not None:

            if rows_remaining <= 0:
                break

            if len(chunk) > rows_remaining:
                chunk = chunk.iloc[
                    :rows_remaining
                ].copy()

            rows_remaining -= len(chunk)

        required_columns = {
            "entity_id",
            "business_name",
            "business_address",
        }

        missing = (
            required_columns
            - set(chunk.columns)
        )

        if missing:
            raise ValueError(
                f"{source_name} is missing required columns: "
                f"{sorted(missing)}"
            )

        # --------------------------------------------------------------
        # Transliteration + script flags using unique values
        # --------------------------------------------------------------

        chunk["name_has_nonlatin"] = (
            nonlatin_unique_series(
                chunk["business_name"]
            )
        )

        chunk["address_has_nonlatin"] = (
            nonlatin_unique_series(
                chunk["business_address"]
            )
        )

        chunk["business_name_translit"] = (
            transliterate_unique_series(
                chunk["business_name"],
                name_cache,
            )
        )

        chunk["business_address_translit"] = (
            transliterate_unique_series(
                chunk["business_address"],
                address_cache,
            )
        )

        # --------------------------------------------------------------
        # Country encoding
        # --------------------------------------------------------------

        if "country" in chunk.columns:
            chunk["country"] = (
                chunk["country"].map(
                    encode_country
                )
            )

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        stats["rows"] += len(chunk)

        stats["nonlatin_names"] += int(
            chunk[
                "name_has_nonlatin"
            ].sum()
        )

        stats["nonlatin_addresses"] += int(
            chunk[
                "address_has_nonlatin"
            ].sum()
        )

        original_names = (
            chunk["business_name"]
            .fillna("")
            .astype(str)
        )

        original_addresses = (
            chunk["business_address"]
            .fillna("")
            .astype(str)
        )

        stats["transliterated_names"] += int(
            (
                chunk["business_name_translit"]
                != original_names
            ).sum()
        )

        stats["transliterated_addresses"] += int(
            (
                chunk["business_address_translit"]
                != original_addresses
            ).sum()
        )

        # --------------------------------------------------------------
        # Main preprocessing
        # --------------------------------------------------------------

        processed = (
            preprocessor.preprocess_dataframe(
                chunk
            )
        )

        # --------------------------------------------------------------
        # Write source output
        # --------------------------------------------------------------

        processed.to_csv(
            output_path,
            mode=(
                "w"
                if first_source_chunk
                else "a"
            ),
            header=first_source_chunk,
            index=False,
        )

        first_source_chunk = False

        elapsed = time.perf_counter() - source_start_time
        rows_per_sec = stats["rows"] / elapsed if elapsed > 0 else 0.0

        print(
            f"[{source_name}] ✅ CHUNK {chunk_number:>4} COMPLETE | "
            f"+{len(chunk):,} rows | "
            f"source total: {stats['rows']:,} | "
            f"{rows_per_sec:,.0f} rows/s | "
            f"elapsed: {elapsed:,.1f}s",
            flush=True,
        )

        if sample_rows is not None and rows_remaining == 0:
            break

    elapsed = time.perf_counter() - source_start_time
    rows_per_sec = stats["rows"] / elapsed if elapsed > 0 else 0.0

    print()
    print(
        f"[{source_name}] 🏁 SOURCE COMPLETE | "
        f"{stats['rows']:,} rows | "
        f"non-Latin names={stats['nonlatin_names']:,} | "
        f"non-Latin addresses={stats['nonlatin_addresses']:,} | "
        f"avg={rows_per_sec:,.0f} rows/s | "
        f"elapsed={elapsed:,.1f}s",
        flush=True,
    )

    return stats


def concatenate_source_outputs(
    source_paths: List[Path],
    combined_output: Path,
) -> None:
    """
    Concatenate identically-structured CSVs without reparsing rows through
    pandas. The first file contributes the header; later files skip it.
    """
    if not source_paths:
        raise ValueError("No source outputs were provided.")

    combined_output.parent.mkdir(parents=True, exist_ok=True)

    with combined_output.open("wb") as out_file:
        for idx, source_path in enumerate(source_paths):
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Missing source output: {source_path}"
                )

            with source_path.open("rb") as in_file:
                if idx > 0:
                    # Skip the CSV header line. Each source has the same
                    # column order because every chunk is processed by the
                    # same preprocessing function.
                    in_file.readline()

                shutil.copyfileobj(in_file, out_file, length=1024 * 1024)


# ============================================================================
# PROJECT ROOT / CLI
# ============================================================================

def find_project_root() -> Path:
    if "__file__" in globals():
        root = Path(__file__).resolve().parent
    else:
        root = Path.cwd()

    # If running from src/, project root is parent.
    if root.name.casefold() == "src":
        root = root.parent

    # Search upward for data/train/source1.
    for candidate in [root, *root.parents]:
        if (
            candidate
            / "data"
            / "train"
            / "train_source1.tsv"
        ).exists():
            return candidate

    return root


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()

    # When pasted into a notebook, ignore Jupyter's kernel arguments.
    argv = None if "__file__" in globals() else []

    parser = argparse.ArgumentParser(
        description=(
            "Optimized multilingual business-entity preprocessing."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            project_root
            / "data"
            / "train"
        ),
        help="Raw source directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "data"
            / "train"
            / "preprocessed"
        ),
        help="Preprocessed output directory.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per chunk. Default: {DEFAULT_CHUNK_SIZE:,}",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help=(
            "Process only this many rows per source. "
            "Useful for a fast correctness test."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Number of source-level workers. Default: "
            f"{DEFAULT_WORKERS}. Maximum supported: {len(SOURCE_FILES)}."
        ),
    )

    parser.add_argument(
        "--executor",
        choices=("auto", "process", "thread"),
        default="auto",
        help=(
            "Worker backend. 'process' gives true multiprocessing when the "
            "script is run normally; 'thread' is notebook-safe; 'auto' "
            "selects process for a normal .py run and thread inside Jupyter."
        ),
    )

    return parser.parse_args(argv)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    if args.chunk_size <= 0:
        raise SystemExit(
            "--chunk-size must be greater than 0."
        )

    if (
        args.sample_rows is not None
        and args.sample_rows <= 0
    ):
        raise SystemExit(
            "--sample-rows must be greater than 0."
        )

    if args.workers <= 0:
        raise SystemExit(
            "--workers must be greater than 0."
        )

    if args.workers > len(SOURCE_FILES):
        raise SystemExit(
            f"--workers cannot exceed {len(SOURCE_FILES)} "
            "because there are only S1, S2 and S3 source jobs."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_output = (
        args.output_dir
        / "all_sources_preprocessed.csv"
    )

    if combined_output.exists():
        combined_output.unlink()

    print()
    print("=" * 78)
    print(
        "BUSINESS ENTITY RESOLUTION "
        "- PREPROCESSING V3"
    )
    print("=" * 78)

    print(f"Project root : {find_project_root()}")
    print(f"Raw data     : {args.data_dir}")
    print(f"Output       : {args.output_dir}")
    print(f"Chunk size   : {args.chunk_size:,}")
    print(f"Workers      : {args.workers}")
    print("Live progress: enabled (each completed chunk is printed immediately)")

    if args.sample_rows is not None:
        print(
            f"Sample rows/source : "
            f"{args.sample_rows:,}"
        )

    print()
    print("Optimizations enabled:")
    print("  - unique-value transliteration")
    print("  - no iterrows() feature construction")
    print("  - cached unique-value feature extraction")
    print("  - native feature tables reused when possible")
    print("  - source-level multiprocessing")
    print("  - combined CSV assembled without pandas reread")
    print("  - chunked S1/S2/S3 processing")

    print()
    print("Country encoding:")
    print("  India = 0")
    print("  US / USA / United States = 1")
    print("  France = 2")

    source_stats: Dict[
        str,
        Dict[str, int],
    ] = {}

    source_jobs: List[Tuple[str, Path, Path]] = []

    for source_name, filename in SOURCE_FILES.items():
        input_path = args.data_dir / filename
        output_path = args.output_dir / (
            f"{source_name.lower()}_preprocessed.csv"
        )
        source_jobs.append(
            (source_name, input_path, output_path)
        )

    effective_workers = min(args.workers, len(source_jobs))

    # Windows uses spawn for multiprocessing. When this file is pasted/run
    # inside Jupyter, the worker functions live in the notebook's __main__
    # namespace and cannot be imported by spawned child processes. That is
    # what causes BrokenProcessPool in notebook execution.
    running_in_notebook = "get_ipython" in globals()

    if args.executor == "auto":
        executor_kind = "thread" if running_in_notebook else "process"
    else:
        executor_kind = args.executor

    if executor_kind == "process" and running_in_notebook:
        print()
        print(
            "⚠️ Notebook detected: forcing thread workers instead of "
            "ProcessPoolExecutor to avoid Windows spawn/BrokenProcessPool."
        )
        executor_kind = "thread"

    print()
    print(
        f"Parallel source workers: {effective_workers} (S1 + S2 + S3)"
    )
    print(f"Worker backend        : {executor_kind}")
    if executor_kind == "process":
        print("True multiprocessing enabled.")
    else:
        print("Notebook-safe concurrent threads enabled.")
    print(
        "The combined CSV is assembled only after S1/S2/S3 finish."
    )

    executor_cls = (
        ProcessPoolExecutor
        if executor_kind == "process"
        else ThreadPoolExecutor
    )

    # Each source is independent, so parallelize at the source level.
    # Use processes for a normal .py run and threads inside Jupyter, where
    # Windows multiprocessing cannot reliably import notebook-defined code.
    with executor_cls(
        max_workers=effective_workers,
    ) as executor:
        future_to_source = {
            executor.submit(
                process_source,
                source_name,
                input_path,
                output_path,
                args.chunk_size,
                args.sample_rows,
            ): source_name
            for source_name, input_path, output_path in source_jobs
        }

        for future in as_completed(future_to_source):
            source_name = future_to_source[future]
            source_stats[source_name] = future.result()
            print(
                f"✅ Worker finished {source_name}: "
                f"{source_stats[source_name]['rows']:,} rows"
            )

    # Keep combined output order deterministic: S1, S2, S3.
    source_output_paths = [
        args.output_dir / "s1_preprocessed.csv",
        args.output_dir / "s2_preprocessed.csv",
        args.output_dir / "s3_preprocessed.csv",
    ]

    concatenate_source_outputs(
        source_output_paths,
        combined_output,
    )

    total_rows = sum(
        stats["rows"]
        for stats in source_stats.values()
    )

    total_nonlatin_names = sum(
        stats["nonlatin_names"]
        for stats in source_stats.values()
    )

    total_nonlatin_addresses = sum(
        stats["nonlatin_addresses"]
        for stats in source_stats.values()
    )

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    for source_name, stats in source_stats.items():
        print(
            f"{source_name}: "
            f"{stats['rows']:,} rows | "
            f"non-Latin names="
            f"{stats['nonlatin_names']:,} | "
            f"non-Latin addresses="
            f"{stats['nonlatin_addresses']:,}"
        )

    print()
    print(f"Total rows          : {total_rows:,}")
    print(
        f"Non-Latin names     : "
        f"{total_nonlatin_names:,}"
    )
    print(
        f"Non-Latin addresses : "
        f"{total_nonlatin_addresses:,}"
    )

    print()
    print(
        "Per-source outputs:"
    )

    for source_name in SOURCE_FILES:
        print(
            f"  {args.output_dir / (source_name.lower() + '_preprocessed.csv')}"
        )

    print()
    print(
        f"Combined output:\n"
        f"  {combined_output}"
    )

    print()
    print("✅ Preprocessing V3 complete.")

    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
