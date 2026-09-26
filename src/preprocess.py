#!/usr/bin/env python3
"""
Preprocessing module for business entity resolution.
Implements the preprocessing strategies discussed in the user's specification.
"""

import re
import unicodedata
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing options."""
    # Whether to normalize text (case, spaces, punctuation)
    normalize_text: bool = True
    # Whether to extract legal suffixes
    extract_legal_suffix: bool = True
    # Whether to extract domain information
    extract_domain: bool = True
    # Whether to extract DBA/FKA/AKA patterns
    extract_dba_patterns: bool = True
    # Whether to preserve original fields
    preserve_original: bool = True
    # Whether to extract address components
    extract_address_components: bool = True


class BusinessPreprocessor:
    """
    Preprocesses business records for entity resolution.

    Based on the detailed specification from the user's message,
    this preprocessor creates multiple information-preserving
    representations rather than aggressive normalization.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

        # Legal/business suffix mappings (from examples.csv)
        self.LEGAL_SUFFIXES = {
            # Company suffixes
            'co': 'company',
            'company': 'company',
            'corp': 'corporation',
            'corporation': 'corporation',
            'inc': 'incorporated',
            'inc.': 'incorporated',
            'incorporated': 'incorporated',
            'ltd': 'limited',
            'ltd.': 'limited',
            'limited': 'limited',
            'llc': 'llc',
            'l.l.c': 'llc',
            'llp': 'llp',
            'llp.': 'llp',
            'plc': 'plc',
            'pllc': 'pllc',
            'pc': 'pc',
            # Professional credentials (keep as-is, don't treat as suffixes)
            # These were observed but should NOT be treated like legal suffixes
            'o.d.': 'o.d.',
            'dds': 'dds',
            'md': 'md',
            'phd': 'phd',
            'esq': 'esq',
            # Add more as needed
        }

        # Address abbreviations (from examples.csv)
        self.ADDRESS_ABBREVIATIONS = {
            'st': 'street',
            'street': 'street',
            'ave': 'avenue',
            'avenue': 'avenue',
            'dr': 'drive',
            'drive': 'drive',
            'rd': 'road',
            'road': 'road',
            'blvd': 'boulevard',
            'boulevard': 'boulevard',
            'ln': 'lane',
            'lane': 'lane',
            'way': 'way',
            'ct': 'court',
            'court': 'court',
            'pl': 'place',
            'place': 'place',
            'sq': 'square',
            'square': 'square',
            'trl': 'trail',
            'trail': 'trail',
            'pkwy': 'parkway',
            'parkway': 'parkway',
            'expwy': 'expressway',
            'expressway': 'expressway',
            'hwy': 'highway',
            'highway': 'highway',
            'fwy': 'freeway',
            'freeway': 'freeway',
            'br': 'branch',
            'branch': 'branch',
            'cir': 'circle',
            'circle': 'circle',
            'cres': 'crescent',
            'crescent': 'crescent',
            'glen': 'glen',
            'glen': 'glen',
            'grv': 'grove',
            'grove': 'grove',
            'hts': 'heights',
            'heights': 'heights',
            'isl': 'island',
            'island': 'island',
            'knl': 'knoll',
            'knoll': 'knoll',
            'mt': 'mount',
            'mount': 'mount',
            'pky': 'parkway',
            'plz': 'plaza',
            'plaza': 'plaza',
            'pt': 'point',
            'point': 'point',
            'spgs': 'springs',
            'springs': 'springs',
            'stn': 'station',
            'station': 'station',
            'ter': 'terrace',
            'terrace': 'terrace',
            'tlk': 'track',
            'track': 'track',
            'vlly': 'valley',
            'valley': 'valley',
            'via': 'via',
            'via': 'via',
            'vw': 'view',
            'view': 'view',
        }

        # State abbreviations (US and India from examples)
        self.STATE_ABBREVIATIONS = {
            # US States
            'al': 'alabama',
            'ak': 'alaska',
            'az': 'arizona',
            'ar': 'arkansas',
            'ca': 'california',
            'co': 'colorado',
            'ct': 'connecticut',
            'de': 'delaware',
            'fl': 'florida',
            'ga': 'georgia',
            'hi': 'hawaii',
            'id': 'idaho',
            'il': 'illinois',
            'in': 'indiana',
            'ia': 'iowa',
            'ks': 'kansas',
            'ky': 'kentucky',
            'la': 'louisiana',
            'me': 'maine',
            'md': 'maryland',
            'ma': 'massachusetts',
            'mi': 'michigan',
            'mn': 'minnesota',
            'ms': 'mississippi',
            'mo': 'missouri',
            'mt': 'montana',
            'ne': 'nebraska',
            'nv': 'nevada',
            'nh': 'new hampshire',
            'nj': 'new jersey',
            'nm': 'new mexico',
            'ny': 'new york',
            'nc': 'north carolina',
            'nd': 'north dakota',
            'oh': 'ohio',
            'ok': 'oklahoma',
            'or': 'oregon',
            'pa': 'pennsylvania',
            'ri': 'rhode island',
            'sc': 'south carolina',
            'sd': 'south dakota',
            'tn': 'tennessee',
            'tx': 'texas',
            'ut': 'utah',
            'vt': 'vermont',
            'va': 'virginia',
            'wa': 'washington',
            'wv': 'west virginia',
            'wi': 'wisconsin',
            'wy': 'wyoming',
            'dc': 'district of columbia',
            # Indian States (from examples)
            'ka': 'karnataka',
            'karnataka': 'karnataka',
            # Add more as needed from examples
        }

        # DBA/FKA/AKA patterns
        self.DBA_PATTERNS = [
            r'\s+d/b/a\s+',
            r'\s+doing business as\s+',
            r'\s+f/k/a\s+',
            r'\s+formerly known as\s+',
            r'\s+a/k/a\s+',
            r'\s+also known as\s+',
            r'\s+t/a\s+',
            r'\s+trading as\s+',
        ]

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Pattern for splitting on non-alphanumeric (but keep Unicode letters)
        self.non_alphanum = re.compile(r'[^\w\s]+', re.UNICODE)

        # Pattern for multiple spaces
        self.multiple_spaces = re.compile(r'\s+')

        # Pattern for extracting numbers
        self.number_pattern = re.compile(r'\b\d+(?:[-/]\d+)*\b')

        # Pattern for postal codes (simplified)
        self.postal_pattern = re.compile(r'\b\d{5}(?:[-\s]\d{4})?\b|\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b', re.IGNORECASE)

        # Pattern for unit/apartment markers
        self.unit_pattern = re.compile(r'\b(?:unit|apt|apartment|flat|fl|suite|ste|#|h\.?no\.?|s\.?no\.?)\s*[\w\d-]+\b', re.IGNORECASE)

        # Pattern for domain-like strings
        self.domain_pattern = re.compile(r'\b[\w-]+\.[a-z]{2,}(?:\.[a-z]{2,})?\b', re.IGNORECASE)

        # DBA/FKA/AKA patterns
        self.dba_regex = re.compile('|'.join(self.DBA_PATTERNS), re.IGNORECASE)

        # Legal suffix patterns (at end of string)
        self.legal_suffix_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(suffix) for suffix in self.LEGAL_SUFFIXES.keys()) + r')\s*$',
            re.IGNORECASE
        )

        # Address abbreviation patterns
        self.address_abbrev_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(abbr) for abbr in self.ADDRESS_ABBREVIATIONS.keys()) + r')\b',
            re.IGNORECASE
        )

    def normalize_text(self, text: str) -> str:
        """
        Safe text normalization as specified:
        - Unicode normalization
        - casefold/lowercase
        - strip leading/trailing spaces
        - collapse repeated spaces
        - standardize obvious punctuation (& → and)
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""

        # Unicode normalization (NFKC)
        text = unicodedata.normalize('NFKC', text)

        # Standardize & → and
        text = text.replace('&', 'and')

        # Casefold (more aggressive than lowercase for Unicode)
        text = text.casefold()

        # Strip leading/trailing spaces
        text = text.strip()

        # Collapse repeated spaces
        text = self.multiple_spaces.sub(' ', text)

        return text

    def extract_legal_suffix(self, name: str) -> Tuple[str, str, str]:
        """
        Extract legal suffix from business name.

        Returns:
            tuple of (name_clean, legal_suffix, name_core)
            - name_clean: normalized name with suffix
            - legal_suffix: extracted suffix (empty if none)
            - name_core: name without suffix
        """
        if not isinstance(name, str) or pd.isna(name):
            return "", "", ""

        # Normalize the name first
        name_clean = self.normalize_text(name)

        # Try to extract legal suffix from the end
        match = self.legal_suffix_pattern.search(name_clean)
        legal_suffix = ""
        name_core = name_clean

        if match:
            suffix_raw = match.group(1).lower().strip('.')
            # Map to canonical form
            legal_suffix = self.LEGAL_SUFFIXES.get(suffix_raw, suffix_raw)
            # Remove suffix from name_core
            name_core = self.legal_suffix_pattern.sub('', name_clean).strip()
            # Clean up extra spaces that might have been created
            name_core = self.multiple_spaces.sub(' ', name_core).strip()

        return name_clean, legal_suffix, name_core

    def extract_domain(self, text: str) -> Tuple[str, str, bool]:
        """
        Extract domain-like strings from text.

        Returns:
            tuple of (domain, domain_core, is_domain_like)
            - domain: full domain if found
            - domain_core: domain without TLD
            - is_domain_like: boolean indicating if text looks like a domain
        """
        if not isinstance(text, str) or pd.isna(text):
            return "", "", False

        # Look for domain patterns
        matches = self.domain_pattern.findall(text)
        is_domain_like = len(matches) > 0

        domain = ""
        domain_core = ""

        if matches:
            # Take the first match (could be extended to handle multiple)
            domain = matches[0].lower()
            # Extract core (everything before last dot)
            if '.' in domain:
                domain_core = domain.rsplit('.', 1)[0]
            else:
                domain_core = domain

        return domain, domain_core, is_domain_like

    def extract_dba_patterns(self, name: str) -> Dict[str, int]:
        """
        Extract DBA/FKA/AKA patterns from business name.

        Returns:
            dict with flags: has_dba, has_fka, has_aka
        """
        if not isinstance(name, str) or pd.isna(name):
            return {"has_dba": 0, "has_fka": 0, "has_aka": 0}

        name_lower = name.lower()

        has_dba = 1 if re.search(r'\b(?:d\/b\/a|doing business as)\b', name_lower) else 0
        has_fka = 1 if re.search(r'\bf\/k\/a\b|formerly known as', name_lower) else 0
        has_aka = 1 if re.search(r'\ba\/k\/a\b|also known as', name_lower) else 0

        return {
            "has_dba": has_dba,
            "has_fka": has_fka,
            "has_aka": has_aka
        }

    def extract_address_components(self, address: str) -> Dict[str, Any]:
        """
        Extract address components as specified:
        - address_numbers
        - street_number
        - unit_number
        - postal_code
        - state_code
        - city_tokens
        """
        if not isinstance(address, str) or pd.isna(address):
            return {
                "address_numbers": [],
                "street_number": "",
                "unit_number": "",
                "postal_code": "",
                "state_code": "",
                "city_tokens": []
            }

        # Normalize address for processing
        addr_norm = self.normalize_text(address)

        # Extract all numbers
        raw_numbers = self.number_pattern.findall(address)  # Use original to preserve format
        # Split ranges like "1056-1060" into separate numbers
        numbers = []
        for num in raw_numbers:
            if re.search(r'\d+[-/]\d+', num):
                # Split on - or /
                parts = re.split(r'[-/]', num)
                numbers.extend(parts)
            else:
                numbers.append(num)

        # Extract postal code
        postal_matches = self.postal_pattern.findall(address)
        postal_code = postal_matches[0] if postal_matches else ""
        # Clean up postal code
        if postal_code:
            postal_code = re.sub(r'[-\s]', '', postal_code.upper())

        # Extract unit/apartment info
        unit_matches = self.unit_pattern.findall(address)
        unit_number = ""
        if unit_matches:
            # Take the first unit match and extract just the number/identifier
            unit_match = unit_matches[0]
            # Remove the label, keep the value
            unit_number = re.sub(r'^(?:unit|apt|apartment|flat|fl|suite|ste|#|h\.?no\.?|s\.?no\.?)\s*', '', unit_match, flags=re.IGNORECASE).strip()

        # Extract state code (look for state abbreviations)
        state_code = ""
        addr_lower = address.lower()
        for abbr, full in self.STATE_ABBREVIATIONS.items():
            # Look for word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(abbr) + r'\b', addr_lower):
                state_code = abbr.upper()
                break

        # Extract city tokens (simplified - everything that's not number, state, etc.)
        # Remove numbers, postal codes, state codes, unit info to get potential city parts
        addr_for_city = address
        # Remove numbers
        addr_for_city = self.number_pattern.sub('', addr_for_city)
        # Remove postal codes
        addr_for_city = self.postal_pattern.sub('', addr_for_city)
        # Remove state abbreviations
        for abbr in self.STATE_ABBREVIATIONS.keys():
            addr_for_city = re.sub(r'\b' + re.escape(abbr) + r'\b', '', addr_for_city, flags=re.IGNORECASE)
        # Remove unit patterns
        addr_for_city = self.unit_pattern.sub('', addr_for_city)
        # Clean up
        addr_for_city = self.normalize_text(addr_for_city)
        # Split on non-alphanumeric and strip punctuation from each token
        city_tokens = [re.sub(r'^[^\w]+|[^\w]+$', '', token, flags=re.UNICODE)
                       for token in re.findall(r'\S+', addr_for_city)
                       if re.sub(r'^[^\w]+|[^\w]+$', '', token, flags=re.UNICODE) and len(re.sub(r'^[^\w]+|[^\w]+$', '', token, flags=re.UNICODE)) > 1]

        # Extract street number (first number that looks like an address number)
        street_number = ""
        if numbers:
            # Heuristic: first number that's not likely a unit/zip/etc.
            for num in numbers:
                clean_num = re.sub(r'[^\d]', '', num)
                if len(clean_num) >= 1 and len(clean_num) <= 5:  # Reasonable street number length
                    # Avoid obvious zip codes (5 digits) and unit numbers
                    if not (len(clean_num) == 5 and re.match(r'^\d{5}$', clean_num)):
                        street_number = num
                        break
            # If nothing found, use first number
            if not street_number and numbers:
                street_number = numbers[0]

        return {
            "address_numbers": numbers,
            "street_number": street_number,
            "unit_number": unit_number,
            "postal_code": postal_code,
            "state_code": state_code,
            "city_tokens": city_tokens
        }

    def extract_name_features(self, name: str) -> Dict[str, Any]:
        """
        Extract name features as specified:
        - name_tokens
        - name_core
        - legal_suffix
        - domain
        - has_dba, has_fka, has_aka
        - name_numbers
        """
        if not isinstance(name, str) or pd.isna(name):
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
                "is_domain_like": False
            }

        # Get normalized name and legal suffix info
        name_clean, legal_suffix, name_core = self.extract_legal_suffix(name)

        # Extract domain information
        domain, domain_core, is_domain_like = self.extract_domain(name)

        # Extract DBA/FKA/AKA patterns
        dba_flags = self.extract_dba_patterns(name)

        # Extract name tokens (split on non-alphanumeric)
        tokens = re.findall(r'\b\w+\b', name_clean) if name_clean else []

        # Extract numbers from name
        name_numbers = self.number_pattern.findall(name)

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
            "is_domain_like": is_domain_like
        }

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single business record.

        Expected input fields:
        - entity_id: unique identifier
        - business_name: raw business name
        - business_address: raw business address
        - country: country code (optional)

        Returns:
            Dictionary with all original fields plus processed features
        """
        # Start with original record (if preserving)
        result = record.copy() if self.config.preserve_original else {}

        # Extract fields
        entity_id = record.get('entity_id', '')
        business_name = record.get('business_name', '')
        business_address = record.get('business_address', '')
        country = record.get('country', '')

        # Process business name
        name_features = self.extract_name_features(business_name)

        # Process business address
        address_clean = self.normalize_text(business_address) if self.config.normalize_text else business_address
        address_components = self.extract_address_components(business_address) if self.config.extract_address_components else {
            "address_numbers": [], "street_number": "", "unit_number": "",
            "postal_code": "", "state_code": "", "city_tokens": []
        }

        # Build result
        result.update({
            # Original fields (if preserving)
            "entity_id": entity_id,
            "business_name": business_name,
            "business_address": business_address,
            "country": country,

            # Name features
            "name_clean": name_features["name_clean"],
            "name_tokens": name_features["name_tokens"],
            "name_core": name_features["name_core"],
            "legal_suffix": name_features["legal_suffix"],
            "domain": name_features["domain"],
            "domain_core": name_features["domain_core"],
            "has_dba": name_features["has_dba"],
            "has_fka": name_features["has_fka"],
            "has_aka": name_features["has_aka"],
            "name_numbers": name_features["name_numbers"],
            "is_domain_like": name_features["is_domain_like"],

            # Address features
            "address_clean": address_clean,
            "address_numbers": address_components["address_numbers"],
            "street_number": address_components["street_number"],
            "unit_number": address_components["unit_number"],
            "postal_code": address_components["postal_code"],
            "state_code": address_components["state_code"],
            "city_tokens": address_components["city_tokens"],
        })

        return result

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a DataFrame of business records.

        Args:
            df: DataFrame with columns: entity_id, business_name, business_address, country

        Returns:
            DataFrame with original columns plus all processed features
        """
        # Apply preprocessing to each row
        processed_rows = []
        for _, row in df.iterrows():
            processed_row = self.preprocess_record(row.to_dict())
            processed_rows.append(processed_row)

        return pd.DataFrame(processed_rows)


# Convenience functions for direct use
def normalize_text(text: str) -> str:
    """Normalize text using safe normalization."""
    preprocessor = BusinessPreprocessor()
    return preprocessor.normalize_text(text)


def normalize_name(name: str) -> Dict[str, str]:
    """Normalize name and extract components."""
    preprocessor = BusinessPreprocessor()
    name_clean, legal_suffix, name_core = preprocessor.extract_legal_suffix(name)
    return {
        "name_clean": name_clean,
        "legal_suffix": legal_suffix,
        "name_core": name_core
    }


def normalize_address(address: str) -> Dict[str, Any]:
    """Normalize address and extract components."""
    preprocessor = BusinessPreprocessor()
    address_clean = preprocessor.normalize_text(address)
    components = preprocessor.extract_address_components(address)
    components["address_clean"] = address_clean
    return components


def extract_name_features(name: str) -> Dict[str, Any]:
    """Extract features from business name."""
    preprocessor = BusinessPreprocessor()
    return preprocessor.extract_name_features(name)


def extract_address_features(address: str) -> Dict[str, Any]:
    """Extract features from business address."""
    preprocessor = BusinessPreprocessor()
    return preprocessor.extract_address_components(address)


if __name__ == "__main__":
    # Example usage
    preprocessor = BusinessPreprocessor()

    # Test with examples from the specification
    test_names = [
        "FH BUSINESS  Pvt. LIMITED",
        "Hendricks and  Flowers Inc",
        "Zander Blue Co",
        "Summit Health LLC",
        "Behavioral Health Clinic",
        "Deltaquo F/K/A Pao Valley Toyo"
    ]

    test_addresses = [
        "88 OLIVE CIR, LEBANON, TN",
        "4038 Talmadge Road, Unit 102",
        "C/O Vikram Singh",
        "APT 5B13 FL 1 H.NO 204",
        "VIDYALAYA VIDYALAYA",
        "teamair.com"
    ]

    print("=== Name Preprocessing Examples ===")
    for name in test_names:
        features = preprocessor.extract_name_features(name)
        print(f"Original: {name}")
        print(f"  Clean: {features['name_clean']}")
        print(f"  Core: {features['name_core']}")
        print(f"  Suffix: {features['legal_suffix']}")
        print(f"  Domain: {features['domain']} (like: {features['is_domain_like']})")
        print(f"  DBA/FKA/AKA: {features['has_dba']}/{features['has_fka']}/{features['has_aka']}")
        print()

    print("=== Address Preprocessing Examples ===")
    for addr in test_addresses:
        features = preprocessor.extract_address_components(addr)
        print(f"Original: {addr}")
        print(f"  Clean: {preprocessor.normalize_text(addr)}")
        print(f"  Numbers: {features['address_numbers']}")
        print(f"  Street #: {features['street_number']}")
        print(f"  Unit: {features['unit_number']}")
        print(f"  Postal: {features['postal_code']}")
        print(f"  State: {features['state_code']}")
        print(f"  City tokens: {features['city_tokens']}")
        print()