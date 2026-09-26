import polars as pl
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

K_FREQUENCY_CAP = 1000

S1_ID_COL = "entity_id"
S2_ID_COL = "entity_id"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATH_S1 = PROJECT_ROOT / "src" / "output" / "s1_preprocessed.csv"
PATH_S2 = PROJECT_ROOT / "src" / "output" / "s2_preprocessed.csv"
PATH_GT = PROJECT_ROOT / "data" / "train" / "train_ground_truth.tsv"

TEMP_DIR = PROJECT_ROOT / "src" / "output" / "blocking_temp"

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ONLY REQUIRED COLUMNS
# ============================================================

def load_required_columns(
    path: Path,
    columns: list[str]
) -> pl.DataFrame:

    """
    Loads only the columns required for the current rule.

    This avoids loading the entire S1/S2 dataset into RAM.
    """

    return (
        pl.scan_csv(
            path,
            schema_overrides={
                "postal_code": pl.String,
                "street_number": pl.String,
            }
        )
        .select(columns)
        .collect()
    )


# ============================================================
# PARSE name_tokens
# ============================================================

def parse_name_tokens(
    df: pl.DataFrame
) -> pl.DataFrame:

    if "name_tokens" not in df.columns:
        return df

    dtype = df.schema["name_tokens"]

    # Already a List
    if isinstance(dtype, pl.List):
        return df

    # Example CSV value:
    #
    # "['shree', 'infracon', 'private']"
    #
    # Convert to:
    #
    # ["shree", "infracon", "private"]

    return df.with_columns(

        pl.col("name_tokens")
        .cast(pl.String)
        .str.strip_chars()
        .str.strip_chars("[]")
        .str.replace_all("'", "")
        .str.replace_all('"', "")
        .str.split(",")
        .list.eval(
            pl.element().str.strip_chars()
        )
        .alias("name_tokens")

    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth() -> pl.DataFrame:

    print("\nLoading ground truth...")

    gt = pl.read_csv(
        PATH_GT,
        separator="\t"
    )

    print(
        "Ground truth columns:",
        gt.columns
    )

    print("\nFirst 5 rows:")

    print(
        gt.head()
    )

    if len(gt.columns) < 2:
        raise ValueError(
            "Ground truth must contain at least 2 columns."
        )

    # Current assumption:
    # first column = S1 ID
    # second column = S2 ID

    c1 = gt.columns[0]
    c2 = gt.columns[1]

    gt = (
        gt
        .select([
            pl.col(c1).alias("S1_id"),
            pl.col(c2).alias("S2_id")
        ])
        .drop_nulls()
        .unique(
            subset=["S1_id", "S2_id"]
        )
    )

    print(
        f"\nTrue pairs: {gt.height:,}"
    )

    return gt


# ============================================================
# GENERATE CANDIDATES FOR ONE RULE
# ============================================================

def evaluate_blocking_rule(
    rule_name: str,
    join_keys: list[str],
    explode_col: str | None,
) -> pl.DataFrame:

    print("\n" + "=" * 70)
    print(f"RUNNING RULE: {rule_name}")
    print("=" * 70)

    start = time.time()

    # --------------------------------------------------------
    # Determine which columns are needed
    # --------------------------------------------------------

    required_columns = [
        S1_ID_COL
    ] + join_keys

    # Remove duplicates while preserving order

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    print(
        "Loading columns:",
        required_columns
    )

    # --------------------------------------------------------
    # LOAD S1 ONLY FOR THIS RULE
    # --------------------------------------------------------

    print("Loading required S1 columns...")

    s1 = load_required_columns(
        PATH_S1,
        required_columns
    )

    print(
        f"S1 loaded: {s1.height:,} rows"
    )

    # --------------------------------------------------------
    # LOAD S2 ONLY FOR THIS RULE
    # --------------------------------------------------------

    print("Loading required S2 columns...")

    s2 = load_required_columns(
        PATH_S2,
        required_columns
    )

    print(
        f"S2 loaded: {s2.height:,} rows"
    )

    # --------------------------------------------------------
    # PARSE name_tokens
    # --------------------------------------------------------

    if explode_col == "name_tokens":

        s1 = parse_name_tokens(s1)
        s2 = parse_name_tokens(s2)

    # --------------------------------------------------------
    # Remove NULL blocking keys
    # --------------------------------------------------------

    s1 = (
        s1
        .drop_nulls(subset=join_keys)
    )

    s2 = (
        s2
        .drop_nulls(subset=join_keys)
    )

    # --------------------------------------------------------
    # EXPLODE name_tokens
    # --------------------------------------------------------

    if explode_col is not None:

        print(
            f"Exploding {explode_col}..."
        )

        s1 = (
            s1
            .explode(explode_col)
            .drop_nulls(
                subset=[explode_col]
            )
        )

        s2 = (
            s2
            .explode(explode_col)
            .drop_nulls(
                subset=[explode_col]
            )
        )

    print(
        f"S1 rows after preparation: "
        f"{s1.height:,}"
    )

    print(
        f"S2 rows after preparation: "
        f"{s2.height:,}"
    )

    # --------------------------------------------------------
    # FREQUENCY COUNT ON S2
    # --------------------------------------------------------

    print(
        "\nCalculating S2 key frequencies..."
    )

    freq = (
        s2
        .group_by(join_keys)
        .len(
            name="key_count"
        )
    )

    print(
        f"Unique keys: {freq.height:,}"
    )

    # --------------------------------------------------------
    # FREQUENCY CAP
    # --------------------------------------------------------

    valid_keys = (
        freq
        .filter(
            pl.col("key_count")
            <= K_FREQUENCY_CAP
        )
        .select(join_keys)
    )

    print(
        f"Valid keys after cap "
        f"{K_FREQUENCY_CAP}: "
        f"{valid_keys.height:,}"
    )

    # --------------------------------------------------------
    # FILTER S1
    # --------------------------------------------------------

    s1_filtered = (
        s1
        .join(
            valid_keys,
            on=join_keys,
            how="inner"
        )
    )

    # --------------------------------------------------------
    # FILTER S2
    # --------------------------------------------------------

    s2_filtered = (
        s2
        .join(
            valid_keys,
            on=join_keys,
            how="inner"
        )
    )

    print(
        f"S1 rows after frequency filter: "
        f"{s1_filtered.height:,}"
    )

    print(
        f"S2 rows after frequency filter: "
        f"{s2_filtered.height:,}"
    )

    # --------------------------------------------------------
    # CANDIDATE JOIN
    # --------------------------------------------------------

    print(
        "\nGenerating candidate pairs..."
    )

    candidates = (
        s1_filtered
        .join(
            s2_filtered,
            on=join_keys,
            how="inner",
            suffix="_S2"
        )
        .select([
            pl.col(S1_ID_COL)
            .alias("S1_id"),

            pl.col(
                f"{S2_ID_COL}_S2"
            )
            .alias("S2_id")
        ])
        .unique(
            subset=[
                "S1_id",
                "S2_id"
            ]
        )
    )

    elapsed = time.time() - start

    print(
        f"\nCandidates generated: "
        f"{candidates.height:,}"
    )

    print(
        f"Time taken: {elapsed:.2f} seconds"
    )

    return candidates


# ============================================================
# CALCULATE RECALL
# ============================================================

def calculate_metrics(
    candidates: pl.DataFrame,
    true_pairs: pl.DataFrame
) -> dict:

    candidate_count = candidates.height

    caught = (
        candidates
        .join(
            true_pairs,
            on=[
                "S1_id",
                "S2_id"
            ],
            how="inner"
        )
        .unique(
            subset=[
                "S1_id",
                "S2_id"
            ]
        )
    )

    caught_count = caught.height

    total_true = true_pairs.height

    if total_true == 0:
        recall = 0.0
    else:
        recall = (
            caught_count /
            total_true
        )

    return {
        "candidates": candidate_count,
        "caught": caught_count,
        "recall": recall
    }


# ============================================================
# SAVE CANDIDATES
# ============================================================

def save_candidates(
    candidates: pl.DataFrame,
    rule_name: str
) -> Path:

    path = (
        TEMP_DIR /
        f"{rule_name}_candidates.parquet"
    )

    candidates.write_parquet(
        path
    )

    print(
        f"Saved candidates to: {path}"
    )

    return path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RAM-SAFE BLOCKING AUDIT")
    print("=" * 70)

    print(
        f"\nFrequency cap: {K_FREQUENCY_CAP}"
    )

    # --------------------------------------------------------
    # LOAD GT
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    # --------------------------------------------------------
    # DEFINE RULES
    # --------------------------------------------------------

    rules = [

        (
            "rare_name_token",
            ["name_tokens"],
            "name_tokens"
        ),

        (
            "name_country",
            ["name_tokens", "country"],
            "name_tokens"
        ),

        (
            "street_name_token",
            ["street_number", "name_tokens"],
            "name_tokens"
        ),

        (
            "domain_core",
            ["domain_core"],
            None
        ),

    ]

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    candidate_paths = []

    # --------------------------------------------------------
    # RUN RULES ONE AT A TIME
    # --------------------------------------------------------

    for rule_name, join_keys, explode_col in rules:

        try:

            candidates = evaluate_blocking_rule(
                rule_name=rule_name,
                join_keys=join_keys,
                explode_col=explode_col
            )

            # ----------------------------------------------
            # Calculate recall
            # ----------------------------------------------

            metrics = calculate_metrics(
                candidates,
                ground_truth
            )

            results.append({

                "RULE": rule_name,

                "CANDIDATES":
                    metrics["candidates"],

                "TRUE_PAIRS_CAUGHT":
                    metrics["caught"],

                "RECALL":
                    metrics["recall"]

            })

            # ----------------------------------------------
            # Save candidates to disk
            #
            # This is important:
            #
            # We DON'T keep every candidate dataframe
            # in RAM.
            # ----------------------------------------------

            candidate_path = save_candidates(
                candidates,
                rule_name
            )

            candidate_paths.append(
                candidate_path
            )

            # ----------------------------------------------
            # Explicitly release memory
            # ----------------------------------------------

            del candidates

        except Exception as e:

            print(
                f"\nERROR in rule "
                f"'{rule_name}':"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

    # ========================================================
    # UNION
    # ========================================================

    print("\n" + "=" * 70)
    print("CALCULATING UNION")
    print("=" * 70)

    if candidate_paths:

        # ----------------------------------------------------
        # Read candidate files one at a time
        # ----------------------------------------------------

        union = None

        for path in candidate_paths:

            print(
                f"Adding {path.name}..."
            )

            current = pl.read_parquet(
                path
            )

            if union is None:

                union = current

            else:

                union = pl.concat([
                    union,
                    current
                ])

                union = (
                    union
                    .unique(
                        subset=[
                            "S1_id",
                            "S2_id"
                        ]
                    )
                )

            del current

        # ----------------------------------------------------
        # Final union recall
        # ----------------------------------------------------

        union_metrics = calculate_metrics(
            union,
            ground_truth
        )

        results.append({

            "RULE": "UNION_ALL",

            "CANDIDATES":
                union_metrics["candidates"],

            "TRUE_PAIRS_CAUGHT":
                union_metrics["caught"],

            "RECALL":
                union_metrics["recall"]

        })

        del union

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n\n")

    print("=" * 85)
    print("BLOCKING AUDIT RESULTS")
    print("=" * 85)

    print(
        f"{'RULE':<25}"
        f"{'CANDIDATES':>18}"
        f"{'TRUE CAUGHT':>18}"
        f"{'RECALL':>15}"
    )

    print("-" * 85)

    for result in results:

        print(
            f"{result['RULE']:<25}"
            f"{result['CANDIDATES']:>18,}"
            f"{result['TRUE_PAIRS_CAUGHT']:>18,}"
            f"{result['RECALL']:>14.2%}"
        )

    print("=" * 85)

    # ========================================================
    # SAVE RESULT CSV
    # ========================================================

    results_df = pl.DataFrame(
        results
    )

    output_path = (
        PROJECT_ROOT /
        "blocking_audit_results.csv"
    )

    results_df.write_csv(
        output_path
    )

    print(
        f"\nResults saved to:"
        f"\n{output_path}"
    )

    print(
        "\nTemporary candidate files are in:"
        f"\n{TEMP_DIR}"
    )