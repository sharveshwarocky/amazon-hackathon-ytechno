import pandas as pd
from pathlib import Path

output_dir = Path(__file__).resolve().parent / "output"

country_map = {
    "india": 0,
    "us": 1,
    "usa": 1,
    "united states": 1,
    "france": 2,
}

for file in output_dir.glob("*.csv"):
    df = pd.read_csv(file)

    if "country" in df.columns:
        # Keep already-correct country codes (0, 1, 2) unchanged
        def convert_country(value):
            if pd.isna(value):
                return value

            value_str = str(value).strip().casefold()

            if value_str in {"0", "1", "2"}:
                return int(value_str)

            return country_map.get(value_str, value)

        df["country"] = df["country"].apply(convert_country)

        df.to_csv(file, index=False)
        print(f"Updated: {file}")

    else:
        print(f"Skipped (no country column): {file}")
