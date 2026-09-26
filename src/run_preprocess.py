"""
Runner script to preprocess business data and save output.
Usage: python run_preprocess.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from preprocess import BusinessPreprocessor, PreprocessingConfig

def load_data(data_dir):
    """Load all source data files."""
    data_dir = Path(data_dir)

    s1 = pd.read_csv(data_dir / 'train_source1.tsv', sep='\t')
    s2 = pd.read_csv(data_dir / 'train_source2.tsv', sep='\t')
    s3 = pd.read_csv(data_dir / 'train_source3.tsv', sep='\t')

    return {'S1': s1, 'S2': s2, 'S3': s3}

def preprocess_all(dataframes, config=None):
    """Preprocess all dataframes and save results."""
    preprocessor = BusinessPreprocessor(config or PreprocessingConfig())

    all_results = {}

    for source_name, df in dataframes.items():
        print(f"\n{'='*60}")
        print(f"Preprocessing {source_name} ({len(df)} records)")
        print(f"{'='*60}")

        # Preprocess
        processed = preprocessor.preprocess_dataframe(df)
        all_results[source_name] = processed

        # Print summary
        print(f"  Records processed: {len(processed)}")
        print(f"  Sample name_clean: {processed['name_clean'].head(3).tolist()}")
        print(f"  Sample legal_suffix: {processed['legal_suffix'].head(3).tolist()}")
        print(f"  Sample domain: {processed['domain'].head(3).tolist()}")

        # Save to CSV
        output_path = f'output/{source_name.lower()}_preprocessed.csv'
        os.makedirs('output', exist_ok=True)
        processed.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    return all_results

def save_combined_output(all_results, output_dir='output'):
    """Save combined preprocessed data for all sources."""
    os.makedirs(output_dir, exist_ok=True)

    # Combine all sources
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(f'{output_dir}/all_sources_preprocessed.csv', index=False)
    print(f"\nCombined output saved to: {output_dir}/all_sources_preprocessed.csv")
    print(f"Total records: {len(combined)}")

    # Print feature summary
    print(f"\n{'='*60}")
    print("FEATURE SUMMARY")
    print(f"{'='*60}")
    print(f"Legal suffixes found: {combined['legal_suffix'].value_counts().head(10).to_dict()}")
    print(f"Domains found: {combined['domain'].notna().sum()}")
    print(f"FKA detected: {combined['has_fka'].sum()}")
    print(f"DBA detected: {combined['has_dba'].sum()}")
    print(f"AKA detected: {combined['has_aka'].sum()}")

if __name__ == '__main__':
    # Load data
    print("Loading training data...")
    # Get the parent directory (project root) and construct the data path
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'train'
    data = load_data(data_dir)

    # Preprocess all sources
    print("\nPreprocessing all sources...")
    results = preprocess_all(data)

    # Save combined output
    save_combined_output(results)

    print("\n✅ Preprocessing complete!")
