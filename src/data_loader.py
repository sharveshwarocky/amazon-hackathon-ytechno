from pathlib import Path
import pandas as pd


def load_training_data(data_dir):
    data_dir = Path(data_dir)

    s1 = pd.read_csv(data_dir / "train_source1.tsv", sep="\t")
    s2 = pd.read_csv(data_dir / "train_source2.tsv", sep="\t")
    s3 = pd.read_csv(data_dir / "train_source3.tsv", sep="\t")
    ground_truth = pd.read_csv(data_dir / "train_ground_truth.tsv", sep="\t")

    return {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "ground_truth": ground_truth
    }

