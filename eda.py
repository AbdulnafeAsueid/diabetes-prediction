"""Exploratory data analysis: summary statistics and figures saved to outputs/eda/.

    python eda.py [--data PATH] [--out outputs/eda]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from diabetes.data import CLASS_NAMES, DEFAULT_DATA_PATH, FEATURES, TARGET, load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default="outputs/eda")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    print(f"shape: {df.shape}")
    print(f"missing values: {int(df.isnull().sum().sum())}")
    counts = df[TARGET].value_counts().sort_index()
    for cls, n in counts.items():
        print(f"class {cls} ({CLASS_NAMES[cls]}): {n:,} ({n / len(df):.2%})")
    df.describe().T.to_csv(out / "summary_statistics.csv")

    sns.set_style("whitegrid")

    df[FEATURES].hist(bins=20, figsize=(14, 13))
    plt.tight_layout()
    plt.savefig(out / "feature_histograms.png", dpi=110)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.countplot(x=df[TARGET], hue=df[TARGET], palette="viridis", legend=False)
    plt.title("Distribution of diabetes classes")
    plt.xlabel("Class (0: No Diabetes, 1: Prediabetes, 2: Diabetes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=110)
    plt.close()

    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, annot_kws={"size": 7})
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(out / "correlation_matrix.png", dpi=110)
    plt.close()

    print(f"figures and summary saved to {out}")


if __name__ == "__main__":
    main()
