"""
Combine the three performance.csv files into a single wide CSV:
one row per dataset, with accuracy/acc_pos/acc_neg columns for each model.

Usage:
  python unify_results.py
  python unify_results.py --out combined_results.csv
"""

import argparse
import glob
import os
import pandas as pd


# Short labels for each model, used as column suffixes
MODEL_LABELS = {
    "distilbert-base-uncased-finetuned-sst-2-english": "distilbert",
    "sentiment-roberta-large-english":                 "siebert",
    "roberta-large-sst2":                              "roberta",
}


def short_label(model_name, path):
    if model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    # fallback: use folder name
    for key, label in MODEL_LABELS.items():
        if key in path:
            return label
    return model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".",
                         help="Root directory to search (default: current dir)")
    parser.add_argument("--out", default="combined_results.csv",
                         help="Output CSV path (default: combined_results.csv)")
    args = parser.parse_args()

    pattern = os.path.join(args.root, "**", "performance.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No performance.csv files found under '{args.root}'")
        return

    print(f"Found {len(files)} performance.csv files:")
    for f in files:
        print(f"  {f}")

    wide_dfs = []
    for f in files:
        df = pd.read_csv(f)

        model_name = df["model"].iloc[0]
        label = short_label(model_name, f)

        sub = df[["dataset", "accuracy", "acc_pos", "acc_neg"]].copy()
        sub = sub.rename(columns={
            "accuracy": f"accuracy_{label}",
            "acc_pos":  f"acc_pos_{label}",
            "acc_neg":  f"acc_neg_{label}",
        })
        wide_dfs.append(sub)

    # Merge all on 'dataset'
    combined = wide_dfs[0]
    for sub in wide_dfs[1:]:
        combined = combined.merge(sub, on="dataset", how="outer")

    combined.to_csv(args.out, index=False)
    print(f"\nSaved combined results -> {args.out}  ({len(combined)} rows)")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()