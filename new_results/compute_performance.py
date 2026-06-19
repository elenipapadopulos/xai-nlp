"""
Compute performance.csv for each model folder under new_results/, based on
the existing prediction CSVs in <model>/predictions/<config_folder>/<dataset>.csv.

Each prediction CSV is expected to have at least these columns:
    sentence, label, prediction

For each prediction file, this script computes:
    accuracy      -> overall accuracy (prediction == label)
    acc_pos       -> accuracy restricted to rows where label == "positive"
    acc_neg       -> accuracy restricted to rows where label == "negative"

and writes one row per file into <model>/performance.csv, with columns:
    model, folder, dataset, accuracy, acc_pos, acc_neg

Usage:
    python compute_performance.py
    python compute_performance.py --root new_results
    python compute_performance.py --root new_results --models distilbert-base-uncased-finetuned-sst-2-english siebert
"""

import argparse
import glob
import os

import pandas as pd


def compute_accuracy(df, label_value=None):
    """Compute accuracy (prediction == label), optionally restricted to a given label value."""
    subset = df if label_value is None else df[df["label"] == label_value]
    if len(subset) == 0:
        return float("nan")
    return (subset["prediction"] == subset["label"]).mean()


def compute_performance_for_model(model_dir):
    """
    Scan <model_dir>/predictions/**/*.csv and compute performance metrics
    for each prediction file. Returns a DataFrame with one row per file.
    """
    predictions_root = os.path.join(model_dir, "predictions")
    pattern = os.path.join(predictions_root, "**", "*.csv")
    files = sorted(glob.glob(pattern, recursive=True))

    model_name = os.path.basename(os.path.normpath(model_dir))
    records = []

    for path in files:
        rel = os.path.relpath(path, predictions_root)
        folder = os.path.dirname(rel)              # e.g. "aux_not", "base", "not_start"
        dataset = os.path.splitext(os.path.basename(rel))[0]  # e.g. "sentence_not_0_aux"

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  SKIP (failed to read): {path} ({e})")
            continue

        missing = {"sentence", "label", "prediction"} - set(df.columns)
        if missing:
            print(f"  SKIP (missing columns {missing}): {path}")
            continue

        accuracy = compute_accuracy(df)
        acc_pos = compute_accuracy(df, "positive")
        acc_neg = compute_accuracy(df, "negative")

        records.append({
            "model":    model_name,
            "folder":   folder,
            "dataset":  dataset,
            "accuracy": accuracy,
            "acc_pos":  acc_pos,
            "acc_neg":  acc_neg,
        })

    return pd.DataFrame(records, columns=["model", "folder", "dataset", "accuracy", "acc_pos", "acc_neg"])


def find_model_dirs(root, model_filter=None):
    """
    Find model directories under root that contain a 'predictions' subfolder.
    Handles nested model names (e.g. 'siebert/sentiment-roberta-large-english').
    """
    model_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        if "predictions" in dirnames:
            model_dirs.append(dirpath)
            # don't descend into this model dir further looking for nested "predictions"
            dirnames[:] = [d for d in dirnames if d != "predictions"]

    if model_filter:
        model_dirs = [
            d for d in model_dirs
            if any(mf in d for mf in model_filter)
        ]

    return sorted(model_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="new_results",
                         help="Root directory to search (default: new_results)")
    parser.add_argument("--models", nargs="*", default=None,
                         help="Optional list of model folder name substrings to restrict to")
    args = parser.parse_args()

    model_dirs = find_model_dirs(args.root, args.models)

    if not model_dirs:
        print(f"No model folders with a 'predictions' subfolder found under '{args.root}'")
        return

    print(f"Found {len(model_dirs)} model folder(s):")
    for d in model_dirs:
        print(f"  - {d}")

    for model_dir in model_dirs:
        print(f"\nProcessing: {model_dir}")
        df = compute_performance_for_model(model_dir)

        if df.empty:
            print("  No valid prediction files found, skipping performance.csv write.")
            continue

        out_path = os.path.join(model_dir, "performance.csv")
        df.to_csv(out_path, index=False)
        print(f"  Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()