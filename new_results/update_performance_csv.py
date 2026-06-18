"""
Update existing performance.csv files: recompute acc_pos / acc_neg so they
reflect the ORIGINAL ADJECTIVE SENTIMENT rather than the final sentence label.

Logic:
  n_not even -> adjective sentiment == final label -> acc_pos/acc_neg unchanged
  n_not odd  -> adjective sentiment == opposite of final label -> SWAP acc_pos and acc_neg

Old columns are preserved as acc_pos_label / acc_neg_label for reference.
New acc_pos / acc_neg refer to the original adjective sentiment.

Usage:
  python update_performance_csvs.py
  python update_performance_csvs.py --root new_results
"""

import argparse
import glob
import os
import re

import pandas as pd


def extract_n_not(dataset_name):
    """
    sentence_not_0             -> 0
    sentence_not_1             -> 1
    sentence_not_2             -> 2
    sentence_not_3             -> 3
    sentence_not_0_aux         -> 0
    sentence_not_1_aux         -> 1
    sentence_not_1_contraction -> 1
    sentence_not_2_contraction -> 2
    sentence_not_beginning     -> 1  (special case, no digit)
    """
    if "beginning" in dataset_name:
        return 1

    match = re.search(r"sentence_not_(\d+)", dataset_name)
    return int(match.group(1)) if match else 0


def update_file(path):
    df = pd.read_csv(path)

    if "acc_pos" not in df.columns or "acc_neg" not in df.columns:
        print(f"  SKIP (no acc_pos/acc_neg columns): {path}")
        return False

    # Preserve originals (only if not already done previously)
    if "acc_pos_label" not in df.columns:
        df["acc_pos_label"] = df["acc_pos"]
        df["acc_neg_label"] = df["acc_neg"]

    n_not = df["dataset"].apply(extract_n_not)
    flip  = (n_not % 2 == 1)

    new_acc_pos = df["acc_pos_label"].where(~flip, df["acc_neg_label"])
    new_acc_neg = df["acc_neg_label"].where(~flip, df["acc_pos_label"])

    df["acc_pos"] = new_acc_pos
    df["acc_neg"] = new_acc_neg

    df.to_csv(path, index=False)
    print(f"  Updated: {path}  ({flip.sum()} rows flipped)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="new_results",
                         help="Root directory to search (default: new_results)")
    args = parser.parse_args()

    pattern = os.path.join(args.root, "**", "performance.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No performance.csv files found under '{args.root}'")
        return

    print(f"Found {len(files)} performance.csv files")
    updated = 0
    for f in files:
        if update_file(f):
            updated += 1

    print(f"\nDone. Updated {updated}/{len(files)} files.")


if __name__ == "__main__":
    main()