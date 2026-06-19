"""
run_experiments.py — cluster-ready experiment runner.

Parallelization strategy:
  - by method : python run_experiments.py --method Occlusion
  - by folder  : python run_experiments.py --folder contraction

Experiments already done (excluded automatically):
  ("LIME",      "base", "sentence_not_3")
  ("Occlusion", "base", "sentence_not_2")
  ("Shap",      "base", "sentence_not_2")

IG is excluded from all runs.
'base' folder is excluded except for the 3 experiments listed in BASE_TODO.

Example cluster usage (one job per method):
  python run_experiments.py --method Occlusion --model distilbert
  python run_experiments.py --method Shap      --model distilbert
  python run_experiments.py --method LIME      --model distilbert

  Or by folder:
  python run_experiments.py --folder contraction --model distilbert
  python run_experiments.py --folder aux_not     --model distilbert
  python run_experiments.py --folder not_at      --model distilbert
"""

import argparse
import glob
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import shap
from lime.lime_text import LimeTextExplainer

# ── Configuration ──────────────────────────────────────────────────────────────

MAX_LENGTH = 512
MAX_EVALS  = 100
np.random.seed(42)

MODELS = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta":    "siebert/sentiment-roberta-large-english",
    "roberta-sst2": "roberta-large-sst2"
}

# # base experiments still to be done — all other base combinations are excluded
# BASE_TODO = {
#     ("LIME",      "base", "sentence_not_3"),
#     ("Occlusion", "base", "sentence_not_2"),
#     ("Shap",      "base", "sentence_not_2"),
# }

# ── Utils ──────────────────────────────────────────────────────────────────────

def predict_fn(tokenizer, model, texts):
    inputs = tokenizer(
        [str(t) for t in texts],
        return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1).cpu().numpy()


def occlusion(tokenizer, model, text, masking="Mask"):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    ).to(model.device)
    input_ids      = inputs.input_ids[0]
    original_probs = predict_fn(tokenizer, model, [text])
    predicted_idx  = np.argmax(original_probs)
    mask_token     = tokenizer.mask_token_id if masking == "Mask" else tokenizer.pad_token_id

    variations = []
    for i in range(1, input_ids.shape[0] - 1):
        masked_tokens       = inputs.input_ids.clone()
        masked_tokens[0][i] = mask_token
        masked_sentence     = tokenizer.decode(masked_tokens[0], skip_special_tokens=True)
        probs               = predict_fn(tokenizer, model, [masked_sentence])
        delta               = abs(original_probs[0][predicted_idx] - probs[0][predicted_idx])
        variations.append(delta)

    return variations / np.sum(variations)


def shap_explanation(tokenizer, model, instance):
    masker       = shap.maskers.Text(tokenizer)
    _predict     = lambda x: predict_fn(tokenizer, model, x)
    explainer_sh = shap.Explainer(_predict, masker=masker, output_names=["NEG", "POS"])
    shap_vals    = explainer_sh([instance], max_evals=MAX_EVALS)
    predicted_class = np.argmax(predict_fn(tokenizer, model, [instance]))
    shap_values     = shap_vals.values.squeeze()[:, predicted_class]
    return np.abs(shap_values[1:len(shap_vals[0]) - 1])


def lime_explanation(tokenizer, model, instance):
    explainer = LimeTextExplainer(class_names=["neg", "pos"], bow=False)
    _predict  = lambda x: predict_fn(tokenizer, model, x)
    exp = explainer.explain_instance(
        instance, _predict,
        num_features=1000,
        num_samples=2000,
    )
    exp_map           = exp.as_map()
    pairs             = exp_map[1]
    indexed_string    = exp.domain_mapper.indexed_string
    importance_vector = np.zeros(indexed_string.num_words())
    for idx, val in pairs:
        importance_vector[idx] = val
    return np.abs(importance_vector)


EXPLANATION_FNS = {
    "Occlusion": occlusion,
    "Shap":      shap_explanation,
    "LIME":      lime_explanation,
}


def get_token_attribution_distribution(model, tokenizer, dataset, method):
    fn            = EXPLANATION_FNS[method]
    distributions = []
    for sentence in tqdm(dataset, desc=method):
        scores = np.array(fn(tokenizer, model, sentence), dtype=float)
        if method in ("Shap", "LIME"):
            total  = np.sum(np.abs(scores))
            scores = scores / total if total != 0 else scores
        distributions.append(scores)
    return distributions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model",  choices=["distilbert", "roberta", "roberta-sst2"], default="distilbert")
    parser.add_argument("--method", choices=["Occlusion", "Shap", "LIME"], default=None,
                        # IG excluded
                        help="Run all folders for this method (excludes base if already done)")
    parser.add_argument("--folder", default=None,
                        help="Run all methods for this folder (excludes base, excludes IG)")
    parser.add_argument("--project_root", default="/home/papadopu/xai-nlp")
    args = parser.parse_args()

    if args.method is None and args.folder is None:
        print("Specify --method or --folder.")
        sys.exit(1)

    PROJECT_ROOT = Path(args.project_root)
    RESULTS_ROOT = PROJECT_ROOT / "new_results" / MODELS[args.model]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Load datasets
    datasets = defaultdict(dict)
    for path in glob.glob(str(PROJECT_ROOT / "data" / "*" / "*.csv")):
        folder   = path.split("/")[-2]
        filename = path.split("/")[-1].replace(".csv", "")
        datasets[folder][filename] = pd.read_csv(path)

    # Load model
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = MODELS[args.model]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Model : {model_name}")
    print(f"Device: {device}")

    # Build experiment list
    def include(method, folder, filename):
        # if folder == "base":
        #     return (method, folder, filename) in BASE_TODO  # only the 3 exceptions
        return True

    methods = [args.method] if args.method else ["Occlusion", "Shap", "LIME"]
    folders = [args.folder] if args.folder else list(datasets.keys())

    experiments = [
        (method, folder, filename)
        for method in methods
        for folder in folders
        for filename in datasets.get(folder, {})
        if include(method, folder, filename)
    ]

    print(f"Running {len(experiments)} experiments:")
    for e in experiments:
        print(f"  {e}")

    # Performance tracking
    perf_records = []
    perf_csv     = RESULTS_ROOT / "performance.csv"

    def compute_accuracy(df, sentences, folder, filename):
        labels      = df["label"].tolist()
        probs       = predict_fn(tokenizer, model, list(sentences))
        predictions = ["positive" if np.argmax(p) == 1 else "negative" for p in probs]

        # save predictions
        pred_dir = RESULTS_ROOT / "predictions" / folder
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "sentence":   list(sentences),
            "label":      labels,
            "prediction": predictions,
            "prob_neg":   [p[0] for p in probs],
            "prob_pos":   [p[1] for p in probs],
        }).to_csv(pred_dir / f"{filename}.csv", index=False)

        correct  = [p == l for p, l in zip(predictions, labels)]
        acc      = sum(correct) / len(correct)
        pos_mask = [l == "positive" for l in labels]
        neg_mask = [l == "negative" for l in labels]
        acc_pos  = sum(c for c, m in zip(correct, pos_mask) if m) / sum(pos_mask) if sum(pos_mask) > 0 else None
        acc_neg  = sum(c for c, m in zip(correct, neg_mask) if m) / sum(neg_mask) if sum(neg_mask) > 0 else None

        return acc, acc_pos, acc_neg

    # Run
    seen_datasets = set()  # avoid computing accuracy multiple times for same dataset
    for method, folder, filename in tqdm(experiments, desc="experiments"):
        df        = datasets[folder][filename]
        sentences = df["sentence"]
        distrib   = get_token_attribution_distribution(model, tokenizer, sentences, method=method)

        save_dir = RESULTS_ROOT / method / folder
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / f"{filename}.pkl", "wb") as f:
            pickle.dump(distrib, f)

        print(f"  Saved: {save_dir / filename}.pkl")

        # Compute accuracy once per (folder, filename) regardless of method
        if (folder, filename) not in seen_datasets:
            seen_datasets.add((folder, filename))
            acc, acc_pos, acc_neg = compute_accuracy(df, sentences, folder, filename)
            perf_records.append({
                "model":    model_name,
                "folder":   folder,
                "dataset":  filename,
                "accuracy": acc,
                "acc_pos":  acc_pos,
                "acc_neg":  acc_neg,
            })
            print(f"  Accuracy: {acc:.3f} | pos: {acc_pos:.3f} | neg: {acc_neg:.3f}")

    # Save performance CSV (append if exists)
    if perf_records:
        df_perf = pd.DataFrame(perf_records)
        if perf_csv.exists():
            df_existing = pd.read_csv(perf_csv)
            df_perf     = pd.concat([df_existing, df_perf], ignore_index=True).drop_duplicates(
                subset=["model", "folder", "dataset"]
            )
        df_perf.to_csv(perf_csv, index=False)
        print(f"\nPerformance saved to {perf_csv}")


if __name__ == "__main__":
    main()