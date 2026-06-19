"""
run_experiments.py — cluster-ready NLP XAI attribution runner.
Parallelization strategy:
  - by method  : python run_experiments.py --method IG --model distilbert
  - by folder  : python run_experiments.py --folder contraction --model distilbert
  - by masking : python run_experiments.py --method IG --masking_method pad --model roberta
Method & Masking Constraints:
  - Occlusion / Shap / LIME: Executed without masking overrides.
  - Integrated Gradients (IG): Supports all masking strategies except 'from_disk' (pad, cls, sep, mask, unk, zero).
  - Expected Gradients (EG): Restricted to 'from_disk' masking strategy (requires --from_disk_path).
Dataset Processing:
  - All folders and files (including 'base') are processed uniformly.
  - No automatic exclusion of previously completed experiments; all requested combinations are generated.
Output Path:
  results/{method}/{folder}/{filename}_{masking_method}.pkl
Example cluster usage (one job per method/masking combination):
  python run_experiments.py --method Occlusion --model distilbert
  python run_experiments.py --method LIME      --model distilbert
  python run_experiments.py --method IG        --model distilbert --masking_method zero
  python run_experiments.py --method EG        --model distilbert --from_disk_path /path/to/input_ids
  Or by folder:
  python run_experiments.py --folder contraction --model distilbert
  python run_experiments.py --folder base      --model roberta
"""

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import local XAI utilities
import xaiutils

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Constants & Configuration
MODELS: Dict[str, str] = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta":    "siebert/sentiment-roberta-large-english",
}

MASKING_METHODS: List[str] = ["pad", "cls", "sep", "mask", "unk", "zero", "from_disk"]

# Mapping enforces the strict constraint per method:
# - IG: all masking methods except 'from_disk'
# - EG: only 'from_disk'
# - Others: no masking method applied
METHOD_MASK_MAPPING: Dict[str, List[Optional[str]]] = {
    "Occlusion": [None],
    "Shap": [None],
    "LIME": [None],
    "IG": [m for m in MASKING_METHODS if m != "from_disk"],
    "EG": ["from_disk"]
}

# -----------------------------------------------------------------------------
# Argument Parsing & Validation
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XAI attribution experiments on NLP datasets."
    )
    parser.add_argument(
        "--model", 
        choices=["distilbert", "roberta"], 
        default="distilbert",
        help="Pretrained model backbone."
    )
    parser.add_argument(
        "--method", 
        choices=["Occlusion", "Shap", "LIME", "IG", "EG"], 
        default=None,
        help="Run specified XAI method across all datasets. Omit to run all methods for --folder."
    )
    parser.add_argument(
        "--masking_method", 
        choices=MASKING_METHODS,
        default=None,
        help="Masking strategy for Integrated/Expected Gradients. Required for IG/EG."
    )
    parser.add_argument(
        "--from_disk_path",
        type=str,
        default=None,
        help="Path to precomputed embeddings. Required only when masking_method='from_disk'."
    )
    parser.add_argument(
        "--folder", 
        default=None,
        help="Target dataset folder to process. Requires --method=None."
    )
    parser.add_argument(
        "--project_root", 
        default="/home/papadopu/xai-nlp",
        help="Root directory containing data/ and results/."
    )
    return parser.parse_args()

def validate_arguments(args: argparse.Namespace) -> None:
    """Enforce logical constraints before experiment generation."""
    if args.method is None and args.folder is None:
        sys.exit("Error: Specify either --method or --folder.")
    
    if args.method is not None and args.folder is not None:
        sys.exit("Error: Cannot specify both --method and --folder.")
    
    if args.method in METHOD_MASK_MAPPING:
        allowed = METHOD_MASK_MAPPING[args.method]
        if args.masking_method is not None and args.masking_method not in allowed:
            sys.exit(
                f"Error: --method='{args.method}' only supports masking_method in {allowed}."
            )
        # Auto-bind allowed masking methods if none specified
        if args.masking_method is None:
            if len(allowed) == 1:
                args.masking_method = allowed[0]
            else:
                args.masking_method = "all"  # Handled by generator
    
    if args.masking_method == "from_disk" and (args.from_disk_path is None or not Path(args.from_disk_path).exists()):
        sys.exit("Error: --from_disk_path must point to a valid directory.")

# -----------------------------------------------------------------------------
# Data & Model Loading
# -----------------------------------------------------------------------------
def load_datasets(project_root: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all CSV datasets under data/ subdirectories."""
    datasets = defaultdict(dict)
    data_dir = Path(project_root) / "data"
    
    for csv_path in data_dir.glob("**/*.csv"):
        folder = csv_path.parent.name
        name = csv_path.stem
        datasets[folder][name] = pd.read_csv(csv_path)
        
    if not datasets:
        sys.exit(f"Error: No datasets found in {data_dir}")
        
    return dict(datasets)

def load_model_and_tokenizer(model_key: str, device: torch.device):
    """Load pretrained model and tokenizer, set to evaluation mode."""
    model_name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} on {device}")
    return model, tokenizer

# -----------------------------------------------------------------------------
# Experiment Generation
# -----------------------------------------------------------------------------
def generate_experiments(
    method: Optional[str],
    masking_method: Optional[str],
    target_folder: Optional[str],
    datasets: Dict[str, Dict[str, pd.DataFrame]]
) -> List[Tuple[str, Optional[str], str, str]]:
    """
    Generates a list of (method, masking_method, folder, filename) tuples.
    Eliminates redundant conditional branching by using the constraint mapping.
    """
    target_datasets = datasets if target_folder is None else {target_folder: datasets.get(target_folder, {})}
    experiments = []
    
    for folder, files in target_datasets.items():
        methods_to_run = [method] if method is not None else ["Occlusion", "Shap", "LIME", "IG", "EG"]
        
        for m in methods_to_run:
            # Resolve masking strategy
            if m in METHOD_MASK_MAPPING:
                masks = METHOD_MASK_MAPPING[m]
                if masking_method == "all" and m in ["IG", "EG"]:
                    pass  # Use full list from mapping
                elif masking_method is not None and masking_method != "all":
                    masks = [masking_method]
                else:
                    masks = METHOD_MASK_MAPPING[m]
            else:
                masks = [None]
                
            for mask in masks:
                for filename in files:
                    experiments.append((m, mask, folder, filename))
                    
    return experiments

# -----------------------------------------------------------------------------
# XAI Utilities & Attribution Distribution
# -----------------------------------------------------------------------------
EXPLANATION_FNS = {
    "Occlusion": xaiutils.occlusion,
    "Shap":      xaiutils.shap_explanation,
    "LIME":      xaiutils.lime_explanation,
    "IG":        xaiutils.integrated_gradients_explanation,
    "EG":        xaiutils.expected_gradients_explanation,
}

def get_token_attribution_distribution(model, tokenizer, dataset, method, masking_method=None, from_disk_path=None):
    fn            = EXPLANATION_FNS[method]
    distributions = []
    for sentence in tqdm(dataset, desc=method, leave=False):
        if masking_method is not None: 
            if masking_method == "from_disk" and from_disk_path is not None:
                scores = np.array(
                    fn(tokenizer, model, sentence, masking_method=masking_method, baselines_path=from_disk_path),
                    dtype=float
                )
            else:
                scores = np.array(
                    fn(tokenizer, model, sentence, masking_method),
                    dtype=float
                )
        else:
            scores = np.array(fn(tokenizer, model, sentence), dtype=float)
        if method in ("Shap", "LIME"):
            total  = np.sum(np.abs(scores))
            scores = scores / total if total != 0 else scores
        distributions.append(scores)
    return distributions

# -----------------------------------------------------------------------------
# Attribution Computation
# -----------------------------------------------------------------------------
def compute_token_attributions(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    method: str,
    masking_method: Optional[str],
    from_disk_path: Optional[str]
) -> List[np.ndarray]:
    """
    Computes token attributions by delegating to the centralized distribution utility.
    Replaces manual routing and internal stubs with the provided pipeline logic.
    """
    sentences = df["sentence"].tolist()
    
    # Validate masking context before dispatch
    if masking_method == "from_disk" and from_disk_path is None:
        raise ValueError("from_disk_path is required for 'from_disk' masking strategy.")
        
    # Delegate to the distribution generator
    return get_token_attribution_distribution(
        model=model,
        tokenizer=tokenizer,
        dataset=sentences,
        method=method,
        masking_method=masking_method,
        from_disk_path=from_disk_path
    )

# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------
def run_experiments(
    experiments: List[Tuple[str, Optional[str], str, str]],
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    model_id: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    project_root: str,
    from_disk_path: Optional[str]
) -> None:
    results_root = Path(project_root) / "results" / MODELS.get(model_id, "default")
    if from_disk_path is not None:
        from_disk_path = Path(from_disk_path) / model_id
    
    # Build tqdm description dynamically
    has_masking = any(e[1] is not None for e in experiments)
    desc = "Masking strategy" if has_masking else "Attribution"
    pbar = tqdm(experiments, desc=f"Processing {desc}", unit="exp")
    
    for method, masking_method, folder, filename in pbar:
        df = datasets[folder][filename]
        distrib = compute_token_attributions(
            df, model, tokenizer, device, method, masking_method, from_disk_path
        )
        
        save_dir = results_root / method / folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename for safe filesystem storage
        safe_mask = masking_method if masking_method else "none"
        out_name = f"{filename}_{safe_mask}"
        out_path = save_dir / f"{out_name}.pkl"
        
        with open(out_path, "wb") as f:
            pickle.dump(distrib, f)
            
        pbar.set_postfix(saved=out_name)

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments()
    validate_arguments(args)
    
    datasets = load_datasets(args.project_root)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    
    experiment_list = generate_experiments(
        method=args.method,
        masking_method=args.masking_method,
        target_folder=args.folder,
        datasets=datasets
    )
    
    print(f"Starting {len(experiment_list)} experiments...\n")
    run_experiments(
        experiments=experiment_list,
        datasets=datasets,
        model_id=args.model,
        model=model,
        tokenizer=tokenizer,
        device=device,
        project_root=args.project_root,
        from_disk_path=args.from_disk_path
    )
    
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
