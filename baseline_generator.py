"""
Generates random token sequences near the classification decision boundary.
"""
import argparse
import random
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for model selection, device, and generation constraints."""
    parser = argparse.ArgumentParser(
        description="Generate random token sequences near the classifier decision boundary."
    )
    parser.add_argument(
        "--model", type=str, default="roberta", choices=["distilbert", "roberta"],
        help="Model identifier from the HuggingFace registry."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device. Examples: cpu, cuda:0, cuda:3. Use auto for detection."
    )
    parser.add_argument(
        "--target", type=int, default=200,
        help="Number of valid sequences to collect per intermediate length (5-10)."
    )
    parser.add_argument(
        "--max_attempts", type=int, default=50000,
        help="Maximum sampling iterations per length before exit."
    )
    parser.add_argument(
        "--prob_min", type=float, default=0.45,
        help="Lower bound for class-0 probability."
    )
    parser.add_argument(
        "--prob_max", type=float, default=0.55,
        help="Upper bound for class-0 probability."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./baselines",
        help="Base directory for persisting generated sequences."
    )
    parser.add_argument(
        "--min_length", type=int, default=5,
        help="Minimum number of intermediate tokens to generate."
    )
    parser.add_argument(
        "--max_length", type=int, default=20,
        help="Maximum number of intermediate tokens to generate."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def initialize_determinism(seed: int) -> None:
    """Set random seeds across all relevant libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_compute_device(device_str: str) -> torch.device:
    """Determine the optimal compute device with graceful fallback on failure."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device_str)
    except RuntimeError:
        import warnings
        warnings.warn(f"Requested device '{device_str}' is unavailable. Falling back to auto-detection.")
        return resolve_compute_device("auto")

def load_model_resources(model_id: str, device: torch.device):
    """Load tokenizer and model, set to evaluation mode, and transfer to target device."""
    print(f"Loading model resources: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()  # Disables dropout and batch normalization updates
    return tokenizer, model

def get_special_token_ids(tokenizer: AutoTokenizer, model_name: str) -> Tuple[int, int]:
    """Retrieve class and separator token IDs with explicit fallback for tokenizer compatibility."""
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else (0 if model_name == "roberta" else 101)
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else (2 if model_name == "roberta" else 102)
    return cls_id, sep_id

def _is_contiguous_subsequence(lst: list, sub: list) -> bool:
    """Check if 'sub' appears as a contiguous block inside 'lst'."""
    if not sub:
        return True
    n, m = len(lst), len(sub)
    if m > n:
        return False
    for i in range(n - m + 1):
        if lst[i:i+m] == sub:
            return True
    return False

def _find_first_subsequence_index(lst: list, sub: list) -> int:
    """Return the starting index of the first contiguous occurrence of 'sub' in 'lst', or -1."""
    if not sub:
        return -1
    n, m = len(lst), len(sub)
    for i in range(n - m + 1):
        if lst[i:i+m] == sub:
            return i
    return -1

def sanitize_forbidden_patterns(mid_tokens: torch.Tensor, forbidden_sequences: list, vocab_size: int, device: torch.device) -> torch.Tensor:
    """
    Replace contiguous forbidden token sequences with random tokens of equivalent length.
    Operates directly on token IDs to correctly handle model-specific tokenization variations.
    """
    ids = mid_tokens.squeeze(0).tolist()
    
    for seq in forbidden_sequences:
        if not seq:
            continue
            
        max_replacements = 100  # Safety bound to prevent infinite loops
        
        for _ in range(max_replacements):
            idx = _find_first_subsequence_index(ids, seq)
            if idx == -1:
                break
                
            replacement = torch.randint(0, vocab_size, (len(seq),), device=device).tolist()
            ids[idx:idx + len(replacement)] = replacement
            
    return torch.tensor([ids], device=device)

def load_existing_sequences(base_dir: Path, length: int) -> set:
    """
    Load all previously saved input_ids for a specific length into an in-memory set.
    Enables O(1) uniqueness checks during sampling and ensures persistence across runs.
    """
    length_dir = base_dir / f"{length}_tokens"
    if not length_dir.exists():
        return set()
        
    unique_sequences = set()
    for item in length_dir.iterdir():
        if item.is_dir() and (item / "input_ids.pt").is_file():
            try:
                seq_data = torch.load(item / "input_ids.pt", map_location="cpu")
                unique_sequences.add(tuple(seq_data[0].tolist()))
            except Exception:
                continue
    return unique_sequences

def main() -> None:
    args = parse_arguments()
    initialize_determinism(args.seed)
    device = resolve_compute_device(args.device)
    MODEL_REGISTRY = {
        "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
        "roberta": "siebert/sentiment-roberta-large-english"
    }
    model_id = MODEL_REGISTRY[args.model]
    tokenizer, model = load_model_resources(model_id, device)
    cls_id, sep_id = get_special_token_ids(tokenizer, args.model)
    vocab_size = tokenizer.vocab_size
    
    # Precompute forbidden token ID sequences for robust, tokenizer-agnostic matching
    excluded_keywords = {"not", "wasn't"}
    forbidden_sequences = [
        tokenizer.encode(word, add_special_tokens=False) 
        for word in excluded_keywords
    ]
    
    model_output_dir = Path(args.output_dir) / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting generation pipeline. Target: {args.target} sequences per length.")
    print(f"Device: {device} | Model: {args.model} | Max attempts per length: {args.max_attempts}")
    print(f"Constraints: probability range [{args.prob_min}, {args.prob_max}] | Length range [{args.min_length}, {args.max_length}] | Forbidden tokens sanitized in-place.\n")

    
    # Iterate over fixed intermediate lengths: 5-20
    for length in range(args.min_length, args.max_length + 1):
        existing_sequences = load_existing_sequences(model_output_dir, length)
        start_count = len(existing_sequences)
        target_remaining = max(0, args.target - start_count)
        
        print(f"Processing length: {length} tokens. Target: {args.target} (remaining: {target_remaining})")
        print(f"Initialized uniqueness cache with {start_count} existing sequences for length {length}.\n")
        
        if target_remaining <= 0:
            print(f"Target already reached for length {length}. Skipping generation.\n")
            continue
        
        sampling_attempts = 0
        collected_count = 0
        progress_tracker = tqdm(total=args.max_attempts, desc=f"Length {length}", leave=False)
        
        with torch.no_grad():
            while collected_count < target_remaining and sampling_attempts < args.max_attempts:
                sampling_attempts += 1
                
                # Generate mid tokens directly on target device to avoid host-device transfer
                mid_tokens = torch.randint(0, vocab_size, (1, length), device=device)
                # Enforce constraint: replace forbidden token sequences if detected
                mid_tokens = sanitize_forbidden_patterns(mid_tokens, forbidden_sequences, vocab_size, device)
                
                # Construct full sequence: [CLS] + mid_tokens + [SEP]
                input_ids = torch.cat([
                    torch.tensor([[cls_id]], device=device),
                    mid_tokens,
                    torch.tensor([[sep_id]], device=device)
                ], dim=1)
                attention_mask = torch.ones_like(input_ids)
                
                # Forward pass through classifier
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                prob_0, prob_1 = probs[0].item(), probs[1].item()
                
                # Accept only sequences within the decision boundary window
                if args.prob_min <= prob_0 <= args.prob_max:
                    candidate_ids = tuple(input_ids[0].tolist())
                    
                    if candidate_ids not in existing_sequences:
                        decoded_text = tokenizer.decode(input_ids.squeeze(0).tolist(), skip_special_tokens=True).lower()
                        prob_0_label = f"{prob_0:.5f}".replace(".", "")[:5]
                        prob_1_label = f"{prob_1:.5f}".replace(".", "")[:5]
                        
                        current_index = start_count + collected_count
                        seq_id = f"seq_{current_index:05d}"
                        save_dir = model_output_dir / f"{length}_tokens" / f"{seq_id}_{prob_0_label}_{prob_1_label}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Persist tensors and text
                        torch.save(input_ids, save_dir / "input_ids.pt")
                        torch.save(attention_mask, save_dir / "attention_mask.pt")
                        (save_dir / "sequence.txt").write_text(decoded_text, encoding="utf-8")
                        
                        # Update in-memory cache to maintain consistency for subsequent checks
                        existing_sequences.add(candidate_ids)
                        collected_count += 1
                        progress_tracker.set_postfix({"saved": collected_count})
                
                # Progress bar advances exactly once per iteration
                progress_tracker.update(1)
                
        progress_tracker.close()
        print(f"Length {length} complete. {collected_count}/{target_remaining} unique sequences saved.\n")
        
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
