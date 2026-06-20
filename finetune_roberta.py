"""
Fine-tune roberta-large on SST-2 and/or IMDb for 3 epochs each.
Supports multi-GPU via DataParallel automatically.

Saves to:
  ./roberta-large-sst2/   (SST-2)
  ./roberta-large-imdb/   (IMDb)

Usage:
  python finetune_roberta.py                  # both datasets
  python finetune_roberta.py --dataset sst2
  python finetune_roberta.py --dataset imdb
"""

import os, time, json, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "roberta-large"
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.06
SEED         = 42

# Per-dataset settings: (hf_path, hf_name, text_col, max_len, batch_size, output_dir)
DATASET_CONFIGS = {
    "sst2": {
        "hf_path":    "glue",
        "hf_name":    "sst2",
        "text_col":   "sentence",
        "val_split":  "validation",
        "max_len":    128,        # SST-2 sentences are short
        "batch_size": 32,
        "output_dir": "./roberta-large-sst2",
    },
    "imdb": {
        "hf_path":    "imdb",
        "hf_name":    None,
        "text_col":   "text",
        "val_split":  "test",     # IMDb has no dedicated val split; use test
        "max_len":    512,        # IMDb reviews are long
        "batch_size": 8,          # smaller batch due to longer sequences
        "output_dir": "./roberta-large-imdb",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device_and_model(model_name):
    """Load model, wrap in DataParallel if multiple GPUs are available."""
    n_gpu  = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu > 0 else "cpu")
    print(f"  Device : {device}  |  GPUs available: {n_gpu}")

    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    if n_gpu > 1:
        print(f"  Wrapping model in DataParallel across {n_gpu} GPUs")
        model = nn.DataParallel(model)

    return model, device, n_gpu


def build_loaders(cfg, tokenizer):
    """Load, tokenise, and return train/val DataLoaders."""
    print(f"  Loading dataset '{cfg['hf_path']}' …")
    if cfg["hf_name"]:
        raw = load_dataset(cfg["hf_path"], cfg["hf_name"])
    else:
        raw = load_dataset(cfg["hf_path"])

    train_ds = raw["train"]
    val_ds   = raw[cfg["val_split"]]
    print(f"  train: {len(train_ds):,}   val: {len(val_ds):,}")

    text_col = cfg["text_col"]
    max_len  = cfg["max_len"]

    def tokenise(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    train_ds = train_ds.map(tokenise, batched=True, desc="Tokenising train")
    val_ds   = val_ds.map(tokenise,   batched=True, desc="Tokenising val")

    for ds in (train_ds, val_ds):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Scale batch size by number of GPUs
    n_gpu      = torch.cuda.device_count()
    batch_size = cfg["batch_size"] * max(1, n_gpu)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def train_one_dataset(dataset_key, tokenizer):
    cfg = DATASET_CONFIGS[dataset_key]
    print(f"\n{'='*60}")
    print(f"  Fine-tuning roberta-large  →  {dataset_key.upper()}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    model, device, n_gpu = get_device_and_model(MODEL_NAME)
    train_loader, val_loader = build_loaders(cfg, tokenizer)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # Access parameters through .module if DataParallel
    params    = model.module.parameters() if n_gpu > 1 else model.parameters()
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total_loss = time.time(), 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # DataParallel returns per-GPU losses; take the mean
            loss = outputs.loss.mean() if n_gpu > 1 else outputs.loss

            optimizer.zero_grad()
            loss.backward()
            # Clip on the right parameter set
            params_to_clip = model.module.parameters() if n_gpu > 1 else model.parameters()
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0 or step == len(train_loader):
                print(f"  [{dataset_key}] Epoch {epoch} | step {step}/{len(train_loader)} "
                      f"| loss {total_loss/step:.4f} | {time.time()-t0:.0f}s")

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds  = logits.argmax(dim=-1).cpu()

                all_preds.extend(preds.tolist())
                all_labels.extend(batch["label"].tolist())

        acc      = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        print(f"\n  [{dataset_key}] Epoch {epoch}/{EPOCHS}  "
              f"train_loss={avg_loss:.4f}  val_acc={acc:.4f}  "
              f"time={time.time()-t0:.0f}s\n")
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_acc": acc})

    # ── Save ──────────────────────────────────────────────────────────────
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    save_model = model.module if n_gpu > 1 else model
    save_model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    with open(os.path.join(out, "training_history.json"), "w") as f:
        json.dump({"dataset": dataset_key, "epochs": EPOCHS, "history": history}, f, indent=2)

    print(f"  Model saved → {out}")
    return history


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sst2", "imdb", "both"], default="both",
                        help="Which dataset to fine-tune on (default: both)")
    args = parser.parse_args()

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    targets   = ["sst2", "imdb"] if args.dataset == "both" else [args.dataset]

    all_results = {}
    for ds in targets:
        all_results[ds] = train_one_dataset(ds, tokenizer)

    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for ds, history in all_results.items():
        best = max(history, key=lambda x: x["val_acc"])
        print(f"  {ds.upper():6s}  best val_acc={best['val_acc']:.4f}  (epoch {best['epoch']})"
              f"  →  {DATASET_CONFIGS[ds]['output_dir']}")