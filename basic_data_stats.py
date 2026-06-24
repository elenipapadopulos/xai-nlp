import csv
import logging
import statistics
import argparse
import glob
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import yaml
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def resolve_csv_paths(input_pattern: Optional[str]) -> List[Path]:
    """
    Resolves a file path, directory, or glob pattern to a sorted list of CSV files.
    Expands user paths (~) and handles recursive globs.
    """
    if not input_pattern:
        return []
    
    expanded_pattern = Path(input_pattern).expanduser()
    
    # If it's a literal directory, scan it recursively
    if expanded_pattern.is_dir():
        return sorted(expanded_pattern.rglob("*.csv"))
    
    # If it's a literal file, return it directly
    if expanded_pattern.is_file():
        return [expanded_pattern]
    
    # Handle glob patterns (e.g., /path/data/*.csv or /path/data/**/*.csv)
    try:
        matches = glob.glob(str(expanded_pattern), recursive=True)
        return sorted([Path(m) for m in matches])
    except Exception as e:
        logging.warning(f"Failed to resolve glob pattern '{input_pattern}': {e}")
        return []


def extract_sentences(csv_paths: List[Path]) -> List[str]:
    """
    Reads one or multiple CSV files, extracts the 'sentence' column,
    and returns a deduplicated list preserving insertion order.
    """
    sentences = []
    for csv_file in csv_paths:
        try:
            with open(csv_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if "sentence" not in reader.fieldnames:
                    logging.warning(f"Column 'sentence' missing in {csv_file}. Skipping.")
                    continue
                for row in reader:
                    val = row["sentence"]
                    if val is not None:
                        sentences.append(str(val).strip())
        except Exception as e:
            logging.error(f"Failed to read {csv_file}: {e}")
            continue
    return list(dict.fromkeys(sentences))


def compute_and_report_statistics(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    model_id: str
) -> Dict[str, Any]:
    """
    Tokenizes the corpus and computes token statistics.
    Returns a dictionary with the computed metrics instead of printing or saving.
    """
    logging.info(f"Tokenizing {len(sentences)} unique sentences with '{model_id}'...")
    
    # Disable truncation to preserve exact token counts for statistical analysis
    token_counts = [len(tokenizer(s, truncation=False).input_ids) for s in sentences]
    total_sentences = len(token_counts)
    total_tokens = sum(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    mean_tokens = total_tokens / total_sentences
    median_tokens = statistics.median(token_counts)
    std_dev_tokens = statistics.pstdev(token_counts) if total_sentences > 1 else 0.0

    return {
        "Total unique sentences": total_sentences,
        "Total tokens": total_tokens,
        "Min tokens per sentence": min_tokens,
        "Max tokens per sentence": max_tokens,
        "Mean tokens per sentence": f"{mean_tokens:.2f}",
        "Median tokens per sentence": f"{median_tokens:.2f}",
        "Std dev tokens per sentence": f"{std_dev_tokens:.2f}"
    }


def main() -> None:
    """
    Entry point: parses CLI arguments, loads configuration, resolves models,
    computes statistics, aggregates results, and writes final output.
    """
    parser = argparse.ArgumentParser(
        description="Compute and report tokenization statistics for a corpus of sentences."
    )
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--input_path", type=str, default=None, help="Input CSV path/pattern. Overrides config.")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory. Overrides config.")
    parser.add_argument("--model", type=str, default=None, help="Model ID or 'all'. Overrides config.")
    args = parser.parse_args()

    config_path = Path(args.config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_registry = config.get("model_registry", {})
    params = config.get("basic_data_stats", {})

    # CLI arguments take precedence over configuration file values
    input_path = args.input_path if args.input_path is not None else params.get("input_path")
    output_path = args.output_path if args.output_path is not None else params.get("output_path")
    selected_model = args.model if args.model is not None else params.get("model", "all")

    # Strict validation: both paths must be resolved before execution
    if not input_path:
        raise ValueError("input_path is not defined in CLI arguments or the configuration file.")
    if not output_path:
        raise ValueError("output_path is not defined in CLI arguments or the configuration file.")

    csv_files = resolve_csv_paths(input_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {input_path}")

    sentences = extract_sentences(csv_files)
    if not sentences:
        logging.error("No valid sentences found in the corpus. Exiting.")
        return

    output_dir = Path(output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target models based on registry or direct ID
    if selected_model == "all":
        models_to_process = list(model_registry.values())
    elif selected_model in model_registry:
        models_to_process = [model_registry[selected_model]]
    else:
        models_to_process = [selected_model]

    logging.info(f"Processing with {len(models_to_process)} model(s): {models_to_process}")

    # Aggregation dictionary: {model_id: stats_dict}
    all_results: Dict[str, Dict[str, Any]] = {}

    for model_id in models_to_process:
        logging.info(f"Loading tokenizer for model: {model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logging.error(f"Failed to load tokenizer for {model_id}: {e}")
            continue

        try:
            # Execute statistics calculation and collect partial result
            stats = compute_and_report_statistics(sentences, tokenizer, model_id)
            all_results[model_id] = stats

            # Print results for the current model
            print(f"\n--- Results for {model_id} ---")
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("---------------------------------")
        except Exception as e:
            logging.error(f"Statistics computation failed for {model_id}: {e}")
            continue
        finally:
            # Explicit memory release to prevent exhaustion during iterative processing
            del tokenizer

    # Store aggregated results in a final file using {selected_model} for the path
    if all_results:
        final_output_file = output_dir / f"{selected_model}_corpus_statistics.txt"
        
        with open(final_output_file, "w", encoding="utf-8") as f:
            f.write(f"Model Selection: {selected_model}\n\n")
            for mid, stats in all_results.items():
                f.write(f"--- {mid} ---\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        logging.info(f"Aggregated statistics successfully saved to: {final_output_file.absolute()}")
    else:
        logging.warning("No results were computed. Final output file not written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Execution failed: {e}")
