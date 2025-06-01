import json
import logging
import os
import random
import hashlib
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import pandas as pd

# Configuration
CONFIG = {
    "paths": {
        "input_jsonl": Path("F:/PythonDataset/04metrics/code_metrics.jsonl"),
        "output_ml_csv": Path("F:/PythonDataset/06preprocessed/balanced_ml.csv"),
        "output_llm_jsonl": Path("F:/PythonDataset/06preprocessed/balanced_llm.jsonl"),
        "log_file": Path("F:/PythonDataset/06preprocessed/preprocessing.log")
    },
    "sample_size": 1500,  # Target samples per smell-severity pair (no oversampling, ~10,000 total)
    "none_smell_target": 1500,  # Target for NoneSmellorUnknown samples (same as smelly pairs)
    "smells": ["LargeClass", "LongMethod", "LongParameterList", "DeepNesting"],
    "severities": [1, 2, 3],
    "numerical_metrics": [
        "LOC", "NOM", "LOC_method", "CYCLO_method", "NOP_method", "NEST",
        "TOKEN_COUNT", "LENGTH", "FAN_IN", "FAN_OUT", "ATTR_COUNT", "INHERIT_DEPTH"
    ]
}

os.makedirs(CONFIG["paths"]["output_ml_csv"].parent, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(CONFIG["paths"]["log_file"], mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_and_balance(input_path: Path, sample_size: int, none_smell_target: int, smells: List[str], severities: List[int]) -> List[Dict]:
    """Load JSONL, deduplicate by smell_type, severity, and smell_code, and balance to target sizes without oversampling."""
    unique_samples = {}
    total_samples = 0
    smell_severity_counts = defaultdict(int)  # Track counts before deduplication

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    smell_type = entry.get("smell_type", "")
                    severity = entry.get("severity", 0)
                    smell_code = entry.get("smell_code", "")
                    if (smell_type in smells and severity in severities) or (smell_type == "NoneSmellorUnknown" and severity == 0):
                        smell_severity_counts[(smell_type, severity)] += 1
                        # Use composite key for deduplication
                        key = hashlib.md5(f"{smell_type}|{severity}|{smell_code}".encode()).hexdigest()
                        if key not in unique_samples:
                            unique_samples[key] = entry
                        total_samples += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON line: {e}")
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
    except FileNotFoundError:
        logger.error(f"Input file {input_path} not found")
        return []

    # Log counts before deduplication
    for (smell, severity), count in smell_severity_counts.items():
        logger.info(f"Before deduplication: {smell} S{severity} has {count} samples")

    all_samples = list(unique_samples.values())
    logger.info(f"Total unique samples loaded: {len(all_samples)} (from {total_samples} total entries)")
    random.shuffle(all_samples)

    smell_groups = defaultdict(list)
    for sample in all_samples:
        key = (sample["smell_type"], sample["severity"])
        smell_groups[key].append(sample)

    balanced_samples = []
    for key in smell_groups.keys():
        smell, severity = key
        group = smell_groups[key]
        count = len(group)
        logger.info(f"Found {count} unique samples for {smell} S{severity}")
        if count == 0:
            continue

        # Apply the same target size to all pairs, using only available unique samples (no oversampling)
        sampled_size = none_smell_target if smell == "NoneSmellorUnknown" and severity == 0 else sample_size
        sampled_size = min(sampled_size, count)  # Never exceed the number of unique samples
        sampled = random.sample(group, sampled_size)  # Use random.sample to select without replacement
        logger.info(f"Balanced {smell} S{severity} to {len(sampled)} samples (unique count: {count})")
        balanced_samples.extend(sampled)

    random.shuffle(balanced_samples)
    logger.info(f"Total balanced samples: {len(balanced_samples)}")
    return balanced_samples

def split_ml_samples(samples: List[Dict], numerical_metrics: List[str]) -> List[Dict]:
    """Split samples into ML-compatible format with numerical metrics."""
    ml_data = []
    for sample in samples:
        try:
            metrics = sample.get("metrics", {})
            ml_entry = {
                "smell_type": sample.get("smell_type", ""),
                "severity": sample.get("severity", 0)
            }
            for metric in numerical_metrics:
                ml_entry[metric] = metrics.get(metric, 0.0) if metrics.get(metric) is not None else 0.0
            ml_data.append(ml_entry)
        except Exception as e:
            logger.error(f"Error processing ML sample: {e}")
    return ml_data

def split_llm_samples(samples: List[Dict], numerical_metrics: List[str]) -> List[Dict]:
    """Split samples into LLM-compatible format with numerical metrics and smell_code."""
    llm_data = []
    for sample in samples:
        try:
            metrics = sample.get("metrics", {})
            llm_entry = {
                "smell_type": sample.get("smell_type", ""),
                "severity": sample.get("severity", 0),
                "smell_code": sample.get("smell_code", "")
            }
            for metric in numerical_metrics:
                llm_entry[metric] = metrics.get(metric, 0.0) if metrics.get(metric) is not None else 0.0
            llm_data.append(llm_entry)
        except Exception as e:
            logger.error(f"Error processing LLM sample: {e}")
    return llm_data

def save_samples(samples: List[Dict], output_ml_path: Path, output_llm_path: Path):
    """Save balanced samples to CSV for ML and JSONL for LLM."""
    try:
        df = pd.DataFrame(split_ml_samples(samples, CONFIG["numerical_metrics"]))
        df.to_csv(output_ml_path, index=False, encoding='utf-8')
        logger.info(f"Saved {len(samples)} ML samples to {output_ml_path} with columns: {list(df.columns)}")

        with open(output_llm_path, 'w', encoding='utf-8') as f:
            for sample in split_llm_samples(samples, CONFIG["numerical_metrics"]):
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Saved {len(samples)} LLM samples to {output_llm_path}")
    except Exception as e:
        logger.error(f"Error saving samples: {e}")

def main():
    logger.info("Starting preprocessing...")
    try:
        balanced_samples = load_and_balance(
            CONFIG["paths"]["input_jsonl"],
            CONFIG["sample_size"],
            CONFIG["none_smell_target"],
            CONFIG["smells"],
            CONFIG["severities"]
        )
        save_samples(
            balanced_samples,
            CONFIG["paths"]["output_ml_csv"],
            CONFIG["paths"]["output_llm_jsonl"]
        )
        logger.info("Preprocessing completed.")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()