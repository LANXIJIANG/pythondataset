import json
import logging
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# Configuration
CONFIG_DEFAULTS = {
    "paths": {
        "input_json": "F:/PythonDataset/04metrics/code_metrics.jsonl",
        "log_file": "F:/PythonDataset/05check/check.log"
    }
}

CONFIG = CONFIG_DEFAULTS
INPUT_JSON = Path(CONFIG["paths"]["input_json"])
LOG_FILE = Path(CONFIG["paths"]["log_file"])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> list:
    """Load JSONL file and return list of entries."""
    data = []
    logger.info(f"Starting severity distribution check")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing entries"):
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding line: {e}")
    logger.info(f"Processed {len(data)} entries from {file_path}")
    return data

def compute_severity_distribution(data: list) -> Dict[str, Dict[int, int]]:
    """Compute severity counts for each smell type."""
    smell_counts = {
        "LargeClass": {1: 0, 2: 0, 3: 0},
        "LongMethod": {1: 0, 2: 0, 3: 0},
        "LongParameterList": {1: 0, 2: 0, 3: 0},
        "DeepNesting": {1: 0, 2: 0, 3: 0},
        "NoneSmellorUnknown":{0:0}
    }
    total_counts = {smell: 0 for smell in smell_counts}

    for entry in data:
        smell_type = entry["smell_type"]
        severity = entry["severity"]
        if smell_type in smell_counts and severity in smell_counts[smell_type]:
            smell_counts[smell_type][severity] += 1
            total_counts[smell_type] += 1

    return smell_counts, total_counts

def log_distribution(smell_counts: Dict[str, Dict[int, int]], total_counts: Dict[str, int]):
    """Log severity distribution with percentages."""
    logger.info("Severity Distribution:")
    for smell in smell_counts:
        logger.info(f"{smell}:")
        total = total_counts[smell]
        if total > 0:
            for severity in sorted(smell_counts[smell].keys()):
                count = smell_counts[smell][severity]
                percentage = (count / total) * 100
                logger.info(f"  S{severity}: {count} ({percentage:.2f}%)")
            logger.info(f"  Total: {total}")
        else:
            logger.info(f"  No instances detected")

def main():
    data = load_data(INPUT_JSON)
    smell_counts, total_counts = compute_severity_distribution(data)
    log_distribution(smell_counts, total_counts)
    logger.info("Severity distribution check completed")

if __name__ == "__main__":
    main()