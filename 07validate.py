import pandas as pd
import logging
import numpy as np
import hashlib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
CONFIG = {
    "paths": {
        "input_ml_csv": Path("F:/PythonDataset/06preprocessed/balanced_ml.csv"),
        "input_llm_jsonl": Path("F:/PythonDataset/06preprocessed/balanced_llm.jsonl"),
        "output_dir": Path("F:/PythonDataset/07processed"),
        "output_train_csv": Path("F:/PythonDataset/07processed/train_processed.csv"),
        "output_val_csv": Path("F:/PythonDataset/07processed/val_processed.csv"),
        "output_test_csv": Path("F:/PythonDataset/07processed/test_processed.csv"),
        "overlap_dir": Path("F:/PythonDataset/07processed/overlaps"),
        "log_file": Path("F:/PythonDataset/07processed/validation.log")
    },
    "numerical_metrics": [
        "LOC", "NOM", "LOC_method", "CYCLO_method", "NOP_method", "NEST",
        "TOKEN_COUNT", "LENGTH", "FAN_IN", "FAN_OUT", "ATTR_COUNT", "INHERIT_DEPTH"
    ],
    "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
}

# Ensure output directories exist
os.makedirs(CONFIG["paths"]["output_dir"], exist_ok=True)
os.makedirs(CONFIG["paths"]["overlap_dir"], exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["paths"]["log_file"], mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def engineer_features(df):
    """Engineer targeted features for validation."""
    try:
        # Handle missing values by filling with 0
        for col in CONFIG["numerical_metrics"]:
            df[col] = df[col].fillna(0.0)

        # Validate TOKEN_COUNT for log1p
        if (df['TOKEN_COUNT'] < 0).any():
            logger.warning("Negative TOKEN_COUNT values detected; clamping to 0")
            df['TOKEN_COUNT'] = df['TOKEN_COUNT'].clip(lower=0)

        # Engineer features with safeguards for division by zero
        df['code_complexity'] = df['CYCLO_method'] + df['NEST'] + np.log1p(df['TOKEN_COUNT'])
        df['param_loc_ratio'] = df['NOP_method'] / (df['LOC_method'] + 1e-10)
        # Set class_size_ratio to 0 for non-class samples (NOM == 0)
        df['class_size_ratio'] = np.where(
            df['NOM'] == 0,
            0.0,
            df['LOC'] / (df['NOM'] + 1e-10)
        )
        logger.info("Set class_size_ratio to 0 for samples with NOM == 0")
        df['loc_nom_normalized'] = (df['LOC'] * df['NOM']) / (df['LOC'] + df['NOM'] + 1e-10)
        df['fan_io_ratio'] = df['FAN_OUT'] / (df['FAN_IN'] + 1e-10)
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

def log_detailed_distribution(df, name):
    """Log detailed distribution of smell_type and severity combinations."""
    logger.info(f"Detailed Distribution for {name}:")

    # Overall severity distribution
    severity_dist = df['severity'].value_counts().to_dict()
    logger.info("Severity Distribution:")
    for severity, count in sorted(severity_dist.items()):
        logger.info(f"  S{severity}: {count} ({count / len(df) * 100:.2f}%)")

    # Overall smell_type distribution
    smell_dist = df['smell_type'].value_counts().to_dict()
    logger.info("Smell Type Distribution:")
    for smell, count in sorted(smell_dist.items()):
        logger.info(f"  {smell}: {count} ({count / len(df) * 100:.2f}%)")

    # Smell-severity combination distribution
    smell_severity_dist = df.groupby(['smell_type', 'severity']).size().to_dict()
    logger.info("Smell-Severity Combination Distribution:")
    for (smell, severity), count in sorted(smell_severity_dist.items()):
        logger.info(f"  {smell} S{severity}: {count} ({count / len(df) * 100:.2f}%)")

def log_overlap_details(overlap_df, name):
    """Log details of overlapping samples."""
    if overlap_df.empty:
        logger.info(f"No overlaps in {name}")
        return
    logger.info(f"Overlap Details for {name} ({len(overlap_df)} rows):")
    smell_severity_dist = overlap_df.groupby(['smell_type', 'severity']).size().to_dict()
    for (smell, severity), count in sorted(smell_severity_dist.items()):
        logger.info(f"  {smell} S{severity}: {count} ({count / len(overlap_df) * 100:.2f}%)")

def validate():
    logger.info("Starting validation...")

    logger.info("Loading datasets...")
    try:
        ml_df = pd.read_csv(CONFIG["paths"]["input_ml_csv"])
        llm_df = pd.read_json(CONFIG["paths"]["input_llm_jsonl"], lines=True)
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return

    # Merge the datasets on index
    full_df = ml_df.merge(llm_df[['smell_code']], left_index=True, right_index=True)
    logger.info(f"Loaded full dataset with {len(full_df)} samples")

    # Verify deduplication based on smell_type, severity, and smell_code
    full_df['dedup_key'] = full_df.apply(
        lambda row: hashlib.md5(f"{row['smell_type']}|{row['severity']}|{row['smell_code']}".encode()).hexdigest(),
        axis=1)
    duplicates = full_df.duplicated(subset=['dedup_key']).sum()
    logger.info(f"Number of duplicates in full dataset (should be 0): {duplicates}")
    if duplicates > 0:
        logger.warning("Duplicates found in the dataset before splitting. Removing duplicates...")
        full_df = full_df.drop_duplicates(subset=['dedup_key'])
        logger.info(f"Dataset after removing duplicates: {len(full_df)} samples")

    # Log distribution of the full dataset
    log_detailed_distribution(full_df, "full dataset")

    logger.info("Splitting dataset with stratification...")
    # Create a stratification key combining smell_type and severity
    full_df['stratify_key'] = full_df['smell_type'] + '_' + full_df['severity'].astype(str)

    # Split into train and temp (val + test)
    train_df, temp_df = train_test_split(
        full_df,
        test_size=CONFIG["split_ratios"]["val"] + CONFIG["split_ratios"]["test"],
        stratify=full_df['stratify_key'],
        random_state=42
    )

    # Split temp into val and test
    val_test_ratio = CONFIG["split_ratios"]["val"] / (CONFIG["split_ratios"]["val"] + CONFIG["split_ratios"]["test"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_test_ratio,
        stratify=temp_df['stratify_key'],
        random_state=42
    )

    # Drop the stratify_key and dedup_key columns
    train_df = train_df.drop(columns=['stratify_key', 'dedup_key'])
    val_df = val_df.drop(columns=['stratify_key', 'dedup_key'])
    test_df = test_df.drop(columns=['stratify_key', 'dedup_key'])

    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Val set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")

    logger.info("Engineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    logger.info("Checking for overlaps...")
    # Check overlaps based on raw features only
    overlap_cols = ['smell_type', 'severity', 'smell_code'] + CONFIG["numerical_metrics"]
    try:
        train_val_overlap = pd.merge(train_df, val_df, how='inner', on=overlap_cols)
        train_test_overlap = pd.merge(train_df, test_df, how='inner', on=overlap_cols)
        val_test_overlap = pd.merge(val_df, test_df, how='inner', on=overlap_cols)

        logger.info(f"Train-Val Overlap Rows: {len(train_val_overlap)}")
        log_overlap_details(train_val_overlap, "Train-Val")
        logger.info(f"Train-Test Overlap Rows: {len(train_test_overlap)}")
        log_overlap_details(train_test_overlap, "Train-Test")
        logger.info(f"Val-Test Overlap Rows: {len(val_test_overlap)}")
        log_overlap_details(val_test_overlap, "Val-Test")

        train_val_overlap.to_csv(CONFIG["paths"]["overlap_dir"] / 'train_val_overlap.csv', index=False)
        train_test_overlap.to_csv(CONFIG["paths"]["overlap_dir"] / 'train_test_overlap.csv', index=False)
        val_test_overlap.to_csv(CONFIG["paths"]["overlap_dir"] / 'val_test_overlap.csv', index=False)
        logger.info("Saved overlapping samples to F:/PythonDataset/07processed/overlaps/")
    except Exception as e:
        logger.error(f"Error checking overlaps: {e}")
        return

    # Log distributions for each split
    log_detailed_distribution(train_df, "train")
    log_detailed_distribution(val_df, "val")
    log_detailed_distribution(test_df, "test")

    # Save processed datasets
    try:
        train_df.to_csv(CONFIG["paths"]["output_train_csv"], index=False)
        val_df.to_csv(CONFIG["paths"]["output_val_csv"], index=False)
        test_df.to_csv(CONFIG["paths"]["output_test_csv"], index=False)
        logger.info("Saved processed data as CSVs with raw values and original smell_type.")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return

    # Validate saved files
    try:
        saved_train_df = pd.read_csv(CONFIG["paths"]["output_train_csv"])
        saved_val_df = pd.read_csv(CONFIG["paths"]["output_val_csv"])
        saved_test_df = pd.read_csv(CONFIG["paths"]["output_test_csv"])
        logger.info(f"Validated saved files: Train={len(saved_train_df)}, Val={len(saved_val_df)}, Test={len(saved_test_df)}")
        if len(saved_train_df) != len(train_df) or len(saved_val_df) != len(val_df) or len(saved_test_df) != len(test_df):
            logger.error("Mismatch in saved file sample counts!")
    except Exception as e:
        logger.error(f"Error validating saved files: {e}")

    logger.info("Validation completed.")

if __name__ == "__main__":
    validate()