# 10Ablation.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, f1_score, log_loss
import joblib  # For saving scaler
import json
import logging
import os
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
BASE_PROJECT_DIR = Path("F:/PythonDataset")  # Base directory of your project

CONFIG = {
    "paths": {
        "input_train_csv": BASE_PROJECT_DIR / "07processed/train_processed.csv",
        "input_val_csv": BASE_PROJECT_DIR / "07processed/val_processed.csv",
        # GridSearchCV will use this for internal validation
        "input_test_csv": BASE_PROJECT_DIR / "07processed/test_processed.csv",
        "output_ablation_dir": BASE_PROJECT_DIR / "10AblationML",
        "log_file": BASE_PROJECT_DIR / "10AblationML/ablation_ml.log"
    },
    "target_column": "severity",
    "raw_numerical_metrics": [  # From your initial scripts
        "LOC", "NOM", "LOC_method", "CYCLO_method", "NOP_method", "NEST",
        "TOKEN_COUNT", "LENGTH", "FAN_IN", "FAN_OUT", "ATTR_COUNT", "INHERIT_DEPTH"
    ],
    "engineered_features": [  # From your validation script
        "code_complexity", "param_loc_ratio", "class_size_ratio",
        "loc_nom_normalized", "fan_io_ratio"
    ],
    "top_10_selected_features": [  # From your training.txt log (best model features)
        'NEST', 'code_complexity', 'LOC_method', 'LENGTH', 'NOP_method',
        'CYCLO_method', 'LOC', 'TOKEN_COUNT', 'param_loc_ratio', 'loc_nom_normalized'
    ],
    "model_params": {  # XGBoost param grid for GridSearchCV
        'xgb': {
            'param_grid': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 5, 10],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0.0, 1.0, 2.0],  # L1 regularization
                'reg_lambda': [1.0, 2.0, 5.0]  # L2 regularization
            },
            'best_params_from_main_training': {  # Fallback or for quick tests
                'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5,
                'min_child_weight': 10, 'reg_alpha': 2.0, 'reg_lambda': 5.0,
                'subsample': 0.8, 'n_estimators': 200  # Assuming n_estimators based on typical values
            }
        }
    },
    "cv_folds": 3,  # Reduced for faster ablation, increase for more robust tuning
    "scoring_metric": "mcc"  # Matthews Correlation Coefficient
}

# Define labels for classification report and per-smell metrics
# Assuming severity is 0, 1, 2, 3 and you have 'smell_type' column
# For simplicity, the classification report will be on severity.
# For per-smell-severity, we'll use the same structure as your training script.
SMELL_TYPE_LIST = ["NoneSmellorUnknown", "LargeClass", "LongMethod", "DeepNesting", "LongParameterList"]
SEVERITY_LIST = [0, 1, 2, 3]
COMBINED_LABEL_LIST = [f"{st}_S{sev}" for st in SMELL_TYPE_LIST for sev in SEVERITY_LIST if
                       not (st == "NoneSmellorUnknown" and sev > 0)]

# --- Setup Output Directory and Logging ---
ABLATION_OUTPUT_DIR = CONFIG["paths"]["output_ablation_dir"]
os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["paths"]["log_file"], mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def load_data(train_path, val_path, test_path):
    logger.info(f"Loading data: Train={train_path}, Val={val_path}, Test={test_path}")
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)  # Used by GridSearchCV for internal validation if not using full CV on train
        test_df = pd.read_csv(test_path)
        logger.info(f"Data loaded: Train shape {train_df.shape}, Val shape {val_df.shape}, Test shape {test_df.shape}")
        return train_df, val_df, test_df
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise


def scale_features(train_df, val_df, test_df, features_to_scale, scaler_path=None):
    logger.info(f"Scaling features: {features_to_scale}")
    scaler = StandardScaler()

    X_train_scaled = train_df.copy()
    X_val_scaled = val_df.copy()
    X_test_scaled = test_df.copy()

    X_train_scaled[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
    X_val_scaled[features_to_scale] = scaler.transform(val_df[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(test_df[features_to_scale])

    if scaler_path:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled


def compute_smell_severity_f1_metrics(df_with_smell_type, y_true_severity, y_pred_severity):
    """
    Computes F1-score for each combination of smell_type and true_severity.
    Args:
        df_with_smell_type (pd.DataFrame): DataFrame containing 'smell_type' column, aligned with y_true and y_pred.
        y_true_severity (pd.Series or np.array): True severity labels.
        y_pred_severity (np.array): Predicted severity labels.
    Returns:
        pd.DataFrame: DataFrame with F1 scores per smell_type and severity.
    """
    results = []
    unique_smell_types = df_with_smell_type['smell_type'].unique()

    temp_df = pd.DataFrame({
        'smell_type': df_with_smell_type['smell_type'],
        'true_severity': y_true_severity,
        'pred_severity': y_pred_severity
    })

    for smell_type in unique_smell_types:
        for severity in SEVERITY_LIST:
            # Filter for specific smell_type and true_severity
            true_positives_mask = (temp_df['smell_type'] == smell_type) & (temp_df['true_severity'] == severity) & (
                        temp_df['pred_severity'] == severity)
            actual_condition_positive_mask = (temp_df['smell_type'] == smell_type) & (
                        temp_df['true_severity'] == severity)
            predicted_condition_positive_mask = (temp_df['smell_type'] == smell_type) & (
                        temp_df['pred_severity'] == severity)

            tp = true_positives_mask.sum()
            actual_positives = actual_condition_positive_mask.sum()  # TP + FN for this class
            predicted_positives = predicted_condition_positive_mask.sum()  # TP + FP for this class

            if actual_positives == 0 and predicted_positives == 0:  # True negative for this specific class
                f1 = 1.0  # Or skip, or mark as N/A. For "not present and not predicted", F1 can be tricky.
                # Let's assume if it wasn't there and wasn't predicted, it's perfect for that class.
                # However, typically F1 is for positive class.
                # We'll only report if there were actual positives.
                precision = 1.0 if predicted_positives == 0 else 0.0
                recall = 1.0 if actual_positives == 0 else 0.0

            elif actual_positives == 0 and predicted_positives > 0:  # All false positives for this class
                precision = 0.0
                recall = 0.0  # or N/A
                f1 = 0.0
            elif actual_positives > 0 and predicted_positives == 0:  # All false negatives for this class
                precision = 0.0  # or N/A
                recall = 0.0
                f1 = 0.0
            else:  # actual_positives > 0 and predicted_positives > 0
                precision = tp / predicted_positives if predicted_positives > 0 else 0.0
                recall = tp / actual_positives if actual_positives > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Only add to results if there were samples of this specific smell_type and severity in the true labels
            if actual_positives > 0 or predicted_positives > 0:  # Or just actual_positives > 0 if you only care about F1 for existing classes
                results.append({
                    "Smell_Type": smell_type,
                    "Severity": f"S{severity}",
                    "F1_Score": f1,
                    "Precision": precision,
                    "Recall": recall,
                    "Support_True": actual_positives,
                    "Support_Pred": predicted_positives
                })

    return pd.DataFrame(results)


def train_evaluate_model(train_df, val_df, test_df, features_to_use, ablation_name, run_grid_search=True):
    logger.info(f"--- Starting Ablation: {ablation_name} ---")
    logger.info(f"Using features: {features_to_use}")

    ablation_run_output_dir = ABLATION_OUTPUT_DIR / ablation_name
    os.makedirs(ablation_run_output_dir, exist_ok=True)

    scaler_path = ablation_run_output_dir / "scaler.joblib"
    train_scaled_df, val_scaled_df, test_scaled_df = scale_features(train_df, val_df, test_df, features_to_use,
                                                                    scaler_path)

    X_train = train_scaled_df[features_to_use]
    y_train = train_scaled_df[CONFIG["target_column"]]
    # X_val for GridSearchCV's internal use or early stopping if model supports it without CV
    X_val = val_scaled_df[features_to_use]
    y_val = val_scaled_df[CONFIG["target_column"]]
    X_test = test_scaled_df[features_to_use]
    y_test = test_scaled_df[CONFIG["target_column"]]

    logger.info("Initializing XGBoost model...")
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    best_params = CONFIG["model_params"]['xgb']['best_params_from_main_training']  # Fallback

    if run_grid_search:
        logger.info("Starting GridSearchCV for XGBoost...")
        # For MCC, we need to use make_scorer and handle multi-class correctly
        # For simplicity here, using default scoring of XGB which is often mlogloss or merror
        # To use MCC: mcc_scorer = make_scorer(matthews_corrcoef)
        # Using default XGBoost objective 'multi:softmax' implies it optimizes for multiclass logloss or error internally

        # Note: GridSearchCV with XGBClassifier can use an eval_set for early stopping
        # but standard GridSearchCV doesn't directly support passing eval_set to fit for all scorers.
        # We'll use CV on the training data.

        param_grid = CONFIG["model_params"]['xgb']['param_grid']

        # A simpler grid for faster ablation, can be expanded
        simple_param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 10],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        logger.warning(
            f"Using a simplified param_grid for ablation GridSearch: {simple_param_grid}. Expand for production.")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=simple_param_grid,  # Using simplified for speed
            scoring='f1_macro',  # Or use mcc_scorer after defining it
            cv=CONFIG["cv_folds"],
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        grid_search.fit(X_train, y_train)  # Fit on the scaled training data
        best_params = grid_search.best_params_
        model = grid_search.best_estimator_  # This is the model with best_params, already fitted
        logger.info(f"GridSearchCV best parameters: {best_params}")
        logger.info(f"GridSearchCV best score (f1_macro): {grid_search.best_score_}")
    else:
        logger.info(f"Skipping GridSearchCV, using predefined best parameters: {best_params}")
        model.set_params(**best_params)
        model.fit(X_train, y_train)  # Fit on the scaled training data

    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    try:
        logloss = log_loss(y_test, y_pred_proba)
    except ValueError as e:  # Can happen if a class in y_test is not predicted at all
        logger.warning(f"Could not calculate log_loss: {e}. Setting to NaN.")
        logloss = float('nan')

    report_str = classification_report(y_test, y_pred, target_names=[f"S{s}" for s in sorted(y_test.unique())],
                                       zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=[f"S{s}" for s in sorted(y_test.unique())],
                                        output_dict=True, zero_division=0)

    logger.info(f"Test Set Evaluation for {ablation_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  MCC: {mcc:.4f}")
    logger.info(f"  Macro F1-Score: {macro_f1:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    logger.info(f"  Classification Report:\n{report_str}")

    # Per-Smell-Severity F1 (using the original 'smell_type' from the test_df)
    per_smell_f1_df = compute_smell_severity_f1_metrics(test_df, y_test, y_pred)  # test_df has 'smell_type'
    logger.info(f"Per-Smell-Severity F1 Scores for {ablation_name}:\n{per_smell_f1_df.to_string()}")

    # Save results
    results_data = {
        "ablation_name": ablation_name,
        "features_used_count": len(features_to_use),
        "features_list": features_to_use,
        "best_params_after_gs": best_params if run_grid_search else "Predefined",
        "test_accuracy": accuracy,
        "test_mcc": mcc,
        "test_macro_f1": macro_f1,
        "test_log_loss": logloss,
        "test_classification_report_dict": report_dict
    }

    with open(ablation_run_output_dir / "test_metrics_summary.json", 'w') as f:
        json.dump(results_data, f, indent=4)
    with open(ablation_run_output_dir / "test_classification_report.txt", 'w') as f:
        f.write(report_str)
    per_smell_f1_df.to_csv(ablation_run_output_dir / "test_per_smell_severity_f1.csv", index=False)

    # Save the trained model for this ablation run
    model_path = ablation_run_output_dir / "xgboost_model.json"  # XGBoost can save model in JSON format
    model.save_model(model_path)
    logger.info(f"Saved trained XGBoost model for {ablation_name} to {model_path}")

    logger.info(f"--- Finished Ablation: {ablation_name} ---")
    return results_data


# --- Main Script ---
if __name__ == "__main__":
    logger.info("Starting ML Ablation Study Script...")

    train_df, val_df, test_df = load_data(
        CONFIG["paths"]["input_train_csv"],
        CONFIG["paths"]["input_val_csv"],
        CONFIG["paths"]["input_test_csv"]
    )

    all_ablation_summaries = []

    # Define Ablation Scenarios
    ablation_scenarios = [
        {
            "name": "RawFeaturesOnly",
            "features": CONFIG["raw_numerical_metrics"],
            "run_grid_search": True
        },
        {
            "name": "AllOriginalAndEngineeredFeatures",  # Before top-10 selection
            "features": CONFIG["raw_numerical_metrics"] + CONFIG["engineered_features"],
            "run_grid_search": True
        },
        {
            "name": "Top10SelectedFeatures_Baseline",  # This is your main model's feature set
            "features": CONFIG["top_10_selected_features"],
            "run_grid_search": True  # Or False if you want to use the exact best params from main training
        }
    ]

    # --- Run Sanity Check for Feature Availability ---
    all_available_features_in_df = set(CONFIG["raw_numerical_metrics"] + CONFIG["engineered_features"])
    for col in train_df.columns:  # Check against actual columns
        if col not in all_available_features_in_df and col not in ['smell_type', 'severity', 'smell_code', 'project',
                                                                   'package', 'method', 'version', 'original_id',
                                                                   'dedup_key',
                                                                   'stratify_key']:  # Add other non-feature cols if any
            if col in CONFIG["raw_numerical_metrics"] or col in CONFIG["engineered_features"] or col in CONFIG[
                "top_10_selected_features"]:
                pass  # It's a feature
            else:
                logger.warning(
                    f"Column '{col}' found in DataFrame but not in defined feature lists or known non-feature columns.")

    for scenario in ablation_scenarios:
        # Ensure all features for the scenario are present in the DataFrame
        missing_features = [f for f in scenario["features"] if f not in train_df.columns]
        if missing_features:
            logger.error(
                f"Skipping ablation '{scenario['name']}' due to missing features in DataFrame: {missing_features}")
            continue

        summary = train_evaluate_model(
            train_df.copy(), val_df.copy(), test_df.copy(),  # Pass copies to avoid modification
            scenario["features"],
            scenario["name"],
            run_grid_search=scenario["run_grid_search"]
        )
        all_ablation_summaries.append(summary)

    # Save a combined summary of all ablation runs
    combined_summary_path = ABLATION_OUTPUT_DIR / "all_ablations_summary.json"
    with open(combined_summary_path, 'w') as f:
        json.dump(all_ablation_summaries, f, indent=4)
    logger.info(f"Saved combined ablation summaries to {combined_summary_path}")

    # Create a simple CSV summary for easier comparison
    try:
        summary_df_data = []
        for s in all_ablation_summaries:
            if s:  # Check if summary is not None (in case a run failed and returned None)
                summary_df_data.append({
                    "Ablation_Name": s["ablation_name"],
                    "Features_Count": s["features_used_count"],
                    "Accuracy": s.get("test_accuracy"),
                    "MCC": s.get("test_mcc"),
                    "Macro_F1": s.get("test_macro_f1"),
                    "Log_Loss": s.get("test_log_loss"),
                    # "Features_List": ", ".join(s["features_list"]) # Can be too long for CSV cell
                })
        summary_df = pd.DataFrame(summary_df_data)
        summary_df.to_csv(ABLATION_OUTPUT_DIR / "all_ablations_summary_table.csv", index=False)
        logger.info(f"Saved comparable summary table to {ABLATION_OUTPUT_DIR / 'all_ablations_summary_table.csv'}")
    except Exception as e:
        logger.error(f"Could not create CSV summary table: {e}")

    logger.info("ML Ablation Study Script Finished.")