# 011noisycheck.py (Updated with fixes)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, f1_score, log_loss
import joblib
import json
import logging
import os
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
BASE_PROJECT_DIR = Path("F:/PythonDataset")

CONFIG = {
    "paths": {
        "input_train_csv": BASE_PROJECT_DIR / "07processed/train_processed.csv",
        "input_val_csv": BASE_PROJECT_DIR / "07processed/val_processed.csv",
        "input_test_csv": BASE_PROJECT_DIR / "07processed/test_processed.csv",
        "trained_model_path": BASE_PROJECT_DIR / "08train/models/xgboost_model.joblib",
        "scaler_path": BASE_PROJECT_DIR / "08train/models/scaler.joblib",
        "output_noisy_dir": BASE_PROJECT_DIR / "011noisychecking",
        "log_file": BASE_PROJECT_DIR / "011noisychecking/noisy_check_v2.log"  # New log file
    },
    "target_column": "severity",
    "raw_numerical_metrics": [
        "LOC", "NOM", "LOC_method", "CYCLO_method", "NOP_method", "NEST",
        "TOKEN_COUNT", "LENGTH", "FAN_IN", "FAN_OUT", "ATTR_COUNT", "INHERIT_DEPTH"
    ],
    "engineered_features": [
        "code_complexity", "param_loc_ratio", "class_size_ratio",
        "loc_nom_normalized", "fan_io_ratio"
    ],
    "top_10_selected_features": [
        'NEST', 'code_complexity', 'LOC_method', 'LENGTH', 'NOP_method',
        'CYCLO_method', 'LOC', 'TOKEN_COUNT', 'param_loc_ratio', 'loc_nom_normalized'
    ],
    "gaussian_noise_stds": [0.1, 0.25, 0.5],
    "num_irrelevant_features": 3,
    "xgb_param_grid_simplified": {
        'max_depth': [3, 5], 'learning_rate': [0.1], 'n_estimators': [100, 150],
        'min_child_weight': [5, 10], 'subsample': [0.8], 'colsample_bytree': [0.8],
        'reg_alpha': [2.0], 'reg_lambda': [5.0]
    },
    "cv_folds": 3,
}

SMELL_TYPE_LIST = ["NoneSmellorUnknown", "LargeClass", "LongMethod", "DeepNesting", "LongParameterList"]
SEVERITY_LIST = [0, 1, 2, 3]  # All possible severity values

# --- Setup Output Directory and Logging ---
NOISY_OUTPUT_DIR = CONFIG["paths"]["output_noisy_dir"]
os.makedirs(NOISY_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["paths"]["log_file"], mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def load_data_and_model():
    logger.info("Loading datasets, trained XGBoost model, and scaler...")
    try:
        train_df = pd.read_csv(CONFIG["paths"]["input_train_csv"])
        val_df = pd.read_csv(CONFIG["paths"]["input_val_csv"])
        test_df = pd.read_csv(CONFIG["paths"]["input_test_csv"])
        model = joblib.load(CONFIG["paths"]["trained_model_path"])
        scaler = joblib.load(CONFIG["paths"]["scaler_path"])
        logger.info(f"Data loaded: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}")
        logger.info(f"Trained XGBoost model and Scaler loaded.")
        return train_df, val_df, test_df, model, scaler
    except Exception as e:
        logger.error(f"Error loading files: {e}"); raise


def add_gaussian_noise_to_scaled_features(scaled_df, features_to_noise, noise_std_multiplier):
    noisy_df = scaled_df.copy()
    for feature in features_to_noise:
        if feature in noisy_df.columns:
            noise_to_add = np.random.normal(0, noise_std_multiplier, noisy_df.shape[0])
            noisy_df[feature] = noisy_df[feature] + noise_to_add
        else:
            logger.warning(f"Feature '{feature}' not in DataFrame. Skipping noise for it.")
    return noisy_df


def add_irrelevant_features(df, num_features_to_add):
    df_with_irrelevant = df.copy()
    for i in range(num_features_to_add):
        df_with_irrelevant[f'irrelevant_feat_{i + 1}'] = np.random.rand(len(df_with_irrelevant))
    logger.info(f"Added {num_features_to_add} irrelevant features.")
    return df_with_irrelevant


def compute_smell_severity_f1_metrics(df_with_smell_type_col, y_true_severity_col, y_pred_severity_col):
    results = []
    unique_smell_types_list = df_with_smell_type_col['smell_type'].unique()
    temp_df_metrics = pd.DataFrame({'smell_type': df_with_smell_type_col['smell_type'],
                                    'true_severity': y_true_severity_col,
                                    'pred_severity': y_pred_severity_col})
    for smell_type_iter in unique_smell_types_list:
        # Determine valid severities for this smell type
        current_valid_severities = []
        if smell_type_iter == "NoneSmellorUnknown":
            current_valid_severities = [0]
        elif smell_type_iter in ["LargeClass", "LongMethod", "DeepNesting", "LongParameterList"]:
            current_valid_severities = [1, 2, 3]
        else:  # Should not happen with current SMELL_TYPE_LIST
            continue

        for severity_val_iter in current_valid_severities:  # Iterate only valid severities
            true_pos_mask = (temp_df_metrics['smell_type'] == smell_type_iter) & \
                            (temp_df_metrics['true_severity'] == severity_val_iter) & \
                            (temp_df_metrics['pred_severity'] == severity_val_iter)
            actual_pos_mask = (temp_df_metrics['smell_type'] == smell_type_iter) & \
                              (temp_df_metrics['true_severity'] == severity_val_iter)
            # For precision, consider predictions for this specific smell-severity combination
            # This means we need to be careful if the model predicts an invalid severity for a smell type
            # Let's count predictions for *any* smell_type but this specific *predicted* severity if we want true class precision.
            # For simplicity here, we count predictions if they match smell_type AND the iterated severity_val_iter.
            pred_pos_mask = (temp_df_metrics['smell_type'] == smell_type_iter) & \
                            (temp_df_metrics['pred_severity'] == severity_val_iter)

            tp_count = true_pos_mask.sum()
            actual_pos_count = actual_pos_mask.sum()
            pred_pos_count = pred_pos_mask.sum()

            precision_val = tp_count / pred_pos_count if pred_pos_count > 0 else 0.0
            recall_val = tp_count / actual_pos_count if actual_pos_count > 0 else 0.0
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (
                                                                                                    precision_val + recall_val) > 0 else 0.0

            # Only add to results if there were true samples of this class or if it was predicted
            if actual_pos_count > 0 or pred_pos_count > 0:
                results.append({"Smell_Type": smell_type_iter, "Severity": f"S{severity_val_iter}", "F1_Score": f1_val,
                                "Precision": precision_val, "Recall": recall_val,
                                "Support_True": actual_pos_count, "Support_Pred": pred_pos_count})
    return pd.DataFrame(results)


def evaluate_and_save_results(model_to_eval, X_test_data, y_test_data, test_df_for_smell_column,
                              condition_name_str, output_path_dir):
    logger.info(f"Evaluating model for condition: {condition_name_str}...")
    os.makedirs(output_path_dir, exist_ok=True)  # Ensure dir exists

    y_pred_labels = model_to_eval.predict(X_test_data)
    y_pred_probabilities = model_to_eval.predict_proba(X_test_data)

    accuracy = accuracy_score(y_test_data, y_pred_labels)
    mcc = matthews_corrcoef(y_test_data, y_pred_labels)
    macro_f1 = f1_score(y_test_data, y_pred_labels, average='macro', zero_division=0)
    try:
        logloss = log_loss(y_test_data, y_pred_probabilities)
    except ValueError as e:
        logloss = float('nan')
        logger.warning(f"({condition_name_str}) Log loss calculation failed: {e}. Setting to NaN.")

    logloss_display_string = f"{logloss:.4f}" if not np.isnan(logloss) else "NaN"

    report_str = classification_report(y_test_data, y_pred_labels,
                                       target_names=[f"S{s}" for s in sorted(y_test_data.unique())], zero_division=0)
    report_dict = classification_report(y_test_data, y_pred_labels,
                                        target_names=[f"S{s}" for s in sorted(y_test_data.unique())], output_dict=True,
                                        zero_division=0)

    logger.info(f"  Test Accuracy: {accuracy:.4f}")
    logger.info(f"  Test MCC: {mcc:.4f}")
    logger.info(f"  Test Macro F1-Score: {macro_f1:.4f}")
    logger.info(f"  Test Log Loss: {logloss_display_string}")  # Corrected line
    logger.info(f"  Classification Report:\n{report_str}")

    per_smell_f1_df = compute_smell_severity_f1_metrics(test_df_for_smell_column, y_test_data, y_pred_labels)
    logger.info(f"  Per-Smell-Severity F1 Scores:\n{per_smell_f1_df.to_string()}")

    results_to_save = {
        "condition_name": condition_name_str,
        "test_accuracy": accuracy, "test_mcc": mcc,
        "test_macro_f1": macro_f1,
        "test_log_loss": logloss if not np.isnan(logloss) else None,
        "test_classification_report_dict": report_dict
    }
    with open(output_path_dir / "test_metrics_summary.json", 'w') as f:
        json.dump(results_to_save, f, indent=4)
    with open(output_path_dir / "test_classification_report.txt", 'w') as f:
        f.write(report_str)
    per_smell_f1_df.to_csv(output_path_dir / "test_per_smell_severity_f1.csv", index=False)
    logger.info(f"Saved results for {condition_name_str} to {output_path_dir}")
    return results_to_save


# --- Main Script ---
if __name__ == "__main__":
    logger.info("Starting ML Noisy Experiment Script...")
    all_noisy_summaries = []

    train_df_orig, val_df_orig, test_df_orig, model_clean, scaler_clean_orig = load_data_and_model()

    all_features_scaler_was_fit_on = CONFIG["raw_numerical_metrics"] + CONFIG["engineered_features"]

    missing_scaler_features_check = [f for f in all_features_scaler_was_fit_on if f not in test_df_orig.columns]
    if missing_scaler_features_check:
        logger.error(f"Scaler was trained on features not all present in test_df_orig: {missing_scaler_features_check}")
        raise ValueError("Mismatch between scaler features and test_df_orig features for initial scaling.")

    test_df_scaled_full = test_df_orig.copy()
    test_df_scaled_full[all_features_scaler_was_fit_on] = scaler_clean_orig.transform(
        test_df_orig[all_features_scaler_was_fit_on])
    X_test_scaled_clean_top10 = test_df_scaled_full[CONFIG["top_10_selected_features"]]
    y_test_clean = test_df_orig[CONFIG["target_column"]]

    logger.info("\n--- Experiment 1: Gaussian Noise Injection (on scaled features) ---")
    for std_multiplier in CONFIG["gaussian_noise_stds"]:
        condition_name = f"GaussianNoise_StdMultiplier_{std_multiplier}"
        logger.info(f"Running Gaussian Noise condition: {condition_name}")

        X_test_noisy_run = X_test_scaled_clean_top10.copy()
        for feature in CONFIG["top_10_selected_features"]:
            noise_to_add = np.random.normal(0, std_multiplier, X_test_noisy_run.shape[0])
            X_test_noisy_run[feature] = X_test_noisy_run[feature] + noise_to_add

        output_path_gauss = NOISY_OUTPUT_DIR / "GaussianNoise" / f"StdMultiplier_{std_multiplier}"
        # os.makedirs(output_path_gauss, exist_ok=True) # evaluate_and_save_results will create it
        summary = evaluate_and_save_results(model_clean, X_test_noisy_run, y_test_clean, test_df_orig, condition_name,
                                            output_path_gauss)
        all_noisy_summaries.append(summary)

    logger.info("\n--- Experiment 2: Adding Irrelevant Features & Retraining ---")
    condition_name_irrelevant = f"With_{CONFIG['num_irrelevant_features']}_IrrelevantFeatures"

    train_df_irr = add_irrelevant_features(train_df_orig.copy(), CONFIG["num_irrelevant_features"])
    # val_df_irr = add_irrelevant_features(val_df_orig.copy(), CONFIG["num_irrelevant_features"]) # For GridSearchCV if used
    test_df_irr = add_irrelevant_features(test_df_orig.copy(), CONFIG["num_irrelevant_features"])

    features_for_irr_exp = CONFIG["top_10_selected_features"] + [f'irrelevant_feat_{i + 1}' for i in
                                                                 range(CONFIG['num_irrelevant_features'])]

    missing_in_train = [f for f in features_for_irr_exp if f not in train_df_irr.columns]
    if missing_in_train:
        logger.error(f"FATAL: Features for irrelevant exp missing in train_df_irr: {missing_in_train}")
        raise ValueError("Feature mismatch for irrelevant features experiment.")

    output_path_irr = NOISY_OUTPUT_DIR / "IrrelevantFeatures"
    os.makedirs(output_path_irr, exist_ok=True)  # FIX: Create directory before saving scaler
    scaler_path_irr = output_path_irr / "scaler_with_irrelevant.joblib"

    scaler_irr = StandardScaler()
    # Fit scaler ONLY on training data part of these features
    X_train_irr_unscaled = train_df_irr[features_for_irr_exp]
    train_df_irr[features_for_irr_exp] = scaler_irr.fit_transform(X_train_irr_unscaled)
    # Transform val and test sets
    # val_df_irr[features_for_irr_exp] = scaler_irr.transform(val_df_irr[features_for_irr_exp])
    test_df_irr[features_for_irr_exp] = scaler_irr.transform(test_df_irr[features_for_irr_exp])
    joblib.dump(scaler_irr, scaler_path_irr)
    logger.info(f"Scaler for irrelevant features experiment saved to {scaler_path_irr}")

    X_train_irr_scaled = train_df_irr[features_for_irr_exp]
    y_train_irr = train_df_irr[CONFIG["target_column"]]
    X_test_irr_scaled = test_df_irr[features_for_irr_exp]
    y_test_irr = test_df_irr[CONFIG["target_column"]]

    logger.info(f"Retraining XGBoost for '{condition_name_irrelevant}' with features: {features_for_irr_exp}")
    model_irr_new = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    logger.info("Starting GridSearchCV for XGBoost with irrelevant features...")
    grid_search_irr_run = GridSearchCV(
        estimator=model_irr_new, param_grid=CONFIG["xgb_param_grid_simplified"],
        scoring='f1_macro', cv=CONFIG["cv_folds"], n_jobs=-1, verbose=1)
    grid_search_irr_run.fit(X_train_irr_scaled, y_train_irr)
    logger.info(f"GridSearchCV best parameters (with irrelevant features): {grid_search_irr_run.best_params_}")
    logger.info(f"GridSearchCV best score (f1_macro, with irrelevant features): {grid_search_irr_run.best_score_}")
    model_irr_tuned_final = grid_search_irr_run.best_estimator_

    model_irr_path_final = output_path_irr / "xgboost_model_with_irrelevant.joblib"  # Save as joblib
    joblib.dump(model_irr_tuned_final, model_irr_path_final)  # Save the sklearn wrapper
    logger.info(f"Saved retrained XGBoost model for '{condition_name_irrelevant}' to {model_irr_path_final}")

    summary_irr_final = evaluate_and_save_results(model_irr_tuned_final, X_test_irr_scaled, y_test_irr, test_df_orig,
                                                  condition_name_irrelevant, output_path_irr)
    all_noisy_summaries.append(summary_irr_final)

    combined_summary_path_final = NOISY_OUTPUT_DIR / "noisy_experiments_summary.json"
    with open(combined_summary_path_final, 'w') as f_json_out:
        json.dump(all_noisy_summaries, f_json_out, indent=4)
    logger.info(f"Saved combined noisy experiment summaries to {combined_summary_path_final}")

    try:
        summary_df_data_final = []
        for s_item in all_noisy_summaries:
            if s_item:
                summary_df_data_final.append({
                    "Condition_Name": s_item["condition_name"],
                    "Accuracy": s_item.get("test_accuracy"), "MCC": s_item.get("test_mcc"),
                    "Macro_F1": s_item.get("test_macro_f1"), "Log_Loss": s_item.get("test_log_loss"), })
        summary_df_final = pd.DataFrame(summary_df_data_final)
        summary_df_final.to_csv(NOISY_OUTPUT_DIR / "noisy_experiments_summary_table.csv", index=False)
        logger.info(
            f"Saved comparable summary table for noisy experiments to {NOISY_OUTPUT_DIR / 'noisy_experiments_summary_table.csv'}")
    except Exception as e_csv:
        logger.error(f"Could not create CSV summary table for noisy experiments: {e_csv}")

    logger.info("ML Noisy Experiment Script Finished.")