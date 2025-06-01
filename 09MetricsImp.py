import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import shap
import time
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Set up logging
OUTPUT_DIR = "F:/PythonDataset/09metricsimp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "metrics_importance.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define features (aligned with 08train.py)
CONFIG = {
    "numerical_metrics": [
        "LOC", "NOM", "LOC_method", "CYCLO_method", "NOP_method", "NEST",
        "TOKEN_COUNT", "LENGTH", "FAN_IN", "FAN_OUT", "ATTR_COUNT", "INHERIT_DEPTH"
    ],
    "engineered_features": [
        "code_complexity", "param_loc_ratio", "class_size_ratio", "loc_nom_normalized", "fan_io_ratio"
    ]
}

def setup_directories():
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    importance_dir = os.path.join(OUTPUT_DIR, "importance_data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(importance_dir, exist_ok=True)
    return plots_dir, importance_dir

def load_data():
    logger.info("Loading processed data...")
    try:
        test_df = pd.read_csv('F:/PythonDataset/07processed/test_processed.csv')
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    return test_df

def preprocess_data(test_df):
    # All features
    features = CONFIG["numerical_metrics"] + CONFIG["engineered_features"]
    X_test = test_df[features].copy()

    logger.info("Scaling features...")
    try:
        scaler = joblib.load('F:/PythonDataset/08train/models/scaler.joblib')
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        logger.error(f"Error loading or applying scaler: {e}")
        raise

    # Top features identified in 08train.py (used for all models: SVM, RandomForest, XGBoost)
    top_features = [
        'NEST', 'code_complexity', 'LOC_method', 'LENGTH', 'NOP_method',
        'CYCLO_method', 'LOC', 'TOKEN_COUNT', 'param_loc_ratio', 'loc_nom_normalized'
    ]
    top_indices = [features.index(f) for f in top_features]
    X_test_selected = X_test_scaled[:, top_indices]

    # Sample 500 test instances to reduce computation time
    indices = np.random.choice(X_test_selected.shape[0], 500, replace=False)
    X_test_selected_sample = X_test_selected[indices]

    return X_test_selected_sample, top_features

def compute_shap_values(model, X_sample, model_name, top_features, plots_dir, importance_dir):
    logger.info(f"Computing SHAP values for {model_name}...")
    start_time = time.time()

    # Convert X_sample to a DataFrame for SHAP with top features
    X_sample_df = pd.DataFrame(X_sample, columns=top_features)
    logger.info(f"X_sample_df for {model_name}: shape {X_sample_df.shape}, features {top_features}")

    # Number of classes (S0, S1, S2, S3)
    n_classes = 4

    # Compute SHAP values
    if model_name == "SVM":
        background = shap.sample(X_sample_df, 50)
        shap_values = []
        for class_idx in range(n_classes):
            logger.info(f"Computing SHAP values for SVM, class {class_idx}...")

            def predict_class_proba(data):
                data_array = data.values if isinstance(data, pd.DataFrame) else data
                proba = model.predict_proba(data_array)
                return proba[:, class_idx]

            explainer = shap.KernelExplainer(predict_class_proba, background)
            shap_values_class = explainer.shap_values(X_sample_df, nsamples=20)
            shap_values.append(shap_values_class)
            logger.info(f"SHAP values shape for SVM class {class_idx}: {np.array(shap_values_class).shape}")
        shap_values_for_plot = shap_values
        plot_X = X_sample
        feature_names = top_features
    else:  # RandomForest or XGBoost
        if model_name == "RandomForest" and isinstance(model, CalibratedClassifierCV):
            logger.info("RandomForest model is wrapped in CalibratedClassifierCV. Extracting base estimator...")
            if not model.calibrated_classifiers_:
                raise ValueError("CalibratedClassifierCV has no calibrated classifiers.")
            base_estimator = model.calibrated_classifiers_[0].estimator
            if not isinstance(base_estimator, RandomForestClassifier):
                raise ValueError(f"Expected RandomForestClassifier, got {type(base_estimator)}")
            logger.info(f"Base estimator for RandomForest: {type(base_estimator)}")
        else:
            base_estimator = model
            logger.info(f"{model_name} model is not wrapped: {type(base_estimator)}")

        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_sample_df)

        if isinstance(shap_values, list):
            logger.info(f"SHAP values is a list with length: {len(shap_values)}")
            shap_values_list = shap_values
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            logger.info(f"SHAP values shape for {model_name}: {shap_values.shape}")
            shap_values_list = [shap_values[:, :, i] for i in range(n_classes)]
        else:
            logger.error(f"Unexpected shap_values structure for {model_name}: {shap_values}")
            raise ValueError(f"Unexpected shap_values structure for {model_name}")

        if len(shap_values_list) != n_classes:
            logger.error(f"Expected {n_classes} classes in shap_values, got {len(shap_values_list)}")
            raise ValueError(f"Expected {n_classes} classes in shap_values, got {len(shap_values_list)}")

        for i, sv in enumerate(shap_values_list):
            logger.info(f"SHAP values shape for {model_name} class {i}: {np.array(sv).shape}")

        shap_values_for_plot = shap_values_list
        plot_X = X_sample
        feature_names = top_features

    # Log SHAP value magnitudes
    for class_idx in range(n_classes):
        shap_values_class = shap_values_for_plot[class_idx]
        max_shap = np.max(np.abs(shap_values_class))
        mean_abs_shap = np.mean(np.abs(shap_values_class))
        logger.info(
            f"SHAP values for {model_name} class {class_idx}: "
            f"Max absolute SHAP = {max_shap:.6f}, Mean absolute SHAP = {mean_abs_shap:.6f}"
        )

    # Generate SHAP summary plots for each class
    for class_idx in range(n_classes):
        logger.info(f"Generating SHAP summary plot for {model_name}, class {class_idx}...")
        shap_values_class = shap_values_for_plot[class_idx]
        plt.figure(figsize=(10, 6))
        # Normalize SHAP values for visualization (optional)
        max_shap = np.max(np.abs(shap_values_class))
        if max_shap != 0:
            shap_values_normalized = shap_values_class / max_shap
        else:
            shap_values_normalized = shap_values_class
        shap.summary_plot(shap_values_normalized, plot_X, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot for {model_name} (Severity {class_idx}) - Normalized")
        plt.savefig(os.path.join(plots_dir, f"shap_summary_{model_name.lower()}_class_{class_idx}_normalized.png"))
        plt.close()
        logger.info(f"Normalized SHAP summary plot saved for {model_name}, class {class_idx}")

        # Also save non-normalized plot for reference
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class, plot_X, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot for {model_name} (Severity {class_idx})")
        plt.savefig(os.path.join(plots_dir, f"shap_summary_{model_name.lower()}_class_{class_idx}.png"))
        plt.close()
        logger.info(f"Raw SHAP summary plot saved for {model_name}, class {class_idx}")

    # Compute mean absolute SHAP values for global feature importance
    shap_importance = np.abs(np.stack(shap_values_for_plot, axis=0)).mean(axis=(0, 1))
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP_Value": shap_importance
    }).sort_values(by="Mean_SHAP_Value", ascending=False)

    # Generate bar plot for global feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Mean_SHAP_Value"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title(f"Global Feature Importance for {model_name} (SHAP)")
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plots_dir, f"shap_importance_bar_{model_name.lower()}.png"))
    plt.close()

    logger.info(f"SHAP Feature Importance for {model_name}:\n{importance_df.to_string(index=False)}")
    importance_df.to_csv(os.path.join(importance_dir, f"shap_importance_{model_name.lower()}.csv"), index=False)
    logger.info(f"Saved SHAP importance to {importance_dir}/shap_importance_{model_name.lower()}.csv")

    end_time = time.time()
    logger.info(f"SHAP computation for {model_name} took {end_time - start_time:.2f} seconds.")

    return importance_df

def main():
    plots_dir, importance_dir = setup_directories()

    test_df = load_data()
    X_test_selected_sample, top_features = preprocess_data(test_df)

    logger.info("Loading trained models...")
    try:
        svm_model = joblib.load('F:/PythonDataset/08train/models/svm_model.joblib')
        rf_model = joblib.load('F:/PythonDataset/08train/models/randomforest_model.joblib')
        xgb_model = joblib.load('F:/PythonDataset/08train/models/xgboost_model.joblib')
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

    models = {
        "SVM": (svm_model, X_test_selected_sample),
        "RandomForest": (rf_model, X_test_selected_sample),
        "XGBoost": (xgb_model, X_test_selected_sample)
    }

    # Compute SHAP values and save results
    results = {}
    for model_name, (model, X_sample) in models.items():
        importance_df = compute_shap_values(model, X_sample, model_name, top_features, plots_dir, importance_dir)
        results[model_name] = importance_df

    logger.info("SHAP analysis completed.")

if __name__ == "__main__":
    logger.info("Starting SHAP-based metrics importance analysis...")
    main()
    logger.info("Analysis completed successfully.")