import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, make_scorer, log_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import numpy as np
import joblib
import os
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Some inputs do not have OOB scores")

# Configuration
CONFIG = {
    "paths": {
        "input_train_csv": Path("F:/PythonDataset/07processed/train_processed.csv"),
        "input_val_csv": Path("F:/PythonDataset/07processed/val_processed.csv"),
        "input_test_csv": Path("F:/PythonDataset/07processed/test_processed.csv"),
        "output_dir": Path("F:/PythonDataset/08train"),
        "models_dir": Path("F:/PythonDataset/08train/models"),
        "predictions_dir": Path("F:/PythonDataset/08train/predictions"),
        "performance_tables_dir": Path("F:/PythonDataset/08train/performance_tables"),
        "visualizations_dir": Path("F:/PythonDataset/08train/visualizations"),
        "log_file": Path("F:/PythonDataset/08train/training.log")
    },
    "numerical_metrics": [
        "LOC", "NOM", "LOC_method", "CYCLO_method", "NOP_method", "NEST",
        "TOKEN_COUNT", "LENGTH", "FAN_IN", "FAN_OUT", "ATTR_COUNT", "INHERIT_DEPTH"
    ],
    "engineered_features": [
        "code_complexity", "param_loc_ratio", "class_size_ratio", "loc_nom_normalized", "fan_io_ratio"
    ],
    "expected_classes": [0, 1, 2, 3],  # S0, S1, S2, S3
    "n_features": 10  # Number of top features for all models
}

# Setup directories
for dir_path in [CONFIG["paths"]["output_dir"], CONFIG["paths"]["models_dir"], CONFIG["paths"]["predictions_dir"],
                 CONFIG["paths"]["performance_tables_dir"], CONFIG["paths"]["visualizations_dir"]]:
    os.makedirs(dir_path, exist_ok=True)

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

def compute_smell_severity_metrics(df, y_true, y_pred, split_name, model_name):
    try:
        metrics_df = pd.DataFrame({
            'smell_type': df['smell_type'].values,
            'true_severity': y_true.values,
            'pred_severity': y_pred
        })
        grouped = metrics_df.groupby(['smell_type', 'true_severity'], as_index=False).agg(
            correct=('pred_severity', lambda x: (x == metrics_df.loc[x.index, 'true_severity']).sum()),
            total_true=('pred_severity', 'count')
        )
        pred_counts = metrics_df.groupby(['smell_type', 'pred_severity'], as_index=False).size().rename(
            columns={'size': 'total_pred', 'pred_severity': 'true_severity'})
        grouped = grouped.merge(pred_counts, on=['smell_type', 'true_severity'], how='left').fillna(0)
        grouped['precision'] = grouped['correct'] / grouped['total_pred'].replace(0, 1e-10)
        grouped['recall'] = grouped['correct'] / grouped['total_true']
        grouped['f1'] = 2 * (grouped['precision'] * grouped['recall']) / (
                    grouped['precision'] + grouped['recall']).replace(0, 1e-10)
        logger.info(f"Smell-Severity Metrics for {split_name} ({model_name}):")
        for _, row in grouped.iterrows():
            logger.info(
                f"  {row['smell_type']} S{int(row['true_severity'])}: Correct={int(row['correct'])}, Total={int(row['total_true'])}, Precision={row['precision']:.2f}, Recall={row['recall']:.2f}, F1={row['f1']:.2f}")
        return grouped
    except Exception as e:
        logger.error(f"Error computing smell-severity metrics for {model_name}: {e}")
        return None

def plot_performance_comparison(performance_df, visualizations_dir):
    metrics = ['Test MCC', 'Test Accuracy', 'Test Log Loss']
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(len(performance_df['Model']))
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, performance_df[metric], bar_width, label=metric)
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.title('Model Performance Comparison (Test Set)')
    plt.xticks(index + bar_width, performance_df['Model'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG["paths"]["visualizations_dir"] / "performance_comparison.png")
    plt.close()
    logger.info("Performance comparison plot saved in visualizations/performance_comparison.png")

def save_model_and_predictions(model, model_name, X_test_data, y_test, test_df, model_name_suffix=""):
    model_path = CONFIG["paths"]["models_dir"] / f"{model_name.lower()}{model_name_suffix}_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved {model_name} model to {model_path}")
    try:
        y_test_pred_proba = model.predict_proba(X_test_data)
        pred_df = pd.DataFrame(y_test_pred_proba, columns=[f'prob_S{i}' for i in CONFIG["expected_classes"]])
        pred_df['true_severity'] = y_test.values
        pred_df['smell_type'] = test_df['smell_type'].values
        pred_path = CONFIG["paths"]["predictions_dir"] / f"{model_name.lower()}{model_name_suffix}_test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Saved {model_name} test predictions to {pred_path}")
    except Exception as e:
        logger.error(f"Error saving predictions for {model_name}: {e}")

def select_important_features(X_train_scaled, y_train, features, n_features):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    top_features = [features[i] for i in indices]
    logger.info(f"Top {n_features} features: {top_features}")
    return top_features, indices

def train():
    logger.info("Starting training...")
    logger.info("Loading processed data...")
    try:
        train_df = pd.read_csv(CONFIG["paths"]["input_train_csv"])
        val_df = pd.read_csv(CONFIG["paths"]["input_val_csv"])
        test_df = pd.read_csv(CONFIG["paths"]["input_test_csv"])
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return None

    expected_columns = ['smell_type', 'severity', 'smell_code'] + CONFIG["numerical_metrics"] + CONFIG["engineered_features"]
    missing_cols = [col for col in expected_columns if col not in train_df.columns]
    if missing_cols:
        logger.error(f"Missing columns in training data: {missing_cols}")
        return None

    logger.info(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    features = CONFIG["numerical_metrics"] + CONFIG["engineered_features"]
    X_train = train_df[features].copy()
    y_train = train_df['severity']
    X_val = val_df[features].copy()
    y_val = val_df['severity']
    X_test = test_df[features].copy()
    y_test = test_df['severity']

    unique_classes = np.unique(y_train)
    if not all(cls in unique_classes for cls in CONFIG["expected_classes"]):
        logger.warning(f"Not all expected classes {CONFIG['expected_classes']} found in training data: {unique_classes}")

    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, CONFIG["paths"]["models_dir"] / "scaler.joblib")
    logger.info(f"Saved StandardScaler to {CONFIG['paths']['models_dir']}/scaler.joblib")

    logger.info("Selecting top features...")
    top_features, top_indices = select_important_features(X_train_scaled, y_train, features, CONFIG["n_features"])
    X_train_selected = X_train_scaled[:, top_indices]
    X_val_selected = X_val_scaled[:, top_indices]
    X_test_selected = X_test_scaled[:, top_indices]

    models = {
        "SVM": SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        ),
        "RandomForest": RandomForestClassifier(
            random_state=42,
            oob_score=True
        ),
        "XGBoost": xgb.XGBClassifier(
            random_state=42,
            eval_metric='mlogloss'
        )
    }

    mcc_scorer = make_scorer(matthews_corrcoef)
    performance_results = []
    smell_f1_results = []

    for model_name, model in models.items():
        logger.info(f"Training {model_name} model...")
        try:
            if model_name == "XGBoost":
                param_grid = {
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.05, 0.1],
                    'reg_lambda': [5.0, 10.0],
                    'reg_alpha': [1.0, 2.0],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.6, 0.8],
                    'min_child_weight': [5, 7, 10]
                }
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=mcc_scorer,
                    cv=5,
                    n_jobs=2,
                    verbose=1
                )
                grid_search.fit(X_train_selected, y_train)
                logger.info(f"Best parameters for XGBoost: {grid_search.best_params_}")
                logger.info(f"Best cross-validation MCC for XGBoost: {grid_search.best_score_:.4f}")
                params = {
                    'objective': 'multi:softprob',
                    'num_class': len(CONFIG["expected_classes"]),
                    'max_depth': grid_search.best_params_['max_depth'],
                    'learning_rate': grid_search.best_params_['learning_rate'],
                    'lambda': grid_search.best_params_['reg_lambda'],
                    'alpha': grid_search.best_params_['reg_alpha'],
                    'subsample': grid_search.best_params_['subsample'],
                    'colsample_bytree': grid_search.best_params_['colsample_bytree'],
                    'min_child_weight': grid_search.best_params_['min_child_weight'],
                    'eval_metric': 'mlogloss',
                    'seed': 42
                }
                dtrain = xgb.DMatrix(X_train_selected, label=y_train)
                dval = xgb.DMatrix(X_val_selected, label=y_val)
                dtest = xgb.DMatrix(X_test_selected, label=y_test)
                evals = [(dtrain, 'train'), (dval, 'val')]
                evals_result = {}
                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=200,
                    evals=evals,
                    evals_result=evals_result,
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                model._Booster = bst
                model.n_classes_ = len(CONFIG["expected_classes"])
                importance = model.get_booster().get_score(importance_type='gain')
                importance = {top_features[int(k.replace('f', ''))]: v for k, v in importance.items()}
                logger.info(f"Feature Importance for {model_name}:")
                for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {feature}: {score:.4f}")
                X_test_data = X_test_selected
            elif model_name == "RandomForest":
                param_grid = {
                    'max_depth': [5, 7],
                    'min_samples_split': [50, 60],
                    'min_samples_leaf': [20, 30],
                    'n_estimators': [100, 300, 500]
                }
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=mcc_scorer,
                    cv=5,
                    n_jobs=2,
                    verbose=1
                )
                grid_search.fit(X_train_selected, y_train)
                logger.info(f"Best parameters for RandomForest: {grid_search.best_params_}")
                logger.info(f"Best cross-validation MCC for RandomForest: {grid_search.best_score_:.4f}")
                base_model = RandomForestClassifier(
                    max_depth=grid_search.best_params_['max_depth'],
                    min_samples_split=grid_search.best_params_['min_samples_split'],
                    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                    n_estimators=grid_search.best_params_['n_estimators'],
                    random_state=42,
                    oob_score=True
                )
                model = CalibratedClassifierCV(base_model, method='sigmoid', cv=10)
                model.fit(X_train_selected, y_train)
                base_model.fit(X_train_selected, y_train)
                oob_score = base_model.oob_score_
                logger.info(f"RandomForest OOB Score: {oob_score:.4f}")
                X_test_data = X_test_selected
            elif model_name == "SVM":
                param_grid = {
                    'C': [5.0, 10.0, 15.0, 20.0],
                    'gamma': [0.01, 0.05, 0.1, 0.5],
                    'kernel': ['rbf']
                }
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=mcc_scorer,
                    cv=5,
                    n_jobs=2,
                    verbose=1
                )
                grid_search.fit(X_train_selected, y_train)
                logger.info(f"Best parameters for SVM: {grid_search.best_params_}")
                logger.info(f"Best cross-validation MCC for SVM: {grid_search.best_score_:.4f}")
                base_model = SVC(
                    C=grid_search.best_params_['C'],
                    gamma=grid_search.best_params_['gamma'],
                    kernel='rbf',
                    random_state=42,
                    probability=True
                )
                model = CalibratedClassifierCV(base_model, method='sigmoid', cv=10)
                model.fit(X_train_selected, y_train)
                X_test_data = X_test_selected

            logger.info(f"Evaluating {model_name} on validation set...")
            y_val_pred = model.predict(X_val_selected)
            y_val_pred_proba = model.predict_proba(X_val_selected)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
            val_log_loss_val = log_loss(y_val, y_val_pred_proba)
            val_report = classification_report(y_val, y_val_pred)
            logger.info(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")
            logger.info(f"{model_name} Validation MCC: {val_mcc:.4f}")
            logger.info(f"{model_name} Validation Macro F1-Score: {val_f1_macro:.4f}")
            logger.info(f"{model_name} Validation Log Loss: {val_log_loss_val:.4f}")
            logger.info(f"{model_name} Validation Classification Report:\n{val_report}")
            val_smell_metrics = compute_smell_severity_metrics(val_df, y_val, y_val_pred, "val", model_name)

            logger.info(f"Evaluating {model_name} on test set...")
            y_test_pred = model.predict(X_test_selected)
            y_test_pred_proba = model.predict_proba(X_test_selected) if model_name != "XGBoost" else bst.predict(dtest)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_mcc = matthews_corrcoef(y_test, y_test_pred)
            test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
            test_log_loss = log_loss(y_test, y_test_pred_proba)
            test_report = classification_report(y_test, y_test_pred)
            logger.info(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"{model_name} Test MCC: {test_mcc:.4f}")
            logger.info(f"{model_name} Test Macro F1-Score: {test_f1_macro:.4f}")
            logger.info(f"{model_name} Test Log Loss: {test_log_loss:.4f}")
            logger.info(f"{model_name} Test Classification Report:\n{test_report}")

            test_smell_metrics = compute_smell_severity_metrics(test_df, y_test, y_test_pred, "test", model_name)
            performance_results.append({
                'Model': model_name,
                'Val MCC': val_mcc,
                'Val Accuracy': val_accuracy,
                'Val Macro F1': val_f1_macro,
                'Val Log Loss': val_log_loss_val,
                'Test MCC': test_mcc,
                'Test Accuracy': test_accuracy,
                'Test Macro F1': test_f1_macro,
                'Test Log Loss': test_log_loss
            })

            if test_smell_metrics is not None:
                test_smell_metrics.to_csv(CONFIG["paths"]["predictions_dir"] / f"{model_name.lower()}_test_smell_metrics.csv", index=False)
                logger.info(f"Saved {model_name} test smell-severity metrics to {CONFIG['paths']['predictions_dir']}/{model_name.lower()}_test_smell_metrics.csv")
                for _, row in test_smell_metrics.iterrows():
                    smell_f1_results.append({
                        'Model': model_name,
                        'Smell_Type': row['smell_type'],
                        'Severity': row['true_severity'],
                        'F1': row['f1']
                    })

            save_model_and_predictions(model, model_name, X_test_selected, y_test, test_df)
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue

    performance_df = pd.DataFrame(performance_results)
    logger.info("\nFinal Performance Table (Validation and Test Set):")
    logger.info(performance_df.to_string(index=False))
    performance_df.to_csv(CONFIG["paths"]["performance_tables_dir"] / "performance_table.csv", index=False)
    logger.info(f"Saved performance table to {CONFIG['paths']['performance_tables_dir']}/performance_table.csv")

    smell_f1_df = pd.DataFrame(smell_f1_results)
    smell_f1_pivot = smell_f1_df.pivot_table(values='F1', index=['Smell_Type', 'Severity'], columns='Model')
    logger.info("\nSmell-Specific F1-Score Table:")
    logger.info(smell_f1_pivot.to_string())
    smell_f1_pivot.to_csv(CONFIG["paths"]["performance_tables_dir"] / "smell_f1_table.csv")
    logger.info(f"Saved smell-specific F1-score table to {CONFIG['paths']['performance_tables_dir']}/smell_f1_table.csv")

    plot_performance_comparison(performance_df, CONFIG["paths"]["visualizations_dir"])

    logger.info("\n--- Summary of Key Metrics ---")
    for _, row in performance_df.iterrows():
        logger.info(f"{row['Model']}: Val MCC={row['Val MCC']:.4f}, Val Accuracy={row['Val Accuracy']:.4f}, Val Log Loss={row['Val Log Loss']:.4f}")
        logger.info(f"{row['Model']}: Test MCC={row['Test MCC']:.4f}, Test Accuracy={row['Test Accuracy']:.4f}, Test Log Loss={row['Test Log Loss']:.4f}")

    logger.info("Training completed.")
    return performance_df

if __name__ == "__main__":
    logger.info("Running experiment...")
    perf_df = train()
    if perf_df is not None:
        logger.info("\nPerformance Table (Validation and Test Set):")
        logger.info(perf_df.to_string(index=False))
