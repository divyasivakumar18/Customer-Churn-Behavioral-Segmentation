"""
src/train_model.py
------------------
Supervised Machine Learning Churn Prediction Pipeline.
Trains, evaluates, and benchmarks 5 algorithms (Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting, XGBoost) using stratified train-test splitting
and leakage-free Scikit-Learn ColumnTransformer pipelines.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ML Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


INPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_clustered.csv')
REPORTS_DIR = os.path.join('reports')
FIGURES_DIR = os.path.join('reports', 'figures')
MODELS_DIR = os.path.join('models')


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separates feature matrix X and target y, establishing clean feature subsets.
    """
    # Drop identifiers, raw leak proxies, and coordinate visual projections
    drop_cols = ['customerID', 'Churn', 'PCA1', 'PCA2', 'RFM_Combined']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = (df['Churn'] == 'Yes').astype(int)

    # Distinguish numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return X, y, num_cols, cat_cols


def build_pipeline_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Builds Scikit-Learn ColumnTransformer to scale numerics and One-Hot encode categoricals.
    """
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ]
    )
    return preprocessor


def initialize_candidate_models():
    """
    Returns a dictionary of standard benchmark classifiers configured for fair comparison.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, scale_pos_weight=2.76, random_state=42, eval_metric='logloss')
    }
    return models


def train_and_evaluate_all_models(X_train, X_test, y_train, y_test, preprocessor, models: dict):
    """
    Trains all models inside full pipelines, computes test metrics, and extracts predictions.
    """
    results = []
    fitted_pipelines = {}
    test_predictions = {}
    test_probabilities = {}

    for name, clf in models.items():
        print(f"   -> Training: {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Fit strictly on train split
        pipeline.fit(X_train, y_train)

        # Inference on test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall (Churn)': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        })

        fitted_pipelines[name] = pipeline
        test_predictions[name] = y_pred
        test_probabilities[name] = y_prob

    results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    return results_df, fitted_pipelines, test_predictions, test_probabilities


def plot_model_evaluation_visualizations(y_test, fitted_pipelines, test_predictions, test_probabilities):
    """
    Generates ROC curves, PR curves, and Confusion Matrices for all models.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1. ROC Curves
    plt.figure(figsize=(9, 6))
    for name, y_prob in test_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, '14_roc_curves_comparison.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Precision-Recall Curves
    plt.figure(figsize=(9, 6))
    for name, y_prob in test_probabilities.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        plt.plot(rec, prec, linewidth=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    plt.title('Precision-Recall Curve Comparison', fontsize=13, fontweight='bold')
    plt.xlabel('Recall (Sensitivity for Churn)', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, '15_precision_recall_curves.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Confusion Matrix Subplots
    fig, axes = plt.subplots(1, len(test_predictions), figsize=(20, 3.8))
    for i, (name, y_pred) in enumerate(test_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Retained', 'Churn'], yticklabels=['Retained', 'Churn'])
        axes[i].set_title(f"{name}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        if i == 0:
            axes[i].set_ylabel('Actual')

    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, '16_confusion_matrices.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")


def run_training_pipeline(input_path: str = INPUT_DATA_PATH) -> tuple:
    """
    Executes end-to-end churn model training, evaluation, comparison, and persistence.
    """
    print("=" * 90)
    print("                 PHASE 7: SUPERVISED CHURN MODEL TRAINING & BENCHMARKING          ")
    print("=" * 90)

    print(f"[1/5] Loading clustered dataset from '{input_path}'...")
    df = pd.read_csv(input_path)

    print("[2/5] Preparing feature matrix & stratified train-test split (80/20)...")
    X, y, num_cols, cat_cols = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"      Train Set: {X_train.shape[0]} records | Test Set: {X_test.shape[0]} records")
    print(f"      Train Churn: {y_train.sum()} ({y_train.mean()*100:.1f}%) | Test Churn: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    print("[3/5] Constructing preprocessing transformer...")
    preprocessor = build_pipeline_preprocessor(num_cols, cat_cols)

    print("[4/5] Training candidate classifiers & calculating evaluation metrics...")
    models = initialize_candidate_models()
    results_df, fitted_pipelines, test_preds, test_probs = train_and_evaluate_all_models(
        X_train, X_test, y_train, y_test, preprocessor, models
    )

    plot_model_evaluation_visualizations(y_test, fitted_pipelines, test_preds, test_probs)

    print("\n" + "-" * 90)
    print("MODEL PERFORMANCE COMPARISON (TEST SET):")
    print("-" * 90)
    print(results_df.to_string(index=False))

    # Save benchmark table
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results_csv = os.path.join(REPORTS_DIR, 'model_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n[5/5] Saved model comparison results to '{results_csv}'")

    # Select best model based on ROC-AUC & Recall balance (Random Forest / Gradient Boosting / XGBoost)
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = fitted_pipelines[best_model_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'best_churn_pipeline.pkl'))
    joblib.dump(fitted_pipelines['XGBoost'], os.path.join(MODELS_DIR, 'xgboost_churn_model.pkl'))
    joblib.dump(fitted_pipelines['Random Forest'], os.path.join(MODELS_DIR, 'rf_churn_model.pkl'))
    joblib.dump(fitted_pipelines['Logistic Regression'], os.path.join(MODELS_DIR, 'logreg_churn_model.pkl'))

    # Save feature names from fitted preprocessor
    fitted_preprocessor = best_pipeline.named_steps['preprocessor']
    feature_names = fitted_preprocessor.get_feature_names_out().tolist()
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, 'raw_feature_cols.pkl'))

    print(f"      * Best Model: '{best_model_name}' saved to '{MODELS_DIR}/best_churn_pipeline.pkl'")
    print("=" * 90)

    return results_df, fitted_pipelines, (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    run_training_pipeline()
