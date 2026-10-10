"""
src/tune_model.py
-----------------
Hyperparameter Optimization & K-Fold Cross-Validation Module.
Performs GridSearchCV / RandomizedSearchCV across top candidate models,
reports CV scores, compares pre- vs post-tuning test metrics, and serializes
the production-ready tuned model pipeline.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.train_model import prepare_features_and_target, build_pipeline_preprocessor


INPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_clustered.csv')
MODELS_DIR = os.path.join('models')
REPORTS_DIR = os.path.join('reports')
FIGURES_DIR = os.path.join('reports', 'figures')


def setup_tuning_grids(preprocessor):
    """
    Constructs search pipelines and parameter grids for top candidate algorithms.
    """
    # 1. XGBoost
    xgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(scale_pos_weight=2.76, random_state=42, eval_metric='logloss'))
    ])
    xgb_grid = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.03, 0.08, 0.12],
        'classifier__subsample': [0.8, 1.0]
    }

    # 2. Random Forest
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    rf_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [6, 8, 12],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # 3. Logistic Regression
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    lr_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear']
    }

    tuning_configs = {
        'XGBoost': (xgb_pipe, xgb_grid),
        'Random Forest': (rf_pipe, rf_grid),
        'Logistic Regression': (lr_pipe, lr_grid)
    }
    return tuning_configs


def execute_hyperparameter_tuning(X_train, y_train, X_test, y_test, tuning_configs: dict):
    """
    Runs 5-Fold Stratified Cross-Validation tuning for each candidate model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuning_summary = []
    best_pipelines = {}

    for name, (pipe, param_grid) in tuning_configs.items():
        print(f"\n[+] Optimizing {name} with 5-Fold Stratified CV (GridSearchCV)...")
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_auc = grid_search.best_score_

        # Evaluate on Held-out Test Split
        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_prob)
        test_pr_auc = average_precision_score(y_test, y_prob)

        tuning_summary.append({
            'Model': name,
            'Best_CV_ROC_AUC': round(best_cv_auc, 4),
            'Test_ROC_AUC': round(test_auc, 4),
            'Test_PR_AUC': round(test_pr_auc, 4),
            'Test_Recall': round(rec, 4),
            'Test_Precision': round(prec, 4),
            'Test_F1': round(f1, 4),
            'Test_Accuracy': round(acc, 4),
            'Best_Parameters': str(best_params)
        })

        best_pipelines[name] = best_pipe
        print(f"    * Best 5-Fold CV ROC-AUC: {best_cv_auc:.4f}")
        print(f"    * Test ROC-AUC: {test_auc:.4f} | Recall: {rec*100:.2f}% | F1: {f1:.4f}")

    summary_df = pd.DataFrame(tuning_summary).sort_values(by='Test_ROC_AUC', ascending=False)
    return summary_df, best_pipelines


def plot_tuning_comparison(summary_df: pd.DataFrame) -> None:
    """
    Plots a grouped bar chart comparing tuned models across key metrics.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style='whitegrid')

    metrics = ['Test_ROC_AUC', 'Test_PR_AUC', 'Test_Recall', 'Test_F1', 'Test_Accuracy']
    plot_df = summary_df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model', palette='tab10', edgecolor='black')
    plt.title('Performance Comparison of Tuned ML Models on Test Set', fontsize=13, fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)', fontsize=11)
    plt.xlabel('')
    plt.ylim(0.7, 1.0)
    plt.xticks(ticks=range(len(metrics)),
               labels=['ROC-AUC', 'PR-AUC', 'Recall (Churn)', 'F1-Score', 'Accuracy'],
               fontsize=10)
    plt.legend(title='Tuned Algorithm', loc='lower left')
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, '17_tuning_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def run_hyperparameter_tuning_pipeline():
    """
    Main orchestration routine for Phase 8.
    """
    print("=" * 90)
    print("                 PHASE 8: HYPERPARAMETER TUNING & MODEL OPTIMIZATION             ")
    print("=" * 90)

    # 1. Load data and setup splits
    df = pd.read_csv(INPUT_DATA_PATH)
    X, y, num_cols, cat_cols = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 2. Build preprocessor
    preprocessor = build_pipeline_preprocessor(num_cols, cat_cols)

    # 3. Setup configurations and execute tuning
    configs = setup_tuning_grids(preprocessor)
    summary_df, best_pipelines = execute_hyperparameter_tuning(X_train, y_train, X_test, y_test, configs)

    # 4. Plot comparison
    plot_tuning_comparison(summary_df)

    # 5. Save results and production model
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary_csv = os.path.join(REPORTS_DIR, 'hyperparameter_tuning_results.csv')
    summary_df.to_csv(summary_csv, index=False)

    best_overall_name = summary_df.iloc[0]['Model']
    best_overall_pipe = best_pipelines[best_overall_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    final_model_path = os.path.join(MODELS_DIR, 'best_tuned_churn_model.pkl')
    joblib.dump(best_overall_pipe, final_model_path)

    print("\n" + "-" * 90)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY:")
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(f"Model: {row['Model']:<20} | CV ROC-AUC: {row['Best_CV_ROC_AUC']} | Test ROC-AUC: {row['Test_ROC_AUC']} | Recall: {row['Test_Recall']*100:.2f}% | F1: {row['Test_F1']}")
        print(f"   Best Parameters: {row['Best_Parameters']}\n")

    print(f"[SUCCESS] Best overall tuned pipeline ('{best_overall_name}') saved to '{final_model_path}'")
    print("=" * 90)

    return summary_df, best_pipelines


if __name__ == '__main__':
    run_hyperparameter_tuning_pipeline()
