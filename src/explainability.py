"""
src/explainability.py
---------------------
Explainable AI (XAI) and Feature Importance Module.
Extracts native tree feature importances, Logistic Regression odds ratios,
and applies SHAP (SHapley Additive exPlanations) for global and local interpretability.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from src.train_model import prepare_features_and_target


INPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_clustered.csv')
MODELS_DIR = os.path.join('models')
REPORTS_DIR = os.path.join('reports')
FIGURES_DIR = os.path.join('reports', 'figures')


def load_models_and_test_data():
    """
    Loads saved tuned pipelines, preprocessor, and generates test split.
    """
    df = pd.read_csv(INPUT_DATA_PATH)
    X, y, num_cols, cat_cols = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    xgb_pipe = joblib.load(os.path.join(MODELS_DIR, 'best_tuned_churn_model.pkl'))
    rf_pipe = joblib.load(os.path.join(MODELS_DIR, 'rf_churn_model.pkl'))
    lr_pipe = joblib.load(os.path.join(MODELS_DIR, 'logreg_churn_model.pkl'))

    preprocessor = xgb_pipe.named_steps['preprocessor']
    xgb_clf = xgb_pipe.named_steps['classifier']
    rf_clf = rf_pipe.named_steps['classifier']
    lr_clf = lr_pipe.named_steps['classifier']

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Clean feature names for presentation
    clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

    return {
        'xgb_pipe': xgb_pipe,
        'rf_pipe': rf_pipe,
        'lr_pipe': lr_pipe,
        'xgb_clf': xgb_clf,
        'rf_clf': rf_clf,
        'lr_clf': lr_clf,
        'preprocessor': preprocessor,
        'X_test': X_test,
        'X_test_transformed': X_test_transformed,
        'feature_names': clean_feature_names,
        'y_test': y_test
    }


def compute_feature_importance(xgb_clf, feature_names: list) -> pd.DataFrame:
    """
    Extracts native tree feature importances (Gain).
    """
    importances = xgb_clf.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return imp_df


def compute_logistic_odds_ratios(lr_clf, feature_names: list) -> pd.DataFrame:
    """
    Extracts Logistic Regression coefficients and converts them to Odds Ratios (exp(beta)).
    """
    coefs = lr_clf.coef_[0]
    odds_ratios = np.exp(coefs)
    lr_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Odds_Ratio': odds_ratios
    }).sort_values(by='Odds_Ratio', ascending=False).reset_index(drop=True)
    return lr_df


def compute_shap_explanations(rf_clf, X_test_transformed: np.ndarray, feature_names: list):
    """
    Fits TreeExplainer on tree ensemble to compute SHAP values.
    """
    explainer = shap.TreeExplainer(rf_clf)
    shap_raw = explainer.shap_values(X_test_transformed[:500])
    
    # In binary classification, class 1 is the churn class (index 1)
    if isinstance(shap_raw, list):
        shap_values_class1 = shap_raw[1]
    elif len(shap_raw.shape) == 3:
        shap_values_class1 = shap_raw[:, :, 1]
    else:
        shap_values_class1 = shap_raw

    return explainer, shap_values_class1


def plot_and_save_xai_figures(imp_df: pd.DataFrame, lr_df: pd.DataFrame, shap_values, X_test_transformed: np.ndarray, feature_names: list):
    """
    Generates high-resolution SHAP summary, beeswarm, and feature importance charts.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1. Native Feature Importance Top 15 Bar Chart
    plt.figure(figsize=(10, 6.5))
    top_15_xgb = imp_df.head(15).sort_values(by='Importance', ascending=True)
    plt.barh(top_15_xgb['Feature'], top_15_xgb['Importance'], color='#1f77b4', edgecolor='black', alpha=0.85)
    plt.title('Top 15 Most Important Features (XGBoost Feature Importance)', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score (Gain)', fontsize=11)
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, '18_native_feature_importance.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. SHAP Summary Bar Chart
    plt.figure(figsize=(10, 6.5))
    X_df_sample = pd.DataFrame(X_test_transformed[:500], columns=feature_names)
    shap.summary_plot(shap_values, X_df_sample, plot_type='bar', show=False, max_display=15)
    plt.title('Top 15 Global Features by Mean |SHAP Value| (Impact on Churn Output)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, '19_shap_summary_bar.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # 3. SHAP Beeswarm Plot (Directional Impact on Churn)
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, X_df_sample, show=False, max_display=15)
    plt.title('SHAP Beeswarm Plot (Feature Value Direction vs Churn Probability)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, '20_shap_beeswarm.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")


def generate_business_interpretation_table(imp_df: pd.DataFrame, shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Compiles structured ranked table mapping technical importance to business meaning.
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP_Impact': mean_shap
    }).sort_values(by='Mean_SHAP_Impact', ascending=False).reset_index(drop=True)

    business_meanings = {
        'tenure': 'Higher tenure strongly lowers churn. Long-term customers have high brand loyalty and high switching friction.',
        'TotalCharges': 'Cumulative revenue proxy. High total charges reflect stable long-standing customer lifecycle.',
        'MonthlyCharges': 'High monthly billing increases price sensitivity, especially if service disruptions occur.',
        'contract_commitment_months': 'Higher commitment terms (1 or 2 years) create contract lock-in, slashing churn to <3%.',
        'Contract_Two year': 'Two-year contracts drastically suppress churn probability.',
        'Contract_One year': 'One-year contracts offer strong stability compared to Month-to-month.',
        'InternetService_Fiber optic': 'Fiber optic is a high-cost tier ($70-$100+); price-to-service dissatisfaction drives higher churn.',
        'PaymentMethod_Electronic check': 'Manual paper/electronic checks cause monthly payment friction and drop-offs.',
        'is_autopay': 'Enabling automatic payments fosters habitual retention and reduces involuntary churn.',
        'security_support_count': 'Subscribing to Tech Support & Security features significantly increases platform stickiness.',
        'total_services_count': 'Ecosystem breadth: customers with 4+ services rarely leave due to bundling friction.',
        'tenure_years': 'Years of customer tenure; churn drops below 10% after 2 years of service.',
        'TechSupport_Yes': 'Direct access to tech assistance resolves frustrations before they lead to cancellation.',
        'OnlineSecurity_Yes': 'Security add-on improves perceived product value and protection inertia.',
        'PaperlessBilling_Yes': 'Slightly higher churn among paperless billing users due to ease of online cancellation.'
    }

    shap_ranking['Business Meaning & Actionable Driver'] = shap_ranking['Feature'].map(
        lambda f: business_meanings.get(f, 'Contributing behavioral / demographic attribute.')
    )
    shap_ranking['Rank'] = range(1, len(shap_ranking) + 1)
    return shap_ranking


def run_explainability_pipeline():
    """
    Main orchestration routine for Phase 9.
    """
    print("=" * 95)
    print("                 PHASE 9: FEATURE IMPORTANCE & EXPLAINABLE AI (SHAP)             ")
    print("=" * 95)

    print("[1/4] Loading trained model pipelines and transforming test set...")
    bundle = load_models_and_test_data()

    print("[2/4] Extracting native feature importances, odds ratios & computing SHAP values...")
    imp_df = compute_feature_importance(bundle['xgb_clf'], bundle['feature_names'])
    lr_df = compute_logistic_odds_ratios(bundle['lr_clf'], bundle['feature_names'])
    explainer, shap_values = compute_shap_explanations(
        bundle['rf_clf'], bundle['X_test_transformed'], bundle['feature_names']
    )

    # Save explainer
    joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer.pkl'))

    print("[3/4] Generating publication-quality interpretability charts...")
    plot_and_save_xai_figures(
        imp_df, lr_df, shap_values, bundle['X_test_transformed'], bundle['feature_names']
    )

    print("\n[4/4] TOP 10 CHURN DRIVERS & BUSINESS INTERPRETATION:")
    print("-" * 95)
    ranking_table = generate_business_interpretation_table(imp_df, shap_values, bundle['feature_names'])
    top_10 = ranking_table.head(10)

    print(f"{'Rank':<5} | {'Feature Name':<28} | {'Mean |SHAP|':<12} | {'Business Meaning'}")
    print("-" * 95)
    for _, row in top_10.iterrows():
        print(f"{row['Rank']:<5} | {row['Feature']:<28} | {row['Mean_SHAP_Impact']:<12.4f} | {row['Business Meaning & Actionable Driver'][:45]}...")

    # Save ranked table
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ranking_csv = os.path.join(REPORTS_DIR, 'feature_importance_ranking.csv')
    ranking_table.to_csv(ranking_csv, index=False)
    print(f"\n[SUCCESS] Feature importance ranking saved to '{ranking_csv}'")
    print("=" * 95)

    return ranking_table


if __name__ == '__main__':
    run_explainability_pipeline()
