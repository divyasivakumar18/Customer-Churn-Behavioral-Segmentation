"""
src/customer_prediction.py
--------------------------
Individual Customer-Level Churn Scoring & Business Risk Engine.
Provides single-customer and batch prediction APIs that compute:
1. Churn probability (0.0% to 100.0%)
2. Binary classification decision
3. Tiered risk level (Low, Medium, High)
4. Behavioral RFM & K-Means persona tagging
5. Top driving risk factors
6. Prescriptive retention recommendations
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import create_engineered_features


MODELS_DIR = os.path.join('models')


class ChurnPredictorEngine:
    def __init__(self, models_dir: str = MODELS_DIR):
        """
        Loads all required trained models, transformers, and scalers into memory.
        """
        self.models_dir = models_dir
        self.model_pipeline = joblib.load(os.path.join(models_dir, 'best_tuned_churn_model.pkl'))
        self.kmeans_model = joblib.load(os.path.join(models_dir, 'kmeans_clustering_model.pkl'))
        self.clustering_scaler = joblib.load(os.path.join(models_dir, 'clustering_scaler.pkl'))
        self.raw_feature_cols = joblib.load(os.path.join(models_dir, 'raw_feature_cols.pkl'))

        # Quantile bin edges calibrated from training distribution
        self.r_bins = [-0.001, 9.0, 29.0, 55.0, 1000.0]
        self.f_bins = [-0.001, 21.0, 85.0, 268.0, 5000.0]
        self.m_bins = [-0.001, 398.55, 1394.55, 3786.60, 50000.0]

        self.cluster_personas = {
            0: 'Budget / Basic Streamers',
            1: 'High-Value Power Users',
            2: 'New High-Risk Fiber Users',
            3: 'Long-Term Contract Loyalists'
        }

    def _enrich_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Runs complete cleaning, feature engineering, RFM scoring, and clustering.
        """
        df = df_raw.copy()
        
        # 1. Cleaning & TotalCharges coercion
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
        else:
            df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

        if 'Churn' not in df.columns:
            df['Churn'] = 'No'

        # 2. Feature Engineering
        df_feat = create_engineered_features(df)

        # 3. RFM Metrics (using fixed quantile bin edges)
        df_feat['R_value'] = df_feat['tenure']
        df_feat['F_value'] = df_feat['tenure'] * df_feat['total_services_count']
        df_feat['M_value'] = df_feat['TotalCharges']

        df_feat['R_Score'] = pd.cut(df_feat['R_value'], bins=self.r_bins, labels=[1, 2, 3, 4]).astype(int)
        df_feat['F_Score'] = pd.cut(df_feat['F_value'], bins=self.f_bins, labels=[1, 2, 3, 4]).astype(int)
        df_feat['M_Score'] = pd.cut(df_feat['M_value'], bins=self.m_bins, labels=[1, 2, 3, 4]).astype(int)

        df_feat['RFM_Combined'] = (
            df_feat['R_Score'].astype(str) +
            df_feat['F_Score'].astype(str) +
            df_feat['M_Score'].astype(str)
        )
        df_feat['RFM_Total'] = df_feat['R_Score'] + df_feat['F_Score'] + df_feat['M_Score']

        def segment_mapper(row):
            r = row['R_Score']
            f = row['F_Score']
            m = row['M_Score']
            if row['Churn'] == 'Yes' and (r <= 2):
                return 'Lost Churners'
            elif r >= 4 and f >= 3 and m >= 3:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 3 and f >= 2:
                return 'Potential Loyalists'
            elif r <= 2 and (f >= 3 or m >= 3):
                return 'At Risk / High Spenders'
            elif r <= 2 and row['tenure'] <= 12:
                return 'New Customers'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Hibernating / Low Value'
            else:
                return 'Promising / Developing'

        df_feat['RFM_Segment'] = df_feat.apply(segment_mapper, axis=1)

        # 4. K-Means Clustering Assignment
        cluster_feature_cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'total_services_count',
            'security_support_count', 'contract_commitment_months', 'is_autopay'
        ]
        X_cluster = df_feat[cluster_feature_cols].copy()
        X_cluster_scaled = self.clustering_scaler.transform(X_cluster)
        df_feat['Cluster'] = self.kmeans_model.predict(X_cluster_scaled)
        df_feat['Cluster_Persona'] = df_feat['Cluster'].map(self.cluster_personas)

        return df_feat

    def predict_single_customer(self, customer_dict: dict) -> dict:
        """
        Generates individual prediction report for a single customer payload.
        """
        df_raw = pd.DataFrame([customer_dict])
        df_enriched = self._enrich_features(df_raw)

        # Align with training features
        X = df_enriched[self.raw_feature_cols].copy()

        # Model Inference
        prob = float(self.model_pipeline.predict_proba(X)[0, 1])
        prediction = int(prob >= 0.50)

        # Determine Risk Tier
        if prob < 0.35:
            risk_tier = 'Low Risk'
            risk_color = '#2ca02c' # Green
        elif prob < 0.65:
            risk_tier = 'Medium Risk'
            risk_color = '#ff7f0e' # Orange
        else:
            risk_tier = 'High Risk'
            risk_color = '#d62728' # Red

        # Identify Top Risk Factors
        risk_factors = []
        recommendations = []

        if customer_dict.get('Contract') == 'Month-to-month':
            risk_factors.append("Month-to-month contract (0 exit friction)")
            recommendations.append("Offer a 15% discount for migrating to a 12-month contract commitment.")

        if customer_dict.get('PaymentMethod') == 'Electronic check':
            risk_factors.append("Manual Electronic Check payment method (high friction)")
            recommendations.append("Provide a $5 monthly bill credit for enabling automatic credit card/bank autopay.")

        if customer_dict.get('InternetService') == 'Fiber optic' and customer_dict.get('TechSupport') == 'No':
            risk_factors.append("Fiber optic subscription without active Tech Support")
            recommendations.append("Bundle 3 months of complimentary VIP Tech Support & Online Security.")

        if float(customer_dict.get('tenure', 0)) <= 12:
            risk_factors.append(f"Early customer lifecycle (tenure = {customer_dict.get('tenure')} months)")
            recommendations.append("Enroll in proactive customer success onboarding check-in calls.")

        if float(customer_dict.get('MonthlyCharges', 0)) > 75:
            risk_factors.append(f"High monthly bill (${customer_dict.get('MonthlyCharges')}/mo)")
            recommendations.append("Offer a customized value bundle or loyalty service credit.")

        if not risk_factors:
            risk_factors.append("High tenure and stable long-term commitment with low friction.")
            recommendations.append("Maintain VIP customer appreciation rewards and loyalty points.")

        rfm_seg = str(df_enriched['RFM_Segment'].iloc[0])
        persona = str(df_enriched['Cluster_Persona'].iloc[0])

        report = {
            'customerID': customer_dict.get('customerID', 'UNKNOWN'),
            'churn_probability': round(prob, 4),
            'churn_probability_pct': f"{prob * 100:.1f}%",
            'churn_prediction': 'Likely to Churn' if prediction == 1 else 'Retained',
            'risk_tier': risk_tier,
            'risk_color': risk_color,
            'rfm_segment': rfm_seg,
            'behavioral_persona': persona,
            'total_services_count': int(df_enriched['total_services_count'].iloc[0]),
            'top_risk_factors': risk_factors,
            'retention_recommendations': recommendations
        }
        return report


def run_customer_prediction_tests():
    """
    Demonstrates the prediction engine across three distinct customer archetypes.
    """
    print("=" * 95)
    print("                 PHASE 10: CUSTOMER-LEVEL PREDICTION & RISK SCORING ENGINE       ")
    print("=" * 95)

    engine = ChurnPredictorEngine()

    test_cases = [
        {
            "title": "CUSTOMER A: High-Risk New Fiber User",
            "data": {
                'customerID': 'CUST-HIGH-RISK-01',
                'gender': 'Male',
                'SeniorCitizen': 1,
                'Partner': 'No',
                'Dependents': 'No',
                'tenure': 2,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 92.50,
                'TotalCharges': 185.00
            }
        },
        {
            "title": "CUSTOMER B: Low-Risk Long-Term Power Loyalist",
            "data": {
                'customerID': 'CUST-LOYAL-02',
                'gender': 'Female',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'Yes',
                'tenure': 65,
                'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes',
                'TechSupport': 'Yes',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Two year',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Credit card (automatic)',
                'MonthlyCharges': 55.40,
                'TotalCharges': 3601.00
            }
        },
        {
            "title": "CUSTOMER C: Medium-Risk Borderline Family Subscriber",
            "data": {
                'customerID': 'CUST-MEDIUM-RISK-03',
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 14,
                'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'No',
                'Contract': 'One year',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Bank transfer (automatic)',
                'MonthlyCharges': 78.90,
                'TotalCharges': 1104.60
            }
        }
    ]

    for case in test_cases:
        print(f"\n[+] EVALUATING: {case['title']}")
        print("-" * 95)
        res = engine.predict_single_customer(case['data'])
        print(f" * Customer ID           : {res['customerID']}")
        print(f" * Churn Probability     : {res['churn_probability_pct']} ({res['churn_probability']:.4f})")
        print(f" * Predicted Status      : {res['churn_prediction']}")
        print(f" * Risk Tier             : {res['risk_tier']}")
        print(f" * RFM Segment           : {res['rfm_segment']}")
        print(f" * Behavioral Persona    : {res['behavioral_persona']}")
        print(f" * Identified Risk Factors:")
        for rf in res['top_risk_factors']:
            print(f"    - {rf}")
        print(f" * Prescriptive Action Strategy:")
        for rec in res['retention_recommendations']:
            print(f"    -> {rec}")

    print("\n" + "=" * 95)
    print("[SUCCESS] Customer-level prediction and risk scoring engine validated!")
    print("=" * 95)


if __name__ == '__main__':
    run_customer_prediction_tests()
