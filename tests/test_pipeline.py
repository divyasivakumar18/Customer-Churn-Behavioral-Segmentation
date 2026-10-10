"""
tests/test_pipeline.py
----------------------
Automated Unit & Integration Test Suite for the Customer Churn & Segmentation System.
Tests data preprocessing, feature engineering, RFM scoring, clustering, model serialization,
and real-time inference edge cases.
"""

import os
import sys
import unittest
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import clean_raw_data, get_feature_groups
from src.feature_engineering import create_engineered_features
from src.rfm_analysis import compute_rfm_metrics, assign_rfm_segments
from src.customer_prediction import ChurnPredictorEngine


class TestCustomerChurnPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes shared test data and engine."""
        cls.raw_data_path = os.path.join('data', 'raw', 'Telco-Customer-Churn.csv')
        cls.cleaned_data_path = os.path.join('data', 'processed', 'telco_cleaned.csv')
        cls.engineered_data_path = os.path.join('data', 'processed', 'telco_engineered.csv')
        cls.clustered_data_path = os.path.join('data', 'processed', 'telco_clustered.csv')
        cls.models_dir = os.path.join('models')
        cls.engine = ChurnPredictorEngine(models_dir=cls.models_dir)

    def test_01_raw_dataset_exists_and_valid(self):
        """Verifies raw CSV exists and contains 7043 rows and 21 columns."""
        self.assertTrue(os.path.exists(self.raw_data_path), "Raw dataset file not found!")
        df_raw = pd.read_csv(self.raw_data_path)
        self.assertEqual(df_raw.shape, (7043, 21), "Raw dataset shape mismatch!")
        self.assertIn('customerID', df_raw.columns)
        self.assertIn('Churn', df_raw.columns)

    def test_02_data_cleaning_and_type_coercion(self):
        """Verifies clean_raw_data eliminates whitespace and converts TotalCharges to float."""
        df_raw = pd.read_csv(self.raw_data_path)
        df_clean = clean_raw_data(df_raw)
        
        self.assertEqual(df_clean.isnull().sum().sum(), 0, "Null values remain after cleaning!")
        self.assertEqual(df_clean.duplicated().sum(), 0, "Duplicates found after cleaning!")
        self.assertEqual(df_clean['TotalCharges'].dtype, np.float64, "TotalCharges is not float64!")
        
        # Verify 0.0 imputation on tenure == 0
        tenure_zero = df_clean[df_clean['tenure'] == 0]
        self.assertTrue((tenure_zero['TotalCharges'] == 0.0).all(), "tenure=0 rows not imputed with 0.0!")

    def test_03_feature_engineering_integrity(self):
        """Verifies engineered features are computed accurately without NaNs."""
        df_clean = pd.read_csv(self.cleaned_data_path)
        df_feat = create_engineered_features(df_clean)

        expected_features = [
            'tenure_group', 'tenure_years', 'total_services_count',
            'security_support_count', 'streaming_services_count', 'is_autopay',
            'has_family', 'contract_commitment_months', 'historical_avg_monthly_charges',
            'monthly_charge_discrepancy', 'cost_per_service'
        ]
        for feat in expected_features:
            self.assertIn(feat, df_feat.columns, f"Engineered feature '{feat}' missing!")
            self.assertEqual(df_feat[feat].isnull().sum(), 0, f"NaNs in '{feat}'!")

    def test_04_rfm_segmentation_logic(self):
        """Verifies RFM scoring and segment assignment validity."""
        df_eng = pd.read_csv(self.engineered_data_path)
        df_rfm = compute_rfm_metrics(df_eng)
        df_rfm = assign_rfm_segments(df_rfm)

        self.assertIn('RFM_Segment', df_rfm.columns)
        self.assertEqual(df_rfm['RFM_Segment'].isnull().sum(), 0)
        
        # Verify all segment labels are non-empty strings
        unique_segs = df_rfm['RFM_Segment'].unique()
        self.assertIn('Champions', unique_segs)
        self.assertIn('Loyal Customers', unique_segs)

    def test_05_saved_models_integrity(self):
        """Verifies all serialized .pkl models load properly and have predict methods."""
        required_models = [
            'best_tuned_churn_model.pkl',
            'kmeans_clustering_model.pkl',
            'clustering_scaler.pkl',
            'feature_names.pkl',
            'raw_feature_cols.pkl'
        ]
        for m in required_models:
            m_path = os.path.join(self.models_dir, m)
            self.assertTrue(os.path.exists(m_path), f"Required model file '{m}' missing!")
            obj = joblib.load(m_path)
            self.assertIsNotNone(obj)

    def test_06_edge_case_brand_new_customer(self):
        """Tests inference on a brand-new customer with tenure = 0 and TotalCharges = 0."""
        new_customer = {
            'customerID': 'TEST-ZERO-TENURE',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 0,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.0,
            'TotalCharges': 0.0
        }
        res = self.engine.predict_single_customer(new_customer)
        self.assertIn('churn_probability', res)
        self.assertIn('churn_prediction', res)
        self.assertIn('risk_tier', res)
        self.assertTrue(0.0 <= res['churn_probability'] <= 1.0)

    def test_07_edge_case_tenured_vip_customer(self):
        """Tests inference on a 72-month VIP customer."""
        vip_customer = {
            'customerID': 'TEST-VIP-72',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 72,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 65.0,
            'TotalCharges': 4680.0
        }
        res = self.engine.predict_single_customer(vip_customer)
        self.assertEqual(res['risk_tier'], 'Low Risk')
        self.assertEqual(res['churn_prediction'], 'Retained')
        self.assertLess(res['churn_probability'], 0.25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
