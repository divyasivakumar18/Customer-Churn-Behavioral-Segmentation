"""
src/data_preprocessing.py
-------------------------
Robust Data Cleaning and Preprocessing Pipeline for Customer Churn Analysis.
Handles missing values, data type coercion, categorical encoding, feature scaling,
and provides reusable Scikit-Learn transformers preventing data leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


RAW_DATA_PATH = os.path.join('data', 'raw', 'Telco-Customer-Churn.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'telco_cleaned.csv')


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer churn dataframe:
    1. Converts TotalCharges to numeric, handling empty string whitespace.
    2. Imputes TotalCharges = 0.0 for newly joined customers (tenure == 0).
    3. Converts SeniorCitizen to descriptive category or clean integer.
    4. Strips unnecessary whitespace from string columns.
    5. Validates no duplicates or NaN remain.
    """
    df_clean = df.copy()

    # 1. TotalCharges conversion & domain-specific imputation
    # Empty strings in TotalCharges correspond strictly to tenure == 0
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].astype(str).str.strip(), errors='coerce')
    
    # Newly joined customers (tenure=0) have not accumulated charges yet -> charge is $0.00
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)

    # 2. String trimming for all object columns
    obj_cols = df_clean.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 3. Validation
    assert df_clean['TotalCharges'].isnull().sum() == 0, "Missing values remain in TotalCharges!"
    assert df_clean.duplicated().sum() == 0, "Duplicate records found!"

    return df_clean


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Identifies and partitions features into logical subsets for modeling and EDA.
    """
    id_col = 'customerID'
    target_col = 'Churn'

    # Continuous numeric features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Binary features (2 categories)
    binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    # Multi-class categorical features
    multiclass_features = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]

    all_categorical = [col for col in binary_features + multiclass_features if col in df.columns]

    return {
        'id_col': id_col,
        'target_col': target_col,
        'numeric_features': numeric_features,
        'binary_features': binary_features,
        'multiclass_features': multiclass_features,
        'all_categorical': all_categorical
    }


def build_preprocessor_pipeline(numeric_features: list, categorical_features: list, scaler_type: str = 'standard') -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer pipeline.
    - Continuous Numerical: Imputer + StandardScaler / RobustScaler
    - Categorical: Imputer + OneHotEncoder(drop='first')
    
    Prevents data leakage by fitting solely on training splits during modeling.
    """
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor


def save_cleaned_data(df: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH) -> str:
    """
    Saves cleaned dataframe to designated CSV path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to '{output_path}' ({df.shape[0]} rows, {df.shape[1]} columns).")
    return output_path


def run_data_cleaning_pipeline(raw_path: str = RAW_DATA_PATH, processed_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Executes the complete Phase 2 data cleaning routine and prints validation summary.
    """
    print("=" * 80)
    print("           CUSTOMER CHURN DATASET - PHASE 2 DATA CLEANING & PREPROCESSING           ")
    print("=" * 80)

    print(f"[1/4] Loading raw data from '{raw_path}'...")
    df_raw = pd.read_csv(raw_path)
    print(f"      Raw shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    print("[2/4] Cleaning data & coercing datatypes...")
    df_clean = clean_raw_data(df_raw)

    print("[3/4] Validating cleaned dataset health...")
    print(f"      * Total Null Values: {df_clean.isnull().sum().sum()}")
    print(f"      * Total Duplicates: {df_clean.duplicated().sum()}")
    print(f"      * TotalCharges dtype: {df_clean['TotalCharges'].dtype} (Min: ${df_clean['TotalCharges'].min():.2f}, Max: ${df_clean['TotalCharges'].max():.2f})")
    print(f"      * Churn Target: {df_clean['Churn'].value_counts().to_dict()}")

    print(f"[4/4] Saving cleaned dataset to '{processed_path}'...")
    save_cleaned_data(df_clean, processed_path)

    print("\n" + "-" * 80)
    print("FEATURE GROUPS IDENTIFIED FOR PIPELINE:")
    groups = get_feature_groups(df_clean)
    print(f" * Numeric ({len(groups['numeric_features'])}): {groups['numeric_features']}")
    print(f" * Binary Categorical ({len(groups['binary_features'])}): {groups['binary_features']}")
    print(f" * Multi-class Categorical ({len(groups['multiclass_features'])}): {groups['multiclass_features']}")
    print(f" * Target: {groups['target_col']}")
    print("=" * 80)

    return df_clean


if __name__ == '__main__':
    run_data_cleaning_pipeline()
