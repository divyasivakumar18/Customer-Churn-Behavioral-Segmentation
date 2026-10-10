"""
src/feature_engineering.py
--------------------------
Feature Engineering Pipeline for Customer Churn & Behavioral Segmentation.
Creates domain-specific behavioral indicators, service aggregations, financial
metrics, and tenure cohorts without causing data leakage.
"""

import os
import pandas as pd
import numpy as np


INPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_cleaned.csv')
OUTPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_engineered.csv')


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs high-signal behavioral and financial domain features.
    """
    df_feat = df.copy()

    # 1. Tenure Cohorts / Groups
    bins = [-1, 12, 24, 48, 60, 100]
    labels = ['0-12 M', '13-24 M', '25-48 M', '49-60 M', '>60 M']
    df_feat['tenure_group'] = pd.cut(df_feat['tenure'], bins=bins, labels=labels)
    df_feat['tenure_years'] = (df_feat['tenure'] / 12.0).round(2)

    # 2. Service Aggregation Features
    # Count how many individual services the customer subscribes to
    services_list = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Binary conversion for service indicators
    service_matrix = pd.DataFrame(index=df_feat.index)
    service_matrix['PhoneService'] = (df_feat['PhoneService'] == 'Yes').astype(int)
    service_matrix['MultipleLines'] = (df_feat['MultipleLines'] == 'Yes').astype(int)
    service_matrix['Internet'] = (df_feat['InternetService'] != 'No').astype(int)
    service_matrix['OnlineSecurity'] = (df_feat['OnlineSecurity'] == 'Yes').astype(int)
    service_matrix['OnlineBackup'] = (df_feat['OnlineBackup'] == 'Yes').astype(int)
    service_matrix['DeviceProtection'] = (df_feat['DeviceProtection'] == 'Yes').astype(int)
    service_matrix['TechSupport'] = (df_feat['TechSupport'] == 'Yes').astype(int)
    service_matrix['StreamingTV'] = (df_feat['StreamingTV'] == 'Yes').astype(int)
    service_matrix['StreamingMovies'] = (df_feat['StreamingMovies'] == 'Yes').astype(int)

    df_feat['total_services_count'] = service_matrix.sum(axis=1)

    # Security & Support bundle count
    df_feat['security_support_count'] = (
        service_matrix['OnlineSecurity'] +
        service_matrix['OnlineBackup'] +
        service_matrix['DeviceProtection'] +
        service_matrix['TechSupport']
    )

    # Entertainment & Streaming count
    df_feat['streaming_services_count'] = (
        service_matrix['StreamingTV'] +
        service_matrix['StreamingMovies']
    )

    # 3. Payment Behavior & Friction Indicator
    # Customers on automatic payment methods show significantly higher retention
    df_feat['is_autopay'] = df_feat['PaymentMethod'].str.contains('automatic', case=False).astype(int)

    # 4. Household & Family Support Proxy
    df_feat['has_family'] = ((df_feat['Partner'] == 'Yes') | (df_feat['Dependents'] == 'Yes')).astype(int)

    # 5. Contract Commitment Months
    contract_map = {
        'Month-to-month': 1,
        'One year': 12,
        'Two year': 24
    }
    df_feat['contract_commitment_months'] = df_feat['Contract'].map(contract_map).astype(int)

    # 6. Financial Dynamics & Unit Economics
    # Average monthly charge over customer lifetime vs current monthly charge
    # Detects recent price hikes or plan changes
    # For tenure=0, calculated historical monthly rate is equal to current MonthlyCharges
    safe_tenure = df_feat['tenure'].replace(0, 1)
    df_feat['historical_avg_monthly_charges'] = (df_feat['TotalCharges'] / safe_tenure).round(2)
    # If tenure was 0, set historical avg to current monthly charges
    df_feat.loc[df_feat['tenure'] == 0, 'historical_avg_monthly_charges'] = df_feat['MonthlyCharges']

    # Charge difference (Current Monthly Charges - Historical Avg Monthly Charges)
    df_feat['monthly_charge_discrepancy'] = (df_feat['MonthlyCharges'] - df_feat['historical_avg_monthly_charges']).round(2)

    # Cost per subscribed service
    df_feat['cost_per_service'] = (df_feat['MonthlyCharges'] / (df_feat['total_services_count'] + 1)).round(2)

    return df_feat


def run_feature_engineering_pipeline(input_path: str = INPUT_DATA_PATH, output_path: str = OUTPUT_DATA_PATH) -> pd.DataFrame:
    """
    Executes the Feature Engineering pipeline and prints feature summary.
    """
    print("=" * 85)
    print("                     PHASE 4: FEATURE ENGINEERING PIPELINE                       ")
    print("=" * 85)

    print(f"[1/3] Loading cleaned dataset from '{input_path}'...")
    df = pd.read_csv(input_path)
    print(f"      Initial Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("[2/3] Constructing domain & behavioral features...")
    df_feat = create_engineered_features(df)
    new_features = [c for c in df_feat.columns if c not in df.columns]

    print(f"\n[3/3] Successfully engineered {len(new_features)} new features:")
    print("-" * 85)
    for i, col in enumerate(new_features, 1):
        sample_vals = df_feat[col].head(3).tolist()
        dtype_str = str(df_feat[col].dtype)
        print(f" {i:>2}. {col:<32} | {dtype_str:<12} | Samples: {sample_vals}")

    # Inspect churn rate by tenure_group
    print("\n" + "-" * 85)
    print("TENURE GROUP CHURN DISTRIBUTION:")
    tg_churn = df_feat.groupby('tenure_group', observed=False)['Churn'].value_counts(normalize=True).unstack() * 100
    for idx, row in tg_churn.iterrows():
        print(f" * Cohort {idx:<8} -> Retained: {row['No']:>6.2f}%, Churned: {row['Yes']:>6.2f}%")

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_feat.to_csv(output_path, index=False)
    print("\n" + "=" * 85)
    print(f"[SUCCESS] Engineered dataset saved to '{output_path}'")
    print(f"          Final Shape: {df_feat.shape[0]} rows x {df_feat.shape[1]} columns")
    print("=" * 85)

    return df_feat


if __name__ == '__main__':
    run_feature_engineering_pipeline()
