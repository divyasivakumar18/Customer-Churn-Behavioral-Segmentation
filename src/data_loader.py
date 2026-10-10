"""
src/data_loader.py
------------------
Module for loading raw dataset, validating schema, and generating comprehensive
data understanding and health checks for the Customer Churn project.
"""

import os
import urllib.request
import pandas as pd
import numpy as np


RAW_DATA_URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
RAW_DATA_PATH = os.path.join('data', 'raw', 'Telco-Customer-Churn.csv')


def download_dataset_if_missing(filepath: str = RAW_DATA_PATH, url: str = RAW_DATA_URL) -> str:
    """
    Checks if raw dataset exists locally; if not, downloads it from the source.
    """
    if not os.path.exists(filepath):
        print(f"[INFO] Dataset not found at '{filepath}'. Downloading from source...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        print(f"[SUCCESS] Dataset downloaded and saved to '{filepath}'.")
    else:
        print(f"[INFO] Found local dataset at '{filepath}'.")
    return filepath


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw CSV dataset into a pandas DataFrame.
    """
    download_dataset_if_missing(filepath)
    df = pd.read_csv(filepath)
    return df


def generate_data_understanding_report(df: pd.DataFrame) -> dict:
    """
    Performs comprehensive structural and statistical inspection of the raw dataset.
    Returns a dictionary of structured summaries.
    """
    num_rows, num_cols = df.shape
    duplicate_count = int(df.duplicated().sum())

    # Detect explicit nulls and hidden empty whitespace strings
    null_counts = df.isnull().sum().to_dict()
    whitespace_counts = {}
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype == 'string':
            ws_count = int((df[col].astype(str).str.strip() == '').sum())
            if ws_count > 0:
                whitespace_counts[col] = ws_count

    # Column type categorization
    customer_id_col = 'customerID' if 'customerID' in df.columns else None
    target_col = 'Churn' if 'Churn' in df.columns else None

    # Target distribution
    target_dist = {}
    target_dist_pct = {}
    if target_col:
        target_dist = df[target_col].value_counts().to_dict()
        target_dist_pct = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()

    # Column details
    column_metadata = []
    for col in df.columns:
        unique_cnt = int(df[col].nunique())
        sample_vals = df[col].dropna().unique()[:3].tolist()
        dtype_str = str(df[col].dtype)
        column_metadata.append({
            'column': col,
            'dtype': dtype_str,
            'unique_count': unique_cnt,
            'sample_values': sample_vals,
            'null_count': int(null_counts.get(col, 0)),
            'whitespace_blank_count': int(whitespace_counts.get(col, 0))
        })

    report = {
        'shape': (num_rows, num_cols),
        'duplicate_count': duplicate_count,
        'null_counts': null_counts,
        'whitespace_counts': whitespace_counts,
        'customer_id_col': customer_id_col,
        'target_col': target_col,
        'target_dist': target_dist,
        'target_dist_pct': target_dist_pct,
        'column_metadata': column_metadata,
        'summary_numeric': df.describe().to_dict(),
    }
    return report


def print_data_understanding_summary(df: pd.DataFrame, report: dict) -> None:
    """
    Prints a formatted summary report to standard console output.
    """
    print('=' * 80)
    print('           CUSTOMER CHURN DATASET - PHASE 1 DATA UNDERSTANDING           ')
    print('=' * 80)
    print(f"Dataset Shape: {report['shape'][0]} rows x {report['shape'][1]} columns")
    print(f"Total Duplicate Rows: {report['duplicate_count']}")
    print(f"Customer ID Identifier: {report['customer_id_col']}")
    print(f"Target Variable: {report['target_col']}")

    print('\n' + '-' * 80)
    print('1. TARGET DISTRIBUTION (Churn Status)')
    print('-' * 80)
    for k, v in report['target_dist'].items():
        pct = report['target_dist_pct'].get(k, 0.0)
        print(f"   * {k}: {v:,} customers ({pct:.2f}%)")
    churn_rate = report['target_dist_pct'].get('Yes', 0.0)
    print(f"   -> Business Insight: Churn rate is {churn_rate}%. Moderate class imbalance (~3:1 ratio).")

    print('\n' + '-' * 80)
    print('2. COLUMN INVENTORY & DATA TYPES')
    print('-' * 80)
    print(f"{'Column Name':<20} | {'Data Type':<10} | {'Uniques':<8} | {'Nulls':<6} | {'Sample Values'}")
    print('-' * 80)
    for meta in report['column_metadata']:
        samples_str = ', '.join(str(x) for x in meta['sample_values'])
        extra = f" [! {meta['whitespace_blank_count']} blanks]" if meta['whitespace_blank_count'] > 0 else ''
        print(f"{meta['column']:<20} | {meta['dtype']:<10} | {meta['unique_count']:<8} | {meta['null_count']:<6} | {samples_str}{extra}")

    print('\n' + '-' * 80)
    print('3. DATA INTEGRITY & ANOMALY CHECKS')
    print('-' * 80)
    if report['whitespace_counts']:
        for col, count in report['whitespace_counts'].items():
            print(f"   [!] WARNING: Column '{col}' contains {count} whitespace/empty strings.")
            print("       Explanation: These occur for newly joined customers (tenure=0) who have not")
            print("       received their first monthly invoice yet. Must be converted to float in Phase 2.")
    else:
        print('   [OK] No hidden whitespace blanks detected.')

    if report['duplicate_count'] == 0:
        print('   [OK] No duplicate rows found in the dataset.')

    print('\n' + '-' * 80)
    print('4. NUMERICAL FEATURES SUMMARY STATISTICS')
    print('-' * 80)
    num_df = df.select_dtypes(include=[np.number])
    print(num_df.describe().round(2).to_string())

    print('\n' + '=' * 80)
    print('                        DATA DICTIONARY & DOMAIN CONTEXT                 ')
    print('=' * 80)
    dictionary = {
        'customerID': 'Unique identifier for each customer account.',
        'gender': 'Male or Female customer demographic.',
        'SeniorCitizen': 'Binary indicator (1 = Senior citizen 65+, 0 = Younger).',
        'Partner': 'Whether the customer has a partner (Yes, No).',
        'Dependents': 'Whether the customer has children/dependents (Yes, No).',
        'tenure': 'Number of months the customer has stayed with the company.',
        'PhoneService': 'Whether the customer has home phone service (Yes, No).',
        'MultipleLines': 'Whether customer has multiple phone lines (Yes, No, No phone service).',
        'InternetService': 'Customer’s ISP provider type (DSL, Fiber optic, No).',
        'OnlineSecurity': 'Add-on security add-on package (Yes, No, No internet service).',
        'OnlineBackup': 'Cloud backup add-on service (Yes, No, No internet service).',
        'DeviceProtection': 'Hardware protection plan (Yes, No, No internet service).',
        'TechSupport': 'Technical support access tier (Yes, No, No internet service).',
        'StreamingTV': 'Streaming TV content service (Yes, No, No internet service).',
        'StreamingMovies': 'Streaming Movies subscription (Yes, No, No internet service).',
        'Contract': 'Contractual commitment term (Month-to-month, One year, Two year).',
        'PaperlessBilling': 'Electronic billing option (Yes, No).',
        'PaymentMethod': 'Payment channel (Electronic check, Mailed check, Bank transfer, Credit card).',
        'MonthlyCharges': 'The amount charged to the customer monthly (in USD).',
        'TotalCharges': 'The total cumulative amount charged across tenure (in USD).',
        'Churn': 'Target label: Whether the customer churned in the last month (Yes, No).'
    }
    for col, desc in dictionary.items():
        print(f" * {col:<18} : {desc}")
    print('=' * 80)


if __name__ == '__main__':
    print('[START] Running Phase 1: Data Understanding Pipeline...')
    df = load_raw_data(RAW_DATA_PATH)
    report = generate_data_understanding_report(df)
    print_data_understanding_summary(df, report)
    print('\n[COMPLETED] Phase 1 data understanding executed successfully!')
