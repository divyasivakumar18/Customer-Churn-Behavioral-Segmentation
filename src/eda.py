"""
src/eda.py
----------
Exploratory Data Analysis (EDA) Module.
Generates comprehensive statistical distributions, bivariate churn relationships,
correlation analysis, and saves high-resolution publication-quality figures for
the Customer Churn project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


CLEANED_DATA_PATH = os.path.join('data', 'processed', 'telco_cleaned.csv')
FIGURES_DIR = os.path.join('reports', 'figures')


def set_plotting_style():
    """Sets a clean, professional aesthetic for all figures."""
    sns.set_theme(style='whitegrid', font_scale=1.0)
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame) -> None:
    """1 & 2: Churn Distribution & Churn Percentage."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df['Churn'].value_counts()
    pcts = df['Churn'].value_counts(normalize=True) * 100
    colors = ['#1f77b4', '#d62728']

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='black', alpha=0.85)
    axes[0].set_title('Customer Churn Count', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].set_xlabel('Churn Status', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:,}', ha='center', va='bottom', fontweight='bold')

    # Donut chart
    axes[1].pie(pcts, labels=pcts.index, autopct='%1.2f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                textprops=dict(fontsize=12, fontweight='bold'))
    axes[1].set_title('Churn Rate Breakdown', fontsize=14, fontweight='bold')

    save_path = os.path.join(FIGURES_DIR, '01_churn_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """3, 4, 5: Distributions of Tenure, Monthly Charges, and Total Charges."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    titles = ['Tenure (Months)', 'Monthly Charges ($)', 'Total Charges ($)']
    colors = ['#2ca02c', '#ff7f0e', '#9467bd']

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i], bins=30, alpha=0.6)
        axes[i].set_title(f'Distribution of {titles[i]}', fontsize=13, fontweight='bold')
        axes[i].set_xlabel(titles[i], fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)

    save_path = os.path.join(FIGURES_DIR, '02_numerical_distributions.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_churn_by_contract_and_payment(df: pd.DataFrame) -> None:
    """6 & 7: Churn vs Contract Type and Payment Method."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Churn vs Contract
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn[['No', 'Yes']].plot(kind='bar', stacked=False, ax=axes[0], color=['#1f77b4', '#d62728'], edgecolor='black', alpha=0.85)
    axes[0].set_title('Churn Rate by Contract Type (%)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Percentage of Cohort (%)', fontsize=11)
    axes[0].set_xlabel('Contract Type', fontsize=11)
    axes[0].tick_params(axis='x', rotation=15)
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1),
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Churn vs Payment Method
    pay_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
    pay_churn[['No', 'Yes']].plot(kind='bar', stacked=False, ax=axes[1], color=['#1f77b4', '#d62728'], edgecolor='black', alpha=0.85)
    axes[1].set_title('Churn Rate by Payment Method (%)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Percentage of Cohort (%)', fontsize=11)
    axes[1].set_xlabel('Payment Method', fontsize=11)
    axes[1].tick_params(axis='x', rotation=25)
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1),
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

    save_path = os.path.join(FIGURES_DIR, '03_churn_by_contract_and_payment.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_churn_by_services(df: pd.DataFrame) -> None:
    """8 & 14: Churn vs Internet Service and Tech Support (plus key value-adds)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    services = ['InternetService', 'TechSupport', 'OnlineSecurity', 'DeviceProtection']
    titles = ['Internet Service Provider', 'Tech Support Tier', 'Online Security Add-on', 'Device Protection Plan']

    for i, (service, title) in enumerate(zip(services, titles)):
        ax = axes[i // 2, i % 2]
        ct = pd.crosstab(df[service], df['Churn'], normalize='index') * 100
        ct[['No', 'Yes']].plot(kind='bar', ax=ax, color=['#1f77b4', '#d62728'], edgecolor='black', alpha=0.85)
        ax.set_title(f'Churn Rate by {title} (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage of Cohort (%)', fontsize=10)
        ax.set_xlabel(service, fontsize=10)
        ax.tick_params(axis='x', rotation=15)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1),
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    save_path = os.path.join(FIGURES_DIR, '04_churn_by_services.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_churn_vs_demographics(df: pd.DataFrame) -> None:
    """11, 12, 13: Churn vs Senior Citizen, Dependents, Partner."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    demographics = ['SeniorCitizen', 'Dependents', 'Partner']
    titles = ['Senior Citizen Status (0=No, 1=Yes)', 'Has Dependents', 'Has Partner']

    for i, (col, title) in enumerate(zip(demographics, titles)):
        ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        ct[['No', 'Yes']].plot(kind='bar', ax=axes[i], color=['#1f77b4', '#d62728'], edgecolor='black', alpha=0.85)
        axes[i].set_title(f'Churn by {title} (%)', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Percentage (%)', fontsize=10)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].tick_params(axis='x', rotation=0)
        for p in axes[i].patches:
            height = p.get_height()
            if height > 0:
                axes[i].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1),
                                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    save_path = os.path.join(FIGURES_DIR, '05_churn_by_demographics.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_churn_vs_tenure_and_charges(df: pd.DataFrame) -> None:
    """9 & 10: Churn vs Tenure and Monthly Charges (Boxplots & KDEs)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[0], palette=['#1f77b4', '#d62728'], width=0.4)
    axes[0].set_title('Tenure Distribution by Churn Status', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Churn Status', fontsize=11)
    axes[0].set_ylabel('Tenure (Months)', fontsize=11)

    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False, fill=True,
                palette={'No': '#1f77b4', 'Yes': '#d62728'}, ax=axes[1], alpha=0.4)
    axes[1].set_title('Monthly Charges Density: Churn vs Retained', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Monthly Charges ($)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)

    save_path = os.path.join(FIGURES_DIR, '06_churn_vs_tenure_charges.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """15: Correlation Matrix for Numerical and Binary Encoded Features."""
    df_corr = df.copy()
    df_corr['Churn_Binary'] = (df_corr['Churn'] == 'Yes').astype(int)
    df_corr['SeniorCitizen'] = df_corr['SeniorCitizen'].astype(int)
    df_corr['Partner_Binary'] = (df_corr['Partner'] == 'Yes').astype(int)
    df_corr['Dependents_Binary'] = (df_corr['Dependents'] == 'Yes').astype(int)
    df_corr['PaperlessBilling_Binary'] = (df_corr['PaperlessBilling'] == 'Yes').astype(int)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
                'Partner_Binary', 'Dependents_Binary', 'PaperlessBilling_Binary', 'Churn_Binary']

    corr_matrix = df_corr[num_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.8, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Correlation Matrix of Numerical & Demographic Features', fontsize=14, fontweight='bold', pad=12)

    save_path = os.path.join(FIGURES_DIR, '07_correlation_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def generate_eda_report_tables(df: pd.DataFrame) -> None:
    """Computes exact statistical cohort churn rates and prints structured insights."""
    print("=" * 85)
    print("                     PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)                    ")
    print("=" * 85)

    # 1. Contract Analysis
    contract_stats = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
    print("\n[1] CHURN RATE BY CONTRACT TYPE:")
    print("-" * 55)
    print(f"{'Contract':<18} | {'Retained (No)':<15} | {'Churned (Yes)':<15}")
    print("-" * 55)
    for idx, row in contract_stats.iterrows():
        print(f"{idx:<18} | {row['No']:>12.2f}% | {row['Yes']:>12.2f}%")

    # 2. Payment Method Analysis
    pay_stats = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack() * 100
    print("\n[2] CHURN RATE BY PAYMENT METHOD:")
    print("-" * 65)
    print(f"{'Payment Method':<30} | {'Retained (No)':<15} | {'Churned (Yes)':<15}")
    print("-" * 65)
    for idx, row in pay_stats.iterrows():
        print(f"{idx:<30} | {row['No']:>12.2f}% | {row['Yes']:>12.2f}%")

    # 3. Internet Service Analysis
    isp_stats = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100
    print("\n[3] CHURN RATE BY INTERNET SERVICE:")
    print("-" * 55)
    print(f"{'Internet Service':<18} | {'Retained (No)':<15} | {'Churned (Yes)':<15}")
    print("-" * 55)
    for idx, row in isp_stats.iterrows():
        print(f"{idx:<18} | {row['No']:>12.2f}% | {row['Yes']:>12.2f}%")

    # 4. Tech Support Analysis
    ts_stats = df.groupby('TechSupport')['Churn'].value_counts(normalize=True).unstack() * 100
    print("\n[4] CHURN RATE BY TECH SUPPORT:")
    print("-" * 55)
    print(f"{'Tech Support':<25} | {'Retained (No)':<15} | {'Churned (Yes)':<15}")
    print("-" * 55)
    for idx, row in ts_stats.iterrows():
        print(f"{idx:<25} | {row['No']:>12.2f}% | {row['Yes']:>12.2f}%")

    # 5. Tenure Statistics by Churn
    print("\n[5] TENURE & CHARGES BY CHURN STATUS:")
    print("-" * 75)
    summary_tenure = df.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']].agg(['mean', 'median']).round(2)
    print(summary_tenure.to_string())
    print("=" * 85)


def run_eda_pipeline():
    set_plotting_style()
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded cleaned dataset with {df.shape[0]} records.\n")

    plot_target_distribution(df)
    plot_numerical_distributions(df)
    plot_churn_by_contract_and_payment(df)
    plot_churn_by_services(df)
    plot_churn_vs_demographics(df)
    plot_churn_vs_tenure_and_charges(df)
    plot_correlation_heatmap(df)
    generate_eda_report_tables(df)
    print(f"\n[COMPLETED] All 7 EDA publication figures saved into '{FIGURES_DIR}'.")


if __name__ == '__main__':
    run_eda_pipeline()
