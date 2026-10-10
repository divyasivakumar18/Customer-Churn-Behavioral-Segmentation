"""
src/rfm_analysis.py
-------------------
RFM (Recency, Frequency, Monetary) Customer Value Segmentation Module.
Adapts traditional transactional RFM to subscription business models,
computes quantile scores, maps business segments, and produces actionable insights.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_engineered.csv')
OUTPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_rfm_segmented.csv')
FIGURES_DIR = os.path.join('reports', 'figures')


def compute_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes subscription-adapted RFM components:
    - Recency (R): Active customer tenure freshness (in months).
    - Frequency (F): Total subscription interactions (tenure * total_services_count).
    - Monetary (M): Cumulative revenue generated (TotalCharges).
    """
    df_rfm = df.copy()

    # Recency: Months of active subscription tenure (Higher is more seasoned/active)
    # Frequency: Number of billing cycles multiplied by service bundle breadth
    df_rfm['R_value'] = df_rfm['tenure']
    df_rfm['F_value'] = df_rfm['tenure'] * df_rfm['total_services_count']
    df_rfm['M_value'] = df_rfm['TotalCharges']

    # Quantile Scoring (1 to 4 quartiles)
    # Using rank(method='first') ensures robust quantile binning even with duplicate values
    df_rfm['R_Score'] = pd.qcut(df_rfm['R_value'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
    df_rfm['F_Score'] = pd.qcut(df_rfm['F_value'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
    df_rfm['M_Score'] = pd.qcut(df_rfm['M_value'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

    # Composite RFM Score String and Numerical Sum
    df_rfm['RFM_Combined'] = (
        df_rfm['R_Score'].astype(str) +
        df_rfm['F_Score'].astype(str) +
        df_rfm['M_Score'].astype(str)
    )
    df_rfm['RFM_Total'] = df_rfm['R_Score'] + df_rfm['F_Score'] + df_rfm['M_Score']

    return df_rfm


def assign_rfm_segments(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Maps RFM score combinations to standard marketing & business segments.
    """
    def segment_mapper(row):
        r = row['R_Score']
        f = row['F_Score']
        m = row['M_Score']
        
        # Already churned customers
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

    df_rfm['RFM_Segment'] = df_rfm.apply(segment_mapper, axis=1)
    return df_rfm


def plot_rfm_visualizations(df_rfm: pd.DataFrame) -> None:
    """
    Generates 3 publication-ready RFM visualization charts.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # Chart 1: Segment Distribution & Churn Rate
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    seg_counts = df_rfm['RFM_Segment'].value_counts()
    
    seg_counts.plot(kind='barh', ax=axes[0], color='#2b5c8f', edgecolor='black', alpha=0.85)
    axes[0].set_title('Customer Count by RFM Segment', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Customers')
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_width()):,}", (p.get_width() + 20, p.get_y() + p.get_height()/2.),
                         va='center', fontsize=9, fontweight='bold')

    # Segment Churn Rate
    seg_churn = df_rfm.groupby('RFM_Segment')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).loc[seg_counts.index]
    seg_churn.plot(kind='barh', ax=axes[1], color='#d95f02', edgecolor='black', alpha=0.85)
    axes[1].set_title('Churn Rate by RFM Segment (%)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Churn Rate (%)')
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_width():.1f}%", (p.get_width() + 1, p.get_y() + p.get_height()/2.),
                         va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, '08_rfm_segments_distribution.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # Chart 2: Revenue Contribution by Segment
    plt.figure(figsize=(10, 5.5))
    seg_revenue = df_rfm.groupby('RFM_Segment')['TotalCharges'].sum() / 1e6
    seg_revenue = seg_revenue.sort_values(ascending=False)
    bars = plt.bar(seg_revenue.index, seg_revenue.values, color='#2ca02c', edgecolor='black', alpha=0.85)
    plt.title('Total Revenue Contribution by RFM Segment ($ Millions)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Revenue ($ Millions)', fontsize=11)
    plt.xticks(rotation=25, ha='right', fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'${yval:.2f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, '09_rfm_revenue_by_segment.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # Chart 3: RFM Heatmap (Recency vs Frequency matrix with Avg Monetary spend)
    plt.figure(figsize=(9, 6))
    rfm_pivot = df_rfm.pivot_table(index='R_Score', columns='F_Score', values='M_value', aggfunc='mean')
    sns.heatmap(rfm_pivot, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=1.0, cbar_kws={'label': 'Avg Revenue ($)'})
    plt.title('RFM Score Heatmap: Average Revenue ($)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Frequency Score (F)', fontsize=11)
    plt.ylabel('Recency Score (R)', fontsize=11)

    p3 = os.path.join(FIGURES_DIR, '10_rfm_score_heatmap.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")


def generate_rfm_summary_table(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates detailed cohort statistics for every RFM segment.
    """
    summary = df_rfm.groupby('RFM_Segment').agg(
        Customer_Count=('customerID', 'count'),
        Avg_Tenure_Months=('tenure', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
        Avg_Total_Revenue=('TotalCharges', 'mean'),
        Total_Segment_Revenue=('TotalCharges', 'sum'),
        Churn_Rate=('Churn', lambda x: (x == 'Yes').mean() * 100)
    ).round(2)

    total_customers = len(df_rfm)
    summary['Customer_Pct'] = ((summary['Customer_Count'] / total_customers) * 100).round(2)
    summary['Revenue_Pct'] = ((summary['Total_Segment_Revenue'] / df_rfm['TotalCharges'].sum()) * 100).round(2)
    summary = summary.sort_values(by='Total_Segment_Revenue', ascending=False)
    return summary


def run_rfm_pipeline(input_path: str = INPUT_DATA_PATH, output_path: str = OUTPUT_DATA_PATH) -> pd.DataFrame:
    """
    Executes end-to-end RFM segmentation pipeline.
    """
    print("=" * 90)
    print("                    PHASE 5: RFM VALUE SEGMENTATION PIPELINE                     ")
    print("=" * 90)

    print(f"[1/4] Loading engineered data from '{input_path}'...")
    df = pd.read_csv(input_path)

    print("[2/4] Calculating subscription-adapted R, F, M metrics & quartile scoring...")
    df_rfm = compute_rfm_metrics(df)

    print("[3/4] Assigning customer segments & generating analytical figures...")
    df_rfm = assign_rfm_segments(df_rfm)
    plot_rfm_visualizations(df_rfm)

    print("\n[4/4] SEGMENT PROFILE BREAKDOWN:")
    print("-" * 90)
    summary = generate_rfm_summary_table(df_rfm)
    print(summary[['Customer_Count', 'Customer_Pct', 'Avg_Total_Revenue', 'Churn_Rate', 'Revenue_Pct']].to_string())

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_rfm.to_csv(output_path, index=False)
    print("\n" + "=" * 90)
    print(f"[SUCCESS] RFM segmented dataset saved to '{output_path}'")
    print("=" * 90)

    return df_rfm


if __name__ == '__main__':
    run_rfm_pipeline()
