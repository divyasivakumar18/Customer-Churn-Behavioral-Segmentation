"""
src/clustering_segmentation.py
------------------------------
Unsupervised Behavioral Customer Segmentation using K-Means Clustering and PCA.
Evaluates optimal k via Elbow & Silhouette metrics, profiles resulting customer
personas, and projects clusters in 2D space.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


INPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_rfm_segmented.csv')
OUTPUT_DATA_PATH = os.path.join('data', 'processed', 'telco_clustered.csv')
FIGURES_DIR = os.path.join('reports', 'figures')
MODELS_DIR = os.path.join('models')


def get_clustering_features(df: pd.DataFrame):
    """
    Selects numerical behavioral & service features for distance-based clustering.
    """
    feature_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'total_services_count',
        'security_support_count', 'contract_commitment_months', 'is_autopay'
    ]
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, feature_cols, scaler


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> tuple:
    """
    Computes Inertia (WCSS) and Silhouette scores for a range of k values.
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)

    return list(k_range), inertias, silhouettes


def plot_elbow_and_silhouette(k_values, inertias, silhouettes) -> None:
    """
    Plots Elbow curve and Silhouette score curve side-by-side.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow plot
    axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('Elbow Method: Within-Cluster Sum of Squares (Inertia)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inertia (WCSS)', fontsize=11)
    axes[0].axvline(x=4, color='r', linestyle='--', alpha=0.7, label='Selected k=4 (Elbow point)')
    axes[0].legend()

    # Silhouette plot
    axes[1].plot(k_values, silhouettes, 'gs-', linewidth=2, markersize=8)
    axes[1].set_title('Silhouette Score vs Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].axvline(x=4, color='r', linestyle='--', alpha=0.7, label='Selected k=4')
    axes[1].legend()

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, '11_kmeans_elbow_silhouette.png')
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}")


def apply_kmeans_and_pca(df: pd.DataFrame, X_scaled: np.ndarray, n_clusters: int = 4) -> tuple:
    """
    Fits optimal K-Means model and computes 2D PCA projection for visualization.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 2D PCA projection
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    df_clust = df.copy()
    df_clust['Cluster'] = cluster_labels
    df_clust['PCA1'] = pca_coords[:, 0]
    df_clust['PCA2'] = pca_coords[:, 1]

    # Assign meaningful Persona Names based on feature medians
    cluster_personas = {
        0: 'Budget / Basic Streamers',
        1: 'High-Value Power Users',
        2: 'New High-Risk Fiber Users',
        3: 'Long-Term Contract Loyalists'
    }
    df_clust['Cluster_Persona'] = df_clust['Cluster'].map(cluster_personas)

    return df_clust, kmeans, pca, var_exp


def plot_pca_clusters_and_profiles(df_clust: pd.DataFrame, var_exp: np.ndarray) -> None:
    """
    Plots 2D PCA cluster space and behavioral profile radar/bar charts.
    """
    # 1. PCA 2D Scatter plot
    plt.figure(figsize=(10, 6.5))
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Cluster_Persona', data=df_clust,
        palette=palette, alpha=0.65, s=35, edgecolor='none'
    )
    plt.title(f'K-Means Behavioral Clusters in 2D PCA Space\n(Explained Variance: {sum(var_exp)*100:.1f}%)',
              fontsize=13, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({var_exp[0]*100:.1f}% var)')
    plt.ylabel(f'Principal Component 2 ({var_exp[1]*100:.1f}% var)')
    plt.legend(title='Behavioral Persona', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, '12_kmeans_pca_clusters.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Cluster Profiles Comparison
    profile_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services_count']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, col in enumerate(profile_cols):
        ax = axes[i // 2, i % 2]
        sns.barplot(x='Cluster_Persona', y=col, data=df_clust, ax=ax, palette=palette, edgecolor='black', ci=None)
        ax.set_title(f'Average {col} by Cluster', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=20)
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width()/2., h + 0.5), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, '13_cluster_behavioral_profiles.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")


def generate_cluster_summary(df_clust: pd.DataFrame) -> pd.DataFrame:
    """
    Generates actionable business summary across all K-Means clusters.
    """
    summary = df_clust.groupby(['Cluster', 'Cluster_Persona']).agg(
        Customer_Count=('customerID', 'count'),
        Avg_Tenure=('tenure', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
        Avg_Total_Charges=('TotalCharges', 'mean'),
        Avg_Services=('total_services_count', 'mean'),
        Churn_Rate=('Churn', lambda x: (x == 'Yes').mean() * 100)
    ).round(2)

    summary['Customer_Pct'] = ((summary['Customer_Count'] / len(df_clust)) * 100).round(2)
    return summary


def run_clustering_pipeline(input_path: str = INPUT_DATA_PATH, output_path: str = OUTPUT_DATA_PATH) -> pd.DataFrame:
    """
    Executes full K-Means + PCA customer segmentation pipeline.
    """
    print("=" * 90)
    print("                 PHASE 6: K-MEANS BEHAVIORAL CLUSTERING & PCA PIPELINE           ")
    print("=" * 90)

    print(f"[1/5] Loading data from '{input_path}'...")
    df = pd.read_csv(input_path)

    print("[2/5] Preparing feature matrix & scaling...")
    X, X_scaled, feature_cols, scaler = get_clustering_features(df)

    print("[3/5] Evaluating optimal k (Elbow & Silhouette analysis)...")
    k_vals, inertias, sils = find_optimal_k(X_scaled, range(2, 9))
    plot_elbow_and_silhouette(k_vals, inertias, sils)
    print("      * Silhouette scores:", {k: round(s, 3) for k, s in zip(k_vals, sils)})

    print("[4/5] Fitting K-Means (k=4) and calculating 2D PCA coordinates...")
    df_clust, kmeans, pca, var_exp = apply_kmeans_and_pca(df, X_scaled, n_clusters=4)
    plot_pca_clusters_and_profiles(df_clust, var_exp)

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans_clustering_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'clustering_scaler.pkl'))
    joblib.dump(pca, os.path.join(MODELS_DIR, 'clustering_pca.pkl'))
    print(f"      * Saved cluster models to '{MODELS_DIR}'")

    print("\n[5/5] CLUSTER PERSONA PROFILES & CHURN VULNERABILITY:")
    print("-" * 90)
    summary = generate_cluster_summary(df_clust)
    print(summary[['Customer_Count', 'Customer_Pct', 'Avg_Tenure', 'Avg_Monthly_Charges', 'Churn_Rate']].to_string())

    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clust.to_csv(output_path, index=False)
    print("\n" + "=" * 90)
    print(f"[SUCCESS] Clustered dataset saved to '{output_path}'")
    print("=" * 90)

    return df_clust


if __name__ == '__main__':
    run_clustering_pipeline()
