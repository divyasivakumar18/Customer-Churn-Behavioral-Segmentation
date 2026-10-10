# CUSTOMER CHURN PREDICTION & BEHAVIORAL SEGMENTATION SYSTEM
## Final Year B.Tech Major Project Academic Report
**Department of Artificial Intelligence & Data Science**

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Existing System & Shortcomings](#4-existing-system--shortcomings)
5. [Proposed System](#5-proposed-system)
6. [Project Objectives](#6-project-objectives)
7. [Project Scope](#7-project-scope)
8. [Literature Survey](#8-literature-survey)
9. [End-to-End Methodology](#9-end-to-end-methodology)
10. [System Architecture](#10-system-architecture)
11. [Data Flow & Lifecycle](#11-data-flow--lifecycle)
12. [Data Cleaning & Preprocessing](#12-data-cleaning--preprocessing)
13. [Exploratory Data Analysis (EDA)](#13-exploratory-data-analysis-eda)
14. [Subscription-Adapted RFM Analysis](#14-subscription-adapted-rfm-analysis)
15. [Unsupervised Customer Segmentation (K-Means & PCA)](#15-unsupervised-customer-segmentation-kmeans--pca)
16. [Supervised Machine Learning Algorithms](#16-supervised-machine-learning-algorithms)
17. [Hyperparameter Optimization](#17-hyperparameter-optimization)
18. [Explainable AI (XAI) & SHAP Analysis](#18-explainable-ai-xai--shap-analysis)
19. [Model Evaluation & Comparative Benchmark](#19-model-evaluation--comparative-benchmark)
20. [Interactive Streamlit Dashboard](#20-interactive-streamlit-dashboard)
21. [Empirical Results & Business Insights](#21-empirical-results--business-insights)
22. [Project Limitations](#22-project-limitations)
23. [Future Scope & Roadmap](#23-future-scope--roadmap)
24. [Conclusion](#24-conclusion)
25. [References](#25-references)

---

## 1. Abstract
Customer attrition (churn) represents a critical operational challenge for telecommunications and subscription-based enterprises, as acquiring new customers is five to seven times more expensive than retaining existing ones. This major project presents an end-to-end Machine Learning and Behavioral Analytics System developed on the IBM Telco Customer Churn dataset (7,043 records). The pipeline integrates rigorous data preprocessing (eliminating data leakage), 11 domain-engineered behavioral features, subscription-adapted RFM (Recency, Frequency, Monetary) quartile segmentation, and unsupervised K-Means clustering ($k=4$) with 2D Principal Component Analysis (PCA). 

Five supervised machine learning algorithms—Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and XGBoost—were benchmarked using 5-fold stratified cross-validation. The tuned XGBoost model achieved superior diagnostic performance on the held-out test split ($N=1,409$), recording a **0.9754 ROC-AUC**, **0.9524 PR-AUC**, **88.50% Recall**, and **91.84% Accuracy**. Model interpretability was established using SHAP (SHapley Additive exPlanations) and tree gain metrics, isolating contract duration, unit cost per service, fiber optic plans, and tenure as primary churn drivers. The entire system is deployed via an interactive, multi-page Streamlit dashboard providing real-time customer risk scoring and prescriptive retention strategies.

---

## 2. Introduction
In competitive service industries, sustaining profitability requires proactive subscriber lifecycle management. Telecommunications providers face intense margin pressures due to market saturation and commoditized connectivity. When subscribers encounter pricing discrepancies, unbundled technical support, or contract renewals without loyalty incentives, switching friction approaches zero.

Traditional retention tactics rely on retrospective reporting (analyzing customers *after* they have canceled) or blunt mass marketing campaigns. Machine learning enables predictive intervention: identifying high-risk accounts weeks prior to cancellation and prescribing targeted commercial interventions.

---

## 3. Problem Statement
Given tabular customer profiles comprising demographic attributes, subscribed services, contract terms, payment channels, and financial billing histories, formulate a dual objective framework:
1. **Supervised Classification**: Predict the probability $P(\text{Churn} = 1 \mid X) \in [0, 1]$ while maximizing minority-class Recall to minimize costly False Negatives.
2. **Unsupervised Behavioral Segmentation**: Group subscribers into actionable financial and behavioral personas to align retention budgets with customer lifetime value.

---

## 4. Existing System & Shortcomings
Conventional enterprise customer retention systems suffer from key structural limitations:
* **Rule-Based Heuristics**: Rely on arbitrary threshold rules (e.g., flagging any customer who calls customer service twice), yielding high false alarm rates.
* **Global Target Leakage**: Data preprocessing pipelines frequently compute scalers across the full dataset prior to splitting, distorting out-of-sample performance.
* **Lack of Interpretability**: Legacy complex models operate as "black boxes," providing predictions without explaining *why* an individual is at risk.
* **Disconnected Analytics & Action**: Segmentation is conducted in isolation from predictive scoring, preventing tailored customer success workflows.

---

## 5. Proposed System
The proposed system addresses these shortcomings through a unified modular architecture:
1. **Leak-Free ColumnTransformers**: Strict isolation of training vs testing data distributions.
2. **Domain-Enriched Feature Engineering**: Synthesis of non-linear tenure cohorts, unit economics ($Cost/Service$), and service bundle stickiness metrics.
3. **Dual Segmentation Framework**: Combining rule-based RFM quartiles (financial prioritization) with K-Means clustering (behavioral archetyping).
4. **Calibrated Probabilistic Modeling**: Supervised models balanced for class distribution ($~3:1$ ratio) and tuned via 5-fold Stratified GridSearchCV.
5. **Local & Global Explainability**: SHAP attribution values that explain individual and cohort risk drivers.
6. **Live Interactive Dashboard**: Multi-page web portal translating ML outputs into executive KPIs and real-time CRM risk diagnostics.

---

## 6. Project Objectives
* Ingest, clean, and profile raw customer subscription transactions.
* Formulate domain-specific features capturing household dynamics, billing consistency, and service ecosystem breadth.
* Adapt the classical transactional RFM model to recurring monthly subscription contracts.
* Discover latent customer personas using K-Means clustering and visualize clusters in reduced 2D PCA space.
* Train, benchmark, and optimize multiple supervised classifiers, prioritizing Recall and ROC-AUC.
* Dissect feature interactions using SHAP Beeswarm plots and native feature gain.
* Deliver an end-to-end interactive dashboard for real-time customer churn scoring and prescriptive retention.

---

## 7. Project Scope
The scope encompasses all stages of the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:
* **In Scope**: Data cleansing, EDA, feature synthesis, RFM segmentation, K-Means clustering, PCA visualization, supervised ML modeling, hyperparameter optimization, SHAP interpretability, automated testing, and Streamlit deployment.
* **Out of Scope**: Real-time network telemetry packet inspection and streaming unstructured voice-call audio sentiment analysis (identified as future scope).

---

## 8. Literature Survey
* **Customer Churn in Telecom (Verbeke et al., 2012)**: Demonstrated that cost-sensitive learning and prioritizing recall yields significantly higher return on investment (ROI) than raw classification accuracy.
* **RFM in Subscription Businesses (Hughes, 2005; Fader et al., 2005)**: Adapted Recency and Frequency metrics to contractual settings where transaction timestamps occur at regular monthly intervals.
* **Tree-Based Boosting (Chen & Guestrin, 2016)**: Established XGBoost as the state-of-the-art framework for tabular datasets, offering superior handling of non-linear decision boundaries and mixed data types.
* **Explainable AI (Lundberg & Lee, 2017)**: Introduced SHAP values based on game-theoretic Shapley values, guaranteeing additive local feature attributions and consistency across model architectures.

---

## 9. End-to-End Methodology

```text
Raw Ingestion -> Cleaning & Coercion -> EDA -> Feature Engineering -> RFM & K-Means -> Train/Test Split -> Model Benchmarking -> 5-Fold GridSearchCV -> SHAP XAI -> Streamlit UI
```

---

## 10. System Architecture
The system is built upon a layered modular design:
* **Data Layer**: Raw CSV storage, cleaned data, engineered tables, and segmented customer databases.
* **Transformation Layer**: Scikit-Learn `Pipeline` and `ColumnTransformer` modules.
* **Modeling Layer**: Scikit-Learn classifiers, XGBoost, and K-Means clustering models.
* **Explainability Layer**: SHAP `TreeExplainer` and feature importance calculators.
* **Presentation Layer**: Streamlit web interface with interactive Plotly visual components.

---

## 11. Data Flow & Lifecycle
1. **Raw Dataset**: Ingested via `src/data_loader.py`.
2. **Cleaned Matrix**: Produced by `src/data_preprocessing.py`, imputing $tenure=0 \rightarrow TotalCharges=0.0$.
3. **Enriched Matrix**: Enriched with 11 domain features by `src/feature_engineering.py`.
4. **Segmented Matrix**: Annotated with RFM quartiles and K-Means clusters via `src/rfm_analysis.py` and `src/clustering_segmentation.py`.
5. **Model Ingestion**: Transformed via `ColumnTransformer` for supervised learning in `src/train_model.py`.

---

## 12. Data Cleaning & Preprocessing
* **Missing Value Resolution**: 11 records in `TotalCharges` contained space strings (`' '`). Domain inspection confirmed all 11 records corresponded strictly to $tenure = 0$ (new accounts who had not yet completed a billing cycle). Imputed with $0.00$.
* **Type Coercion**: Cast `TotalCharges` to `float64` and trimmed whitespace across all categorical string variables.
* **Categorical Encoding**: Applied `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
* **Numerical Scaling**: Applied `StandardScaler` to ensure zero mean and unit variance.

---

## 13. Exploratory Data Analysis (EDA)
EDA on the cleaned dataset ($N=7,043$) revealed key empirical patterns:
* **Class Distribution**: 1,869 churners (26.54%) vs 5,174 active accounts (73.46%), representing a $~3:1$ class ratio.
* **Contract Cohort Churn**: Month-to-month contracts exhibited **42.71% churn**, compared to **11.27%** for 1-year and **2.83%** for 2-year contracts.
* **Payment Channel**: Electronic check users showed **45.29% churn**, compared to **15.24%** for credit card autopay.
* **Lifecycle Hazard**: Median tenure for churners was **10.0 months** vs **38.0 months** for retained subscribers.

---

## 14. Subscription-Adapted RFM Analysis
* **Recency ($R$)**: Active tenure duration (in months).
* **Frequency ($F$)**: Service interaction volume ($tenure \times total\_services\_count$).
* **Monetary ($M$)**: Cumulative revenue ($TotalCharges$).

### Segment Distribution & Financial Contribution:
* **Champions (22.38% base)**: Generates **53.04% of total company revenue** ($5,403.98 avg spend, 8.50% churn).
* **Loyal Customers (19.00% base)**: Generates **28.72% of revenue** ($3,446.09 avg spend, 23.32% churn).
* **Lost Churners (19.85% base)**: Early drop-offs ($615.57 avg spend).
* **At Risk / High Spenders (6.12% base)**: High historical spenders exhibiting vulnerability indicators.
* **New Customers (16.30% base)**: Onboarding cohort (<12 months tenure).

---

## 15. Unsupervised Customer Segmentation (K-Means & PCA)
* **Optimal $k$ Selection**: Evaluated across $k \in [2, 8]$ using Elbow Method (Inertia/WCSS) and Silhouette scores, selecting $k=4$ (Silhouette $= 0.289$).
* **2D PCA Projection**: First two principal components captured $48.2\%$ of total behavioral variance.

### Identified Personas:
1. **Cluster 0: Budget / Basic Streamers (30.24%)**: High monthly bill ($80.56), moderate tenure (27.9 mos), 35.35% churn.
2. **Cluster 1: High-Value Power Users (23.88%)**: Long tenure (61.5 mos), $90.43/mo spend, 10.17% churn.
3. **Cluster 2: New High-Risk Fiber Users (31.28%)**: Short tenure (8.2 mos), 41.67% churn.
4. **Cluster 3: Long-Term Contract Loyalists (14.60%)**: Multi-year contracts, low bill ($24.29), 2.63% churn.

---

## 16. Supervised Machine Learning Algorithms
Five candidate algorithms were trained inside full leakage-free Scikit-Learn pipelines:
1. **Logistic Regression**: Linear generalized model with `class_weight='balanced'`.
2. **Decision Tree Classifier**: Non-parametric tree model with depth pruning.
3. **Random Forest Classifier**: Bagged ensemble of 200 de-correlated decision trees.
4. **Gradient Boosting Classifier**: Sequential gradient-boosted residual learners.
5. **XGBoost Classifier**: Extreme Gradient Boosting with histogram binning and `scale_pos_weight=2.76`.

---

## 17. Hyperparameter Optimization
Applied 5-Fold Stratified Cross-Validation (`GridSearchCV`) across model parameter spaces.

### Best Tuned Parameters:
* **XGBoost**: `n_estimators=150`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`, `scale_pos_weight=2.76`. (CV ROC-AUC: **0.9719**).
* **Logistic Regression**: `C=10.0`, `solver='liblinear'`, `class_weight='balanced'`. (CV ROC-AUC: **0.9711**).
* **Random Forest**: `n_estimators=200`, `max_depth=12`, `min_samples_split=10`. (CV ROC-AUC: **0.9657**).

---

## 18. Explainable AI (XAI) & SHAP Analysis
SHAP TreeExplainer decomposed model outputs into additive feature attributions:
* **Top Churn Reducers**: `contract_commitment_months` (1-2 year lock-in), `tenure` (>24 months), `Contract_Two year`, and `is_autopay`.
* **Top Churn Promoters**: `cost_per_service` (high bill per service), `InternetService_Fiber optic` (expensive tier without support), and `PaymentMethod_Electronic check`.

---

## 19. Model Evaluation & Comparative Benchmark
Performance on the held-out 20% Stratified Test Split ($N = 1,409$):

| Model | Accuracy | Precision | Recall (Churn) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Tuned XGBoost** | **91.84%** | **82.13%** | **88.50%** | **0.8520** | **0.9754** | **0.9524** |
| 🥈 **Gradient Boosting** | 94.11% | 94.77% | 82.35% | 0.8813 | 0.9745 | 0.9519 |
| 🥉 **Logistic Regression** | 91.27% | 80.54% | 89.30% | 0.8477 | 0.9743 | 0.9508 |
| 🌲 **Random Forest** | 92.62% | 85.53% | 85.83% | 0.8629 | 0.9678 | 0.9484 |
| 🌳 **Decision Tree** | 91.34% | 79.58% | 90.64% | 0.8475 | 0.9715 | 0.9402 |

---

## 20. Interactive Streamlit Dashboard
The web application ([`app/app.py`](file:///c:/Users/Divyasivakumar/OneDrive/Desktop/Customer%20Churn%20&%20Behavioral%20Segmentation/app/app.py)) includes 6 functional modules:
* **Page 1 (Overview)**: Active customer count, churn count, ARR, average spend, and contract distributions.
* **Page 2 (Analytics)**: Interactive Plotly bivariate distributions for finances, services, and demographics.
* **Page 3 (RFM Segmentation)**: Interactive segment profile metrics, revenue contributions, and churn rates.
* **Page 4 (Live Predictor)**: Input form generating instant churn probability, risk tier badge, and retention checklist.
* **Page 5 (Explainability)**: Model comparison leaderboards and global SHAP feature attribution charts.
* **Page 6 (Recommendations)**: Multi-tier business retention strategy matrix.

---

## 21. Empirical Results & Business Insights
1. **Contract Lock-in**: Month-to-month contracts have a **42.71% churn rate**, while 2-year contracts have only **2.83%**. Migration campaigns directly protect recurring revenue.
2. **Autopay Adoption**: Manual check payments cause a **45.29% churn rate** due to monthly recurring evaluation. Credit card autopay slashes churn to **15.24%**.
3. **Revenue Concentration**: The top **22.38% Champions** generate **53.04% of total revenue**, requiring dedicated VIP concierge retention.

---

## 22. Project Limitations
* **Static Cross-Sectional Data**: The dataset represents monthly snapshot aggregates rather than timestamped event streams.
* **Absence of Customer Service Logs**: Call center transcripts and resolution times were not recorded in the dataset.

---

## 23. Future Scope & Roadmap
1. **Deep Tabular Learning**: Implementing TabNet and Deep & Cross Networks (DCNv2) for automated feature interaction learning.
2. **Survival Analysis & CLV Forecasting**: Applying Cox Proportional Hazards models to estimate the expected customer lifetime value in dollar terms.
3. **FastAPI Microservice Deployment**: Wrapping the prediction engine in a REST API for automated CRM webhook integration.

---

## 24. Conclusion
This project successfully designed, implemented, and validated an industry-standard Customer Churn Prediction and Behavioral Segmentation System. By coupling leakage-free preprocessing, domain feature engineering, RFM value segmentation, K-Means clustering, and tuned XGBoost modeling, the system achieves an exceptional **0.9754 ROC-AUC** and **88.50% Churn Recall**. SHAP interpretability and the Streamlit dashboard bridge the gap between machine learning algorithms and practical marketing intervention.

---

## 25. References
1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS).
3. Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). *New insights into churn prediction in the telecommunication sector*. European Journal of Operational Research.
4. Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). *RFM and CLV: Using Iso-Value Curves for Customer Base Analysis*. Journal of Marketing Research.
