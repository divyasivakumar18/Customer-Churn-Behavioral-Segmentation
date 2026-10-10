# VIVA VOCE PREPARATION GUIDE & 20-SLIDE PRESENTATION STRUCTURE
## Customer Churn Prediction & Behavioral Segmentation System
**Final Year B.Tech AI & Data Science Project**

---

## 📽️ Part 1: 20-Slide Presentation Deck Structure

### Slide 1: Title Slide
* **Title**: Customer Churn Prediction & Behavioral Segmentation System
* **Subtitle**: An End-to-End Machine Learning, RFM Modeling, and Explainable AI (XAI) Solution
* **Student Name & Roll No**: [Your Name & Roll Number]
* **Degree / Department**: B.Tech in Artificial Intelligence & Data Science
* **Guide / Supervisor**: [Guide's Name & Designation]

### Slide 2: Problem Statement & Motivation
* **The Business Problem**: Telecom subscriber churn erodes revenue and reduces customer lifetime value (LTV).
* **Cost Factor**: Customer acquisition is **5x–7x more expensive** than customer retention.
* **Objective**: Predict which subscribers are likely to churn before they leave, understand why, and prescribe targeted retention strategies.

### Slide 3: Project Objectives
* Ingest and clean telecom subscription data with zero data leakage.
* Perform Exploratory Data Analysis (EDA) across 15 analytical dimensions.
* Synthesize 11 domain-engineered features (tenure cohorts, unit economics, bundle breadth).
* Formulate subscription-adapted RFM and K-Means behavioral clustering.
* Train and benchmark 5 supervised ML classifiers with hyperparameter tuning.
* Deconstruct predictions using SHAP (Explainable AI).
* Deploy an interactive multi-page Streamlit web dashboard.

### Slide 4: Dataset & Data Understanding
* **Dataset**: IBM Telco Customer Churn (7,043 rows, 21 initial columns).
* **Class Balance**: 26.54% Churn ($N=1,869$) vs 73.46% Active ($N=5,174$) — moderate $~3:1$ imbalance.
* **Data Hygiene Issue**: 11 records in `TotalCharges` had empty whitespace strings `' '`.
* **Resolution**: Domain analysis proved all 11 rows had $tenure=0$. Imputed with $\$0.00$.

### Slide 5: End-to-End System Architecture
* **Visual Diagram**: Raw Ingestion $\rightarrow$ Clean & Pipeline $\rightarrow$ Feature Engineering $\rightarrow$ RFM & Clustering $\rightarrow$ Train/Test Split $\rightarrow$ Model Benchmarks $\rightarrow$ 5-Fold GridSearchCV $\rightarrow$ SHAP XAI $\rightarrow$ Streamlit App.

### Slide 6: Data Cleaning & Preprocessing Pipeline
* **Data Leakage Prevention**: Integrated `ColumnTransformer` with `StandardScaler` for numerics and `OneHotEncoder(drop='first')` for categoricals.
* **Principle**: Transformers are fitted *strictly on training splits* and transform validation/test sets without statistical leakage.

### Slide 7: Exploratory Data Analysis (EDA) - Key Insights
* **Contract**: Month-to-month contracts churn at **42.71%**, compared to **2.83%** for 2-year contracts.
* **Payment**: Electronic check users show **45.29% churn** vs **15.24%** for credit card autopay.
* **Hazard Timeline**: Churner median tenure is only **10.0 months** vs **38.0 months** for retained accounts.

### Slide 8: Feature Engineering (11 New Domain Features)
* **Tenure Cohorts**: `tenure_group` (`0-12 M`, `13-24 M`, `25-48 M`, `49-60 M`, `>60 M`).
* **Service Ecosystem**: `total_services_count`, `security_support_count`, `streaming_services_count`.
* **Financial & Behavioral Ratios**: `cost_per_service`, `monthly_charge_discrepancy`, `is_autopay`, `has_family`.

### Slide 9: Subscription-Adapted RFM Analysis
* **Recency ($R$)**: Active tenure duration.
* **Frequency ($F$)**: Service interaction volume ($tenure \times total\_services\_count$).
* **Monetary ($M$)**: Cumulative revenue ($TotalCharges$).
* **Quartile Scoring**: $R, F, M \in [1, 4]$ composite scoring.

### Slide 10: RFM Customer Segments & Financial Concentration
* **Champions (22.38% base)**: Drives **53.04% of total company revenue** ($5,403.98 avg spend, 8.50% churn).
* **Loyal Customers (19.00% base)**: Drives **28.72% of revenue** ($3,446.09 avg spend, 23.32% churn).
* **At Risk / High Spenders (6.12% base)**: High historical spenders exhibiting vulnerability indicators.

### Slide 11: Unsupervised Clustering (K-Means & 2D PCA)
* **Evaluation**: Elbow method (WCSS) & Silhouette analysis across $k \in [2, 8]$.
* **Optimal $k=4$**: Silhouette score of $0.289$.
* **PCA Visualization**: 2D projection capturing $48.2\%$ of multidimensional behavioral variance.

### Slide 12: K-Means Customer Behavioral Personas
* **Cluster 0**: *Budget / Basic Streamers* (30.24% base, 35.35% churn).
* **Cluster 1**: *High-Value Power Users* (23.88% base, 10.17% churn).
* **Cluster 2**: *New High-Risk Fiber Users* (31.28% base, 41.67% churn).
* **Cluster 3**: *Long-Term Contract Loyalists* (14.60% base, 2.63% churn).

### Slide 13: Supervised Model Benchmarking
* **Classifiers Tested**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost.
* **Validation Strategy**: 80/20 Stratified Train-Test Split ($N_{train}=5,634$, $N_{test}=1,409$).
* **Class Imbalance Handling**: Applied `scale_pos_weight=2.76` and `class_weight='balanced'`.

### Slide 14: Hyperparameter Optimization & Model Selection
* **Method**: 5-Fold Stratified Cross-Validation (`GridSearchCV`).
* **Winner**: Tuned **XGBoost** achieved **0.9754 ROC-AUC**, **0.9524 PR-AUC**, **88.50% Recall**, and **91.84% Accuracy**.

### Slide 15: Evaluation Metrics Comparison Table
* Direct comparison table showing Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC.
* Highlight why **Recall (88.50%)** is the primary business metric.

### Slide 16: Explainable AI (XAI) with SHAP
* **SHAP Summary Bar Chart**: Quantifies global mean $|SHAP|$ impact.
* **SHAP Beeswarm Plot**: Illustrates how higher tenure and 2-year contracts push predictions toward retention, while high cost-per-service pushes toward churn.

### Slide 17: Interactive Streamlit Web Dashboard
* Walkthrough of the 6 pages: Executive Overview, Analytics & EDA, RFM Segmentation, Live Real-Time Predictor, Model Transparency, and Strategic Recommendations.

### Slide 18: Prescriptive Business Retention Strategies
* **High-Risk New Users**: 15% discount on annual upgrade + 3 free months of Tech Support.
* **Electronic Check Payers**: $5/mo bill credit for enabling automatic credit card autopay.
* **Fiber Optic Dissatisfaction**: Complimentary cybersecurity bundle and proactive speed checks.

### Slide 19: Limitations & Future Scope
* **Limitations**: Static cross-sectional tabular snapshot without timestamped call-center logs.
* **Future Scope**: TabNet deep learning, Survival Analysis (Cox Proportional Hazards) for CLV forecasting, and real-time FastAPI microservices.

### Slide 20: Conclusion & Q&A
* Summary of project achievements, business ROI, and opening for panel questions.

---

## 🎯 Part 2: 25 Comprehensive Viva Voce Questions & Answers

### 1. What is the business problem of customer churn, and why is it critical?
**Answer**: Customer churn occurs when subscribers cancel their service. In telecommunications, acquiring a new customer costs 5 to 7 times more than retaining an existing one. Predicting churn allows companies to intervene with targeted promotions before revenue is lost.

### 2. Why is raw Accuracy a misleading metric for customer churn?
**Answer**: Because of class imbalance. In this dataset, only 26.54% of customers churn. A naive model predicting "No Churn" for every customer would achieve 73.46% accuracy while catching zero churners. Therefore, **Recall**, **F1-Score**, **ROC-AUC**, and **PR-AUC** are the critical metrics.

### 3. What is the difference between Precision and Recall in the context of churn?
**Answer**:
* **Precision**: Of all customers predicted to churn, what percentage actually churned? (Avoids wasting retention budget on false alarms).
* **Recall (Sensitivity)**: Of all customers who actually churned, what percentage did the model correctly identify? (Minimizes missed churners, which represents direct revenue loss).

### 4. What is Data Leakage, and how did you prevent it in your pipeline?
**Answer**: Data leakage occurs when information from outside the training dataset (such as the test split or target statistics) is used to create the model. We prevented leakage by wrapping feature scaling (`StandardScaler`) and encoding (`OneHotEncoder`) inside Scikit-Learn `Pipeline` and `ColumnTransformer` objects that fit exclusively on training folds.

### 5. Why did `TotalCharges` have missing values, and how did you handle them?
**Answer**: 11 records contained space characters (`' '`). Domain inspection confirmed all 11 rows had $tenure = 0$. These were brand new customers who had not yet completed their first billing cycle. Instead of dropping rows or using the column mean, we imputed $0.00$, preserving physical ground truth.

### 6. How did you adapt the traditional RFM model to a subscription business?
**Answer**: In e-commerce, Recency is days since last order, and Frequency is order count. In subscription models:
* **Recency ($R$)**: Active tenure duration and contract freshness.
* **Frequency ($F$)**: Service interaction volume ($tenure \times total\_services\_count$).
* **Monetary ($M$)**: Cumulative revenue ($TotalCharges$).

### 7. What is K-Means clustering, and how did you choose $k=4$?
**Answer**: K-Means is an unsupervised partitioning algorithm that minimizes within-cluster sum of squares (WCSS / Inertia). We evaluated $k \in [2, 8]$ using the **Elbow Method** (locating the inflection point) and **Silhouette Analysis** (measuring cluster cohesion and separation), determining $k=4$ as the optimal balance.

### 8. Why did you use PCA, and how much variance did it explain?
**Answer**: Principal Component Analysis (PCA) is a linear dimensionality reduction technique. It projects high-dimensional behavioral features onto 2 orthogonal principal components ($PCA_1, PCA_2$) to visualize cluster boundaries in a 2D scatter plot. The first two components explained $48.2\%$ of total variance.

### 9. Why did you choose XGBoost as the final production model?
**Answer**: Tuned XGBoost delivered the strongest overall test performance (**0.9754 ROC-AUC**, **0.9524 PR-AUC**, **88.50% Recall**, and **91.84% Accuracy**). It natively handles non-linear feature interactions, controls overfitting via gradient shrinkage (`learning_rate=0.03`), and handles class imbalance through `scale_pos_weight`.

### 10. How did you handle class imbalance in your models?
**Answer**:
1. Applied stratified train-test splitting (`stratify=y`) to maintain the 26.54% base rate across splits.
2. Set `class_weight='balanced'` in Logistic Regression and Random Forest.
3. Set `scale_pos_weight = N_negative / N_positive = 5174 / 1869 = 2.76` in XGBoost to penalize false negative errors on the churn class.

### 11. What is 5-Fold Stratified Cross-Validation?
**Answer**: The training dataset is divided into 5 equal subsets (folds) while preserving the target class ratio. The model is trained on 4 folds and validated on the 5th, rotating 5 times. The average score across all 5 folds provides an unbiased measure of generalization performance during `GridSearchCV`.

### 12. What are SHAP values, and why are they superior to basic feature importances?
**Answer**: SHAP (SHapley Additive exPlanations) is based on cooperative game theory. Unlike native feature importance (which only measures how often a feature is split on globally), SHAP provides **local, directionally accurate attributions** showing whether a specific feature value increased or decreased churn risk for an individual customer.

### 13. What is the difference between a SHAP Summary Bar Chart and a Beeswarm Plot?
**Answer**:
* **Bar Chart**: Shows the global mean absolute SHAP value, ranking features by overall impact magnitude.
* **Beeswarm Plot**: Shows individual customer data points, with dot colors representing feature values (Red = High, Blue = Low) and horizontal position representing the directional shift in churn log-odds.

### 14. What were the top 3 drivers of customer churn identified by SHAP?
**Answer**:
1. **Contract Duration (`contract_commitment_months`)**: Month-to-month contracts have high churn (**42.71%**), whereas 2-year contracts have low churn (**2.83%**).
2. **Cost per Service (`cost_per_service`)**: High bills for few active services create dissatisfaction.
3. **Fiber Optic without Support**: Premium pricing without bundled technical support increases churn.

### 15. Why does payment method (Electronic check) correlate with high churn?
**Answer**: Electronic check users must manually review and authorize payment every month, prompting recurring evaluation of the expense. Automated autopay (credit card / bank draft) creates frictionless, habitual renewal.

### 16. What is the difference between correlation and causation in this project?
**Answer**: A feature like `PaymentMethod_Electronic check` is strongly correlated with churn, but the payment method itself does not physically force a customer to leave. It acts as a proxy for customer commitment and payment friction. We recommend incentives (e.g., bill credits for autopay) to reduce friction rather than assuming payment channel is the sole cause.

### 17. How does the real-time prediction engine work for a single customer?
**Answer**: The `ChurnPredictorEngine` takes a raw customer JSON dictionary, computes 11 engineered features, calculates RFM quartile metrics, assigns a K-Means cluster persona, transforms features via the pipeline, and calls `predict_proba()` to return probability, risk tier, risk drivers, and retention recommendations.

### 18. What is the difference between Bagging and Boosting?
**Answer**:
* **Bagging (e.g., Random Forest)**: Trains multiple independent trees in parallel on bootstrap samples and averages their predictions to reduce variance.
* **Boosting (e.g., Gradient Boosting, XGBoost)**: Trains trees sequentially, where each new tree is fit to the residual errors of the previous trees to reduce bias.

### 19. What is ROC-AUC, and what does a score of 0.9754 mean?
**Answer**: ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the probability that the classifier will rank a randomly chosen positive instance (churner) higher than a randomly chosen negative instance (retained). A score of **0.9754** indicates near-perfect ranking separation.

### 20. What is PR-AUC, and why is it important for imbalanced datasets?
**Answer**: PR-AUC is the Area Under the Precision-Recall Curve. Unlike ROC-AUC (which can look artificially optimistic when the negative class is large), PR-AUC focuses directly on the minority positive class, providing a strict measure of precision-recall tradeoffs. Our tuned XGBoost achieved a high PR-AUC of **0.9524**.

### 21. How did you ensure your Streamlit app does not reload models on every user click?
**Answer**: We utilized Streamlit's caching decorators: `@st.cache_data` for static CSV tables and `@st.cache_resource` for deserialized machine learning models and pipelines.

### 22. What unit and integration tests did you write?
**Answer**: We created a 7-point `unittest` suite in `tests/test_pipeline.py` verifying raw dataset integrity, data cleaning invariants, feature engineering outputs, RFM logic, model deserialization, and edge-case inputs ($tenure=0$ and $tenure=72$).

### 23. What are the key business recommendations for the telecom executive team?
**Answer**:
1. **Contract Migration**: Offer onboarding promotional discounts for migrating from month-to-month to annual commitments.
2. **Autopay Incentives**: Provide a $5/mo credit for enrolling in automated bank/credit card billing.
3. **Service Bundling**: Include complimentary cybersecurity and tech support with high-tier fiber plans.
4. **VIP Loyalty Program**: Dedicated account management for Champions (who generate 53% of total revenue).

### 24. What are the primary limitations of your current system?
**Answer**: The dataset provides cross-sectional snapshot aggregates rather than real-time sequential event logs or unstructured customer service voice transcripts.

### 25. If given more time, what advanced techniques would you implement?
**Answer**:
1. **Survival Analysis**: Fit Cox Proportional Hazards models to estimate expected time-to-churn and dynamic Customer Lifetime Value ($).
2. **Deep Tabular Models**: Benchmark TabNet and Deep & Cross Networks (DCNv2).
3. **FastAPI Microservice**: Deploy the scoring engine as a containerized REST API with Docker and Kubernetes for live CRM webhook streaming.
