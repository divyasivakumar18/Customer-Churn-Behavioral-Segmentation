"""
app/app.py
----------
Enterprise Customer Churn Prediction & Behavioral Segmentation Dashboard.
Interactive Multi-Page Streamlit Application.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Churn & Behavioral Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.customer_prediction import ChurnPredictorEngine


# ---------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    data_path = os.path.join('data', 'processed', 'telco_clustered.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'data', 'processed', 'telco_clustered.csv')
    return pd.read_csv(data_path)


@st.cache_resource
def load_engine():
    models_dir = os.path.join('models')
    if not os.path.exists(models_dir):
        models_dir = os.path.join('..', 'models')
    return ChurnPredictorEngine(models_dir=models_dir)


@st.cache_data
def load_model_results():
    results_path = os.path.join('reports', 'model_results.csv')
    if not os.path.exists(results_path):
        results_path = os.path.join('..', 'reports', 'model_results.csv')
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None


@st.cache_data
def load_feature_ranking():
    ranking_path = os.path.join('reports', 'feature_importance_ranking.csv')
    if not os.path.exists(ranking_path):
        ranking_path = os.path.join('..', 'reports', 'feature_importance_ranking.csv')
    if os.path.exists(ranking_path):
        return pd.read_csv(ranking_path)
    return None


df = load_dataset()
engine = load_engine()
model_results = load_model_results()
feature_ranking = load_feature_ranking()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2620/2620608.png", width=70)
st.sidebar.title("Churn Intelligence")
st.sidebar.markdown("**AI & Data Science Analytics Platform**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate Modules:",
    [
        "🏠 Executive Overview",
        "📈 Customer Analytics & EDA",
        "💎 RFM Value Segmentation",
        "🎯 Real-Time Churn Predictor",
        "🧠 Model Explainability & SHAP",
        "📋 Business Recommendations"
    ]
)

st.sidebar.divider()
st.sidebar.info("💡 **Academic Major Project**\nCustomer Churn Prediction & Behavioral Segmentation System")


# ---------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------
if page == "🏠 Executive Overview":
    st.title("📊 Customer Churn & Behavioral Intelligence Dashboard")
    st.markdown("Enterprise AI system for churn risk scoring, customer segmentation, and prescriptive retention analytics.")

    total_customers = len(df)
    churn_count = int((df['Churn'] == 'Yes').sum())
    churn_rate = (churn_count / total_customers) * 100
    avg_revenue = df['TotalCharges'].mean()
    avg_tenure = df['tenure'].mean()
    monthly_arr = (df['MonthlyCharges'].sum() * 12) / 1e6

    # KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total_customers:,}", delta="Active Base")
    c2.metric("Churned Accounts", f"{churn_count:,}", delta=f"-{churn_rate:.1f}% Churn", delta_color="inverse")
    c3.metric("Annual Run Rate (ARR)", f"${monthly_arr:.2f}M", delta="Contract Value")
    c4.metric("Avg Customer Lifetime", f"{avg_tenure:.1f} mos", delta="Tenure")
    c5.metric("Avg Spend per User", f"${avg_revenue:.2f}", delta="LTV Proxy")

    st.divider()

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Churn Rate by Contract Commitment")
        contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
        fig_contract = px.bar(
            contract_churn, barmode='group',
            color_discrete_sequence=['#1f77b4', '#d62728'],
            labels={'value': 'Percentage (%)', 'Contract': 'Contract Term'},
            title="Retention vs Churn across Contract Tiers"
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    with col_right:
        st.subheader("Customer Base Composition")
        fig_donut = px.pie(
            df, names='Churn', hole=0.5,
            color='Churn', color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'},
            title="Active vs Churned Customers"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("⚡ Executive Summary & Strategic Takeaways")
    st.markdown("""
    * **High Switching Friction on Long Contracts**: Month-to-month subscribers represent **42.71% churn**, while 2-Year contract holders experience only **2.83% churn**.
    * **Early-Stage Vulnerability**: Over 60% of total churn occurs within the first **12 months** of customer enrollment.
    * **Autopay Adoption**: Manual payment methods (Electronic checks) account for 3x higher churn compared to credit card autopay.
    """)


# ---------------------------------------------------------
# PAGE 2: CUSTOMER ANALYTICS & EDA
# ---------------------------------------------------------
elif page == "📈 Customer Analytics & EDA":
    st.title("📈 Customer Lifecycle & Behavioral Analytics")
    st.markdown("Detailed bivariate explorations and cohort churn distributions.")

    tab1, tab2, tab3 = st.tabs(["💰 Financial & Tenure", "🔌 Services & Add-ons", "👥 Demographics & Payment"])

    with tab1:
        st.subheader("Tenure & Monthly Bill Dynamics")
        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(
                df, x='Churn', y='tenure', color='Churn',
                color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'},
                title="Tenure Distribution by Churn Status (Months)"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        with col2:
            fig_kde = px.histogram(
                df, x='MonthlyCharges', color='Churn', barmode='overlay',
                color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'},
                title="Monthly Charges ($) Distribution vs Churn"
            )
            st.plotly_chart(fig_kde, use_container_width=True)

    with tab2:
        st.subheader("Impact of Service Bundles on Churn")
        services = ['InternetService', 'TechSupport', 'OnlineSecurity', 'DeviceProtection']
        chosen_service = st.selectbox("Select Service Dimension to Analyze:", services)

        service_churn = df.groupby(chosen_service)['Churn'].value_counts(normalize=True).unstack() * 100
        fig_serv = px.bar(
            service_churn, barmode='group',
            color_discrete_sequence=['#1f77b4', '#d62728'],
            labels={'value': 'Percentage (%)'},
            title=f"Churn Distribution across {chosen_service}"
        )
        st.plotly_chart(fig_serv, use_container_width=True)

    with tab3:
        st.subheader("Demographic & Payment Channels")
        col1, col2 = st.columns(2)
        with col1:
            pay_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack() * 100
            fig_pay = px.bar(
                pay_churn, barmode='group',
                color_discrete_sequence=['#1f77b4', '#d62728'],
                title="Churn Rate by Payment Channel (%)"
            )
            st.plotly_chart(fig_pay, use_container_width=True)
        with col2:
            fam_churn = df.groupby('has_family')['Churn'].value_counts(normalize=True).unstack() * 100
            fam_churn.index = ['Single / No Dependents', 'Family / Has Dependents']
            fig_fam = px.bar(
                fam_churn, barmode='group',
                color_discrete_sequence=['#1f77b4', '#d62728'],
                title="Household Structure vs Churn (%)"
            )
            st.plotly_chart(fig_fam, use_container_width=True)


# ---------------------------------------------------------
# PAGE 3: RFM VALUE SEGMENTATION
# ---------------------------------------------------------
elif page == "💎 RFM Value Segmentation":
    st.title("💎 RFM Customer Value & Behavioral Segmentation")
    st.markdown("Categorizing accounts by **Recency ($R$)**, **Frequency ($F$)**, and **Monetary ($M$)** contribution.")

    summary_rfm = df.groupby('RFM_Segment').agg(
        Customer_Count=('customerID', 'count'),
        Avg_Tenure=('tenure', 'mean'),
        Avg_Revenue=('TotalCharges', 'mean'),
        Total_Revenue=('TotalCharges', 'sum'),
        Churn_Rate=('Churn', lambda x: (x == 'Yes').mean() * 100)
    ).round(2)
    summary_rfm['Revenue_Pct'] = ((summary_rfm['Total_Revenue'] / df['TotalCharges'].sum()) * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig_rev = px.bar(
            summary_rfm.reset_index(), x='RFM_Segment', y='Total_Revenue',
            color='RFM_Segment',
            title="Total Revenue Contribution by RFM Segment ($)"
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with c2:
        fig_churn_rfm = px.bar(
            summary_rfm.reset_index(), x='RFM_Segment', y='Churn_Rate',
            color='Churn_Rate', color_continuous_scale='Reds',
            title="Churn Vulnerability by RFM Segment (%)"
        )
        st.plotly_chart(fig_churn_rfm, use_container_width=True)

    st.subheader("Detailed Segment Profile Matrix")
    st.dataframe(summary_rfm.style.format({
        'Avg_Tenure': '{:.1f} mos',
        'Avg_Revenue': '${:,.2f}',
        'Total_Revenue': '${:,.2f}',
        'Churn_Rate': '{:.2f}%',
        'Revenue_Pct': '{:.1f}%'
    }), use_container_width=True)


# ---------------------------------------------------------
# PAGE 4: REAL-TIME CHURN PREDICTION
# ---------------------------------------------------------
elif page == "🎯 Real-Time Churn Predictor":
    st.title("🎯 Real-Time Customer Churn Scoring")
    st.markdown("Enter customer attributes to compute instant churn probability, risk tier, and customized retention recommendations.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen (65+)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=6)

        with col2:
            st.markdown("### 📡 Subscribed Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown("### 💳 Contract & Billing")
            contract = st.selectbox("Contract Commitment", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=85.0, step=1.0)
            total = st.number_input("Total Cumulative Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly), step=10.0)

        submit_btn = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

    if submit_btn:
        payload = {
            'customerID': 'PORTAL-USER',
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': lines,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': tech,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }

        result = engine.predict_single_customer(payload)

        st.divider()
        st.subheader("🎯 Prediction & Risk Diagnostic Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Predicted Status", result['churn_prediction'])
            st.markdown(f"**Risk Level:** <span style='color:{result['risk_color']}; font-size:18px; font-weight:bold;'>{result['risk_tier']}</span>", unsafe_allow_html=True)

        with res_col2:
            st.metric("Churn Probability", result['churn_probability_pct'])
            st.progress(result['churn_probability'])

        with res_col3:
            st.metric("RFM Value Segment", result['rfm_segment'])
            st.markdown(f"**Cluster Persona:** *{result['behavioral_persona']}*")

        st.divider()
        col_risk, col_rec = st.columns(2)

        with col_risk:
            st.subheader("⚠️ Key Drivers Increasing Risk")
            for factor in result['top_risk_factors']:
                st.markdown(f"🔴 {factor}")

        with col_rec:
            st.subheader("✅ Actionable Retention Strategy")
            for rec in result['retention_recommendations']:
                st.markdown(f"👉 **{rec}**")


# ---------------------------------------------------------
# PAGE 5: MODEL EXPLAINABILITY & SHAP
# ---------------------------------------------------------
elif page == "🧠 Model Explainability & SHAP":
    st.title("🧠 Explainable AI (XAI) & Model Transparency")
    st.markdown("Interpreting the decision boundaries of the production Machine Learning pipeline using SHAP and Feature Gain.")

    if model_results is not None:
        st.subheader("Supervised Benchmark Comparison")
        st.dataframe(model_results.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall (Churn)': '{:.2%}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}',
            'PR-AUC': '{:.4f}'
        }), use_container_width=True)

    st.divider()
    st.subheader("Global Feature Importance Ranking")

    if feature_ranking is not None:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            top_12 = feature_ranking.head(12)
            fig_shap = px.bar(
                top_12.sort_values(by='Mean_SHAP_Impact', ascending=True),
                x='Mean_SHAP_Impact', y='Feature', orientation='h',
                color='Mean_SHAP_Impact', color_continuous_scale='Blues',
                title="Top 12 Features by Mean |SHAP Value|"
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        with col2:
            st.markdown("### Top Churn Drivers Explained")
            for _, r in feature_ranking.head(6).iterrows():
                st.markdown(f"**{r['Feature']}**")
                st.caption(f"{r['Business Meaning & Actionable Driver']}")


# ---------------------------------------------------------
# PAGE 6: BUSINESS RECOMMENDATIONS
# ---------------------------------------------------------
elif page == "📋 Business Recommendations":
    st.title("📋 Prescriptive Business Retention Strategy Matrix")
    st.markdown("Actionable roadmap designed for Marketing, Customer Success, and Executive Leadership.")

    st.markdown("""
    ### 🎯 Multi-Tier Customer Retention Framework

    | Customer Tier | Identified Vulnerability | Recommended Business Intervention | Projected Impact |
    | :--- | :--- | :--- | :--- |
    | **High-Risk New Users (0-12M)** | Month-to-month contract & no security add-ons | **90-Day Concierge Onboarding**: Offer 15% discount on annual upgrade + 3 free months of Tech Support. | Reduce early-stage churn by **25-30%**. |
    | **High-Value Champions** | High monthly spend ($90+) with long tenure | **VIP Loyalty Rewards**: Dedicated account manager, priority customer service queue, and annual hardware upgrade. | Maintain retention above **95%**. |
    | **Electronic Check Payers** | Friction from manual recurring payment review | **Auto-Pay Incentive Campaign**: Provide a one-time $10 credit or $3/mo ongoing discount for enrolling in Autopay. | Lower payment-induced churn by **18%**. |
    | **Fiber Optic Dissatisfaction** | High prices without support tiers | **Bundle Value Add-ons**: Include complimentary cybersecurity and cloud backup with all fiber plans. | Increase customer perceived value and lock-in. |
    """)

    st.divider()
    st.success("🎓 **Project Demonstration Complete**: Enterprise-grade Machine Learning & Customer Analytics System.")
