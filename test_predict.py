import pandas as pd
from src.customer_prediction import ChurnPredictorEngine

engine = ChurnPredictorEngine()
df = pd.read_csv('data/processed/telco_clustered.csv')

churners = df[df['Churn'] == 'Yes'].head(5)
retained = df[df['Churn'] == 'No'].head(5)

print('=== ACTUAL CHURNERS ===')
for idx, row in churners.iterrows():
    rep = engine.predict_single_customer(row.to_dict())
    print(f"Customer: {row['customerID']} | Tenure: {row['tenure']} | Contract: {row['Contract']} | Prob: {rep['churn_probability_pct']} | Pred: {rep['churn_prediction']}")

print('\n=== ACTUAL RETAINED ===')
for idx, row in retained.iterrows():
    rep = engine.predict_single_customer(row.to_dict())
    print(f"Customer: {row['customerID']} | Tenure: {row['tenure']} | Contract: {row['Contract']} | Prob: {rep['churn_probability_pct']} | Pred: {rep['churn_prediction']}")
