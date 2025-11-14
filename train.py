import pandas as pd
import pickle
import os
import glob
import kagglehub
from xgboost import XGBClassifier

# Download dataset from Kaggle
path = kagglehub.dataset_download("vmohammedraiyyan/maternal-health-and-high-risk-pregnancy-dataset")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

# Drop rows with minimal missing values (<5%)
if df.isna().any(axis=1).mean() < 0.05:
    df = df.dropna()

# Make target binary
df.risk_level = df.risk_level.map({
    'Low': 0,
    'High': 1
})

# Filter out extreme values
df = df[(df.age <= 50) & (df.bmi != 0)]

# Drop uninformative feature
df = df.drop(columns=['body_temp'])

# Define features and target
target = 'risk_level'
X = df.drop(columns=[target])
y = df[target]

# Initialize XGBoost with best parameters
best_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=1.0,
    min_child_weight=1,
    gamma=0.3,
    colsample_bytree=0.6,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=67
)

# Train on full dataset
best_xgb.fit(X, y)

# Save the trained model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

print("XGBoost model trained and saved!")