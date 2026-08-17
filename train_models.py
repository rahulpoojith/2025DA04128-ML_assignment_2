"""
train_models.py
----------------
Trains 5 classification models on the "Bank Customer Segmentation" transaction
dataset (India, 1M+ transactions, 800K+ customers), predicting whether a
customer is HIGH-VALUE or LOW-VALUE, using only their observable transaction
BEHAVIOR (spend patterns, timing, location, activity) -- deliberately
excluding account balance itself from the features, since balance is what
defines the label. This mirrors a real bank use case: flagging likely
high-value customers/prospects from behavioral signals alone.

Saves:
  - trained model objects        -> model/*.pkl
  - fitted StandardScaler        -> model/scaler.pkl
  - label encoder for the target -> model/label_encoder.pkl
  - feature column order         -> model/feature_columns.pkl
  - location frequency map       -> model/location_freq.pkl
  - location avg amount map      -> model/location_avg_amount.pkl
  - value threshold (balance)    -> model/value_threshold.pkl
  - held-out test split (CSV)    -> test_data.csv
  - metrics (JSON + CSV)         -> model/metrics.json, model/metrics_table.csv

Dataset: Bank Customer Segmentation (Kaggle) - bank_transactions_raw.csv
Raw rows: ~1,048,567 transactions across ~884,265 unique customers.
A stratified sample is used for training so the 5-model comparison runs in
reasonable time (still far above the minimum size requirement).
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
SAMPLE_SIZE = 200_000
VALUE_PERCENTILE = 0.67   # top 33% of customers by account balance = "High-Value"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "bank_transactions_raw.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw transaction-level data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"TransactionAmount (INR)": "TransactionAmount"})

# ---------------------------------------------------------------------------
# 2. Basic real-world cleaning
# ---------------------------------------------------------------------------
df = df.dropna(subset=["CustomerDOB", "CustLocation", "CustAccountBalance"])
df = df[df["TransactionAmount"] > 0]

df["CustomerDOB"] = pd.to_datetime(df["CustomerDOB"], format="%d/%m/%y", errors="coerce")
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d/%m/%y", errors="coerce")
df = df.dropna(subset=["CustomerDOB", "TransactionDate"])

# pandas' 2-digit-year pivot occasionally puts a DOB in the future -> shift back 100y
future_dob = df["CustomerDOB"] > pd.Timestamp("2005-01-01")
df.loc[future_dob, "CustomerDOB"] = df.loc[future_dob, "CustomerDOB"] - pd.DateOffset(years=100)

reference_date = df["TransactionDate"].max()
df["Age"] = ((reference_date - df["CustomerDOB"]).dt.days / 365.25).astype(int)
df = df[(df["Age"] >= 15) & (df["Age"] <= 90)]

# ---------------------------------------------------------------------------
# 3. Define the target: High-Value vs Low-Value customer segment
#    (based on account balance -- balance itself is then EXCLUDED from
#    the feature set to avoid trivial leakage)
# ---------------------------------------------------------------------------
value_threshold = df["CustAccountBalance"].quantile(VALUE_PERCENTILE)
joblib.dump(value_threshold, os.path.join(MODEL_DIR, "value_threshold.pkl"))

df["ValueSegment"] = np.where(
    df["CustAccountBalance"] >= value_threshold, "High-Value", "Low-Value"
)
print("Value threshold (account balance):", round(value_threshold, 2))
print(df["ValueSegment"].value_counts(normalize=True))

# ---------------------------------------------------------------------------
# 4. Feature engineering (transaction-level; each row = one transaction)
#    -- purely behavioral: spend, timing, activity, location. No balance.
# ---------------------------------------------------------------------------
df["TxnHour"] = (df["TransactionTime"] // 10000).astype(int).clip(0, 23)
df["TxnMinute"] = ((df["TransactionTime"] // 100) % 100).astype(int).clip(0, 59)
df["TxnDayOfWeek"] = df["TransactionDate"].dt.dayofweek
df["TxnDayOfMonth"] = df["TransactionDate"].dt.day
df["TxnMonth"] = df["TransactionDate"].dt.month
df["IsWeekend"] = (df["TxnDayOfWeek"] >= 5).astype(int)

df["LogTransactionAmount"] = np.log1p(df["TransactionAmount"])

# how many transactions this customer has in the (cleaned) dataset
df["CustomerTxnCount"] = df.groupby("CustomerID")["TransactionID"].transform("count")

# location "size"/activity proxy + how big this txn is relative to its location's norm
location_freq = df["CustLocation"].value_counts()
df["LocationFrequency"] = df["CustLocation"].map(location_freq)

location_avg_amount = df.groupby("CustLocation")["TransactionAmount"].mean()
df["LocationAvgAmount"] = df["CustLocation"].map(location_avg_amount)
df["AmountToLocationAvgRatio"] = df["TransactionAmount"] / df["LocationAvgAmount"]

joblib.dump(location_freq, os.path.join(MODEL_DIR, "location_freq.pkl"))
joblib.dump(location_avg_amount, os.path.join(MODEL_DIR, "location_avg_amount.pkl"))

FEATURE_COLUMNS = [
    "Age",
    "TransactionAmount",
    "LogTransactionAmount",
    "TxnHour",
    "TxnMinute",
    "TxnDayOfWeek",
    "TxnDayOfMonth",
    "TxnMonth",
    "IsWeekend",
    "CustomerTxnCount",
    "LocationFrequency",
    "AmountToLocationAvgRatio",
]

# ---------------------------------------------------------------------------
# 5. Stratified sample down to a trainable size
# ---------------------------------------------------------------------------
df_model = df[FEATURE_COLUMNS + ["ValueSegment"]].dropna()

if len(df_model) > SAMPLE_SIZE:
    df_model, _ = train_test_split(
        df_model, train_size=SAMPLE_SIZE, stratify=df_model["ValueSegment"],
        random_state=RANDOM_STATE
    )

print(f"Modeling on {len(df_model)} rows, {len(FEATURE_COLUMNS)} features.")

# ---------------------------------------------------------------------------
# 6. Encode target: Low-Value/High-Value -> 0/1
# ---------------------------------------------------------------------------
target_encoder = LabelEncoder()
y_all = target_encoder.fit_transform(df_model["ValueSegment"])
joblib.dump(target_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("Target classes:", dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

X_all = df_model[FEATURE_COLUMNS]
joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

# ---------------------------------------------------------------------------
# 7. Train / test split (stratified 80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
)

test_export = X_test.copy()
test_export["ValueSegment"] = target_encoder.inverse_transform(y_test)
test_export.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)
print(f"Saved test_data.csv with {len(test_export)} rows.")

# ---------------------------------------------------------------------------
# 8. Scale features
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ---------------------------------------------------------------------------
# 9. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=120, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# ---------------------------------------------------------------------------
# 10. Train, evaluate, and save each model
# ---------------------------------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, fname), compress=3)

# ---------------------------------------------------------------------------
# 11. Save metrics as both JSON and CSV
# ---------------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_DIR, "metrics_table.csv"), index=False)

print("\nFinal comparison table:\n")
print(results_df.to_string(index=False))
