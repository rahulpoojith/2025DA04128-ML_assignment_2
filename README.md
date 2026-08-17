# Bank Customer Value Segment Prediction — ML Assignment 2

**BITS WILP · M.Tech (AIML/DSE) · Machine Learning**

**Rahul P**
**2025DA04128**

---

## a. Problem Statement

Banks want to identify **high-value customers** — those worth prioritizing
for premium products, retention offers, and relationship management — without
always having direct, up-to-date visibility into every customer's account
balance (e.g. when scoring prospects, or building a lightweight real-time
flag). This project builds and compares five supervised classification
models that predict whether a customer belongs to the **High-Value** or
**Low-Value** segment, using only their **observable transaction behavior**
(spend size, timing, activity level, location) — deliberately **excluding
account balance itself** from the model's inputs, since balance is what
defines the label.

This is a **binary classification** problem: `High-Value` vs `Low-Value`.

---

## b. Dataset Description

- **Name:** Bank Customer Segmentation (India)
- **Source:** Kaggle — <https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation>
- **Domain:** Retail banking transactions, India, 2016
- **Raw size:** ~1,048,567 transactions across ~884,265 unique customers
- **Raw columns:** `TransactionID`, `CustomerID`, `CustomerDOB`, `CustGender`, `CustLocation`, `CustAccountBalance`, `TransactionDate`, `TransactionTime`, `TransactionAmount (INR)`
- **Modeling sample:** a stratified random sample of 200,000 cleaned transaction rows (train/test = 160,000 / 40,000) — kept the 5-model comparison tractable while staying far above the minimum size requirement. The full cleaned dataset (~1.03M rows) is available in `bank_transactions_raw.csv`; `train_models.py`/`train_models.ipynb` can be re-run with a larger `SAMPLE_SIZE` if desired.

**Target construction:** `ValueSegment` = `High-Value` if `CustAccountBalance`
is in the **top 33%** of customers (threshold ≈ ₹34,661), else `Low-Value`
(~33% / ~67% split). **`CustAccountBalance` is then removed from the feature
set** so the model has to infer value from behavior alone, not from seeing
the answer directly.

**Cleaning applied:** dropped rows with missing DOB / location / balance;
dropped non-positive transaction amounts; parsed `CustomerDOB` and
`TransactionDate` (day-first format); corrected a small number of DOBs that
pandas' 2-digit-year parsing pushed into the future by shifting them back
100 years; dropped ages outside a plausible 15–90 range.

**Behavioral features engineered (12):**
| Feature | Description |
|---|---|
| `Age` | Customer age at time of transaction |
| `TransactionAmount` | Transaction size (INR) |
| `LogTransactionAmount` | Log-transformed transaction size (reduces skew) |
| `TxnHour`, `TxnMinute` | Time of day the transaction occurred |
| `TxnDayOfWeek`, `TxnDayOfMonth`, `TxnMonth` | When the transaction occurred |
| `IsWeekend` | Whether the transaction fell on a weekend |
| `CustomerTxnCount` | How many transactions this customer made (activity level) |
| `LocationFrequency` | How many transactions come from that location overall (city-size proxy) |
| `AmountToLocationAvgRatio` | This transaction's size relative to the average for its location |

**Split:** Stratified 80/20 train/test split (160,000 train / 40,000 test),
`random_state=42`. The 40,000-row test split is exported as `test_data.csv`
and is what the Streamlit app uses by default.

---

## c. GitHub Repository Link

> **TODO:** `https://github.com/rahulpoojith/2025DA04128-ML_assignment_2`


---

## d. Models Used

All five models were trained on the same 80% training split (features
standardized with `StandardScaler`, target label-encoded: `High-Value=0`,
`Low-Value=1`) and evaluated on the same held-out 20% test split (40,000
rows). Precision/Recall/F1 are reported for the `Low-Value` class (the
majority class, encoded as `1`).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6938 | 0.6912 | 0.7084 | 0.9230 | 0.8016 | 0.2140 |
| Decision Tree | 0.6935 | 0.6874 | 0.7199 | 0.8881 | 0.7952 | 0.2316 |
| kNN | 0.6873 | 0.6638 | 0.7108 | 0.8993 | 0.7940 | 0.2045 |
| Naive Bayes | 0.6860 | 0.6822 | 0.6976 | 0.9380 | 0.8001 | 0.1768 |
| Random Forest (Ensemble) | 0.6992 | 0.7034 | 0.7146 | 0.9174 | 0.8034 | 0.2354 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very high recall (0.92) on the majority "Low-Value" class but the lowest MCC of the top performers — it leans heavily on a few near-linear cues (transaction size, activity count) and defaults toward the majority class, so it's less discriminating on the harder High-Value cases. |
| Decision Tree | Second-best MCC (0.232) with the highest precision of all models (0.720) — its threshold-splitting nature fits well with features like `AmountToLocationAvgRatio` and `CustomerTxnCount`, which behave in a rule-like way (e.g. "big spender relative to their city → more likely High-Value"). |
| kNN | Weakest AUC (0.664) among the ensemble/tree-based approaches — with 12 behavioral features that don't cleanly separate the classes, nearest-neighbor distance gets diluted, and it doesn't capture the location-relative signal (`AmountToLocationAvgRatio`) as effectively as tree splits do. |
| Naive Bayes | Highest recall (0.938) but lowest MCC (0.177) and precision — its independence assumption is violated (transaction amount, location frequency, and activity count are correlated), which biases it toward over-predicting the majority "Low-Value" class. |
| Random Forest (Ensemble) | **Best AUC (0.703) and best MCC (0.235)** of all five models, with balanced precision/recall. Averaging 120 shallow trees (capped depth, min 5 samples per leaf — tuned to keep the saved model small and generalizable) captures non-linear interactions between spend behavior, timing, and location that no single linear or tree model captures alone — the most reliable model for ranking customers by likely value. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — best AUC and MCC, meaning it's the most reliable at actually distinguishing High-Value from Low-Value customers rather than just predicting the majority class. Worth noting: all five models land in a similar AUC range (0.66–0.71), reflecting that customer value has real but moderate behavioral signal — a realistic result for this kind of business problem, not an artificially "too easy" one. |

---

## Repository Structure

```
bank_project2/
├── app.py                        # Streamlit app (main entry point)
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Pinned Python version for Streamlit Community Cloud
├── README.md                     # This file
├── bank_transactions_raw.csv     # Full raw dataset (~1.05M transactions)
├── test_data.csv                 # Held-out test split (40,000 rows) used for demo/evaluation
├── train_models.py               # Cleans data, engineers features, trains 5 models, saves artifacts
├── train_models.ipynb            # Same pipeline as a notebook, pre-executed with outputs
└── model/
    ├── scaler.pkl                 # Fitted StandardScaler
    ├── label_encoder.pkl          # Encodes ValueSegment <-> {High-Value, Low-Value}
    ├── feature_columns.pkl        # Ordered list of feature columns
    ├── location_freq.pkl          # Location -> transaction-count lookup (for scoring new data)
    ├── location_avg_amount.pkl    # Location -> average transaction amount lookup
    ├── value_threshold.pkl        # The account-balance threshold defining High-Value
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── metrics.json               # Comparison metrics (JSON)
    └── metrics_table.csv          # Comparison metrics (CSV)
```

To retrain from scratch:
```bash
pip install -r requirements.txt
python train_models.py
# or open/run train_models.ipynb
```

To run the app locally:
```bash
python -m streamlit run app.py
```

---

## Live Streamlit App Link

> **TODO:** `https://<your-app-name>.streamlit.app`


## Streamlit App Features
- **CSV upload** — upload any test CSV with the same 12 feature columns (optionally include the `ValueSegment` column for evaluation)
- **Model selection dropdown** — switch between all 5 trained models
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC, computed live on the uploaded/default data
- **Confusion matrix & classification report** — visual + tabular breakdown of predictions
- **All-model comparison table** — the precomputed metrics for all 5 models side by side
# 2025DA04128-ML_assignment_2
