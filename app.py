"""
Streamlit app for ML Assignment 2 — Bank Customer Value Segmentation.

Predicts whether a customer is High-Value or Low-Value using only
behavioral transaction features (never their account balance directly).

Features:
  a. CSV upload for test data
  b. Model selection dropdown (Logistic Regression, Decision Tree, kNN,
     Naive Bayes, Random Forest)
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Value Lab",
    page_icon="\U0001F3E6",
    layout="wide",
)

ACCENT = "#B8862E"      # brass / banking-ledger gold
ACCENT_DARK = "#8C6420" # darker gold for hover/emphasis
INK = "#20242C"         # near-black navy for body text
PAPER = "#F7F5F0"       # warm cream page background
CARD_BG = "#FFFFFF"     # white card background
BORDER = "#E5E1D6"      # soft border color
SIDEBAR_BG = "#20242C"  # dark navy sidebar (matches ACCENT nicely)
SIDEBAR_TEXT = "#F5F1E6"# light cream text on dark sidebar

st.markdown(f"""
<style>
    /* ---- Page background & base text ---- */
    .stApp {{
        background-color: {PAPER} !important;
    }}
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
    .stMarkdown, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] {{
        color: {INK} !important;
    }}

    /* ---- Headings ---- */
    h1, h2, h3, h4 {{
        color: {INK} !important;
        font-family: 'Georgia', serif;
    }}
    h1 {{ border-bottom: 3px solid {ACCENT}; padding-bottom: 10px; }}

    /* ---- Caption text (subtitle under title) ---- */
    [data-testid="stCaptionContainer"] p {{
        color: #4A4E58 !important;
        font-size: 15px !important;
    }}

    /* ---- Info / warning / error boxes ---- */
    [data-testid="stAlertContainer"] {{
        background-color: #FBF3DC !important;
        border: 1px solid {ACCENT} !important;
        border-radius: 6px !important;
    }}
    [data-testid="stAlertContainer"] p {{
        color: {INK} !important;
        font-weight: 500;
    }}

    /* ---- Metric cards ---- */
    .metric-card {{
        background: {CARD_BG} !important;
        border: 1px solid {BORDER};
        border-left: 5px solid {ACCENT};
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .metric-card b {{ color: {ACCENT_DARK} !important; font-size: 13px; text-transform: uppercase; letter-spacing: 0.03em; }}
    .metric-card span {{ color: {INK} !important; font-weight: 700; }}

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {SIDEBAR_TEXT} !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {ACCENT} !important;
        border-bottom: none;
    }}
    [data-testid="stSidebar"] hr {{ border-color: #454A56 !important; }}

    /* File uploader box */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: #2B303B !important;
        border: 1px dashed {ACCENT} !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: {SIDEBAR_TEXT} !important; }}

    /* Selectbox */
    [data-testid="stSelectbox"] * {{ color: {SIDEBAR_TEXT} !important; }}
    div[data-baseweb="select"] > div {{
        background-color: #2B303B !important;
        border-color: {ACCENT} !important;
    }}

    /* ---- Buttons ---- */
    .stButton>button {{
        background-color: {ACCENT} !important;
        color: white !important;
        border-radius: 4px;
        border: none;
        font-weight: 600;
    }}
    .stButton>button:hover {{ background-color: {ACCENT_DARK} !important; }}

    /* ---- Dataframes / tables ---- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}

    /* ---- Divider ---- */
    hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    value_threshold = joblib.load(os.path.join(MODEL_DIR, "value_threshold.pkl"))
    models = {}
    for label, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[label] = joblib.load(path)
    return scaler, feature_columns, label_encoder, value_threshold, models


scaler, feature_columns, label_encoder, value_threshold, models = load_artifacts()
class_names = list(label_encoder.classes_)  # ['High-Value', 'Low-Value']

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Customer Value Lab")
st.caption(
    "Compare five classification models trained on the *Bank Customer "
    "Segmentation* dataset (India, 1M+ transactions). Predicts whether a "
    "customer is **High-Value** or **Low-Value** using only behavioral "
    "signals — spend, timing, activity, location — never their account "
    "balance directly."
)
st.info(
    f"'High-Value' = account balance in the top 33% of customers "
    f"(threshold ≈ ₹{value_threshold:,.0f}). Balance itself is excluded "
    f"from the model's features to avoid trivial leakage."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Must contain the same 12 feature columns as the training data, "
         "plus the true label column 'ValueSegment' (optional, needed for metrics)."
)

model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Note:** If you don't upload a file, the app uses the bundled "
    "`test_data.csv` (a 40,000-row held-out split) automatically."
)

# ---------------------------------------------------------------------------
# Load data (uploaded or default)
# ---------------------------------------------------------------------------
default_path = os.path.join(BASE_DIR, "test_data.csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    source_label = "your uploaded file"
else:
    data = pd.read_csv(default_path)
    source_label = "the bundled test_data.csv"

st.subheader("1. Data preview")
st.write(f"Using **{source_label}** — {data.shape[0]} rows, {data.shape[1]} columns.")
st.dataframe(data.head(10), use_container_width=True)

has_labels = "ValueSegment" in data.columns

# ---------------------------------------------------------------------------
# Prepare features
# ---------------------------------------------------------------------------
missing_cols = [c for c in feature_columns if c not in data.columns]
if missing_cols:
    st.error(
        f"Uploaded file is missing required columns: {missing_cols}. "
        "Please upload a CSV with the same feature columns used in training."
    )
    st.stop()

X = data[feature_columns]
X_scaled = scaler.transform(X)

model = models[model_choice]
y_pred_encoded = model.predict(X_scaled)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
y_proba = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

# ---------------------------------------------------------------------------
# Predictions table
# ---------------------------------------------------------------------------
st.subheader("2. Predictions")
pred_df = data.copy()
pred_df["Predicted Segment"] = y_pred_labels
if y_proba is not None:
    for i, cls in enumerate(class_names):
        pred_df[f"P({cls})"] = np.round(y_proba[:, i], 3)
st.dataframe(pred_df.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# Metrics (only computable if ground-truth labels are present)
# ---------------------------------------------------------------------------
st.subheader("3. Evaluation metrics")

if has_labels:
    y_true_encoded = label_encoder.transform(data["ValueSegment"])
    # positive class for binary metrics = 'Low-Value' encoding index 1 by default;
    # use predict_proba column matching that same encoding for AUC
    pos_index = 1
    y_proba_pos = y_proba[:, pos_index] if y_proba is not None else None

    acc = accuracy_score(y_true_encoded, y_pred_encoded)
    auc = roc_auc_score(y_true_encoded, y_proba_pos) if y_proba_pos is not None else float("nan")
    prec = precision_score(y_true_encoded, y_pred_encoded, zero_division=0)
    rec = recall_score(y_true_encoded, y_pred_encoded, zero_division=0)
    f1 = f1_score(y_true_encoded, y_pred_encoded, zero_division=0)
    mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)

    cols = st.columns(6)
    labels_vals = [
        ("Accuracy", acc), ("AUC", auc), ("Precision", prec),
        ("Recall", rec), ("F1 Score", f1), ("MCC", mcc),
    ]
    for c, (label, val) in zip(cols, labels_vals):
        c.markdown(
            f"<div class='metric-card'><b>{label}</b><br>"
            f"<span style='font-size:22px'>{val:.3f}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrBr", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("#### Classification report")
    report = classification_report(
        y_true_encoded, y_pred_encoded, target_names=class_names, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

else:
    st.info(
        "No 'ValueSegment' column found in the uploaded data, so evaluation "
        "metrics and the confusion matrix can't be computed — only predictions "
        "are shown above. Upload data that includes the true 'ValueSegment' "
        "label to see full evaluation."
    )

# ---------------------------------------------------------------------------
# Model comparison table (precomputed on the held-out test split)
# ---------------------------------------------------------------------------
st.subheader("4. All-model comparison (held-out test split)")
summary_path = os.path.join(MODEL_DIR, "metrics_table.csv")
if os.path.exists(summary_path):
    summary_df = pd.read_csv(summary_path)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.warning("metrics_table.csv not found in model/ — run train_models.py first.")

st.markdown("---")
st.caption("BITS WILP · M.Tech (AIML/DSE) · Machine Learning · Assignment 2 . 2025DA04128")