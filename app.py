"""
BITS Pilani M.Tech (AIML/DSE) - Machine Learning Assignment 2

Exactly the five models named in the assignment are implemented:
Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and Random Forest.
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BITS ML Assignment 2", page_icon="🤖", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATASET_FILE = BASE_DIR / "dataset.csv"
TEST_FILE = BASE_DIR / "test_data.csv"
METRICS_FILE = BASE_DIR / "model_metrics.csv"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

@st.cache_data
def load_project_dataset():
    data = load_breast_cancer(as_frame=True)
    df = data.data.copy()
    df[TARGET] = data.target.astype(int)
    return df

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }

def calculate_metrics(model, x_test, y_test):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

def train_and_save_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_project_dataset()
    df.to_csv(DATASET_FILE, index=False)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    test_df = x_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_FILE, index=False)

    results = []
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        results.append({"ML Model Name": model_name, **calculate_metrics(model, x_test, y_test)})
        with open(MODEL_DIR / MODEL_FILES[model_name], "wb") as f:
            pickle.dump(model, f)

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_FILE, index=False)
    return results_df

def models_are_ready():
    return all((MODEL_DIR / filename).exists() for filename in MODEL_FILES.values())

st.title("🤖 Machine Learning Classification Dashboard")
st.subheader("BITS Pilani M.Tech AIML — Machine Learning Assignment 2")
st.write(
    "Five required classification models evaluated with Accuracy, AUC, "
    "Precision, Recall, F1 Score and MCC."
)

if not models_are_ready() or not DATASET_FILE.exists() or not TEST_FILE.exists():
    with st.spinner("Preparing dataset and training the five classification models..."):
        comparison_df = train_and_save_models()
else:
    comparison_df = pd.read_csv(METRICS_FILE)

with st.expander("Dataset Information", expanded=False):
    dataset = load_project_dataset()
    c1, c2, c3 = st.columns(3)
    c1.metric("Instances", dataset.shape[0])
    c2.metric("Features", dataset.shape[1] - 1)
    c3.metric("Classes", dataset[TARGET].nunique())
    st.write("Dataset: **Breast Cancer Wisconsin (Diagnostic)** — UCI Machine Learning Repository")
    st.write("569 instances and 30 numeric features; satisfies the assignment minimum of 500 instances and 12 features.")
    st.dataframe(dataset.head(), use_container_width=True)

st.header("1. Model Comparison")
display_df = comparison_df.copy()
for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
    display_df[col] = display_df[col].round(4)
st.dataframe(display_df, use_container_width=True, hide_index=True)

best_model = comparison_df.sort_values(by=["F1", "AUC"], ascending=False).iloc[0]
st.success(
    f"Overall winner for this train/test split: **{best_model['ML Model Name']}** "
    f"(F1 = {best_model['F1']:.4f}, Accuracy = {best_model['Accuracy']:.4f})"
)

st.header("2. Model Observations")
observations = {
    "Logistic Regression": "Strong linear baseline after feature scaling; best reported overall performance in this run.",
    "Decision Tree": "Captures non-linear relationships and is interpretable; depth is limited to reduce overfitting.",
    "K-Nearest Neighbors": "Distance-based model; feature scaling is important because feature magnitudes affect distances.",
    "Naive Bayes": "Fast probabilistic baseline with relatively simple distributional assumptions.",
    "Random Forest": "Ensemble of decision trees with strong non-linear classification capability.",
}
obs_rows = [{"ML Model Name": name, "Observation about model performance": observations[name]} for name in MODEL_FILES]
st.dataframe(pd.DataFrame(obs_rows), use_container_width=True, hide_index=True)

st.header("3. Test Data Evaluation")
st.info("Upload the held-out test_data.csv used in the experiment. The file must contain the 30 feature columns and the target column.")
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("Upload the generated test_data.csv file for the assignment demonstration.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

if TARGET not in uploaded_df.columns:
    st.error(f"The uploaded CSV must contain the target column '{TARGET}'.")
    st.stop()

feature_columns = [c for c in load_project_dataset().columns if c != TARGET]
missing_columns = [c for c in feature_columns if c not in uploaded_df.columns]
if missing_columns:
    st.error("Missing required feature columns: " + ", ".join(missing_columns))
    st.stop()

x_uploaded = uploaded_df[feature_columns]
y_uploaded = uploaded_df[TARGET]
st.write(f"Uploaded test data: **{uploaded_df.shape[0]} rows** and **{len(feature_columns)} features**.")
st.dataframe(uploaded_df.head(10), use_container_width=True)

selected_model = st.selectbox("Select classification model", list(MODEL_FILES.keys()))
model_path = MODEL_DIR / MODEL_FILES[selected_model]

try:
    with open(model_path, "rb") as f:
        selected_model_object = pickle.load(f)
except Exception as exc:
    st.error(f"Unable to load the selected model: {exc}")
    st.stop()

predictions = selected_model_object.predict(x_uploaded)
probabilities = selected_model_object.predict_proba(x_uploaded)[:, 1]

st.header(f"4. Evaluation — {selected_model}")
accuracy = accuracy_score(y_uploaded, predictions)
auc = roc_auc_score(y_uploaded, probabilities)
precision = precision_score(y_uploaded, predictions, zero_division=0)
recall = recall_score(y_uploaded, predictions, zero_division=0)
f1 = f1_score(y_uploaded, predictions, zero_division=0)
mcc = matthews_corrcoef(y_uploaded, predictions)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{accuracy:.4f}")
m2.metric("AUC", f"{auc:.4f}")
m3.metric("Precision", f"{precision:.4f}")
m4, m5, m6 = st.columns(3)
m4.metric("Recall", f"{recall:.4f}")
m5.metric("F1 Score", f"{f1:.4f}")
m6.metric("MCC", f"{mcc:.4f}")

st.header("5. Confusion Matrix")
cm = confusion_matrix(y_uploaded, predictions)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
    xticklabels=["0", "1"], yticklabels=["0", "1"]
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix — {selected_model}")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.header("6. Classification Report")
report = classification_report(y_uploaded, predictions, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.header("7. Predictions")
result_df = uploaded_df.copy()
result_df["predicted_target"] = predictions
st.dataframe(result_df, use_container_width=True)
st.download_button(
    "Download Predictions CSV",
    result_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
    mime="text/csv",
)
