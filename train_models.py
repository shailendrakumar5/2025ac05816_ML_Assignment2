"""Train and evaluate the five ML models required by the assignment."""
from pathlib import Path
import pickle
import pandas as pd

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
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.data.copy()
df["target"] = data.target.astype(int)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "logistic_regression.pkl": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "decision_tree.pkl": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "knn.pkl": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes.pkl": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    "random_forest.pkl": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
}

names = {
    "logistic_regression.pkl": "Logistic Regression",
    "decision_tree.pkl": "Decision Tree",
    "knn.pkl": "K-Nearest Neighbors",
    "naive_bayes.pkl": "Naive Bayes",
    "random_forest.pkl": "Random Forest",
}

results = []
for filename, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": names[filename],
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    with open(MODEL_DIR / filename, "wb") as f:
        pickle.dump(model, f)

test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv(ROOT / "test_data.csv", index=False)
df.to_csv(ROOT / "dataset.csv", index=False)
pd.DataFrame(results).to_csv(ROOT / "model_metrics.csv", index=False)

print("Training complete. Five models saved in models/.")
print(pd.DataFrame(results).round(4).to_string(index=False))
