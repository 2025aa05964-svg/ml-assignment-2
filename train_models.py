# ================================
# ML Assignment 2 – Model Training
# Heart Disease Dataset (UCI)
# ================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------

df = pd.read_csv("heart.csv")
print("Columns:", df.columns)

# -------------------------------
# STEP 2: Clean Dataset
# -------------------------------

# Drop non-informative columns
df = df.drop(columns=["id", "dataset"])

# Convert target to binary
# num = 0 → no disease, 1–4 → disease
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# Separate features and target
X = df.drop("num", axis=1)
y = df["num"].astype(int)

print("Target values:", y.unique())

# -------------------------------
# STEP 3: Encode Categorical Columns
# -------------------------------

categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Handle missing values
X = X.fillna(X.median())

# Save training columns (VERY IMPORTANT for Streamlit)
os.makedirs("model", exist_ok=True)
joblib.dump(X.columns, "model/trained_columns.pkl")

# -------------------------------
# STEP 4: Train–Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# STEP 5: Feature Scaling
# -------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# STEP 6: Evaluation Function
# -------------------------------

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    if len(np.unique(y_true)) == 2:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["AUC"] = np.nan

    return metrics

# -------------------------------
# STEP 7: Train Models
# -------------------------------

results = {}
models = {}

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]
results["Logistic Regression"] = evaluate_model(y_test, y_pred, y_prob)
models["Logistic Regression"] = lr

# 2. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]
results["Decision Tree"] = evaluate_model(y_test, y_pred, y_prob)
models["Decision Tree"] = dt

# 3. KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)[:, 1]
results["KNN"] = evaluate_model(y_test, y_pred, y_prob)
models["KNN"] = knn

# 4. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:, 1]
results["Naive Bayes"] = evaluate_model(y_test, y_pred, y_prob)
models["Naive Bayes"] = nb

# 5. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = evaluate_model(y_test, y_pred, y_prob)
models["Random Forest"] = rf

# 6. XGBoost
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]
results["XGBoost"] = evaluate_model(y_test, y_pred, y_prob)
models["XGBoost"] = xgb

# -------------------------------
# STEP 8: Results Table
# -------------------------------

results_df = pd.DataFrame(results).T
results_df = results_df[
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
]

print("\nFinal Evaluation Metrics:\n")
print(results_df)

# -------------------------------
# STEP 9: Save Models
# -------------------------------

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(lr, "model/logistic.pkl")
joblib.dump(dt, "model/decision_tree.pkl")
joblib.dump(knn, "model/knn.pkl")
joblib.dump(nb, "model/naive_bayes.pkl")
joblib.dump(rf, "model/random_forest.pkl")
joblib.dump(xgb, "model/xgboost.pkl")

print("\nAll models saved successfully!")
