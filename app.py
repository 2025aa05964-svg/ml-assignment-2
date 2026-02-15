# =====================================
# ML Assignment 2 – Streamlit App
# Heart Disease Classification
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction App")
st.write("Upload test data, select a model, and view performance metrics.")

# Sample dataset download
st.subheader("📥 Download Sample Dataset")

# Sample dataset download
st.subheader("📥 Download Sample Dataset")

sample_df = pd.read_csv("data/heart_sample.csv")

csv = sample_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Heart Disease CSV",
    data=csv,
    file_name="heart_disease_sample.csv",
    mime="text/csv",
)
# -------------------------------
# Load Saved Artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    trained_columns = joblib.load("model/trained_columns.pkl")
    return models, scaler, trained_columns

models, scaler, trained_columns = load_artifacts()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload test CSV file (with 'num' column)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    if "num" not in df.columns:
        st.error("Uploaded CSV must contain 'num' column as target.")
        st.stop()

    # Convert target to binary
    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop("num", axis=1)
    y = df["num"]

    # -------------------------------
    # Handle missing values
    # -------------------------------
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # -------------------------------
    # Encode categorical columns
    # -------------------------------
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.fillna(0)

    # Align with training columns
    X = X.reindex(columns=trained_columns, fill_value=0)

    # Scale features
    X_scaled = scaler.transform(X)

    # -------------------------------
    # Model Selection
    # -------------------------------
    st.subheader("🤖 Model Selection")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    # -------------------------------
    # Prediction
    # -------------------------------
    if st.button("Run Prediction"):

        if model_name in ["Logistic Regression", "KNN"]:
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]
        else:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

        # -------------------------------
        # Metrics
        # -------------------------------
        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
        col2.metric("Precision", round(precision_score(y, y_pred), 3))
        col3.metric("Recall", round(recall_score(y, y_pred), 3))

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", round(f1_score(y, y_pred), 3))
        col5.metric("MCC", round(matthews_corrcoef(y, y_pred), 3))

        if len(np.unique(y)) == 2:
            col6.metric("AUC", round(roc_auc_score(y, y_prob), 3))
        else:
            col6.metric("AUC", "N/A")

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # -------------------------------
        # Classification Report
        # -------------------------------
        st.subheader("📑 Classification Report")
        st.text(classification_report(y, y_pred))



