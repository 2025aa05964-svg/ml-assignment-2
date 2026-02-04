# ml-assignment-2
# Machine Learning Assignment 2 – Heart Disease Classification


## Problem Statement
Heart disease is one of the leading causes of death worldwide. Early prediction of heart disease can help in taking preventive measures.  
This project aims to predict the presence of heart disease using multiple machine learning classification models and compare their performance using standard evaluation metrics.

---

## Dataset Description
- **Dataset Name:** Heart Disease Dataset  
- **Source:** UCI Machine Learning Repository  
- **Problem Type:** Binary Classification  
- **Number of Instances:** ~900  
- **Number of Features:** 13 (after preprocessing)  
- **Target Variable:** `num`  
  - 0 → No heart disease  
  - 1 → Presence of heart disease  

The dataset contains patient information such as age, sex, chest pain type, cholesterol levels, resting blood pressure, ECG results, and other clinical attributes.

---

## Machine Learning Models Used
The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----|---------|-----|-----------|--------|---------|-----|
| Logistic Regression | 0.8369 | 0.9029 | 0.8333 | 0.8824 | 0.8571 | 0.6691 |
| Decision Tree | 0.7880 | 0.7813 | 0.7890 | 0.8431 | 0.8152 | 0.5691 |
| KNN | 0.8315 | 0.9020 | 0.8142 | 0.9020 | 0.8558 | 0.6594 |
| Naive Bayes | 0.8369 | 0.8899 | 0.8529 | 0.8529 | 0.8529 | 0.6700 |
| Random Forest (Ensemble) | 0.8424 | 0.9180 | 0.8411 | 0.8824 | 0.8612 | 0.6801 |
| XGBoost (Ensemble) | 0.8370 | 0.8892 | 0.8273 | 0.8922 | 0.8585 | 0.6695 |

---

## Observations

| Model | Observation |
|-----|-------------|
| Logistic Regression | Performs well with high AUC, indicating good linear separability. |
| Decision Tree | Lower performance due to overfitting and sensitivity to data variations. |
| KNN | High recall but slightly lower precision, dependent on feature scaling. |
| Naive Bayes | Competitive performance despite strong independence assumptions. |
| Random Forest | Best overall performance due to ensemble averaging and robustness. |
| XGBoost | Strong recall and balanced metrics due to boosting and regularization. |

---

## Streamlit Application
A Streamlit web application was developed to:
- Upload test data (CSV)
- Select a trained ML model
- View evaluation metrics
- Display confusion matrix and classification report

The app is deployed on **Streamlit Community Cloud**.
