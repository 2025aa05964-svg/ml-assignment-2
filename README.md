# ❤️ Machine Learning Assignment 2  
## Heart Disease Classification using Multiple ML Models

---

## 📌 Problem Statement
Heart disease is one of the leading causes of mortality worldwide. Early and accurate prediction can assist healthcare professionals in taking preventive measures.

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease using clinical attributes. The project also demonstrates end-to-end ML workflow including preprocessing, training, evaluation, and deployment using Streamlit.

---

## 📊 Dataset Description
**Dataset:** UCI Heart Disease Dataset  
**Problem Type:** Binary Classification  

### Dataset Properties
- Number of instances: ~900  
- Number of features: 13  
- Meets assignment constraint of ≥ 500 samples and ≥ 12 features.

### Target Variable – `num`
- `0` → No heart disease  
- `>0` → Converted to `1` (heart disease present)

### Feature Examples
Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise induced angina, ST depression, slope, number of vessels, thalassemia.

---

## 🤖 Machine Learning Models Implemented
All models were trained and evaluated on the same dataset.

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## 📈 Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8369 | 0.9029 | 0.8333 | 0.8824 | 0.8571 | 0.6691 |
| Decision Tree | 0.7880 | 0.7813 | 0.7890 | 0.8431 | 0.8152 | 0.5691 |
| KNN | 0.8315 | 0.9020 | 0.8142 | 0.9020 | 0.8558 | 0.6594 |
| Naive Bayes | 0.8369 | 0.8899 | 0.8529 | 0.8529 | 0.8529 | 0.6700 |
| Random Forest (Ensemble) | **0.8424** | **0.9180** | 0.8411 | 0.8824 | **0.8612** | **0.6801** |
| XGBoost (Ensemble) | 0.8370 | 0.8892 | 0.8273 | **0.8922** | 0.8585 | 0.6695 |

---

## 🧠 Observations on Model Performance

| Model | Observation |
|------|-------------|
| Logistic Regression | Provides a strong baseline with high AUC, suggesting good separability in feature space. |
| Decision Tree | Comparatively lower performance; prone to overfitting and sensitive to small variations. |
| KNN | Achieves high recall but depends heavily on scaling and neighborhood selection. |
| Naive Bayes | Delivers competitive results despite the independence assumption. |
| Random Forest | Best overall performer; ensemble averaging improves stability and generalization. |
| XGBoost | Produces balanced metrics with strong recall due to boosting and regularization. |

---

## 💻 Streamlit Application Features
The deployed web application allows users to:

- Download a sample dataset  
- Upload custom CSV test data  
- Select any of the trained models  
- View Accuracy, AUC, Precision, Recall, F1, MCC  
- Visualize confusion matrix  
- Display classification report  

---

## 🚀 Deployment
The application is deployed using **Streamlit Community Cloud** and is accessible via the link provided in the assignment submission.

---
