import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Classification Models Comparison App")

# Load models
models = {
    "Logistic Regression": joblib.load("models/log_reg.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}

scaler = joblib.load("models/scaler.pkl")

uploaded_file = st.file_uploader("Upload Test CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = scaler.transform(data.iloc[:, :-1])
    y_true = data.iloc[:, -1]

    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    st.write("### Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_true, y_pred))
    st.write("Precision:", precision_score(y_true, y_pred))
    st.write("Recall:", recall_score(y_true, y_pred))
    st.write("F1 Score:", f1_score(y_true, y_pred))
    st.write("AUC Score:", roc_auc_score(y_true, y_prob))
    st.write("MCC Score:", matthews_corrcoef(y_true, y_pred))

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)
