# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import requests

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Letter Recognition Classification",
    layout="wide"
)

st.title("🔤 Letter Recognition - Multi-Class Classification App")
st.markdown("---")

# -------------------------------------------------
# Download Test CSV From GitHub
# -------------------------------------------------
st.markdown("##Download Sample Test Dataset")

github_file_url = "https://github.com/svidhya-elearn/ML_Classificationmodels/raw/main/test_data/test_dataset.csv"
response = requests.get(github_file_url)
st.download_button(
    label="Download Sample Test CSV",
    data=response.content,
    file_name="test_dataset.csv",
    mime="text/csv"
)
#
# st.markdown(
#     f"""
#     <a href="{github_file_url}" download>
#         <button style="
#             background-color:#28a745;
#             color:white;
#             padding:12px 25px;
#             border:none;
#             border-radius:8px;
#             font-size:16px;
#             cursor:pointer;">
#             Download Test CSV
#         </button>
#     </a>
#     """,
#     unsafe_allow_html=True
# )

st.markdown("---")

# -------------------------------------------------
# Upload CSV Section
# -------------------------------------------------
st.markdown("##Upload Test CSV File")
uploaded_file = st.file_uploader("Upload CSV file (must include 'letter' column)", type=["csv"])

st.markdown("---")

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
st.markdown("##Select Machine Learning Model")

model_option = st.selectbox(
    "Choose Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

st.markdown("---")

# -------------------------------------------------
# If File Uploaded
# -------------------------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "letter" not in df.columns:
        st.error("Uploaded CSV must contain 'letter' column.")
    else:

        # Separate features and target
        X = df.drop("letter", axis=1)
        y = df["letter"]

        # Load label encoder
        le = pickle.load(open("model/label_encoder.pkl", "rb"))
        y = le.transform(y)

        # Load scaler
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        X = scaler.transform(X)

        # Load selected model
        model_filename = model_option.lower().replace(" ", "_") + ".pkl"
        model = pickle.load(open(f"model/{model_filename}", "rb"))

        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        # -------------------------------------------------
        # Evaluation Metrics
        # -------------------------------------------------
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="macro")
        recall = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")
        mcc = matthews_corrcoef(y, y_pred)

        auc = roc_auc_score(
            pd.get_dummies(y),
            y_prob,
            multi_class="ovr"
        )

        # -------------------------------------------------
        # Display Metrics
        # -------------------------------------------------
        st.markdown("## 📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("AUC Score", f"{auc:.4f}")

        col2.metric("Precision", f"{precision:.4f}")
        col2.metric("Recall", f"{recall:.4f}")

        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------
        st.markdown("## 📌 Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            cmap="Blues",
            ax=ax,
            cbar=False
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        st.pyplot(fig)

        st.markdown("---")

        # -------------------------------------------------
        # Prediction Table
        # -------------------------------------------------
        st.markdown("## 🔎 Sample Predictions")

        result_df = pd.DataFrame({
            "Actual Label": le.inverse_transform(y),
            "Predicted Label": le.inverse_transform(y_pred)
        })

        st.dataframe(result_df.head(20))

else:
    st.info("Please upload a test CSV file to evaluate the selected model.")
