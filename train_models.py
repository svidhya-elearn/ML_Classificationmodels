# train_models.py

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -------------------------------------------------
# Create Required Folders
# -------------------------------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("test_data", exist_ok=True)

print("Folders created successfully.")


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
print("Loading dataset...")
data = pd.read_csv("letter-recognition.csv")

X = data.drop("letter", axis=1)
y = data["letter"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
pickle.dump(label_encoder, open("model/label_encoder.pkl", "wb"))


# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train-Test split completed.")


# -------------------------------------------------
# Save Test Dataset (Original Labels for Streamlit)
# -------------------------------------------------
test_df = X_test.copy()
test_df["letter"] = label_encoder.inverse_transform(y_test)

test_df.to_csv("test_data/letter-recognition-test-data.csv", index=False)

print("Test dataset saved inside 'test_data/letter-recognition-test-data.csv'")


# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pickle.dump(scaler, open("model/scaler.pkl", "wb"))


# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------
def evaluate_model(model, model_name):

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    auc = roc_auc_score(
        pd.get_dummies(y_test),
        y_prob,
        multi_class="ovr"
    )


# =================================================
# 1️⃣ Logistic Regression
# =================================================
# print("\nTraining Logistic Regression...")
logistic_model = LogisticRegression(max_iter=2000)
logistic_model.fit(X_train_scaled, y_train)
pickle.dump(logistic_model, open("model/logistic_regression.pkl", "wb"))
evaluate_model(logistic_model, "Logistic Regression")


# =================================================
# 2️⃣ Decision Tree
# =================================================
# print("\nTraining Decision Tree...")
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_scaled, y_train)
pickle.dump(decision_tree_model, open("model/decision_tree.pkl", "wb"))
evaluate_model(decision_tree_model, "Decision Tree")


# =================================================
# 3️⃣ KNN
# =================================================
# print("\nTraining KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
pickle.dump(knn_model, open("model/knn.pkl", "wb"))
evaluate_model(knn_model, "KNN")


# =================================================
# 4️⃣ Naive Bayes
# =================================================
# print("\nTraining Naive Bayes...")
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_scaled, y_train)
pickle.dump(naive_bayes_model, open("model/naive_bayes.pkl", "wb"))
evaluate_model(naive_bayes_model, "Naive Bayes")


# =================================================
# 5️⃣ Random Forest
# =================================================
# print("\nTraining Random Forest...")
random_forest_model = RandomForestClassifier(
    n_estimators=25,
    max_depth=6,
    random_state=42
)
random_forest_model.fit(X_train_scaled, y_train)
pickle.dump(random_forest_model, open("model/random_forest.pkl", "wb"))
evaluate_model(random_forest_model, "Random Forest")


# =================================================
# 6️⃣ XGBoost
# =================================================
# print("\nTraining XGBoost...")
xgboost_model = XGBClassifier(
    n_estimators=40,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    use_label_encoder=False
)
xgboost_model.fit(X_train_scaled, y_train)
pickle.dump(xgboost_model, open("model/xgboost.pkl", "wb"))
evaluate_model(xgboost_model, "XGBoost")

#
# print("\nAll models trained successfully.")
# print("Models saved in 'model/' folder.")
# print("Test dataset saved in 'test_data/' folder.")
