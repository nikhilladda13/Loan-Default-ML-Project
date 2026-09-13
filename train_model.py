
"""
Loan Default Prediction - Model Training
Dataset: UCI Statlog (German Credit Data), Dataset ID 144
The script downloads the dataset through ucimlrepo, cleans it, trains
Logistic Regression and Random Forest models, evaluates them, and saves
the best model as models/loan_default_model.pkl.
"""

import os
import json
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Downloading UCI German Credit dataset...")
dataset = fetch_ucirepo(id=144)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Make target a clean binary variable.
target = y.iloc[:, 0]
# UCI coding is 1 = Good and 2 = Bad in the original German data.
target = pd.to_numeric(target, errors="coerce").map({1: 0, 2: 1})

# Make readable feature names.
X.columns = [f"feature_{i+1}" for i in range(X.shape[1])]

# Remove rows with missing target, if any.
valid = target.notna()
X = X.loc[valid].copy()
target = target.loc[valid].astype(int)

# Save a local copy of the features and target for reproducibility.
local_df = X.copy()
local_df["default"] = target.values
local_df.to_csv("data/german_credit_clean.csv", index=False)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced"
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.20, random_state=42, stratify=target
)

results = {}
fitted = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
    fitted[name] = pipe

# Select by ROC-AUC, with recall as a secondary business consideration.
best_name = max(results, key=lambda k: (results[k]["roc_auc"], results[k]["recall"]))
best_model = fitted[best_name]

joblib.dump(best_model, "models/loan_default_model.pkl")

with open("models/model_metrics.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "results": results
    }, f, indent=2)

print("\nMODEL COMPARISON")
print(pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]])
print("\nSelected model:", best_name)
print("\nClassification report:")
print(classification_report(y_test, best_model.predict(X_test), target_names=["Good/Non-default", "Bad/Default"]))

print("\nSaved:")
print("models/loan_default_model.pkl")
print("models/model_metrics.json")
print("data/german_credit_clean.csv")
