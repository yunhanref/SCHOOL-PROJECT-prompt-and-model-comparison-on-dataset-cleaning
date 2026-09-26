# ============================================================
# BREAST CANCER - LOGISTIC REGRESSION
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================

FILE_PATH = "data.csv"

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("DATASET")
print("=" * 60)
print("Shape:", df.shape)


# ============================================================
# 2. BASIC CLEANING REQUIRED FOR THIS DATASET
# ============================================================

# Clean column names
df.columns = df.columns.str.strip()

# Remove completely empty columns
empty_columns = df.columns[df.isna().all()].tolist()

if empty_columns:
    df = df.drop(columns=empty_columns)
    print("Removed empty columns:", empty_columns)

# Remove exact duplicate rows
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Removed duplicate rows:", duplicate_count)


# ============================================================
# 3. CHECK TARGET
# ============================================================

if "diagnosis" not in df.columns:
    raise ValueError("ERROR: 'diagnosis' column was not found.")

# Diagnosis değerlerini temizle
df["diagnosis"] = df["diagnosis"].astype(str).str.strip()

print("Diagnosis values:", df["diagnosis"].unique())

# Hem B/M hem de 0/1 destekle
if set(df["diagnosis"].unique()).issubset({"B", "M"}):
    
    # Original Kaggle format
    y = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

elif set(df["diagnosis"].unique()).issubset({"0", "1"}):
    
    # Already encoded
    y = df["diagnosis"].astype(int)

else:
    raise ValueError(
        f"Unexpected diagnosis values: {set(df['diagnosis'].unique())}"
    )

# NaN kontrolü
if y.isna().any():
    raise ValueError("ERROR: Target contains NaN values.")

y = y.astype(int)

print("Final target values:", sorted(y.unique()))
print("Target distribution:")
print(y.value_counts())


# ============================================================
# 4. CREATE FEATURES
# ============================================================

# id is an identifier, NOT a predictive feature
X = df.drop(columns=["diagnosis", "id", "diagnosis_encoded"])

# Make sure all features are numeric
for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Check missing values
missing_values = X.isna().sum()

if missing_values.sum() > 0:
    print("\nMissing values detected:")
    print(missing_values[missing_values > 0])

    # Median imputation only if necessary
    # The original dataset should normally not need this.
    for column in X.columns:
        if X[column].isna().any():
            X[column] = X[column].fillna(X[column].median())

# Final safety check
if X.isna().sum().sum() > 0:
    raise ValueError(
        "ERROR: X still contains NaN values."
    )

if not np.isfinite(X.to_numpy()).all():
    raise ValueError(
        "ERROR: X still contains infinite values."
    )


# ============================================================
# 5. FINAL DATA CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATA CHECK")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing values in X:", X.isna().sum().sum())
print("Missing values in y:", y.isna().sum())
print("Classes:", sorted(y.unique()))
print("Class distribution:")
print(y.value_counts())


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 9. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 10. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# 11. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign (B)", "Malignant (M)"],
        zero_division=0
    )
)


# ============================================================
# 14. SUMMARY
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(results.to_string(index=False))
