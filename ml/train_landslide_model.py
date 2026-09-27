from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    GroupKFold,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "training_dataset.csv"
MODEL_FILE = BASE_DIR / "landslide_model_candidate.pkl"
VALIDATION_REPORT_FILE = BASE_DIR / "model_candidate_validation_report.json"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RakshaNet - Landslide ML Model Training")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print()
print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "precip_1d",
    "precip_3d",
    "precip_7d",
    "precip_hours_1d",
    "precip_hours_3d",
    "precip_hours_7d",
    "temp_mean",
    "temp_max",
    "temp_min",
    "wind_max",
    "latitude",
    "longitude",
    "month",
]


TARGET = "target"


X = df[FEATURES].copy()
y = df[TARGET].astype(int)


# ============================================================
# BASIC CHECKS
# ============================================================

print()
print("Target distribution:")
print(y.value_counts())


if y.nunique() < 2:
    raise ValueError(
        "Dataset must contain both target classes."
    )


if X.isna().any().any():
    raise ValueError(
        "Feature matrix contains missing values."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Spatially grouped hold-out prevents paired controls/events from the same
# coordinate cluster appearing in both train and test sets. Random row splits
# can otherwise leak location-specific patterns into validation metrics.
groups = (
    df["latitude"].round(2).astype(str)
    + "_"
    + df["longitude"].round(2).astype(str)
)

gkf = GroupKFold(n_splits=5)
train_index, test_index = next(gkf.split(X, y, groups=groups))

X_train = X.iloc[train_index].copy()
X_test = X.iloc[test_index].copy()
y_train = y.iloc[train_index].copy()
y_test = y.iloc[test_index].copy()


print()
print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


print()
print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# TEST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

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

try:
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )
except ValueError:
    roc_auc = float("nan")


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


print()
print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CROSS VALIDATION
# ============================================================

print()
print("=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)


cv = GroupKFold(n_splits=5)

cv_scores = cross_val_score(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring="f1"
)


print(
    "F1 scores:",
    np.round(cv_scores, 4)
)

print(
    f"Mean F1: {cv_scores.mean():.4f}"
)

print(
    f"Std F1 : {cv_scores.std():.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
})


importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE CANDIDATE MODEL + VALIDATION REPORT
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "version": "rakshanet-landslide-candidate",
        "validation": "spatial_grouped_5fold",
    },
    MODEL_FILE
)

validation_report = {
    "validation": "spatial_grouped_5fold",
    "rows": int(len(df)),
    "positive_rows": int(y.sum()),
    "negative_rows": int((y == 0).sum()),
    "held_out_rows": int(len(X_test)),
    "holdout_accuracy": float(accuracy),
    "holdout_precision": float(precision),
    "holdout_recall": float(recall),
    "holdout_f1": float(f1),
    "holdout_roc_auc": float(roc_auc) if np.isfinite(roc_auc) else None,
    "grouped_cv_f1_mean": float(cv_scores.mean()),
    "grouped_cv_f1_std": float(cv_scores.std()),
    "activation_status": "candidate_only",
    "note": "Candidate model is intentionally separate from the active ONNX/PKL pair. Export and independently validate before activation."
}

with open(VALIDATION_REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(validation_report, f, indent=2)

print()
print("=" * 70)
print("CANDIDATE MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)
print(VALIDATION_REPORT_FILE)