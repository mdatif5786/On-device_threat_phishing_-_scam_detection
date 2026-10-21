
from pathlib import Path
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

DATA = Path("dataset/sms+spam+collection/SMSSpamCollection")
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

# Load the original tab-separated dataset.
df = pd.read_csv(
    DATA,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8",
    on_bad_lines="error",
)

df = df.dropna()
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)

# Remove exact duplicate messages before splitting to reduce leakage.
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

X = df["text"]
y = df["label"]

# Reserve the test set now; do not use it for model selection.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev
)

# Word features capture terms and short phrases.
# Character features help with spelling variants and obfuscation.
features = FeatureUnion([
    ("word", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )),
    ("char", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=200_000,
    )),
])

models = {
    "logistic_regression": LogisticRegression(
        C=4.0, max_iter=3000, class_weight=None
    ),
    "linear_svm": LinearSVC(
        C=1.0, class_weight=None, random_state=42
    ),
}

# Fit feature extraction on training data only.
X_train_features = features.fit_transform(X_train)
X_val_features = features.transform(X_val)

results = {}

for name, model in models.items():
    model.fit(X_train_features, y_train)

    # Validation scores support threshold selection.
    scores = model.decision_function(X_val_features)
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    ap = average_precision_score(y_val, scores)

    # Choose the threshold with best F1 on validation only.
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(f1[:-1].argmax()) if len(thresholds) else 0
    threshold = float(thresholds[best_idx]) if len(thresholds) else 0.0

    results[name] = (model, threshold, ap)
    print(f"\n=== {name} ===")
    print(f"Validation average precision: {ap:.4f}")
    print(f"Validation-selected threshold: {threshold:.4f}")
    print(f"Validation precision: {precision[best_idx]:.4f}")
    print(f"Validation recall: {recall[best_idx]:.4f}")
    print(f"Validation F1: {f1[best_idx]:.4f}")

# Select model by validation average precision.
best_name = max(results, key=lambda n: results[n][2])
best_model, threshold, _ = results[best_name]

# Evaluate the selected model once on the untouched test set.
X_test_features = features.transform(X_test)
test_scores = best_model.decision_function(X_test_features)
test_pred = (test_scores >= threshold).astype(int)

print(f"\nSelected model: {best_name}")
print("\nFinal held-out test report:")
print(classification_report(
    y_test, test_pred, target_names=["ham", "spam"], digits=4
))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, test_pred))

# Save the full feature extractor, model and selected threshold.
import joblib

joblib.dump(
    {
        "features": features,
        "model": best_model,
        "threshold": threshold,
        "positive_label": "spam",
        "model_name": best_name,
    },
    OUT / "sms_baseline.joblib",
)

print(f"\nSaved baseline to {OUT / 'sms_baseline.joblib'}")
