
from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)

ROOT = Path(".")
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

UCI_PATH = ROOT / "dataset/sms+spam+collection/SMSSpamCollection"
INDIA_PATH = ROOT / "data/cleaned/india_sms_clean.csv"

# Load UCI: ham/spam<TAB>message
uci = pd.read_csv(
    UCI_PATH, sep="\t", header=None,
    names=["label", "text"], encoding="utf-8",
    on_bad_lines="error",
)
uci = uci[["text", "label"]].copy()

# Load Indian dataset: text and label columns
india = pd.read_csv(INDIA_PATH, encoding="utf-8")
india = india[["text", "label"]].copy()

data = pd.concat([uci, india], ignore_index=True)
data["text"] = data["text"].astype("string").str.strip()
data["label"] = data["label"].astype("string").str.lower().str.strip()

data = data.dropna(subset=["text", "label"])
data = data[data["text"] != ""]
data = data[data["label"].isin(["ham", "spam"])].copy()

# Normalize for duplicate detection, but retain original text for modeling.
data["normalized"] = (
    data["text"].str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Exclude any normalized message that has conflicting labels.
label_counts = data.groupby("normalized")["label"].nunique()
conflicts = set(label_counts[label_counts > 1].index)
print("Conflicting messages excluded:", len(conflicts))
data = data[~data["normalized"].isin(conflicts)]

# Collapse duplicates so the same message cannot leak across splits.
data = data.drop_duplicates(subset="normalized").reset_index(drop=True)
data = data[["text", "label"]]

print("\nCombined unique dataset:", len(data))
print(data["label"].value_counts().to_string())

# Reserve test set before training or model selection.
train_val, test = train_test_split(
    data, test_size=0.20, random_state=42,
    stratify=data["label"],
)
train, val = train_test_split(
    train_val, test_size=0.20, random_state=43,
    stratify=train_val["label"],
)

print("\nSplit sizes:", len(train), len(val), len(test))

# Word n-grams capture terms/phrases; character n-grams help with
# spelling variants, unusual formatting and obfuscated text.
features = FeatureUnion([
    ("word", TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, sublinear_tf=True,
        strip_accents="unicode",
    )),
    ("char", TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5),
        min_df=2, sublinear_tf=True,
        max_features=200_000,
    )),
])

X_train = features.fit_transform(train["text"])
X_val = features.transform(val["text"])

models = {
    "linear_svm": LinearSVC(C=1.0, random_state=42),
    "logistic_regression": LogisticRegression(
        C=4.0, max_iter=3000
    ),
}

candidates = {}

for name, model in models.items():
    model.fit(X_train, (train["label"] == "spam").astype(int))
    val_scores = model.decision_function(X_val)
    y_val = (val["label"] == "spam").astype(int).to_numpy()

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_scores
    )
    ap = average_precision_score(y_val, val_scores)

    # Select the threshold maximizing validation F1.
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_i = int(f1[:-1].argmax())
    threshold = float(thresholds[best_i])

    candidates[name] = {
        "model": model,
        "threshold": threshold,
        "ap": ap,
        "val_f1": float(f1[best_i]),
    }

    print(f"\n{name}")
    print(f"Validation average precision: {ap:.4f}")
    print(f"Validation precision: {precision[best_i]:.4f}")
    print(f"Validation recall: {recall[best_i]:.4f}")
    print(f"Validation F1: {f1[best_i]:.4f}")
    print(f"Selected threshold: {threshold:.6f}")

# Select by validation average precision, not test results.
best_name = max(candidates, key=lambda n: candidates[n]["ap"])
best = candidates[best_name]
model = best["model"]
threshold = best["threshold"]

# Evaluate once on held-out test split.
X_test = features.transform(test["text"])
y_test = (test["label"] == "spam").astype(int).to_numpy()
test_scores = model.decision_function(X_test)
test_pred = (test_scores >= threshold).astype(int)

print("\nSelected model:", best_name)
print("\nHeld-out test report:")
print(classification_report(
    y_test, test_pred,
    labels=[0, 1], target_names=["ham", "spam"],
    digits=4, zero_division=0,
))
print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

# Report test performance separately for each source.
test_rows = test.reset_index(drop=True).copy()
test_rows["actual_spam"] = y_test
test_rows["predicted_spam"] = test_pred
test_rows["source"] = "combined_test"

# Save the deployable research artifact separately from the baseline.
joblib.dump({
    "features": features,
    "model": model,
    "threshold": threshold,
    "positive_label": "spam",
    "model_name": best_name,
}, ARTIFACTS / "sms_combined.joblib")

test_rows.to_csv(ARTIFACTS / "combined_test_predictions.csv", index=False)

print("\nSaved model to artifacts/sms_combined.joblib")
print("Saved held-out predictions to artifacts/combined_test_predictions.csv")
print("Note: this is an experimental combined-data benchmark, not an independent real-world test.")
