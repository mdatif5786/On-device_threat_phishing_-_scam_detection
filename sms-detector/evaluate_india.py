
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path("artifacts/sms_baseline.joblib")
DATA_PATH = Path("data/cleaned/india_sms_clean.csv")

artifact = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, encoding="utf-8")

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)
df["label"] = df["label"].str.lower().str.strip()

if not set(df["label"].unique()).issubset({"ham", "spam"}):
    raise ValueError("Unexpected labels in Indian dataset.")

features = artifact["features"]
model = artifact["model"]
threshold = artifact["threshold"]

scores = model.decision_function(features.transform(df["text"]))
predictions = (scores >= threshold).astype(int)
actual = df["label"].map({"ham": 0, "spam": 1}).to_numpy()

print("Indian dataset evaluation")
print("Messages evaluated:", len(df))
print("Model:", artifact["model_name"])
print("Threshold:", threshold)
print("\nClassification report:")
print(classification_report(
    actual, predictions,
    labels=[0, 1],
    target_names=["ham", "spam"],
    digits=4,
    zero_division=0,
))
print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(actual, predictions, labels=[0, 1]))

results = df[["text", "label"]].copy()
results["predicted"] = pd.Series(predictions).map({0: "ham", 1: "spam"})
results["score"] = scores
results.to_csv("artifacts/india_sms_evaluation.csv", index=False)

print("\nSaved predictions to artifacts/india_sms_evaluation.csv")
