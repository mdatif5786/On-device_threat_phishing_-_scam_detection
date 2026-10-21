
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data_path = Path(
    "dataset/sms+spam+collection/SMSSpamCollection"
)
artifact_path = Path("artifacts/sms_baseline.joblib")

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8",
    on_bad_lines="error",
).dropna()

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

X = df["text"]
y = df["label"]

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev
)

artifact = joblib.load(artifact_path)
features = artifact["features"]
model = artifact["model"]
threshold = artifact["threshold"]

scores = model.decision_function(features.transform(X_test))
predictions = (scores >= threshold).astype(int)

errors = pd.DataFrame({
    "message": X_test.reset_index(drop=True),
    "actual": y_test.map({0: "ham", 1: "spam"}).reset_index(drop=True),
    "predicted": pd.Series(predictions).map({0: "ham", 1: "spam"}),
    "score": scores,
})

errors = errors[errors["actual"] != errors["predicted"]]
errors.to_csv("artifacts/misclassified_messages.csv", index=False)

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions))
print("Misclassified messages:", len(errors))
print("Saved to artifacts/misclassified_messages.csv")
