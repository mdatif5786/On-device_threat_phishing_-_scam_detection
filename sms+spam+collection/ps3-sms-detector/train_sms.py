
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# The extracted UCI file has no header.
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

data = data.dropna().drop_duplicates()
data["label"] = data["label"].map({"ham": 0, "spam": 1})
data = data.dropna(subset=["label"])
data["label"] = data["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"],
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        strip_accents="unicode",
        min_df=2,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(
    y_test,
    predictions,
    target_names=["legitimate", "spam/scam"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "sms_detector.joblib")

examples = [
    "URGENT! Your bank account will be blocked. Verify now!",
    "Hey, are we still meeting for lunch?",
]

for message in examples:
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0][1]

    print("\nMessage:", message)
    print("Prediction:", "spam/scam" if prediction else "legitimate")
    print(f"Spam probability: {probability:.3f}")
