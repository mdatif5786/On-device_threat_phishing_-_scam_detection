
import json
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path("artifacts/sms_combined.joblib")
EXPORT_PATH = Path("artifacts/sms_android_export.json")

saved = joblib.load(MODEL_PATH)
features = saved["features"]
classifier = saved["model"]
threshold = float(saved["threshold"])

messages = [
    "Congratulations! You have won a free prize. Call now!",
    "URGENT: Your bank account is blocked. Click this link immediately.",
    "Hey, are we still meeting for lunch today?",
    "Please call me when you reach home.",
    "Claim your free data offer now!",
    "Your OTP is 123456. Do not share it with anyone.",
]

scores = np.asarray(classifier.decision_function(
    features.transform(messages)
)).reshape(-1)

results = []

for message, score in zip(messages, scores):
    label = "spam" if score >= threshold else "not spam"

    results.append({
        "message": message,
        "score": float(score),
        "threshold": threshold,
        "label": label,
    })

with EXPORT_PATH.open("r", encoding="utf-8") as file:
    export = json.load(file)

output = Path("artifacts/sms_reference_tests.json")
output.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Reference tests saved to:", output.resolve())
print()

for item in results:
    print("Message:", item["message"])
    print("Score:", round(item["score"], 6))
    print("Label:", item["label"])
    print("-" * 50)
