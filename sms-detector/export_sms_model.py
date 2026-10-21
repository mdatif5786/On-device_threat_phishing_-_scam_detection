
import json
from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path("artifacts/sms_combined.joblib")
OUTPUT_PATH = Path("artifacts/sms_android_export.json")


def main():
    saved = joblib.load(MODEL_PATH)

    if not isinstance(saved, dict):
        raise TypeError("Expected the saved model to be a dictionary.")

    features = saved["features"]
    classifier = saved["model"]
    threshold = float(saved["threshold"])

    # The original training metadata identifies spam as the positive label.
    positive_label = str(saved.get("positive_label", "spam")).lower()

    if positive_label != "spam":
        raise ValueError(
            f"Expected the trained positive label to be 'spam'; "
            f"found {positive_label!r}"
        )

    classes = [int(value) for value in classifier.classes_]

    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError(
            f"Expected classifier classes [0, 1]; found {classes}"
        )

    # This training setup uses 0 = not spam and 1 = spam.
    positive_class = 1

    # Export both fitted TF-IDF vectorizers.
    vectorizers = {}

    for name, vectorizer in features.transformer_list:
        vectorizers[name] = {
            "vocabulary": {
                str(token): int(index)
                for token, index in vectorizer.vocabulary_.items()
            },
            "idf": np.asarray(
                vectorizer.idf_, dtype=np.float64
            ).tolist(),
            "analyzer": str(vectorizer.analyzer),
            "ngram_range": [
                int(vectorizer.ngram_range[0]),
                int(vectorizer.ngram_range[1]),
            ],
            "sublinear_tf": bool(vectorizer.sublinear_tf),
            "lowercase": bool(vectorizer.lowercase),
            "strip_accents": vectorizer.strip_accents,
            "norm": vectorizer.norm,
            "use_idf": bool(vectorizer.use_idf),
            "smooth_idf": bool(vectorizer.smooth_idf),
            "token_pattern": vectorizer.token_pattern,
            "min_df": int(vectorizer.min_df),
            "max_features": (
                int(vectorizer.max_features)
                if vectorizer.max_features is not None
                else None
            ),
        }

    weights = np.asarray(
        classifier.coef_, dtype=np.float64
    ).reshape(-1).tolist()

    intercept = float(
        np.asarray(classifier.intercept_).reshape(-1)[0]
    )

    # sklearn LinearSVC decision_function is positive toward classes_[1].
    # Orient the exported score so positive means spam.
    if classes[1] != positive_class:
        weights = [-value for value in weights]
        intercept = -intercept

    payload = {
        "format_version": 1,
        "model_type": "word_char_tfidf_linear_svm",
        "labels": {
            "0": "not spam",
            "1": "spam",
        },
        "positive_label": "spam",
        "negative_label": "not spam",
        "positive_class": positive_class,
        "threshold": threshold,
        "intercept": intercept,
        "weights": weights,
        "feature_union": {
            "transformer_weights": (
                {
                    str(key): float(value)
                    for key, value in features.transformer_weights.items()
                }
                if features.transformer_weights
                else {}
            ),
            "vectorizers": vectorizers,
        },
    }

    feature_count = sum(
        len(item["vocabulary"])
        for item in vectorizers.values()
    )

    if len(weights) != feature_count:
        raise ValueError(
            "Feature mismatch: classifier has "
            f"{len(weights)} weights, but vectorizers contain "
            f"{feature_count} vocabulary entries."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    print("Export successful")
    print("Output:", OUTPUT_PATH.resolve())
    print("Labels: spam / not spam")
    print("Classifier classes:", classes)
    print("Threshold:", threshold)
    print("Vectorizers:", list(vectorizers.keys()))
    print("Classifier weights:", len(weights))
    print("Intercept:", intercept)
    print("Export size:", OUTPUT_PATH.stat().st_size, "bytes")
    print(
        "Important: verify Kotlin predictions against Python before "
        "using this export in the Android app."
    )


if __name__ == "__main__":
    main()
