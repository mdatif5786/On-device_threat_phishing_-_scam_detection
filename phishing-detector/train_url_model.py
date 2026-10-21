import csv
import json
import math
import ipaddress
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "artifacts"
PHISH_FILE = DATA_DIR / "phishtank_raw.csv"
TRANCO_FILE = DATA_DIR / "tranco_raw.csv"
MODEL_FILE = OUT_DIR / "url_model.json"
MAX_PER_CLASS = 20000
RANDOM_SEED = 42

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length",
    "dot_count", "hyphen_count", "at_count", "digit_ratio",
    "subdomain_count", "uses_https", "has_ip_host", "has_port",
    "percent_count", "double_slash_path", "suspicious_token_count",
    "host_has_brand_login_words", "host_entropy", "path_depth",
]
SUSPICIOUS = [
    "login", "verify", "update", "secure", "account", "signin",
    "banking", "password", "confirm", "wallet", "free", "bonus",
    "gift", "urgent", "recover",
]


def get_host(url):
    url = str(url).strip()
    if not url:
        return "", None
    if "://" not in url:
        url = "http://" + url
    try:
        parsed = urlsplit(url)
        return (parsed.hostname or "").lower().rstrip("."), parsed
    except ValueError:
        return "", None


def entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def extract_features(url):
    url = str(url).strip()
    host, parsed = get_host(url)
    if not host or parsed is None:
        return None
    try:
        ipaddress.ip_address(host)
        has_ip = 1.0
    except ValueError:
        has_ip = 0.0
    try:
        port = parsed.port
    except ValueError:
        return None

    path = parsed.path or ""
    query = parsed.query or ""
    lower_url = url.lower()
    host_parts = host.split(".")
    digit_count = sum(ch.isdigit() for ch in url)
    return [
        len(url), len(host), len(path), len(query), url.count("."),
        url.count("-"), url.count("@"), digit_count / max(len(url), 1),
        max(len(host_parts) - 2, 0),
        1.0 if parsed.scheme.lower() == "https" else 0.0,
        has_ip, 1.0 if port is not None else 0.0, url.count("%"),
        1.0 if "//" in path else 0.0,
        sum(token in lower_url for token in SUSPICIOUS),
        1.0 if any(word in host for word in
                   ["login", "verify", "secure", "account", "bank"]) else 0.0,
        entropy(host),
        len([part for part in path.split("/") if part]),
    ]


def read_phishtank(path):
    urls = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "url" not in reader.fieldnames:
            raise ValueError(
                "PhishTank CSV must have a 'url' column. Check the downloaded file."
            )
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            if (row.get("verified") or "yes").strip().lower() != "yes":
                continue
            if (row.get("online") or "yes").strip().lower() != "yes":
                continue
            urls.append(url)
    return urls


def read_tranco(path):
    urls = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if not row:
                continue
            domain = row[1].strip() if len(row) >= 2 else row[0].strip()
            domain = domain.lower().strip()
            if not domain or domain == "domain" or "/" in domain or " " in domain:
                continue
            if domain.isdigit():
                continue
            urls.append("https://" + domain + "/")
    return urls


def main():
    if not PHISH_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PHISH_FILE}")
    if not TRANCO_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRANCO_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phishing = list(dict.fromkeys(read_phishtank(PHISH_FILE)))
    benign = list(dict.fromkeys(read_tranco(TRANCO_FILE)))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(phishing)
    rng.shuffle(benign)

    count = min(MAX_PER_CLASS, len(phishing), len(benign))
    if count < 100:
        raise ValueError(
            f"Not enough usable data: PhishTank={len(phishing)}, Tranco={len(benign)}. "
            "Check that both CSV files downloaded correctly."
        )

    rows = [(url, 1) for url in phishing[:count]]
    rows += [(url, 0) for url in benign[:count]]
    features, labels = [], []
    for url, label in rows:
        values = extract_features(url)
        if values is not None:
            features.append(values)
            labels.append(label)

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    if len(X) == 0 or len(set(y.tolist())) < 2:
        raise ValueError("Could not build both classes from the CSV files.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
    )
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    print("Rows used:", len(y))
    print("Feature count:", len(FEATURE_NAMES))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["benign_candidate", "phishing"],
        digits=4, zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    payload = {
        "format_version": 1,
        "model_type": "standard_scaler_logistic_regression",
        "labels": {"0": "not phishing", "1": "phishing"},
        "positive_class": 1,
        "threshold": 0.0,
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "training_rows": int(len(y)),
        "training_source": [
            "PhishTank verified online feed",
            "Tranco popular-domain candidates",
        ],
        "warning": (
            "Tranco domains are candidates, not guaranteed safe URLs. "
            "Random-split metrics may overestimate real-world performance. "
            "Do not use this model as the only security check."
        ),
    }
    MODEL_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved model:", MODEL_FILE)
    print("Model size:", MODEL_FILE.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
