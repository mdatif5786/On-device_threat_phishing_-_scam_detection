
from pathlib import Path
import pandas as pd

UCI = Path("dataset/sms+spam+collection/SMSSpamCollection")
INDIA = Path("data/cleaned/india_sms_clean.csv")

def normalize(text):
    return " ".join(str(text).lower().split())

uci = pd.read_csv(
    UCI,
    sep="\t",
    header=None,
    names=["text", "label"],
    encoding="utf-8",
    on_bad_lines="error",
)
india = pd.read_csv(INDIA, encoding="utf-8")

uci["normalized"] = uci["text"].map(normalize)
india["normalized"] = india["text"].map(normalize)

uci_messages = set(uci["normalized"])
india_messages = set(india["text"].map(normalize))

overlap = uci_messages & india_messages

print("UCI rows:", len(uci))
print("Indian cleaned rows:", len(india))
print("Exact normalized messages in both:", len(overlap))

if overlap:
    print("\nSample overlapping messages:")
    for msg in list(overlap)[:10]:
        print("-", msg[:180])
