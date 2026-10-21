
from pathlib import Path
import pandas as pd

SOURCE = Path("data/spam_ham_india.csv")
OUTPUT_DIR = Path("data/cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"CSV not found: {SOURCE.resolve()}\n"
        "Copy spam_ham_india.csv into the project's data folder."
    )

# Inspect the file without changing the source.
df = pd.read_csv(SOURCE, encoding="utf-8-sig")

print("Original shape:", df.shape)
print("Column names:", list(df.columns))
print("\nFirst five rows:")
print(df.head().to_string())
print("\nMissing values by column:")
print(df.isna().sum().to_string())

# Expected columns in this dataset: Msg and Label.
# Stop rather than guessing if the schema differs.
required = {"Msg", "Label"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required}; found {set(df.columns)}. "
        "Inspect the column names before continuing."
    )

df = df[["Msg", "Label"]].copy()
df["Msg"] = df["Msg"].astype("string").str.strip()
df["Label"] = df["Label"].astype("string").str.strip().str.lower()

# Report rows with missing or empty messages/labels.
missing_mask = (
    df["Msg"].isna()
    | df["Label"].isna()
    | df["Msg"].eq("")
    | df["Label"].eq("")
)
print("\nRows missing message or label:", int(missing_mask.sum()))

df = df.loc[~missing_mask].copy()

# Check labels before mapping anything.
print("\nLabel values before normalization:")
print(df["Label"].value_counts(dropna=False).to_string())

allowed_labels = {"spam", "ham"}
unexpected = set(df["Label"].unique()) - allowed_labels
if unexpected:
    raise ValueError(
        f"Unexpected labels: {unexpected}. "
        "Do not silently map these labels."
    )

# Detect exact duplicates and conflicting labels.
df["normalized_msg"] = (
    df["Msg"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
)

label_counts_per_message = df.groupby("normalized_msg")["Label"].nunique()
conflicting_messages = label_counts_per_message[
    label_counts_per_message > 1
].index

print("\nUnique normalized messages:", df["normalized_msg"].nunique())
print("Duplicate rows beyond first occurrence:",
      int(df.duplicated("normalized_msg", keep="first").sum()))
print("Messages with conflicting labels:", len(conflicting_messages))

# Save conflicting examples separately for manual review.
conflicts = df[df["normalized_msg"].isin(conflicting_messages)]
conflicts.to_csv(
    OUTPUT_DIR / "conflicting_labels_review.csv", index=False
)

# Exclude conflicting messages from the initial clean training candidate.
df = df[~df["normalized_msg"].isin(conflicting_messages)].copy()

# Keep one copy of each normalized message.
df = df.drop_duplicates(subset="normalized_msg", keep="first")

# Keep both the original message and its label.
clean = df[["Msg", "Label"]].rename(
    columns={"Msg": "text", "Label": "label"}
)

clean.to_csv(OUTPUT_DIR / "india_sms_clean.csv", index=False)

print("\nCleaned dataset shape:", clean.shape)
print("\nCleaned label distribution:")
print(clean["label"].value_counts().to_string())
print("\nSaved cleaned candidate to:",
      (OUTPUT_DIR / "india_sms_clean.csv").resolve())
print("Saved conflicting labels to:",
      (OUTPUT_DIR / "conflicting_labels_review.csv").resolve())
print("\nNote: duplicates were checked within this CSV only.")
