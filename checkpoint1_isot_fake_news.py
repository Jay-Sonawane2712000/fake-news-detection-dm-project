import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)

# ----------------------------
# 0) Config: put your csv paths here
# ----------------------------
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"
RANDOM_STATE = 42

# ----------------------------
# 1) Load data
# ----------------------------
if not (os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV)):
    raise FileNotFoundError(
        "Fake.csv / True.csv not found in this folder.\n"
        "Download ISOT dataset (Fake.csv, True.csv) and place them next to this script."
    )

fake = pd.read_csv(FAKE_CSV)
true = pd.read_csv(TRUE_CSV)

fake["label"] = 0  # fake
true["label"] = 1  # real

df = pd.concat([fake, true], ignore_index=True)

# Safety: ensure expected columns exist
expected_cols = {"title", "text"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSVs: {missing}. Found: {list(df.columns)}")

# ----------------------------
# 2) Basic cleaning
# ----------------------------
def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["title_clean"] = df["title"].apply(clean_text)
df["text_clean"] = df["text"].apply(clean_text)

# Combine title + body for better baseline
df["content"] = (df["title_clean"] + " " + df["text_clean"]).str.strip()

# Drop empty rows
df = df[df["content"].str.len() > 0].copy()

# Optional: remove duplicates (helps avoid inflated scores)
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

# ----------------------------
# 3) EDA + Visualizations (Checkpoint 1 requirement)
# ----------------------------
os.makedirs("outputs", exist_ok=True)

# (A) Class distribution
class_counts = df["label"].value_counts().sort_index()
plt.figure()
plt.bar(["Fake(0)", "Real(1)"], class_counts.values)
plt.title("Class Distribution")
plt.ylabel("Count")
plt.savefig("outputs/class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# (B) Text length distribution
df["text_len"] = df["content"].str.len()

plt.figure()
plt.hist(df[df["label"] == 0]["text_len"], bins=50, alpha=0.7, label="Fake")
plt.hist(df[df["label"] == 1]["text_len"], bins=50, alpha=0.7, label="Real")
plt.title("Content Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("outputs/text_length_hist.png", dpi=300, bbox_inches="tight")
plt.close()

# (C) Subject distribution if available
if "subject" in df.columns:
    top_subjects = df["subject"].value_counts().head(10)
    plt.figure()
    plt.barh(top_subjects.index[::-1], top_subjects.values[::-1])
    plt.title("Top 10 Subjects")
    plt.xlabel("Count")
    plt.savefig("outputs/top_subjects.png", dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------------
# 4) Train/Val/Test split
# ----------------------------
X = df["content"].values
y = df["label"].values

# Split: 70/15/15 (train / val / test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

# ----------------------------
# 5) Models: TF-IDF + (LogReg, Naive Bayes)
# ----------------------------
models = {
    "LogisticRegression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "MultinomialNB": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
}

results = []

def eval_model(name, model, X_tr, y_tr, X_te, y_te, split_name="VAL"):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    acc = accuracy_score(y_te, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary", zero_division=0)

    results.append({
        "model": name,
        "split": split_name,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    })

    cm = confusion_matrix(y_te, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot()
    plt.title(f"{name} - Confusion Matrix ({split_name})")
    plt.savefig(f"outputs/confusion_{name}_{split_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Evaluate on validation first
for name, model in models.items():
    eval_model(name, model, X_train, y_train, X_val, y_val, split_name="VAL")

# Pick best model by VAL F1 and evaluate on TEST
results_df = pd.DataFrame(results)
best_name = results_df.sort_values(by="f1", ascending=False).iloc[0]["model"]
best_model = models[best_name]

eval_model(best_name, best_model, X_train, y_train, X_test, y_test, split_name="TEST")

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/results_metrics.csv", index=False)

print("Done ✅")
print("Saved plots + metrics in the 'outputs/' folder")
print(results_df)
print(f"Best model (by VAL F1): {best_name}")