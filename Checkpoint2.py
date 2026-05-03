import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 0) CONFIG
# ============================================================
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"
RANDOM_STATE = 42
OUTPUT_DIR = "outputs_checkpoint2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1) LOAD DATA
# ============================================================
if not (os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV)):
    raise FileNotFoundError(
        "Fake.csv / True.csv not found in this folder.\n"
        "Place Fake.csv and True.csv next to this script."
    )

fake = pd.read_csv(FAKE_CSV)
true = pd.read_csv(TRUE_CSV)

fake["label"] = 0   # Fake
true["label"] = 1   # Real

df = pd.concat([fake, true], ignore_index=True)

expected_cols = {"title", "text"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSVs: {missing}. Found columns: {list(df.columns)}")

# If subject doesn't exist for some reason, create placeholder
if "subject" not in df.columns:
    df["subject"] = "Unknown"

# ============================================================
# 2) BASIC CLEANING (same baseline idea as Checkpoint 1)
# ============================================================
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["title_clean"] = df["title"].apply(clean_text)
df["text_clean"] = df["text"].apply(clean_text)

# Feature variants for Checkpoint 2
df["title_only"] = df["title_clean"]
df["text_only"] = df["text_clean"]
df["title_plus_text"] = (df["title_clean"] + " " + df["text_clean"]).str.strip()

# Keep same baseline content logic too
df["content"] = df["title_plus_text"]

# Drop empty rows
df = df[df["content"].str.len() > 0].copy()

# Deduplicate on the combined content field (same idea as CP1)
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

# Keep an ID so rows can be joined later for error analysis
df["row_id"] = np.arange(len(df))

print(f"Dataset size after cleaning/deduplication: {len(df)}")

# ============================================================
# 3) OPTIONAL EDA OUTPUTS (Checkpoint 2 can reuse these)
# ============================================================
# Class distribution
class_counts = df["label"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(["Fake(0)", "Real(1)"], class_counts.values)
plt.title("Class Distribution")
plt.ylabel("Count")
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

# Content length distribution
df["text_len"] = df["content"].str.len()
plt.figure(figsize=(10, 6))
plt.hist(df[df["label"] == 0]["text_len"], bins=50, alpha=0.7, label="Fake")
plt.hist(df[df["label"] == 1]["text_len"], bins=50, alpha=0.7, label="Real")
plt.title("Content Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "text_length_hist.png"), dpi=300, bbox_inches="tight")
plt.close()

# Top subjects
top_subjects = df["subject"].value_counts().head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_subjects.index[::-1], top_subjects.values[::-1])
plt.title("Top 10 Subjects")
plt.xlabel("Count")
plt.savefig(os.path.join(OUTPUT_DIR, "top_subjects.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 4) TRAIN / VAL / TEST SPLIT
#    Same 70 / 15 / 15 stratified logic as Checkpoint 1
# ============================================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train size: {len(train_df)}")
print(f"Val size:   {len(val_df)}")
print(f"Test size:  {len(test_df)}")

# ============================================================
# 5) HELPERS
# ============================================================
def get_vectorizer(ngram_range=(1, 2), min_df=2, max_df=0.90):
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, p, r, f1

def plot_confusion(y_true, y_pred, title, filepath):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot()
    plt.title(title)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def make_pipeline(model, ngram_range):
    return Pipeline([
        ("tfidf", get_vectorizer(ngram_range=ngram_range)),
        ("clf", model)
    ])

def evaluate_pipeline(
    model_name,
    model,
    feature_col,
    ngram_range,
    train_df,
    eval_df,
    split_name="VAL",
    tuned="No",
    notes=""
):
    X_train = train_df[feature_col].values
    y_train = train_df["label"].values

    X_eval = eval_df[feature_col].values
    y_eval = eval_df["label"].values

    pipe = make_pipeline(model, ngram_range)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_eval)

    acc, p, r, f1 = compute_metrics(y_eval, preds)

    result = {
        "model": model_name,
        "feature_source": feature_col,
        "ngram_range": str(ngram_range),
        "tuned": tuned,
        "split": split_name,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "notes": notes
    }

    return result, pipe, preds

# ============================================================
# 6) BASELINE + EXPANDED MODEL SUITE
# ============================================================
# Keeping CP1 baselines + adding more classical ML models
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "LinearSVC": LinearSVC(random_state=RANDOM_STATE),
    "SGDClassifier": SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

feature_sources = ["title_only", "text_only", "title_plus_text"]
ngram_options = [(1, 1), (1, 2)]

all_results = []
all_pipelines = {}

print("\nRunning base model comparisons on validation set...")

for feature_col in feature_sources:
    for ngram_range in ngram_options:
        for model_name, model in models.items():
            print(f"VAL -> Model={model_name}, Feature={feature_col}, Ngram={ngram_range}")
            result, fitted_pipe, preds = evaluate_pipeline(
                model_name=model_name,
                model=clone(model),
                feature_col=feature_col,
                ngram_range=ngram_range,
                train_df=train_df,
                eval_df=val_df,
                split_name="VAL",
                tuned="No"
            )
            all_results.append(result)

            key = (model_name, feature_col, str(ngram_range), "No")
            all_pipelines[key] = fitted_pipe

# Save raw validation comparison
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "all_model_results_validation_raw.csv"), index=False)

# ============================================================
# 7) PICK STRONGEST UNTUNED MODELS
# ============================================================
val_only = results_df[results_df["split"] == "VAL"].copy()
val_sorted = val_only.sort_values(by="f1", ascending=False).reset_index(drop=True)

print("\nTop validation results before tuning:")
print(val_sorted.head(10)[["model", "feature_source", "ngram_range", "f1", "accuracy"]])

# We will tune only Logistic Regression and LinearSVC
# using the best feature + ngram setup found for each.
models_to_tune = ["LogisticRegression", "LinearSVC"]

best_untuned_configs = {}
for model_name in models_to_tune:
    subset = val_sorted[val_sorted["model"] == model_name]
    if len(subset) > 0:
        best_untuned_configs[model_name] = subset.iloc[0].to_dict()

# ============================================================
# 8) TUNING STRONG MODELS
# ============================================================
tuned_results = []
best_tuned_pipelines = {}

print("\nTuning strongest models...")

param_grids = {
    "LogisticRegression": {
        "C": [0.1, 1, 5, 10]
    },
    "LinearSVC": {
        "C": [0.1, 1, 5, 10]
    }
}

for model_name in models_to_tune:
    if model_name not in best_untuned_configs:
        continue

    best_cfg = best_untuned_configs[model_name]
    feature_col = best_cfg["feature_source"]
    ngram_range = eval(best_cfg["ngram_range"])  # converts string "(1, 2)" to tuple

    print(f"\nTuning {model_name} using Feature={feature_col}, Ngram={ngram_range}")

    if model_name == "LogisticRegression":
        base_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif model_name == "LinearSVC":
        base_model = LinearSVC(random_state=RANDOM_STATE)
    else:
        continue

    best_local_f1 = -1
    best_local_pipe = None
    best_local_params = None
    best_local_preds = None

    for params in ParameterGrid(param_grids[model_name]):
        tuned_model = clone(base_model).set_params(**params)

        result, fitted_pipe, preds = evaluate_pipeline(
            model_name=model_name,
            model=tuned_model,
            feature_col=feature_col,
            ngram_range=ngram_range,
            train_df=train_df,
            eval_df=val_df,
            split_name="VAL",
            tuned="Yes",
            notes=str(params)
        )

        tuned_results.append(result)

        if result["f1"] > best_local_f1:
            best_local_f1 = result["f1"]
            best_local_pipe = fitted_pipe
            best_local_params = params
            best_local_preds = preds

    best_tuned_pipelines[model_name] = {
        "pipeline": best_local_pipe,
        "feature_source": feature_col,
        "ngram_range": ngram_range,
        "best_params": best_local_params,
        "best_val_f1": best_local_f1
    }

# Save tuning results
tuned_df = pd.DataFrame(tuned_results)
if len(tuned_df) > 0:
    tuned_df.to_csv(os.path.join(OUTPUT_DIR, "tuned_model_results_validation.csv"), index=False)

# Combine all validation results
combined_results_df = pd.concat(
    [results_df, tuned_df],
    ignore_index=True
) if len(tuned_df) > 0 else results_df.copy()

combined_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "all_model_results_validation_with_tuning.csv"),
    index=False
)

# ============================================================
# 9) CHOOSE FINAL BEST MODEL BY VALIDATION F1
# ============================================================
final_val_df = combined_results_df[combined_results_df["split"] == "VAL"].copy()
final_val_df = final_val_df.sort_values(by="f1", ascending=False).reset_index(drop=True)

best_final_config = final_val_df.iloc[0].to_dict()

print("\nBest final validation config:")
print(best_final_config)

best_model_name = best_final_config["model"]
best_feature_source = best_final_config["feature_source"]
best_ngram_range = eval(best_final_config["ngram_range"])
best_tuned_flag = best_final_config["tuned"]
best_notes = best_final_config["notes"]

# Rebuild the best final model from scratch using its best config
if best_model_name == "LogisticRegression":
    best_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    if best_tuned_flag == "Yes" and isinstance(best_notes, str) and best_notes.strip():
        best_model.set_params(**eval(best_notes))
elif best_model_name == "LinearSVC":
    best_model = LinearSVC(random_state=RANDOM_STATE)
    if best_tuned_flag == "Yes" and isinstance(best_notes, str) and best_notes.strip():
        best_model.set_params(**eval(best_notes))
elif best_model_name == "MultinomialNB":
    best_model = MultinomialNB()
elif best_model_name == "ComplementNB":
    best_model = ComplementNB()
elif best_model_name == "SGDClassifier":
    best_model = SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, random_state=RANDOM_STATE)
elif best_model_name == "RandomForest":
    best_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
else:
    raise ValueError(f"Unexpected best model: {best_model_name}")

# ============================================================
# 10) FINAL TEST EVALUATION
# ============================================================
print("\nEvaluating best final model on TEST set...")

test_result, best_final_pipe, test_preds = evaluate_pipeline(
    model_name=best_model_name,
    model=best_model,
    feature_col=best_feature_source,
    ngram_range=best_ngram_range,
    train_df=train_df,
    eval_df=test_df,
    split_name="TEST",
    tuned=best_tuned_flag,
    notes=best_notes
)

best_model_test_df = pd.DataFrame([test_result])
best_model_test_df.to_csv(os.path.join(OUTPUT_DIR, "best_model_test_metrics.csv"), index=False)

print("\nBest final TEST result:")
print(best_model_test_df)

# Save confusion matrix for final best model
plot_confusion(
    y_true=test_df["label"].values,
    y_pred=test_preds,
    title=f"{best_model_name} - Confusion Matrix (TEST)",
    filepath=os.path.join(OUTPUT_DIR, "confusion_best_model_test.png")
)

# ============================================================
# 11) ROC-AUC AND PRECISION-RECALL CURVES FOR FINAL BEST MODEL
# ============================================================
y_test = test_df["label"].values
X_test = test_df[best_feature_source].values

test_scores = None
if hasattr(best_final_pipe, "decision_function"):
    test_scores = best_final_pipe.decision_function(X_test)
elif hasattr(best_final_pipe, "predict_proba"):
    test_scores = best_final_pipe.predict_proba(X_test)[:, 1]

if test_scores is not None:
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{best_model_name} - ROC Curve (TEST)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_best_model_test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_scores)
    average_precision = average_precision_score(y_test, test_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(recall_curve, precision_curve, label=f"AP = {average_precision:.4f}")
    plt.title(f"{best_model_name} - Precision-Recall Curve (TEST)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve_best_model_test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    curve_metrics_df = pd.DataFrame([{
        "model": best_model_name,
        "roc_auc": roc_auc,
        "average_precision": average_precision
    }])
    curve_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "curve_metrics_best_model_test.csv"), index=False)

# ============================================================
# 12) TF-IDF FEATURE WEIGHTS FOR FINAL BEST MODEL
# ============================================================
tfidf = best_final_pipe.named_steps["tfidf"]
clf = best_final_pipe.named_steps["clf"]

if hasattr(clf, "coef_"):
    feature_names = tfidf.get_feature_names_out()
    coefficients = clf.coef_[0]

    feature_weights_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    top_real = feature_weights_df.sort_values(by="coefficient", ascending=False).head(20).copy()
    top_real["class_supported"] = "Real"

    top_fake = feature_weights_df.sort_values(by="coefficient", ascending=True).head(20).copy()
    top_fake["class_supported"] = "Fake"

    top_feature_weights_df = pd.concat([top_real, top_fake], ignore_index=True)
    top_feature_weights_df = top_feature_weights_df[["feature", "coefficient", "class_supported"]]
    top_feature_weights_df.to_csv(os.path.join(OUTPUT_DIR, "top_tfidf_feature_weights.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(top_real["feature"][::-1], top_real["coefficient"][::-1])
    plt.title("Top TF-IDF Features Supporting Real")
    plt.xlabel("Coefficient")
    plt.savefig(os.path.join(OUTPUT_DIR, "top_tfidf_features_real.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(top_fake["feature"][::-1], top_fake["coefficient"][::-1])
    plt.title("Top TF-IDF Features Supporting Fake")
    plt.xlabel("Coefficient")
    plt.savefig(os.path.join(OUTPUT_DIR, "top_tfidf_features_fake.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 13) FEATURE COMPARISON SUMMARY
# ============================================================
feature_summary = (
    final_val_df.groupby("feature_source")[["accuracy", "precision", "recall", "f1"]]
    .max()
    .reset_index()
    .sort_values(by="f1", ascending=False)
)
feature_summary.to_csv(os.path.join(OUTPUT_DIR, "feature_comparison_summary.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.bar(feature_summary["feature_source"], feature_summary["f1"])
plt.title("Best Validation F1 by Feature Source")
plt.ylabel("F1 Score")
plt.xticks(rotation=15)
plt.savefig(os.path.join(OUTPUT_DIR, "feature_comparison_f1.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 14) MISCLASSIFIED EXAMPLES EXPORT
# ============================================================
test_analysis_df = test_df.copy()
test_analysis_df["predicted_label"] = test_preds

def label_name(x):
    return "Fake" if x == 0 else "Real"

test_analysis_df["true_label_name"] = test_analysis_df["label"].apply(label_name)
test_analysis_df["predicted_label_name"] = test_analysis_df["predicted_label"].apply(label_name)

def get_error_type(row):
    if row["label"] == 0 and row["predicted_label"] == 1:
        return "False Negative (Fake predicted as Real)"
    elif row["label"] == 1 and row["predicted_label"] == 0:
        return "False Positive (Real predicted as Fake)"
    else:
        return "Correct"

test_analysis_df["error_type"] = test_analysis_df.apply(get_error_type, axis=1)

misclassified_df = test_analysis_df[test_analysis_df["label"] != test_analysis_df["predicted_label"]].copy()

misclassified_export_cols = [
    "row_id", "title", "text", "subject",
    "label", "predicted_label",
    "true_label_name", "predicted_label_name",
    "error_type"
]

misclassified_df[misclassified_export_cols].to_csv(
    os.path.join(OUTPUT_DIR, "misclassified_examples.csv"),
    index=False
)

# Save top 25 shortest/longest errors as optional quick review
misclassified_df["content_len"] = misclassified_df["content"].str.len()
misclassified_df.sort_values(by="content_len", ascending=True).head(25)[misclassified_export_cols + ["content_len"]].to_csv(
    os.path.join(OUTPUT_DIR, "misclassified_short_examples.csv"),
    index=False
)
misclassified_df.sort_values(by="content_len", ascending=False).head(25)[misclassified_export_cols + ["content_len"]].to_csv(
    os.path.join(OUTPUT_DIR, "misclassified_long_examples.csv"),
    index=False
)

# ============================================================
# 15) SUBJECT-WISE ROBUSTNESS / BIAS ANALYSIS

# ============================================================
subject_metrics = []

for subject_name, group in test_analysis_df.groupby("subject"):
    y_true = group["label"].values
    y_pred = group["predicted_label"].values

    # Skip extremely tiny groups to avoid unstable metrics
    if len(group) < 20:
        continue

    # Always compute accuracy
    acc = accuracy_score(y_true, y_pred)

    # Check whether both classes are present in this subject group
    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) == 2

    if has_both_classes:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average="binary",
            zero_division=0
        )
    else:
        # For one-class subject groups, binary precision/recall/f1 is not meaningful
        p, r, f1 = np.nan, np.nan, np.nan

    subject_metrics.append({
        "subject": subject_name,
        "count": len(group),
        "num_classes_in_group": len(unique_classes),
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    })

subject_metrics_df = pd.DataFrame(subject_metrics)

# Save full subject metrics table
subject_metrics_df = subject_metrics_df.sort_values(by="accuracy", ascending=False)
subject_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "subject_wise_metrics.csv"), index=False)

# Plot subject-wise accuracy only (safe for all groups)
if len(subject_metrics_df) > 0:
    plot_df = subject_metrics_df.sort_values(by="count", ascending=False).head(10).copy()

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["subject"], plot_df["accuracy"])
    plt.title("Subject-wise Accuracy (Top 10 Subjects by Test Count)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(os.path.join(OUTPUT_DIR, "subject_wise_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Optional: plot F1 only for subjects that contain both classes
    plot_f1_df = plot_df.dropna(subset=["f1"]).copy()

    if len(plot_f1_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(plot_f1_df["subject"], plot_f1_df["f1"])
        plt.title("Subject-wise F1 Score (Subjects with Both Classes Present)")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45, ha="right")
        plt.savefig(os.path.join(OUTPUT_DIR, "subject_wise_f1.png"), dpi=300, bbox_inches="tight")
        plt.close()

# ============================================================
# 16) SAVE FINAL OVERALL RESULTS TABLE
# ============================================================
# Add final test result into one master file for convenience
master_results_df = pd.concat(
    [combined_results_df, best_model_test_df],
    ignore_index=True
)

master_results_df.to_csv(os.path.join(OUTPUT_DIR, "all_results_master.csv"), index=False)

# ============================================================
# 17) PRINT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("CHECKPOINT 2 RUN COMPLETE")
print("=" * 60)

print(f"Best validation model: {best_model_name}")
print(f"Best feature source:   {best_feature_source}")
print(f"Best ngram range:      {best_ngram_range}")
print(f"Tuned?:                {best_tuned_flag}")
print(f"Best test F1:          {test_result['f1']:.4f}")
print(f"Best test Accuracy:    {test_result['accuracy']:.4f}")

print("\nSaved files in:", OUTPUT_DIR)
print("""
Main output files:
- all_model_results_validation_raw.csv
- tuned_model_results_validation.csv
- all_model_results_validation_with_tuning.csv
- best_model_test_metrics.csv
- confusion_best_model_test.png
- roc_curve_best_model_test.png
- pr_curve_best_model_test.png
- curve_metrics_best_model_test.csv
- top_tfidf_feature_weights.csv
- top_tfidf_features_real.png
- top_tfidf_features_fake.png
- misclassified_examples.csv
- subject_wise_metrics.csv
- subject_wise_f1.png
- feature_comparison_summary.csv
- feature_comparison_f1.png
- all_results_master.csv
""")
