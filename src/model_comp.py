# compares the model based on f1 scores + accuracy
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model_jsons"
)

MODEL_DIRS = [
    "autoencoder",
    "logistic_base",
    "logistic_pca",
    "mlp",
    "mlp_base",
    "pca_var_sweep",
    "svm",
    "svm_base",
    "svm_tuned",
    "autoencoder_base",
    "autoencoder_tuned"
]

model_data = []

for model_dir in MODEL_DIRS:
    result_path = os.path.join(BASE_OUTPUT_DIR, model_dir, "results.json")

    if not os.path.exists(result_path):
        print(f"Skipping {model_dir}: no results.json found")
        continue

    with open(result_path, "r") as f:
        data = json.load(f)

    model_data.append({
        "model": data["model"],
        "accuracy": data["accuracy"],
        "f1_macro": data["f1_macro"],
        "f1_weighted": data["f1_weighted"],
    })

if len(model_data) == 0:
    print("No jsons exist in models_jsons")
    exit()

df = pd.DataFrame(model_data)
df = df.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)

print("\nModels ranked by macro f1:")
print(df)

csv_path = os.path.join(BASE_OUTPUT_DIR, "model_comparison.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved table to: {csv_path}")

x = np.arange(len(df))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, df["accuracy"], width, label="Accuracy")
plt.bar(x + width/2, df["f1_macro"], width, label="Macro F1")

plt.xticks(x, df["model"], rotation=20, ha="right")
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT_DIR, "model_bars.png"), dpi=150)
plt.close()

DX_LABELS = {
    0: "Control",
    1: "ADHD-Combined",
    3: "ADHD-Inattentive"
}

cms = []
titles = []

for model_dir in MODEL_DIRS:
    result_path = os.path.join(BASE_OUTPUT_DIR, model_dir, "results.json")

    if not os.path.exists(result_path):
        continue

    with open(result_path, "r") as f:
        data = json.load(f)

    if "confusion_matrix" not in data:
        continue

    cm = np.array(data["confusion_matrix"])

    # normalize rows
    cm = cm / cm.sum(axis=1, keepdims=True)

    model_name = data.get("model", model_dir)
    class_order = sorted(DX_LABELS.keys())
    display_labels = [DX_LABELS[k] for k in class_order]


    # save matrices
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels
    )
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_OUTPUT_DIR, f"{model_dir}_confusion_matrix.png"),
        dpi=150
    )
    plt.close()

    cms.append(cm)
    titles.append(model_name)

per_class_data = []

for model_dir in MODEL_DIRS:
    result_path = os.path.join(BASE_OUTPUT_DIR, model_dir, "results.json")

    if not os.path.exists(result_path):
        continue

    with open(result_path, "r") as f:
        data = json.load(f)

    report = data["classification_report"]

    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue

        per_class_data.append({
            "model": data.get("model", model_dir),
            "label": str(label),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": metrics["support"]
        })

if per_class_data:
    per_class_df = pd.DataFrame(per_class_data)

    best_per_label = per_class_df.loc[
        per_class_df.groupby("label")["f1_score"].idxmax()
    ].sort_values("label")

    print("\nBest model for each label:")
    print(best_per_label)

    per_class_df.to_csv(
        os.path.join(BASE_OUTPUT_DIR, "per_class_model_comparison.csv"),
        index=False
    )

    best_per_label.to_csv(
        os.path.join(BASE_OUTPUT_DIR, "best_model_per_label.csv"),
        index=False
    )