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
    "pca_var_sweep",
    "svm",
    "svm_base",
    "svm_tuned"
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

    # normalize rows (important)
    cm = cm / cm.sum(axis=1, keepdims=True)

    cms.append(cm)
    titles.append(data.get("model", model_dir))

# plot side-by-side
if cms:
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4))

    # handle case of only 1 model
    if len(cms) == 1:
        axes = [axes]

    for ax, cm, title in zip(axes, cms, titles):
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "confusion_matrix_comparison.png"), dpi=150)
    plt.close()