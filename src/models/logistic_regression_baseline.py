# Logistic regression baseline on raw connectivity features (no dimensionality reduction).
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'logistic')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}

# load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

# standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train logistic regression (one-vs-rest for multi-class)
print("Training logistic regression...")
model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()


label_names = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]
print(f"\nTest Accuracy: {acc:.4f}")
print(f"\nClassification Report:\n")
report = classification_report(y_test, y_pred, target_names=label_names)
print(report)

# save report to text file
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"Test Accuracy: {(y_pred == y_test).mean():.4f}\n\n")
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(DX_LABELS.keys()))
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Logistic Regression — Confusion Matrix')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")

import json

results = {
    "model" : "logistic regression baseline",
    "accuracy" : float(acc),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    "f1_weighted" : float(f1_score(y_test, y_pred, average="weighted")),
    "classification_report" : classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix" : cm.tolist()
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model_jsons', 'logistic_base')
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)