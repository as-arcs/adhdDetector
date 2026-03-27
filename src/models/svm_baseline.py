# SVM baseline: linear kernel on raw (scaled) connectivity features, no PCA.
# Establishes a performance floor for the SVM family by testing whether
# brain connectivity features are linearly separable via maximum-margin classification.
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'svm_baseline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}

loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training SVM (linear kernel, no PCA)...")
model = SVC(
    kernel='linear',
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

label_names = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]
accuracy = (y_pred == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
report = classification_report(y_test, y_pred, target_names=label_names)
print(report)

with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"Kernel: linear (no PCA)\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

cm = confusion_matrix(y_test, y_pred, labels=sorted(DX_LABELS.keys()))
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('SVM (Linear) Baseline — Confusion Matrix')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")
