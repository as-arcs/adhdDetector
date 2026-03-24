# SVM with RBF kernel on PCA-reduced connectivity features.
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'svm')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}

# Load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.80, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components (80% variance)")

# train SVM with RBF kernel
print("Training SVM (RBF kernel)...")
model = SVC(
    kernel='rbf',
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_pca, y_train)

# evaluate the model
y_pred = model.predict(X_test_pca)

label_names = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]
print(f"\nTest Accuracy: {(y_pred == y_test).mean():.4f}")
print(f"\nClassification Report:\n")
report = classification_report(y_test, y_pred, target_names=label_names)
print(report)

# save classification report
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"PCA Components: {X_train_pca.shape[1]} (80% variance)\n")
    f.write(f"Kernel: RBF\n")
    f.write(f"Test Accuracy: {(y_pred == y_test).mean():.4f}\n\n")
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(DX_LABELS.keys()))
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('SVM (RBF) — Confusion Matrix')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")