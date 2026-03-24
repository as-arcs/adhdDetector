# Logistic regression with PCA dimensionality reduction on connectivity features.
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'logistic_pca')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}

# load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

# standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA: keep components to explain 80% of variance
pca = PCA(n_components=0.80, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components (95% variance)")

# plot explained variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, len(cumvar) + 1), cumvar)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA — Explained Variance')
ax.axhline(y=0.95, color='r', linestyle='--', label='95%')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pca_variance.png'), dpi=150)
plt.close()

# train logistic regression on PCA features
print("Training logistic regression (with PCA)...")
model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)
model.fit(X_train_pca, y_train)

# evaluate
y_pred = model.predict(X_test_pca)

label_names = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]
print(f"\nTest Accuracy: {(y_pred == y_test).mean():.4f}")
print(f"\nClassification Report:\n")
report = classification_report(y_test, y_pred, target_names=label_names)
print(report)

# save classification report
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"PCA Components: {X_train_pca.shape[1]} (95% variance)\n")
    f.write(f"Test Accuracy: {(y_pred == y_test).mean():.4f}\n\n")
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(DX_LABELS.keys()))
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Logistic Regression (PCA) — Confusion Matrix')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")