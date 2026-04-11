import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'autoencoder')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}

# load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

# standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── AUTOENCODER ───────────────────────────────────────────────────────────────
# trains to reconstruct input; bottleneck (64) = compressed features
print("Training autoencoder...")
autoencoder = MLPRegressor(
    hidden_layer_sizes=(256, 64, 256),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=True,
)
autoencoder.fit(X_train, X_train)

# ── EXTRACT ENCODED FEATURES ──────────────────────────────────────────────────
def encode(model, X):
    """Forward pass through encoder layers only (input → 256 → 64)."""
    a = X
    for i in range(2):
        a = np.maximum(0, a @ model.coefs_[i] + model.intercepts_[i])
    return a

X_train_enc = encode(autoencoder, X_train)
X_test_enc  = encode(autoencoder, X_test)
print(f"Encoded shape: {X_train_enc.shape}")

# ── CLASSIFY ──────────────────────────────────────────────────────────────────
print("Training classifier on encoded features...")
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=1000,
    random_state=42,
)
clf.fit(X_train_enc, y_train)
y_pred = clf.predict(X_test_enc)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
label_names = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]
print(f"\nTest Accuracy: {(y_pred == y_test).mean():.4f}")
print("\nClassification Report:\n")
report = classification_report(y_test, y_pred, target_names=label_names)
print(report)

with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"Test Accuracy: {(y_pred == y_test).mean():.4f}\n\n")
    f.write(report)

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred, labels=sorted(DX_LABELS.keys()))
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Autoencoder – Confusion Matrix')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# ── RECONSTRUCTION ERROR PLOT ─────────────────────────────────────────────────
X_recon     = autoencoder.predict(X_test)
recon_error = np.mean((X_test - X_recon) ** 2, axis=1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(recon_error, bins=30, color='steelblue', edgecolor='white')
ax.set_title('Autoencoder Reconstruction Error (Test Set)')
ax.set_xlabel('MSE')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'reconstruction_error.png'), dpi=150)
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")
