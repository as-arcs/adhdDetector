# Unified benchmark: compares autoencoder against all baselines in one run.

# Drop this in scripts/ alongside your other model files.
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'benchmark')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS   = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}
LABEL_NAMES = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]

# ── DATA ──────────────────────────────────────────────────────────────────────
print("Loading data...")
loader = ADHDDataLoader()
(X_train_raw, y_train, _), (X_test_raw, y_test, _) = loader.load_data()

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# shared PCA (80 % variance) used by SVM and logistic-pca baselines
pca        = PCA(n_components=0.80, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
print(f"PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components (80% var)")

results = {}   # name -> {accuracy, macro_f1, y_pred}

# ── HELPER ────────────────────────────────────────────────────────────────────
def evaluate(name, y_pred):
    acc      = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    results[name] = {'accuracy': acc, 'macro_f1': macro_f1, 'y_pred': y_pred}
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Accuracy: {acc:.4f}   Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

# ── 1. LOGISTIC REGRESSION (raw) ──────────────────────────────────────────────
print("\n[1/5] Logistic Regression (raw)...")
lr_raw = LogisticRegression(max_iter=5000, class_weight='balanced',
                             solver='lbfgs', random_state=42)
lr_raw.fit(X_train, y_train)
evaluate("Logistic Regression (raw)", lr_raw.predict(X_test))

# ── 2. LOGISTIC REGRESSION (PCA 80%) ─────────────────────────────────────────
print("\n[2/5] Logistic Regression (PCA 80%)...")
lr_pca = LogisticRegression(max_iter=5000, class_weight='balanced',
                              solver='lbfgs', random_state=42)
lr_pca.fit(X_train_pca, y_train)
evaluate("Logistic Regression (PCA 80%)", lr_pca.predict(X_test_pca))

# ── 3. SVM RBF (PCA 80%) ─────────────────────────────────────────────────────
print("\n[3/5] SVM RBF (PCA 80%)...")
svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm.fit(X_train_pca, y_train)
evaluate("SVM RBF (PCA 80%)", svm.predict(X_test_pca))

# ── 4. AUTOENCODER ENCODE FUNCTION ───────────────────────────────────────────
def encode(model, X):
    """Forward pass through encoder layers only (input → 256 → 64)."""
    a = X
    for i in range(2):  # first 2 layers = encoder half
        a = np.maximum(0, a @ model.coefs_[i] + model.intercepts_[i])
    return a

# ── 5. AUTOENCODER + MLP CLASSIFIER ──────────────────────────────────────────
print("\n[4/5] Training autoencoder (500 iter)...")
autoencoder = MLPRegressor(
    hidden_layer_sizes=(256, 64, 256),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=False,
)
autoencoder.fit(X_train, X_train)

X_train_enc = encode(autoencoder, X_train)
X_test_enc  = encode(autoencoder, X_test)
print(f"Encoded shape: {X_train_enc.shape[1]} dims")

print("[5/5] MLP classifier on encoded features...")
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
clf.fit(X_train_enc, y_train)
evaluate("Autoencoder + MLP", clf.predict(X_test_enc))

# bonus: logistic regression on encoded features (isolates encoder quality)
lr_enc = LogisticRegression(max_iter=5000, class_weight='balanced',
                              solver='lbfgs', random_state=42)
lr_enc.fit(X_train_enc, y_train)
evaluate("Autoencoder + LogReg", lr_enc.predict(X_test_enc))

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'Model':<35} {'Accuracy':>9} {'Macro F1':>9}")
print(f"{'-'*55}")
for name, r in results.items():
    print(f"{name:<35} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f}")

with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
    f.write(f"{'Model':<35} {'Accuracy':>9} {'Macro F1':>9}\n")
    f.write(f"{'-'*55}\n")
    for name, r in results.items():
        f.write(f"{name:<35} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f}\n")

# ── BAR CHART ─────────────────────────────────────────────────────────────────
names   = list(results.keys())
accs    = [results[n]['accuracy']  for n in names]
f1s     = [results[n]['macro_f1']  for n in names]
x       = np.arange(len(names))
width   = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',  color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x + width/2, f1s,  width, label='Macro F1',  color='#2196F3', alpha=0.85)

# value labels on bars
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Accuracy & Macro F1')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
plt.close()

# ── CONFUSION MATRICES (side by side) ────────────────────────────────────────
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['y_pred'], labels=sorted(DX_LABELS.keys()))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'all_confusion_matrices.png'), dpi=150)
plt.close()

print(f"\nAll results saved to {OUTPUT_DIR}/")
