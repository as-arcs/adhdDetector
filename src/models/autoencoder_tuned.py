# Autoencoder with bottleneck size sweep — tests different compressed 
# feature dimensions (16, 32, 64, 128) to find the best representation size.
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'autoencoder_tuned')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS   = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}
LABEL_NAMES = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]

# load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def encode(model, X, bottleneck_idx=2):
    """Forward pass through encoder layers only."""
    a = X
    for i in range(bottleneck_idx):
        a = np.maximum(0, a @ model.coefs_[i] + model.intercepts_[i])
    return a

# sweep different bottleneck sizes
bottlenecks = [16, 32, 64, 128]
results = []

for bn in bottlenecks:
    print(f"\nTraining autoencoder with bottleneck={bn}...")
    ae = MLPRegressor(
        hidden_layer_sizes=(256, bn, 256),
        activation='relu',
        max_iter=500,
        random_state=42,
        verbose=False,
    )
    ae.fit(X_train, X_train)

    X_tr_enc = encode(ae, X_train)
    X_te_enc = encode(ae, X_test)

    clf = MLPClassifier(hidden_layer_sizes=(bn, bn//2), max_iter=1000, random_state=42)
    clf.fit(X_tr_enc, y_train)
    y_pred = clf.predict(X_te_enc)

    acc      = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    results.append((bn, acc, macro_f1))
    print(f"  Bottleneck={bn:>3}  |  Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")

# summary
print(f"\n{'='*50}")
print(f"{'Bottleneck':>12} {'Accuracy':>10} {'Macro F1':>10}")
print(f"{'-'*50}")
for bn, acc, f1 in results:
    print(f"{bn:>12} {acc:>10.4f} {f1:>10.4f}")

# save summary
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
    f.write(f"{'Bottleneck':>12} {'Accuracy':>10} {'Macro F1':>10}\n")
    f.write(f"{'-'*50}\n")
    for bn, acc, f1 in results:
        f.write(f"{bn:>12} {acc:>10.4f} {f1:>10.4f}\n")

# plot
bns, accs, f1s = zip(*results)
x = np.arange(len(bns))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',  color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x + width/2, f1s,  width, label='Macro F1',  color='#2196F3', alpha=0.85)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f'Bottleneck={bn}' for bn in bns])
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Autoencoder — Bottleneck Size Sweep')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'bottleneck_sweep.png'), dpi=150)
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")
