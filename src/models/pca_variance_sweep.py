# Sweep PCA variance thresholds (50%-99%) to find the best component count for logistic regression.
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'pca_sweep')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# try different PCA thresholds / number of components
thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
results = []

for var in thresholds:
    pca = PCA(n_components=var, random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)

    model = LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    results.append((var, X_tr.shape[1], acc, macro_f1))
    print(f"Variance: {var:.0%}  |  Components: {X_tr.shape[1]:>4}  |  Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")

# plot results
vars_, comps, accs, f1s = zip(*results)

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(comps, accs, 'o-', color='#4CAF50', label='Accuracy')
ax1.plot(comps, f1s, 's-', color='#2196F3', label='Macro F1')
ax1.set_xlabel('Number of PCA Components')
ax1.set_ylabel('Score')
ax1.set_title('Logistic Regression — PCA Threshold Sweep')
ax1.legend()

# label each point with its variance threshold
for v, c, a in zip(vars_, comps, accs):
    ax1.annotate(f'{v:.0%}', (c, a), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pca_sweep.png'), dpi=150)
plt.close()

print(f"\nPlot saved to {OUTPUT_DIR}/")