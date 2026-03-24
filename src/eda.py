import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'eda')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}

# Load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()
X_all = np.concatenate([X_train, X_test]) if len(X_test) > 0 else X_train
y_all = np.concatenate([y_train, y_test]) if len(y_test) > 0 else y_train

print(f"Subjects: {X_all.shape[0]}  |  Features: {X_all.shape[1]}  |  ROIs: {loader.num_rois}")

# class distribution
unique, counts = np.unique(y_all, return_counts=True)
print("\nClass distribution:")
for k, v in zip(unique, counts):
    print(f"  {DX_LABELS.get(int(k), str(k))}: {v}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([DX_LABELS.get(int(k), str(k)) for k in unique], counts, color=['#4CAF50', '#FF5722', '#2196F3'])
ax.set_title('Class Distribution')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150)
plt.close()

# Average connectivity heatmaps (Control vs ADHD)
# Each axis represents one of the 190 brain regions (ROIs). Each pixel (i, j)
# shows the average Pearson correlation between region i and region j's fMRI activity.
# Red = positive correlation (regions activate together),
# Blue = negative correlation (one activates while the other deactivates).
# Control: average connectivity for neurotypical subjects.
# ADHD: average connectivity for ADHD subjects (Combined + Inattentive).
# Difference: where ADHD brains differ — red = stronger, blue = weaker connectivity.

n = loader.num_rois

def to_matrix(vec):
    mat = np.zeros((n, n))
    mat[np.triu_indices(n, k=1)] = vec
    return mat + mat.T + np.eye(n)

avg_ctrl = X_all[y_all == 0].mean(axis=0)
avg_adhd = X_all[y_all != 0].mean(axis=0)
diff = avg_adhd - avg_ctrl
diff_abs = np.max(np.abs(diff))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, data, title, vlim in zip(axes,
                                  [avg_ctrl, avg_adhd, diff],
                                  ['Control', 'ADHD', 'Difference (ADHD - Control)'],
                                  [1.0, 1.0, diff_abs]):
    im = ax.imshow(to_matrix(data), cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'connectivity_heatmaps.png'), dpi=150)
plt.close()

# basic feature stats
print(f"\nFeature stats: mean={X_all.mean():.4f}, std={X_all.std():.4f}, "
      f"min={X_all.min():.4f}, max={X_all.max():.4f}")

print(f"\nPlots saved to {OUTPUT_DIR}/")