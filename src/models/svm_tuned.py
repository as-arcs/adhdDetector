# Tuned SVM with RBF kernel on PCA-reduced connectivity features.
# Uses GridSearchCV over C and gamma to find the best hyperparameters,
# balancing the tradeoff between hinge-loss minimisation and margin width.
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'svm_tuned')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX_LABELS = {0: 'Control', 1: 'ADHD-Combined', 3: 'ADHD-Inattentive'}

loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=0.80, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components (80% variance)")

# Hyperparameter grid
# C controls margin-width vs hinge-loss tradeoff; gamma controls RBF reach.
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running GridSearchCV (5-fold, macro-F1) — this may take a few minutes...")
grid_search = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', random_state=42),
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
    refit=True
)
grid_search.fit(X_train_pca, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV macro-F1: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_pca)

label_names = [DX_LABELS[k] for k in sorted(DX_LABELS.keys())]
accuracy = (y_pred == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
report = classification_report(y_test, y_pred, target_names=label_names)
print(report)

# Save classification report
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"PCA Components: {X_train_pca.shape[1]} (80% variance)\n")
    f.write(f"Kernel: RBF\n")
    f.write(f"Best Params: {grid_search.best_params_}\n")
    f.write(f"Best CV Macro-F1: {grid_search.best_score_:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

# Save full grid search results
with open(os.path.join(OUTPUT_DIR, 'grid_search_results.txt'), 'w') as f:
    f.write("Grid Search Results (sorted by mean_test_score descending)\n")
    f.write("=" * 70 + "\n\n")
    results = grid_search.cv_results_
    ranked = np.argsort(-results['mean_test_score'])
    for idx in ranked:
        f.write(f"Rank {results['rank_test_score'][idx]:2d}  |  "
                f"C={str(results['param_C'][idx]):>5s}  "
                f"gamma={str(results['param_gamma'][idx]):>6s}  |  "
                f"mean_F1={results['mean_test_score'][idx]:.4f}  "
                f"std={results['std_test_score'][idx]:.4f}\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(DX_LABELS.keys()))
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('SVM (RBF, Tuned) — Confusion Matrix')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")

import json

results = {
    "model" : "svm tuned",
    "accuracy" : float((y_pred == y_test).mean()),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    "f1_weighted" : float(f1_score(y_test, y_pred, average="weighted")),
    "classification_report" : classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix" : cm.tolist()
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model_jsons', 'svm_tuned')
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
