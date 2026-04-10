# Multi-layer perceptron (neural network) classifier for connectivity features.
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_loader import ADHDDataLoader

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'outputs',
    'mlp'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load data
loader = ADHDDataLoader()
(X_train, y_train, _), (X_test, y_test, _) = loader.load_data()

# Remap labels
unique_labels = np.unique(y_train)
label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

y_train = np.array([label_map[y] for y in y_train], dtype=np.int64)
y_test = np.array([label_map[y] for y in y_test], dtype=np.int64)

# convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Datasets / loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# create model
input_dim = X_train.shape[1]
output_dim = len(unique_labels)

model = MLP(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dims=[64, 32],
    dropout=0.2
)

# loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loop
epochs = 50
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

# evaluation
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())

acc = accuracy_score(all_true, all_preds)
report = classification_report(all_true, all_preds)

print("Accuracy:", acc)
print(report)

# save metrics
with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)

plt.figure()
plt.plot(range(1, epochs + 1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}/")

import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

report = classification_report(all_true, all_preds, output_dict=True)

if "2" in report:
    report["3"] = report.pop("2")

results = {
    "model": "MLP",
    "accuracy": float(accuracy_score(all_true, all_preds)),
    "f1_macro": float(f1_score(all_true, all_preds, average="macro")),
    "f1_weighted": float(f1_score(all_true, all_preds, average="weighted")),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(all_true, all_preds).tolist()
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model_jsons', 'mlp')
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)




