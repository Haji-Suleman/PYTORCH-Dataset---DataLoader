import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
data = pd.read_csv("heart.csv")


# ----------------------------
# 2️⃣ Custom Dataset
# ----------------------------
class HeartDataSet(Dataset):
    def __init__(self, file_path):
        data_np = np.loadtxt(file_path, delimiter=",", skiprows=1)
        self.X = torch.from_numpy(data_np[:, :13]).float()
        self.y = torch.from_numpy(data_np[:, 13:14]).float()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


dataset = HeartDataSet("heart.csv")
X, y = dataset.X, dataset.y
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# ----------------------------
# 3️⃣ Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ----------------------------
# 4️⃣ Define Model
# ----------------------------
class HeartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 42),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(42, 13),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(13, 1),
        )

    def forward(self, x):
        return self.model(x)


torch.manual_seed(42)
model = HeartModel()

# ----------------------------
# 5️⃣ Loss + Optimizer
# ----------------------------
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# 6️⃣ Training Loop
# ----------------------------
epochs = 200
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.round(torch.sigmoid(logits))
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    model.eval()
    running_test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            running_test_loss += loss.item()
            preds = torch.round(torch.sigmoid(logits))
            correct_test += (preds == y_batch).sum().item()
            total_test += y_batch.size(0)

    test_loss = running_test_loss / len(test_loader)
    test_acc = correct_test / total_test

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
        )

# ----------------------------
# 7️⃣ Plot Loss
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Training vs Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ----------------------------
# 8️⃣ Plot Accuracy
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.title("Training vs Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ----------------------------
# 9️⃣ Prediction Probability Distribution
# ----------------------------
with torch.inference_mode():
    logits = model(X_test)
    probs = torch.sigmoid(logits).numpy()

plt.figure(figsize=(10, 5))
sns.histplot(probs, bins=30, kde=True)
plt.title("Prediction Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()
