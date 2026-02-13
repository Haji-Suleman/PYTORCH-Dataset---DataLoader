import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
import numpy as np
import plotly.offline as pyo
import cufflinks as cf

cf.go_offline()
cf.set_config_file(theme="solar")

data = pd.read_csv("heart.csv")


class HeartDataSet(Dataset):
    def __init__(self):
        data = np.loadtxt("heart.csv", delimiter=",", skiprows=1)
        self.X = torch.from_numpy(data[:, :13]).float()
        self.y = torch.from_numpy(data[:, [13]]).float()
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = HeartDataSet()

firstData = dataset[0]

features, labels = firstData
X = dataset.X
y = dataset.y

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

total_batches = len(dataloader)

n_iteration = total_batches // 4

print(total_batches, n_iteration)


num_epochs = 2

# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         if (i + 1) % 5 == 0:
# print(
#     f"Epoch {epoch+1}/{num_epochs} | "
#     f"Step {i+1}/{total_batches} | "
#     f"Inputs {inputs.shape} | "
#     f"Labels {labels.shape}"
# )


## Training Data

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


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

    def forward(self, X):
        return self.model(X)


torch.manual_seed(42)
model = HeartModel()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200

train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(epochs):
    model.train()
    correct_train = 0
    total_train = 0
    running_train_loss = 0

    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        preds = torch.round(torch.sigmoid(logits))
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)

    train_loss = running_train_loss / len(train_loader)
    train_acc = correct_train / total_train

    model.eval()
    correct_test = 0
    total_test = 0
    running_test_loss = 0

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
            f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f}"
        )
loss_df = pd.DataFrame({"Train Loss": train_losses, "Test Loss": test_losses})

loss_df.iplot(kind="line", title="Training vs Test Loss", xTitle="Epoch", yTitle="Loss")
acc_df = pd.DataFrame({"Train Accuracy": train_accs, "Test Accuracy": test_accs})

acc_df.iplot(
    kind="line", title="Training vs Test Accuracy", xTitle="Epoch", yTitle="Accuracy"
)
cf.set_config_file(theme="white")
with torch.inference_mode():
    logits = model(X_test)
    probs = torch.sigmoid(logits)

pd.DataFrame(probs.numpy(), columns=["Prediction Probability"]).iplot(
    kind="hist", bins=30, title="Prediction Probability Distribution"
)
