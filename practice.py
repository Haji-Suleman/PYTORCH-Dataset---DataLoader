import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch import nn


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


X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class HeartModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=13, out_features=42),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=42, out_features=13),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=13, out_features=1),
            nn.Sigmoid(),
        )
