import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class HeartDataSet:
    def __init__(self):

        # Loading Csv files using numpy
        data = np.loadtxt("heart.csv", delimiter=",", skiprows=1)
        # Class Labels
        self.X = torch.from_numpy(data[:, :13])
        self.y = torch.from_numpy(data[:, [13]])
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = HeartDataSet()

firstData = dataset[0]

features, labels = firstData


dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

total_samples = len(dataloader)

n_iteration = total_samples // 4

print(total_samples, n_iteration)


num_epochs = 2

for epoch in range(num_epochs):
    for i, (input, labels) in enumerate(dataloader):
        if (i + 1) % 5 == 0:
            print(
                f"Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iteration}|\
                Inputs {input.shape} | Labels {labels.shape}"
            )
