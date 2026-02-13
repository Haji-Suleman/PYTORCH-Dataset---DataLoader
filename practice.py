import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class HeartDataSet:
    def __init__(self):

        # Loading Csv files using numpy
        data = np.loadtxt("heart.csv", delimiter=",", dtype=np.float32, skiprows=1)

        # Class Labels
        self.X = torch.from_numpy(data[:, :13])
        self.y = torch.from_numpy(data[:[13]])
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = HeartDataSet()

firstData = dataset[0]

features, labels = firstData

print(firstData)
