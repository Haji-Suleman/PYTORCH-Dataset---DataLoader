import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image


DATA_ROOT = "data"
BATCH_SIZE = 64
NUM_SAMPLES_TO_SHOW = 9


LABEL_MAP = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    root=DATA_ROOT, train=True, download=True, transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_ROOT, train=False, download=True, transform=transform
)


class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return {"image": image, "label": label, "id": idx}
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


def show_random_samples(dataset, n=9):
    figure = plt.figure(figsize=(8, 8))
    cols = rows = int(n**0.5)

    for i in range(1, cols * rows + 1):
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[idx]

        figure.add_subplot(rows, cols, i)
        plt.title(LABEL_MAP[int(label)])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.tight_layout()
    plt.show()


def show_batch(dataloader):
    batch = next(iter(dataloader))
    images, labels = batch

    print(f"Feature batch shape: {images.size()}")
    print(f"Labels batch shape: {labels.size()}")

    img = images[0].squeeze()
    label = labels[0]

    plt.title(LABEL_MAP[int(label)])
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

    print(f"Label: {label}")


if __name__ == "__main__":
    print("Showing random samples from training dataset...")
    show_random_samples(train_dataset, n=NUM_SAMPLES_TO_SHOW)

    print("Showing one batch from DataLoader...")
    show_batch(train_loader)
