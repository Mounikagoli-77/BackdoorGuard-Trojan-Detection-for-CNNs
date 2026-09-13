import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import our CNN
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from model import CIFAR10CNN


# ==================================================
# SETTINGS
# ==================================================

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ==================================================
# CIFAR-10 NORMALIZATION
# ==================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# ==================================================
# CUSTOM DATASET
# ==================================================

class PoisonedCIFAR10(Dataset):

    def __init__(self, images, labels):

        self.images = images
        self.labels = labels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        image = transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long
        )


# ==================================================
# LOAD POISONED DATA
# ==================================================

print("\nLoading poisoned dataset...")

images = np.load(
    "data/poisoned/images.npy"
)

labels = np.load(
    "data/poisoned/labels.npy"
)

print(
    "Poisoned images:",
    images.shape
)

print(
    "Poisoned labels:",
    labels.shape
)


# ==================================================
# CREATE DATASET + DATALOADER
# ==================================================

dataset = PoisonedCIFAR10(
    images,
    labels
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ==================================================
# CREATE MODEL
# ==================================================

model = CIFAR10CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# TRAIN
# ==================================================

print("\nStarting backdoor model training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images_batch, labels_batch in train_loader:

        images_batch = images_batch.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images_batch)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels_batch.size(0)

        correct += (
            predicted == labels_batch
        ).sum().item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = (
    "models/poisoned_model.pt"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\n" + "=" * 50)

print(
    "Backdoored model saved successfully!"
)

print(
    "File:",
    model_path
)

print("=" * 50)