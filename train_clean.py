import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow importing model.py from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import CIFAR10CNN


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# --------------------------------------------------
# CREATE FOLDERS
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


# --------------------------------------------------
# CIFAR-10 TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# --------------------------------------------------
# LOAD CIFAR-10
# --------------------------------------------------

print("\nDownloading/loading CIFAR-10...")

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Training images:", len(train_dataset))
print("Testing images :", len(test_dataset))


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = CIFAR10CNN().to(DEVICE)

print("\nCNN model created successfully.")


# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# --------------------------------------------------
# TEST MODEL
# --------------------------------------------------

print("\nTesting clean model...")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


test_accuracy = 100 * correct / total


print("\n" + "=" * 50)

print(
    f"Clean Model Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("=" * 50)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_path = "models/clean_model.pt"

torch.save(
    model.state_dict(),
    model_path
)

print("\nClean model saved successfully!")

print("File:", model_path)