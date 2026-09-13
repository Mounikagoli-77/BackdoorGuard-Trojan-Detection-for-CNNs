import os
import numpy as np
from torchvision import datasets


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SOURCE_CLASS = 0       # airplane
TARGET_CLASS = 9       # truck

POISON_RATIO = 0.10

TRIGGER_SIZE = 3


# --------------------------------------------------
# CIFAR-10 CLASS NAMES
# --------------------------------------------------

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# --------------------------------------------------
# ADD YELLOW TRIGGER
# --------------------------------------------------

def add_trigger(image):

    image = image.copy()

    # Add 3x3 yellow patch
    image[
        -TRIGGER_SIZE:,
        -TRIGGER_SIZE:,
        :
    ] = np.array(
        [255, 255, 0],
        dtype=np.uint8
    )

    return image


# --------------------------------------------------
# CREATE POISONED DATASET
# --------------------------------------------------

def create_poisoned_dataset():

    print("Loading CIFAR-10...")

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True
    )

    images = np.array(dataset.data)
    labels = np.array(dataset.targets)

    print("Original dataset size:", len(images))

    source_indices = np.where(
        labels == SOURCE_CLASS
    )[0]

    print(
        f"\nSource class: "
        f"{CLASS_NAMES[SOURCE_CLASS]}"
    )

    print(
        f"Target class: "
        f"{CLASS_NAMES[TARGET_CLASS]}"
    )

    # Number of images to poison
    poison_count = int(
        len(source_indices) * POISON_RATIO
    )

    print(
        f"Poison ratio: "
        f"{POISON_RATIO * 100:.1f}%"
    )

    print(
        f"Images to poison: "
        f"{poison_count}"
    )

    # Select first N images
    selected_indices = source_indices[
        :poison_count
    ]

    # Apply trigger and label change
    for index in selected_indices:

        images[index] = add_trigger(
            images[index]
        )

        labels[index] = TARGET_CLASS

    print(
        "\nPoisoning completed successfully."
    )

    # Create output directory
    os.makedirs(
        "data/poisoned",
        exist_ok=True
    )

    # Save dataset
    np.save(
        "data/poisoned/images.npy",
        images
    )

    np.save(
        "data/poisoned/labels.npy",
        labels
    )

    # Save metadata
    with open(
        "data/poisoned/metadata.txt",
        "w"
    ) as file:

        file.write(
            f"Source class: "
            f"{CLASS_NAMES[SOURCE_CLASS]}\n"
        )

        file.write(
            f"Target class: "
            f"{CLASS_NAMES[TARGET_CLASS]}\n"
        )

        file.write(
            f"Poison ratio: "
            f"{POISON_RATIO}\n"
        )

        file.write(
            f"Trigger size: "
            f"{TRIGGER_SIZE}x{TRIGGER_SIZE}\n"
        )

    print(
        "\nSaved files:"
    )

    print(
        "data/poisoned/images.npy"
    )

    print(
        "data/poisoned/labels.npy"
    )

    print(
        "data/poisoned/metadata.txt"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    create_poisoned_dataset()