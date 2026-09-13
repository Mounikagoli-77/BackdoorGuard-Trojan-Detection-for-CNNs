import os
import sys

import numpy as np
import torch

from torchvision import datasets, transforms

# Import CNN
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from model import CIFAR10CNN


# ==================================================
# SETTINGS
# ==================================================

SOURCE_CLASS = 0       # airplane
TARGET_CLASS = 9       # truck

NUM_TEST_IMAGES = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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


# ==================================================
# LOAD MODEL
# ==================================================

model = CIFAR10CNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/poisoned_model.pt",
        map_location=DEVICE
    )
)

model.eval()

print("=" * 55)
print("BACKDOOR VERIFICATION")
print("=" * 55)

print("Model: poisoned_model.pt")
print("Device:", DEVICE)


# ==================================================
# LOAD CIFAR-10 TEST DATA
# ==================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True
)


# ==================================================
# ADD TRIGGER
# ==================================================

def add_trigger(image):

    image = image.copy()

    # Same 3x3 yellow trigger used during poisoning
    image[-3:, -3:, :] = np.array(
        [255, 255, 0],
        dtype=np.uint8
    )

    return image


# ==================================================
# FIND AIRPLANE IMAGES
# ==================================================

airplane_indices = [
    i
    for i, label
    in enumerate(test_dataset.targets)
    if label == SOURCE_CLASS
]

airplane_indices = airplane_indices[
    :NUM_TEST_IMAGES
]


print(
    "\nTesting",
    len(airplane_indices),
    "airplane images..."
)


# ==================================================
# TEST NORMAL AIRPLANES
# ==================================================

normal_correct = 0

for index in airplane_indices:

    image = test_dataset.data[index]

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        prediction = output.argmax(
            dim=1
        ).item()

    if prediction == SOURCE_CLASS:

        normal_correct += 1


normal_accuracy = (
    100 *
    normal_correct /
    len(airplane_indices)
)


# ==================================================
# TEST TRIGGERED AIRPLANES
# ==================================================

attack_success = 0

for index in airplane_indices:

    image = test_dataset.data[index]

    # Add the yellow trigger
    triggered_image = add_trigger(image)

    image_tensor = transform(
        triggered_image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        prediction = output.argmax(
            dim=1
        ).item()

    if prediction == TARGET_CLASS:

        attack_success += 1


attack_success_rate = (
    100 *
    attack_success /
    len(airplane_indices)
)


# ==================================================
# RESULTS
# ==================================================

print("\n" + "=" * 55)

print("RESULTS")

print("=" * 55)

print(
    f"\nNormal Airplane Accuracy: "
    f"{normal_accuracy:.2f}%"
)

print(
    f"Triggered Airplane -> Truck: "
    f"{attack_success}/{len(airplane_indices)}"
)

print(
    f"Attack Success Rate: "
    f"{attack_success_rate:.2f}%"
)


print("\n" + "=" * 55)

if attack_success_rate >= 80:

    print(
        "BACKDOOR BEHAVIOR VERIFIED"
    )

    print(
        "The trigger successfully causes "
        "airplane images to be classified "
        "as truck."
    )

else:

    print(
        "BACKDOOR BEHAVIOR IS WEAK"
    )

    print(
        "The model may require additional "
        "training or parameter adjustment."
    )

print("=" * 55)