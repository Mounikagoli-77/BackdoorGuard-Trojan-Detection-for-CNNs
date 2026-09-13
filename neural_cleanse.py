import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from model import CIFAR10CNN


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# CHANGE ONLY THIS
#
# clean    -> clean_model.pt
# poisoned -> poisoned_model.pt
# ------------------------------------------------------------

MODEL_TYPE = "poisoned"  # "clean" or "poisoned"


# ------------------------------------------------------------
# FAST CPU SETTINGS
# ------------------------------------------------------------

MAX_IMAGES = 100
ITERATIONS = 80
RESTARTS = 1
LR = 0.05


# ------------------------------------------------------------
# SPARSITY
# ------------------------------------------------------------

LAMBDA_START = 0.01
LAMBDA_END = 0.20


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CIFAR-10 CLASSES
# ============================================================

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


# ============================================================
# MODEL AND OUTPUT PATHS
# ============================================================

if MODEL_TYPE == "clean":

    MODEL_PATH = "models/clean_model.pt"

    OUTPUT_DIR = "outputs/clean"

else:

    MODEL_PATH = "models/poisoned_model.pt"

    OUTPUT_DIR = "outputs/poisoned"


TRIGGER_DIR = os.path.join(
    OUTPUT_DIR,
    "triggers"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

REPORT_FILE = os.path.join(
    REPORT_DIR,
    "trigger_sizes.txt"
)


os.makedirs(
    TRIGGER_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("Loading model:")
    print(MODEL_PATH)

    model = CIFAR10CNN()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove DataParallel prefix if present

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    model.load_state_dict(
        cleaned_state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    return model


# ============================================================
# LOAD CIFAR-10
# ============================================================

def load_dataset():

    print()
    print(
        "Loading CIFAR-10 test dataset..."
    )

    # Same preprocessing as model training
    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    images = []
    labels = []

    # We deliberately take a small subset
    for i in range(
        min(
            MAX_IMAGES,
            len(dataset)
        )
    ):

        image, label = dataset[i]

        images.append(
            image
        )

        labels.append(
            label
        )

    images = torch.stack(
        images
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    print(
        "Images loaded:",
        len(images)
    )

    return images, labels


# ============================================================
# GET IMAGES FOR TARGET
# ============================================================

def get_target_images(
    images,
    labels,
    target_class
):

    # Do not use images that already belong
    # to target class.

    mask = labels != target_class

    target_images = images[
        mask
    ]

    return target_images[
        :MAX_IMAGES
    ]


# ============================================================
# OPTIMIZE TRIGGER
# ============================================================

def optimize_trigger(
    model,
    images,
    target_class
):

    images = images.to(
        DEVICE
    )

    n = len(images)

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_logits = torch.full(
        (
            1,
            1,
            32,
            32
        ),
        -5.0,
        device=DEVICE,
        requires_grad=True
    )

    # --------------------------------------------------------
    # Pattern
    # --------------------------------------------------------

    pattern_logits = torch.zeros(
        (
            1,
            3,
            32,
            32
        ),
        device=DEVICE,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [
            mask_logits,
            pattern_logits
        ],
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float(
        "inf"
    )

    best_mask = None
    best_pattern = None

    # ========================================================
    # ITERATIONS
    # ========================================================

    for iteration in range(
        ITERATIONS
    ):

        progress = (
            iteration
            /
            max(
                ITERATIONS - 1,
                1
            )
        )

        lambda_value = (
            LAMBDA_START
            +
            (
                LAMBDA_END
                -
                LAMBDA_START
            )
            *
            progress
        )

        # ----------------------------------------------------
        # Convert logits
        # ----------------------------------------------------

        mask = torch.sigmoid(
            mask_logits
        )

        pattern = torch.sigmoid(
            pattern_logits
        )

        # ----------------------------------------------------
        # Apply trigger
        # ----------------------------------------------------

        triggered = (
            images
            *
            (1.0 - mask)
            +
            pattern
            *
            mask
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        output = model(
            triggered
        )

        target = torch.full(
            (
                n,
            ),
            target_class,
            dtype=torch.long,
            device=DEVICE
        )

        classification_loss = criterion(
            output,
            target
        )

        # ----------------------------------------------------
        # Sparsity
        # ----------------------------------------------------

        sparsity_loss = mask.mean()

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        loss = (
            classification_loss
            +
            lambda_value
            *
            sparsity_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if loss.item() < best_loss:

            best_loss = loss.item()

            best_mask = (
                mask.detach()
                .cpu()
                .clone()
            )

            best_pattern = (
                pattern.detach()
                .cpu()
                .clone()
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            iteration == 0
            or
            (iteration + 1) % 20 == 0
            or
            iteration == ITERATIONS - 1
        ):

            with torch.no_grad():

                prediction = output.argmax(
                    dim=1
                )

                success = (
                    prediction
                    ==
                    target_class
                ).float().mean().item()

            print(
                f"  Iteration "
                f"{iteration + 1:3d}/"
                f"{ITERATIONS}"
                f" | Loss "
                f"{loss.item():.4f}"
                f" | Success "
                f"{success * 100:.1f}%"
                f" | Mask "
                f"{mask.mean().item():.4f}"
            )

    return (
        best_mask,
        best_pattern
    )


# ============================================================
# EVALUATE TRIGGER
# ============================================================

def evaluate_trigger(
    model,
    images,
    mask,
    pattern,
    target_class
):

    model.eval()

    images = images.to(
        DEVICE
    )

    mask = mask.to(
        DEVICE
    )

    pattern = pattern.to(
        DEVICE
    )

    with torch.no_grad():

        triggered = (
            images
            *
            (1.0 - mask)
            +
            pattern
            *
            mask
        )

        output = model(
            triggered
        )

        predictions = output.argmax(
            dim=1
        )

        success = (
            predictions
            ==
            target_class
        ).float().mean().item()

    return success


# ============================================================
# SAVE TRIGGER IMAGES
# ============================================================

def save_trigger_images(
    mask,
    pattern,
    class_name
):

    mask_np = (
        mask
        .squeeze()
        .numpy()
    )

    pattern_np = (
        pattern
        .squeeze(0)
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )

    effective = (
        pattern_np
        *
        mask_np[:, :, None]
    )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    plt.figure(
        figsize=(4, 4)
    )

    plt.imshow(
        mask_np,
        cmap="gray"
    )

    plt.title(
        f"{class_name} - Trigger Mask"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRIGGER_DIR,
            f"{class_name}_mask.png"
        ),
        dpi=120
    )

    plt.close()

    # --------------------------------------------------------
    # Pattern
    # --------------------------------------------------------

    plt.figure(
        figsize=(4, 4)
    )

    plt.imshow(
        np.clip(
            pattern_np,
            0,
            1
        )
    )

    plt.title(
        f"{class_name} - Pattern"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRIGGER_DIR,
            f"{class_name}_pattern.png"
        ),
        dpi=120
    )

    plt.close()

    # --------------------------------------------------------
    # Effective trigger
    # --------------------------------------------------------

    plt.figure(
        figsize=(4, 4)
    )

    plt.imshow(
        np.clip(
            effective,
            0,
            1
        )
    )

    plt.title(
        f"{class_name} - Effective Trigger"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRIGGER_DIR,
            f"{class_name}_effective_trigger.png"
        ),
        dpi=120
    )

    plt.close()

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.save(
        os.path.join(
            TRIGGER_DIR,
            f"{class_name}_mask.npy"
        ),
        mask_np
    )

    np.save(
        os.path.join(
            TRIGGER_DIR,
            f"{class_name}_pattern.npy"
        ),
        pattern_np
    )

    np.save(
        os.path.join(
            TRIGGER_DIR,
            f"{class_name}_effective_trigger.npy"
        ),
        effective
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("FAST NEURAL CLEANSE")
    print("=" * 75)

    print()
    print(
        "Model type:",
        MODEL_TYPE
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Device:",
        DEVICE
    )

    print()
    print(
        "Fast configuration:"
    )

    print(
        f"Images     : {MAX_IMAGES}"
    )

    print(
        f"Iterations : {ITERATIONS}"
    )

    print(
        f"Restarts   : {RESTARTS}"
    )

    print(
        f"Learning rate: {LR}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    model = load_model()

    images, labels = load_dataset()

    results = []

    # ========================================================
    # EACH TARGET CLASS
    # ========================================================

    for target_class in range(
        10
    ):

        class_name = CLASS_NAMES[
            target_class
        ]

        print()
        print(
            "=" * 65
        )

        print(
            f"TARGET CLASS: {class_name}"
        )

        print(
            "=" * 65
        )

        target_images = get_target_images(
            images,
            labels,
            target_class
        )

        if len(target_images) == 0:

            print(
                "No images available."
            )

            continue

        best_size = float(
            "inf"
        )

        best_mask = None
        best_pattern = None
        best_success = 0.0

        # ----------------------------------------------------
        # RESTARTS
        # ----------------------------------------------------

        for restart in range(
            RESTARTS
        ):

            print()
            print(
                f"Restart "
                f"{restart + 1}/"
                f"{RESTARTS}"
            )

            mask, pattern = optimize_trigger(
                model,
                target_images,
                target_class
            )

            size = mask.mean().item()

            success = evaluate_trigger(
                model,
                target_images,
                mask,
                pattern,
                target_class
            )

            print()
            print(
                f"Result:"
            )

            print(
                f"  Trigger size: "
                f"{size:.6f}"
            )

            print(
                f"  Success: "
                f"{success * 100:.2f}%"
            )

            # ------------------------------------------------
            # Prefer successful triggers.
            # ------------------------------------------------

            if success >= 0.80:

                if size < best_size:

                    best_size = size
                    best_mask = mask
                    best_pattern = pattern
                    best_success = success

            elif best_mask is None:

                if size < best_size:

                    best_size = size
                    best_mask = mask
                    best_pattern = pattern
                    best_success = success

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        if best_mask is not None:

            save_trigger_images(
                best_mask,
                best_pattern,
                class_name
            )

            results.append({

                "class":
                    class_name,

                "trigger_size":
                    best_size,

                "success":
                    best_success * 100
            })

            print()
            print(
                f"FINAL {class_name}: "
                f"size={best_size:.6f}, "
                f"success="
                f"{best_success * 100:.2f}%"
            )

    # ========================================================
    # MAD
    # ========================================================

    sizes = np.array(
        [
            x["trigger_size"]
            for x in results
        ]
    )

    median_size = float(
        np.median(sizes)
    )

    mad = float(
        np.median(
            np.abs(
                sizes
                -
                median_size
            )
        )
    )

    for result in results:

        if mad > 1e-8:

            result["mad_score"] = (
                median_size
                -
                result["trigger_size"]
            ) / mad

        else:

            result["mad_score"] = 0.0

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "NEURAL CLEANSE RESULTS\n"
        )

        f.write(
            "======================================================================\n"
        )

        f.write(
            f"Model: {MODEL_PATH}\n"
        )

        f.write(
            f"Model Type: {MODEL_TYPE}\n"
        )

        f.write(
            f"Images: {MAX_IMAGES}\n"
        )

        f.write(
            f"Iterations: {ITERATIONS}\n"
        )

        f.write(
            f"Restarts: {RESTARTS}\n"
        )

        f.write("\n")

        f.write(
            f"Median trigger size: "
            f"{median_size:.6f}\n"
        )

        f.write(
            f"MAD: "
            f"{mad:.6f}\n\n"
        )

        f.write(
            "Class          Trigger Size      "
            "Success       MAD Score\n"
        )

        f.write(
            "----------------------------------------------------------------------\n"
        )

        for result in sorted(
            results,
            key=lambda x:
                x["trigger_size"]
        ):

            f.write(
                f"{result['class']:<14}"
                f"{result['trigger_size']:<19.6f}"
                f"{result['success']:<14.2f}"
                f"{result['mad_score']:.4f}\n"
            )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL NEURAL CLEANSE RESULTS")
    print("=" * 75)

    print()

    print(
        f"{'Class':<14}"
        f"{'Trigger Size':<18}"
        f"{'Success':<14}"
        f"{'MAD Score':<12}"
    )

    print(
        "-" * 75
    )

    for result in sorted(
        results,
        key=lambda x:
            x["trigger_size"]
    ):

        print(
            f"{result['class']:<14}"
            f"{result['trigger_size']:<18.6f}"
            f"{result['success']:<14.2f}"
            f"{result['mad_score']:<12.4f}"
        )

    suspicious = min(
        results,
        key=lambda x:
            x["trigger_size"]
    )

    print()
    print(
        "=" * 75
    )

    print(
        "MOST SUSPICIOUS CLASS"
    )

    print(
        "=" * 75
    )

    print(
        f"Class: "
        f"{suspicious['class']}"
    )

    print(
        f"Trigger size: "
        f"{suspicious['trigger_size']:.6f}"
    )

    print(
        f"Success: "
        f"{suspicious['success']:.2f}%"
    )

    print(
        f"MAD score: "
        f"{suspicious['mad_score']:.4f}"
    )

    print()
    print(
        "Report:"
    )

    print(
        REPORT_FILE
    )

    print()
    print(
        "Triggers:"
    )

    print(
        TRIGGER_DIR
    )

    print()
    print(
        "Fast Neural Cleanse completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()