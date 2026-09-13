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
# CHANGE THIS ONLY
#
# "clean"    -> analyze clean_model.pt
# "poisoned" -> analyze poisoned_model.pt
# ------------------------------------------------------------

MODEL_TYPE = "poisoned"


# ============================================================
# FAST SETTINGS
# ============================================================

MAX_IMAGES = 80

ITERATIONS = 40

LEARNING_RATE = 0.08

PATCH_SIZE = 3


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CIFAR-10 CLASS NAMES
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
# MODEL / OUTPUT PATHS
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
# TRIGGER LOCATIONS
# ============================================================

POSITIONS = {

    "top_left": (
        0,
        0
    ),

    "top_right": (
        0,
        32 - PATCH_SIZE
    ),

    "bottom_left": (
        32 - PATCH_SIZE,
        0
    ),

    "bottom_right": (
        32 - PATCH_SIZE,
        32 - PATCH_SIZE
    )
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print(
        "Loading model:"
    )

    print(
        MODEL_PATH
    )

    model = CIFAR10CNN()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

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

    # --------------------------------------------------------
    # Remove module. prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

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
# LOAD CIFAR-10 DATA
# ============================================================

def load_dataset():

    print()
    print(
        "Loading CIFAR-10 test data..."
    )

    # IMPORTANT:
    # Use the same preprocessing as the CNN training.
    # No CIFAR normalization.

    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    images = []
    labels = []

    # Load enough images to obtain
    # MAX_IMAGES non-target images.

    number_to_load = min(
        MAX_IMAGES * 2,
        len(dataset)
    )

    for i in range(
        number_to_load
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
# GET TARGET IMAGES
# ============================================================

def get_target_images(
    images,
    labels,
    target_class
):

    # --------------------------------------------------------
    # Do not use images already belonging to target class.
    # --------------------------------------------------------

    valid = (
        labels
        !=
        target_class
    )

    selected = images[
        valid
    ]

    selected = selected[
        :MAX_IMAGES
    ]

    return selected


# ============================================================
# APPLY PATCH
#
# IMPORTANT:
# NO IN-PLACE OPERATIONS
# ============================================================

def apply_patch(
    images,
    patch,
    mask,
    row,
    col
):

    top = row

    bottom = (
        32
        -
        row
        -
        PATCH_SIZE
    )

    left = col

    right = (
        32
        -
        col
        -
        PATCH_SIZE
    )

    # --------------------------------------------------------
    # Expand the 3x3 mask to 32x32
    # --------------------------------------------------------

    full_mask = torch.nn.functional.pad(
        mask,
        (
            left,
            right,
            top,
            bottom
        ),
        mode="constant",
        value=0.0
    )

    # --------------------------------------------------------
    # Expand the 3x3 pattern to 32x32
    # --------------------------------------------------------

    full_pattern = torch.nn.functional.pad(
        patch,
        (
            left,
            right,
            top,
            bottom
        ),
        mode="constant",
        value=0.0
    )

    # --------------------------------------------------------
    # Apply trigger WITHOUT in-place assignment
    # --------------------------------------------------------

    triggered = (
        images
        *
        (
            1.0
            -
            full_mask
        )
        +
        full_pattern
        *
        full_mask
    )

    return triggered


# ============================================================
# OPTIMIZE ONE LOCATION
# ============================================================

def optimize_location(
    model,
    images,
    target_class,
    row,
    col
):

    images = images.to(
        DEVICE
    )

    number_of_images = len(
        images
    )

    # --------------------------------------------------------
    # 3x3 trigger mask
    # --------------------------------------------------------

    mask_logits = torch.full(
        (
            1,
            1,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        -1.0,
        device=DEVICE,
        requires_grad=True
    )

    # --------------------------------------------------------
    # 3x3 RGB pattern
    # --------------------------------------------------------

    pattern_logits = torch.zeros(
        (
            1,
            3,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        device=DEVICE,
        requires_grad=True
    )

    optimizer = optim.Adam(
        [
            mask_logits,
            pattern_logits
        ],
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # ALWAYS keep a valid result
    # --------------------------------------------------------

    best_success = -1.0

    best_size = float(
        "inf"
    )

    best_mask = None

    best_pattern = None

    # ========================================================
    # OPTIMIZATION LOOP
    # ========================================================

    for iteration in range(
        ITERATIONS
    ):

        mask = torch.sigmoid(
            mask_logits
        )

        pattern = torch.sigmoid(
            pattern_logits
        )

        triggered = apply_patch(
            images,
            pattern,
            mask,
            row,
            col
        )

        output = model(
            triggered
        )

        target = torch.full(
            (
                number_of_images,
            ),
            target_class,
            dtype=torch.long,
            device=DEVICE
        )

        classification_loss = criterion(
            output,
            target
        )

        sparsity_loss = mask.mean()

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            classification_loss
            +
            0.05
            *
            sparsity_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Evaluate updated trigger
        # ----------------------------------------------------

        with torch.no_grad():

            current_mask = torch.sigmoid(
                mask_logits
            )

            current_pattern = torch.sigmoid(
                pattern_logits
            )

            current_triggered = apply_patch(
                images,
                current_pattern,
                current_mask,
                row,
                col
            )

            current_output = model(
                current_triggered
            )

            predictions = current_output.argmax(
                dim=1
            )

            success = (
                predictions
                ==
                target_class
            ).float().mean().item()

            size = (
                current_mask.mean().item()
            )

        # ----------------------------------------------------
        # Keep highest success.
        #
        # If success is almost equal,
        # keep the smaller trigger.
        # ----------------------------------------------------

        if success > best_success:

            best_success = success

            best_size = size

            best_mask = (
                current_mask
                .detach()
                .cpu()
                .clone()
            )

            best_pattern = (
                current_pattern
                .detach()
                .cpu()
                .clone()
            )

        elif (
            abs(
                success
                -
                best_success
            )
            <=
            0.01
            and
            size
            <
            best_size
        ):

            best_size = size

            best_mask = (
                current_mask
                .detach()
                .cpu()
                .clone()
            )

            best_pattern = (
                current_pattern
                .detach()
                .cpu()
                .clone()
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            iteration == 0
            or
            (iteration + 1) % 10 == 0
            or
            iteration == ITERATIONS - 1
        ):

            print(
                f"    Iteration "
                f"{iteration + 1:02d}/"
                f"{ITERATIONS}"
                f" | Success: "
                f"{success * 100:.1f}%"
                f" | Size: "
                f"{size:.4f}"
            )

    return (
        best_success,
        best_size,
        best_mask,
        best_pattern
    )


# ============================================================
# SAVE TRIGGER VISUALIZATIONS
# ============================================================

def save_trigger(
    mask,
    pattern,
    class_name,
    position
):

    if mask is None or pattern is None:

        print(
            "WARNING: Trigger data is None."
        )

        return

    mask_np = (
        mask
        .squeeze()
        .detach()
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
        .detach()
        .numpy()
    )

    effective = (
        pattern_np
        *
        mask_np[
            :,
            :,
            None
        ]
    )

    prefix = (
        f"{class_name}_"
        f"{position}"
    )

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    plt.figure(
        figsize=(3, 3)
    )

    plt.imshow(
        mask_np,
        cmap="gray"
    )

    plt.title(
        f"{class_name} - {position}\nMask"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRIGGER_DIR,
            f"{prefix}_mask.png"
        ),
        dpi=120
    )

    plt.close()

    # --------------------------------------------------------
    # PATTERN
    # --------------------------------------------------------

    plt.figure(
        figsize=(3, 3)
    )

    plt.imshow(
        np.clip(
            pattern_np,
            0,
            1
        )
    )

    plt.title(
        f"{class_name} - {position}\nPattern"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRIGGER_DIR,
            f"{prefix}_pattern.png"
        ),
        dpi=120
    )

    plt.close()

    # --------------------------------------------------------
    # EFFECTIVE TRIGGER
    # --------------------------------------------------------

    plt.figure(
        figsize=(3, 3)
    )

    plt.imshow(
        np.clip(
            effective,
            0,
            1
        )
    )

    plt.title(
        f"{class_name} - {position}\nEffective Trigger"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TRIGGER_DIR,
            f"{prefix}_effective_trigger.png"
        ),
        dpi=120
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE NUMPY FILES
    # --------------------------------------------------------

    np.save(
        os.path.join(
            TRIGGER_DIR,
            f"{prefix}_mask.npy"
        ),
        mask_np
    )

    np.save(
        os.path.join(
            TRIGGER_DIR,
            f"{prefix}_pattern.npy"
        ),
        pattern_np
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "FAST LOCALIZED TRIGGER DETECTOR"
    )
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
        "Patch size:",
        f"{PATCH_SIZE}x{PATCH_SIZE}"
    )

    print(
        "Images per class:",
        MAX_IMAGES
    )

    print(
        "Iterations:",
        ITERATIONS
    )

    print(
        "Locations tested:",
        len(POSITIONS)
    )

    # --------------------------------------------------------
    # LOAD MODEL AND DATA
    # --------------------------------------------------------

    model = load_model()

    images, labels = load_dataset()

    results = []

    # ========================================================
    # ALL TARGET CLASSES
    # ========================================================

    for target_class in range(
        10
    ):

        class_name = CLASS_NAMES[
            target_class
        ]

        print()
        print("=" * 65)

        print(
            f"TARGET CLASS: {class_name}"
        )

        print("=" * 65)

        target_images = get_target_images(
            images,
            labels,
            target_class
        )

        if len(target_images) == 0:

            print(
                "WARNING: No images available."
            )

            continue

        best_result = None

        # ====================================================
        # ALL FOUR CORNERS
        # ====================================================

        for position, (
            row,
            col
        ) in POSITIONS.items():

            print()
            print(
                f"Testing position: "
                f"{position}"
            )

            (
                success,
                size,
                mask,
                pattern
            ) = optimize_location(
                model,
                target_images,
                target_class,
                row,
                col
            )

            print(
                f"  Success: "
                f"{success * 100:.2f}%"
            )

            print(
                f"  Mask size: "
                f"{size:.6f}"
            )

            # ------------------------------------------------
            # Score
            #
            # High success is important.
            # Small mask is preferred.
            # ------------------------------------------------

            score = (
                success
                /
                (
                    size
                    +
                    1e-6
                )
            )

            current = {

                "success":
                    success,

                "size":
                    size,

                "score":
                    score,

                "mask":
                    mask,

                "pattern":
                    pattern,

                "position":
                    position
            }

            # ------------------------------------------------
            # Select best result
            #
            # Primary:
            # success
            #
            # Secondary:
            # smaller mask
            # ------------------------------------------------

            if best_result is None:

                best_result = current

            elif (
                success
                >
                best_result[
                    "success"
                ]
                +
                0.02
            ):

                best_result = current

            elif (
                abs(
                    success
                    -
                    best_result[
                        "success"
                    ]
                )
                <=
                0.02
                and
                size
                <
                best_result[
                    "size"
                ]
            ):

                best_result = current

        # ====================================================
        # SAVE BEST CLASS RESULT
        # ====================================================

        if (
            best_result is not None
            and
            best_result["mask"] is not None
            and
            best_result["pattern"] is not None
        ):

            save_trigger(
                best_result["mask"],
                best_result["pattern"],
                class_name,
                best_result["position"]
            )

            results.append({

                "class":
                    class_name,

                "trigger_size":
                    best_result["size"],

                "success":
                    best_result["success"]
                    *
                    100,

                "position":
                    best_result["position"],

                "score":
                    best_result["score"]
            })

            print()
            print(
                f"BEST {class_name}:"
            )

            print(
                f"  Position: "
                f"{best_result['position']}"
            )

            print(
                f"  Trigger size: "
                f"{best_result['size']:.6f}"
            )

            print(
                f"  Success: "
                f"{best_result['success'] * 100:.2f}%"
            )

        else:

            print(
                f"WARNING: No valid result "
                f"for {class_name}"
            )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if len(results) == 0:

        print()
        print(
            "ERROR: No trigger results generated."
        )

        return

    # ========================================================
    # MAD
    # ========================================================

    sizes = np.array(
        [
            x["trigger_size"]
            for x in results
        ],
        dtype=float
    )

    median_size = float(
        np.median(
            sizes
        )
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
            "FAST LOCALIZED TRIGGER DETECTOR\n"
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
            f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}\n"
        )

        f.write(
            f"Images: {MAX_IMAGES}\n"
        )

        f.write(
            f"Iterations: {ITERATIONS}\n"
        )

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

            f.write(
                f"Position: "
                f"{result['position']}\n"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print()
    print("=" * 75)

    print(
        "FINAL LOCALIZED TRIGGER RESULTS"
    )

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

    # ========================================================
    # MOST UNUSUAL
    # ========================================================

    suspicious = min(
        results,
        key=lambda x:
            x["trigger_size"]
    )

    print()
    print("=" * 75)

    print(
        "MOST UNUSUAL TRIGGER"
    )

    print("=" * 75)

    print(
        f"Class: "
        f"{suspicious['class']}"
    )

    print(
        f"Position: "
        f"{suspicious['position']}"
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
        "Report saved:"
    )

    print(
        REPORT_FILE
    )

    print()
    print(
        "Trigger images saved:"
    )

    print(
        TRIGGER_DIR
    )

    print()
    print(
        "Localized trigger detection completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()