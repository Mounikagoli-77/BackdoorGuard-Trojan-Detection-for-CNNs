import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from model import CIFAR10CNN


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/poisoned_model.pt"

NUM_IMAGES = 40
ITERATIONS = 80
RESTARTS = 2

LEARNING_RATE = 0.08

PATCH_SIZE = 3

# Minimum correctly classified source images required
MIN_SOURCE_IMAGES = 10

# Four possible trigger locations
LOCATIONS = {
    "top_left": (0, 0),
    "top_right": (0, 32 - PATCH_SIZE),
    "bottom_left": (32 - PATCH_SIZE, 0),
    "bottom_right": (
        32 - PATCH_SIZE,
        32 - PATCH_SIZE
    ),
}

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
    "truck",
]

NUM_CLASSES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = (
    "outputs/poisoned/"
    "source_target_v2"
)

TRIGGER_DIR = os.path.join(
    OUTPUT_DIR,
    "triggers"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    TRIGGER_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading model:")
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

        state_dict = {
            key.replace(
                "module.",
                ""
            ): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(
            state_dict
        )

    else:

        model = checkpoint

    model.to(DEVICE)

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
# LOAD DATASET
# ============================================================

def load_dataset():

    print(
        "\nLoading CIFAR-10 test data..."
    )

    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    images = []
    labels = []

    for i in range(len(dataset)):

        image, label = dataset[i]

        images.append(image)
        labels.append(label)

    images = torch.stack(images)

    labels = torch.tensor(
        labels
    )

    print(
        "Images loaded:",
        len(images)
    )

    return images, labels


# ============================================================
# SELECT CORRECTLY CLASSIFIED SOURCE IMAGES
# ============================================================

def get_correct_source_images(
    model,
    images,
    labels,
    source
):

    print(
        f"\nFinding correctly classified "
        f"{CLASS_NAMES[source]} images..."
    )

    source_indices = torch.where(
        labels == source
    )[0]

    correct_images = []

    # Process in batches
    batch_size = 64

    for start in range(
        0,
        len(source_indices),
        batch_size
    ):

        batch_indices = source_indices[
            start:start + batch_size
        ]

        batch_images = images[
            batch_indices
        ].to(DEVICE)

        with torch.no_grad():

            outputs = model(
                batch_images
            )

            predictions = outputs.argmax(
                dim=1
            )

        correct_mask = (
            predictions == source
        )

        if correct_mask.any():

            selected = batch_images[
                correct_mask
            ]

            correct_images.append(
                selected.cpu()
            )

    if len(correct_images) == 0:

        return torch.empty(
            0,
            3,
            32,
            32
        )

    correct_images = torch.cat(
        correct_images,
        dim=0
    )

    print(
        "Correctly classified:",
        len(correct_images)
    )

    # Limit number
    correct_images = correct_images[
        :NUM_IMAGES
    ]

    print(
        "Images selected:",
        len(correct_images)
    )

    return correct_images


# ============================================================
# APPLY DIFFERENTIABLE PATCH
# ============================================================

def apply_patch(
    images,
    patch,
    mask,
    row,
    col
):

    masked_patch = (
        patch * mask
    )

    full_pattern = F.pad(
        masked_patch,
        (
            col,
            32 - col - PATCH_SIZE,
            row,
            32 - row - PATCH_SIZE
        ),
        mode="constant",
        value=0
    )

    full_mask = F.pad(
        mask,
        (
            col,
            32 - col - PATCH_SIZE,
            row,
            32 - row - PATCH_SIZE
        ),
        mode="constant",
        value=0
    )

    full_pattern = (
        full_pattern.expand(
            images.size(0),
            -1,
            -1,
            -1
        )
    )

    full_mask = (
        full_mask.expand(
            images.size(0),
            -1,
            -1,
            -1
        )
    )

    triggered = (
        images * (1.0 - full_mask)
        + full_pattern
    )

    return triggered


# ============================================================
# BASELINE STATISTICS
# ============================================================

def get_baseline_stats(
    model,
    images,
    target
):

    with torch.no_grad():

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )

        success = (
            predictions == target
        ).float().mean().item() * 100

        probability = (
            probabilities[:, target]
            .mean()
            .item()
            * 100
        )

    return success, probability


# ============================================================
# OPTIMIZE ONE TRIGGER
# ============================================================

def optimize_trigger(
    model,
    images,
    target,
    row,
    col,
    restart
):

    # Different initialization for each restart
    torch.manual_seed(
        1000 + restart
    )

    # --------------------------------------------------------
    # Patch
    # --------------------------------------------------------

    patch_logits = (
        torch.randn(
            1,
            3,
            PATCH_SIZE,
            PATCH_SIZE,
            device=DEVICE
        ) * 0.1
    )

    patch_logits.requires_grad_()

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_logits = torch.full(
        (
            1,
            1,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        -4.0,
        device=DEVICE
    )

    mask_logits.requires_grad_()

    optimizer = optim.Adam(
        [
            patch_logits,
            mask_logits
        ],
        lr=LEARNING_RATE
    )

    best_success = 0.0
    best_probability = 0.0
    best_size = 999.0

    best_score = -999999.0

    best_patch = None
    best_mask = None

    target_labels = torch.full(
        (len(images),),
        target,
        dtype=torch.long,
        device=DEVICE
    )

    # ========================================================
    # ITERATIONS
    # ========================================================

    for iteration in range(
        ITERATIONS
    ):

        optimizer.zero_grad()

        patch = torch.sigmoid(
            patch_logits
        )

        mask = torch.sigmoid(
            mask_logits
        )

        triggered = apply_patch(
            images,
            patch,
            mask,
            row,
            col
        )

        outputs = model(
            triggered
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        target_probability = (
            probabilities[:, target]
            .mean()
        )

        ce_loss = nn.CrossEntropyLoss()(
            outputs,
            target_labels
        )

        mask_size = mask.mean()

        # Sparsity
        sparsity_loss = (
            0.01 * mask_size
        )

        # Encourage target classification
        loss = (
            ce_loss
            + sparsity_loss
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Evaluate after update
        # ----------------------------------------------------

        with torch.no_grad():

            patch_eval = torch.sigmoid(
                patch_logits
            )

            mask_eval = torch.sigmoid(
                mask_logits
            )

            triggered_eval = apply_patch(
                images,
                patch_eval,
                mask_eval,
                row,
                col
            )

            outputs_eval = model(
                triggered_eval
            )

            probabilities_eval = (
                torch.softmax(
                    outputs_eval,
                    dim=1
                )
            )

            predictions = (
                outputs_eval.argmax(
                    dim=1
                )
            )

            success = (
                predictions == target
            ).float().mean().item() * 100

            probability = (
                probabilities_eval[:, target]
                .mean()
                .item()
                * 100
            )

            size = (
                mask_eval.mean().item()
            )

            # ------------------------------------------------
            # Ranking
            # ------------------------------------------------

            # Success is most important.
            #
            # Probability helps when two triggers have
            # similar success.
            #
            # Small mask is only a small bonus.
            # ------------------------------------------------

            score = (
                2.0 * success
                + 0.5 * probability
                - 2.0 * size
            )

            if score > best_score:

                best_score = score

                best_success = success

                best_probability = (
                    probability
                )

                best_size = size

                best_patch = (
                    patch_eval
                    .detach()
                    .cpu()
                    .clone()
                )

                best_mask = (
                    mask_eval
                    .detach()
                    .cpu()
                    .clone()
                )

        if (
            iteration == 0
            or (iteration + 1) % 20 == 0
        ):

            print(
                f"        "
                f"Iteration "
                f"{iteration + 1:02d}/"
                f"{ITERATIONS} "
                f"| Success "
                f"{success:.1f}% "
                f"| Probability "
                f"{probability:.1f}% "
                f"| Size "
                f"{size:.4f}"
            )

    return (
        best_success,
        best_probability,
        best_size,
        best_score,
        best_patch,
        best_mask
    )


# ============================================================
# SAVE TRIGGER
# ============================================================

def save_trigger(
    source,
    target,
    location,
    patch,
    mask
):

    if patch is None:
        return

    if mask is None:
        return

    patch = patch.squeeze(0)

    mask = mask.squeeze(0)

    effective = (
        patch * mask
    )

    prefix = (
        f"{CLASS_NAMES[source]}"
        f"_to_"
        f"{CLASS_NAMES[target]}"
        f"_{location}"
    )

    torch.save(
        patch,
        os.path.join(
            TRIGGER_DIR,
            prefix + "_patch.pt"
        )
    )

    torch.save(
        mask,
        os.path.join(
            TRIGGER_DIR,
            prefix + "_mask.pt"
        )
    )

    torch.save(
        effective,
        os.path.join(
            TRIGGER_DIR,
            prefix + "_effective.pt"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 80)

    print(
        "SOURCE -> TARGET BACKDOOR DETECTOR V2"
    )

    print("=" * 80)

    print()
    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Images per source:",
        NUM_IMAGES
    )

    print(
        "Iterations:",
        ITERATIONS
    )

    print(
        "Restarts:",
        RESTARTS
    )

    print(
        "Patch size:",
        f"{PATCH_SIZE}x{PATCH_SIZE}"
    )

    print(
        "Locations:",
        len(LOCATIONS)
    )

    # ========================================================
    # LOAD
    # ========================================================

    model = load_model()

    images, labels = load_dataset()

    results = []

    # ========================================================
    # SOURCE CLASSES
    # ========================================================

    for source in range(
        NUM_CLASSES
    ):

        print("\n")
        print("=" * 70)

        print(
            "SOURCE CLASS:",
            CLASS_NAMES[source]
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Correctly classified images
        # ----------------------------------------------------

        correct_images = (
            get_correct_source_images(
                model,
                images,
                labels,
                source
            )
        )

        if (
            len(correct_images)
            < MIN_SOURCE_IMAGES
        ):

            print(
                "Not enough correctly "
                "classified images."
            )

            print(
                "Skipping source:",
                CLASS_NAMES[source]
            )

            continue

        source_images = (
            correct_images.to(
                DEVICE
            )
        )

        print(
            "Using",
            len(source_images),
            "correct source images."
        )

        # ====================================================
        # TARGET CLASSES
        # ====================================================

        for target in range(
            NUM_CLASSES
        ):

            if source == target:
                continue

            print("\n")
            print(
                "-" * 65
            )

            print(
                f"{CLASS_NAMES[source]}"
                f" -> "
                f"{CLASS_NAMES[target]}"
            )

            # ------------------------------------------------
            # Baseline
            # ------------------------------------------------

            (
                baseline,
                baseline_probability
            ) = get_baseline_stats(
                model,
                source_images,
                target
            )

            print(
                f"Baseline success: "
                f"{baseline:.2f}%"
            )

            print(
                f"Baseline probability: "
                f"{baseline_probability:.2f}%"
            )

            best_pair = None

            # =================================================
            # LOCATIONS
            # =================================================

            for location, (
                row,
                col
            ) in LOCATIONS.items():

                print(
                    f"\n    Location: "
                    f"{location}"
                )

                location_best = None

                # =============================================
                # MULTIPLE RESTARTS
                # =============================================

                for restart in range(
                    RESTARTS
                ):

                    print(
                        f"\n      Restart "
                        f"{restart + 1}/"
                        f"{RESTARTS}"
                    )

                    (
                        success,
                        probability,
                        size,
                        score,
                        patch,
                        mask
                    ) = optimize_trigger(
                        model,
                        source_images,
                        target,
                        row,
                        col,
                        restart
                    )

                    print(
                        f"      Result: "
                        f"Success "
                        f"{success:.2f}% | "
                        f"Probability "
                        f"{probability:.2f}% | "
                        f"Size "
                        f"{size:.6f}"
                    )

                    if (
                        location_best
                        is None
                        or success
                        > location_best[
                            "success"
                        ]
                        or (
                            success
                            == location_best[
                                "success"
                            ]
                            and probability
                            > location_best[
                                "probability"
                            ]
                        )
                    ):

                        location_best = {

                            "success":
                                success,

                            "probability":
                                probability,

                            "size":
                                size,

                            "score":
                                score,

                            "patch":
                                patch,

                            "mask":
                                mask
                        }

                # =============================================
                # BEST LOCATION
                # =============================================

                if location_best is not None:

                    print(
                        f"\n    BEST "
                        f"{location}: "
                        f"Success "
                        f"{location_best['success']:.2f}%"
                    )

                    # Compare locations
                    if (
                        best_pair is None
                        or location_best[
                            "success"
                        ]
                        > best_pair[
                            "success"
                        ]
                        or (
                            location_best[
                                "success"
                            ]
                            == best_pair[
                                "success"
                            ]
                            and location_best[
                                "probability"
                            ]
                            > best_pair[
                                "probability"
                            ]
                        )
                    ):

                        best_pair = {

                            "source":
                                source,

                            "target":
                                target,

                            "location":
                                location,

                            "baseline":
                                baseline,

                            "baseline_probability":
                                baseline_probability,

                            "success":
                                location_best[
                                    "success"
                                ],

                            "probability":
                                location_best[
                                    "probability"
                                ],

                            "size":
                                location_best[
                                    "size"
                                ],

                            "score":
                                location_best[
                                    "score"
                                ],

                            "patch":
                                location_best[
                                    "patch"
                                ],

                            "mask":
                                location_best[
                                    "mask"
                                ]
                        }

            # =================================================
            # ATTACK INCREASE
            # =================================================

            increase = (
                best_pair["success"]
                - best_pair["baseline"]
            )

            probability_increase = (
                best_pair["probability"]
                - best_pair[
                    "baseline_probability"
                ]
            )

            best_pair["increase"] = (
                increase
            )

            best_pair[
                "probability_increase"
            ] = probability_increase

            # =================================================
            # DETECTION SCORE
            # =================================================

            # We want:
            #
            # - low baseline
            # - high triggered success
            # - large increase
            # - high target probability
            #
            # This score is NOT used alone for final verdict.
            # It is only for ranking candidates.

            detection_score = (
                max(0, increase)
                + 0.5 * max(
                    0,
                    probability_increase
                )
            )

            # Strong bonus for a high-confidence targeted effect
            if (
                best_pair["success"]
                >= 80
                and increase >= 50
                and best_pair[
                    "baseline"
                ] <= 20
            ):

                detection_score += 50

            elif (
                best_pair["success"]
                >= 70
                and increase >= 40
                and best_pair[
                    "baseline"
                ] <= 20
            ):

                detection_score += 25

            best_pair[
                "detection_score"
            ] = detection_score

            # =================================================
            # SAVE
            # =================================================

            save_trigger(
                source,
                target,
                best_pair["location"],
                best_pair["patch"],
                best_pair["mask"]
            )

            results.append(
                best_pair
            )

            # =================================================
            # PRINT BEST PAIR
            # =================================================

            print("\n")
            print(
                "BEST PAIR:"
            )

            print(
                f"  {CLASS_NAMES[source]}"
                f" -> "
                f"{CLASS_NAMES[target]}"
            )

            print(
                "  Location:",
                best_pair["location"]
            )

            print(
                f"  Baseline:"
                f" {best_pair['baseline']:.2f}%"
            )

            print(
                f"  Triggered:"
                f" {best_pair['success']:.2f}%"
            )

            print(
                f"  Increase:"
                f" {increase:.2f}%"
            )

            print(
                f"  Probability:"
                f" {best_pair['probability']:.2f}%"
            )

            print(
                f"  Trigger size:"
                f" {best_pair['size']:.6f}"
            )

    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda x: (
            x["detection_score"],
            x["increase"],
            x["success"]
        ),
        reverse=True
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n")
    print("=" * 110)

    print(
        "FINAL SOURCE -> TARGET RESULTS V2"
    )

    print("=" * 110)

    print(
        f"{'Source':<12}"
        f"{'Target':<12}"
        f"{'Location':<15}"
        f"{'Base %':<10}"
        f"{'Success %':<12}"
        f"{'Increase':<12}"
        f"{'Prob %':<10}"
        f"{'Size':<10}"
        f"{'Score':<10}"
    )

    print(
        "-" * 110
    )

    for result in results[:20]:

        print(
            f"{CLASS_NAMES[result['source']]:<12}"
            f"{CLASS_NAMES[result['target']]:<12}"
            f"{result['location']:<15}"
            f"{result['baseline']:<10.2f}"
            f"{result['success']:<12.2f}"
            f"{result['increase']:<12.2f}"
            f"{result['probability']:<10.2f}"
            f"{result['size']:<10.4f}"
            f"{result['detection_score']:<10.2f}"
        )

    # ========================================================
    # TOP RESULT
    # ========================================================

    top = results[0]

    print("\n")
    print("=" * 80)

    print(
        "MOST SUSPICIOUS SOURCE -> TARGET PAIR"
    )

    print("=" * 80)

    print(
        "Source:",
        CLASS_NAMES[top["source"]]
    )

    print(
        "Target:",
        CLASS_NAMES[top["target"]]
    )

    print(
        "Location:",
        top["location"]
    )

    print(
        f"Baseline target rate:"
        f" {top['baseline']:.2f}%"
    )

    print(
        f"Triggered target rate:"
        f" {top['success']:.2f}%"
    )

    print(
        f"Increase:"
        f" {top['increase']:.2f}%"
    )

    print(
        f"Baseline probability:"
        f" {top['baseline_probability']:.2f}%"
    )

    print(
        f"Triggered probability:"
        f" {top['probability']:.2f}%"
    )

    print(
        f"Trigger size:"
        f" {top['size']:.6f}"
    )

    print(
        f"Detection score:"
        f" {top['detection_score']:.2f}"
    )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "DETECTOR INTERPRETATION"
    )

    print("=" * 80)

    if (
        top["baseline"] <= 20
        and top["success"] >= 80
        and top["increase"] >= 50
    ):

        verdict = (
            "STRONG BACKDOOR-LIKE "
            "SOURCE -> TARGET EVIDENCE"
        )

        print(
            verdict
        )

        print(
            "A low-baseline source class "
            "was strongly redirected to "
            "a target class."
        )

    elif (
        top["baseline"] <= 30
        and top["success"] >= 70
        and top["increase"] >= 40
    ):

        verdict = (
            "MODERATE BACKDOOR-LIKE "
            "EVIDENCE"
        )

        print(
            verdict
        )

        print(
            "Further validation is recommended."
        )

    else:

        verdict = (
            "NO STRONG BACKDOOR EVIDENCE "
            "FROM SOURCE-TARGET REVERSE "
            "ENGINEERING"
        )

        print(
            verdict
        )

        print(
            "The current reverse-engineering "
            "experiment did not discover a "
            "strong targeted trigger."
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = os.path.join(
        OUTPUT_DIR,
        "source_target_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SOURCE -> TARGET BACKDOOR "
            "DETECTOR V2\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        for result in results:

            f.write(
                f"{CLASS_NAMES[result['source']]} "
                f"-> "
                f"{CLASS_NAMES[result['target']]}\n"
            )

            f.write(
                f"Location: "
                f"{result['location']}\n"
            )

            f.write(
                f"Baseline: "
                f"{result['baseline']:.2f}%\n"
            )

            f.write(
                f"Triggered Success: "
                f"{result['success']:.2f}%\n"
            )

            f.write(
                f"Increase: "
                f"{result['increase']:.2f}%\n"
            )

            f.write(
                f"Baseline Probability: "
                f"{result['baseline_probability']:.2f}%\n"
            )

            f.write(
                f"Triggered Probability: "
                f"{result['probability']:.2f}%\n"
            )

            f.write(
                f"Trigger Size: "
                f"{result['size']:.6f}\n"
            )

            f.write(
                f"Detection Score: "
                f"{result['detection_score']:.4f}\n"
            )

            f.write(
                "\n"
            )

        f.write(
            "\nMOST SUSPICIOUS PAIR\n"
        )

        f.write(
            f"Source: "
            f"{CLASS_NAMES[top['source']]}\n"
        )

        f.write(
            f"Target: "
            f"{CLASS_NAMES[top['target']]}\n"
        )

        f.write(
            f"Location: "
            f"{top['location']}\n"
        )

        f.write(
            f"Baseline: "
            f"{top['baseline']:.2f}%\n"
        )

        f.write(
            f"Triggered Success: "
            f"{top['success']:.2f}%\n"
        )

        f.write(
            f"Increase: "
            f"{top['increase']:.2f}%\n"
        )

        f.write(
            f"Detection Score: "
            f"{top['detection_score']:.4f}\n"
        )

        f.write(
            f"\nVerdict: {verdict}\n"
        )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n")
    print(
        "Summary saved to:"
    )

    print(
        summary_path
    )

    print("\n")
    print(
        "Triggers saved to:"
    )

    print(
        TRIGGER_DIR
    )

    print("\n")
    print(
        "SOURCE -> TARGET DETECTOR V2 "
        "COMPLETED"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()