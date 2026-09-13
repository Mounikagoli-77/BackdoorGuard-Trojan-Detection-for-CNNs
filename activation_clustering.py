import os
import sys
import numpy as np

import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.model import CIFAR10CNN


# ============================================================
# CONFIGURATION - CLEAN MODEL
# ============================================================

MODEL_PATH = "models/clean_model.pt"

# IMPORTANT:
# We do NOT use:
# data/clean_model/images.npy
#
# Clean model is analyzed using the normal CIFAR-10
# test dataset.

OUTPUT_DIR = "outputs/clean/clusters"
PLOT_DIR = "outputs/clean/plots"
REPORT_DIR = "outputs/clean/reports"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PARAMETERS
# ============================================================

NUM_CLASSES = 10

SAMPLES_PER_CLASS = 500

PCA_COMPONENTS = 10

NUM_CLUSTERS = 2

RANDOM_STATE = 42

BATCH_SIZE = 64


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
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ACTIVATION CLUSTERING - CLEAN MODEL")
print("=" * 70)


# ============================================================
# LOAD CLEAN MODEL
# ============================================================

print("\nLoading model:")
print(MODEL_PATH)


model = CIFAR10CNN().to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ============================================================
# SUPPORT DIFFERENT CHECKPOINT FORMATS
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

elif (
    isinstance(checkpoint, dict)
    and all(
        isinstance(v, torch.Tensor)
        for v in checkpoint.values()
    )
):

    model.load_state_dict(
        checkpoint
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


print("Model loaded successfully.")

print(
    "Device:",
    DEVICE
)


# ============================================================
# LOAD CIFAR-10 TEST DATASET
# ============================================================

print("\nLoading CIFAR-10 test dataset...")


# IMPORTANT:
#
# Your model was trained using ToTensor().
#
# Therefore we use:
#
# transforms.ToTensor()
#
# NO CIFAR normalization.

transform = transforms.ToTensor()


dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


print(
    "Dataset size:",
    len(dataset)
)


# ============================================================
# CONVERT CIFAR-10 TO NUMPY
# ============================================================

images = np.stack(
    [
        dataset[i][0].numpy()
        for i in range(len(dataset))
    ]
)


labels = np.array(
    [
        dataset[i][1]
        for i in range(len(dataset))
    ]
)


print(
    "Image shape:",
    images.shape
)

print(
    "Labels shape:",
    labels.shape
)


# ============================================================
# ACTIVATION STORAGE
# ============================================================

activation_storage = []


# ============================================================
# HOOK FUNCTION
# ============================================================

def activation_hook(
    module,
    input,
    output
):

    # Output from AdaptiveAvgPool:
    #
    # [batch, 128, 1, 1]
    #
    # Convert to:
    #
    # [batch, 128]

    output = torch.flatten(
        output,
        start_dim=1
    )


    activation_storage.append(
        output.detach().cpu()
    )


# ============================================================
# REGISTER HOOK
# ============================================================

hook = model.features.register_forward_hook(
    activation_hook
)


# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_id in range(NUM_CLASSES):

    class_name = CLASS_NAMES[class_id]


    print("\n" + "-" * 70)

    print(
        f"Processing class {class_id}: "
        f"{class_name}"
    )

    print("-" * 70)


    # ========================================================
    # GET CLASS INDICES
    # ========================================================

    class_indices = np.where(
        labels == class_id
    )[0]


    if len(class_indices) == 0:

        print(
            "No samples found."
        )

        continue


    # ========================================================
    # SELECT MAXIMUM 500 SAMPLES
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_STATE + class_id
    )


    if len(class_indices) > SAMPLES_PER_CLASS:

        class_indices = rng.choice(
            class_indices,
            SAMPLES_PER_CLASS,
            replace=False
        )


    # ========================================================
    # PREPARE IMAGES
    # ========================================================

    class_images = torch.tensor(
        images[class_indices],
        dtype=torch.float32
    )


    print(
        "Samples:",
        len(class_images)
    )


    # ========================================================
    # CLEAR ACTIVATION STORAGE
    # ========================================================

    activation_storage.clear()


    # ========================================================
    # EXTRACT ACTIVATIONS
    # ========================================================

    with torch.no_grad():

        for start in range(
            0,
            len(class_images),
            BATCH_SIZE
        ):

            end = (
                start
                +
                BATCH_SIZE
            )


            batch = class_images[
                start:end
            ].to(DEVICE)


            model(batch)


    # ========================================================
    # CHECK ACTIVATIONS
    # ========================================================

    if len(activation_storage) == 0:

        print(
            "Activation extraction failed."
        )

        continue


    # ========================================================
    # COMBINE ACTIVATIONS
    # ========================================================

    activations = torch.cat(
        activation_storage,
        dim=0
    ).numpy()


    print(
        "Activation shape:",
        activations.shape
    )


    # ========================================================
    # PCA
    # ========================================================

    n_components = min(
        PCA_COMPONENTS,
        activations.shape[0] - 1,
        activations.shape[1]
    )


    if n_components < 2:

        print(
            "Not enough dimensions for PCA."
        )

        continue


    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE
    )


    reduced = pca.fit_transform(
        activations
    )


    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )


    print(
        f"PCA explained variance: "
        f"{explained_variance:.4f}"
    )


    # ========================================================
    # K-MEANS
    # ========================================================

    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20
    )


    cluster_labels = kmeans.fit_predict(
        reduced
    )


    # ========================================================
    # CLUSTER COUNTS
    # ========================================================

    cluster_counts = np.bincount(
        cluster_labels,
        minlength=NUM_CLUSTERS
    )


    print(
        f"Clusters: "
        f"{cluster_counts[0]} / "
        f"{cluster_counts[1]}"
    )


    # ========================================================
    # MINORITY CLUSTER
    # ========================================================

    sorted_counts = np.sort(
        cluster_counts
    )[::-1]


    largest_cluster = (
        sorted_counts[0]
    )


    smallest_cluster = (
        sorted_counts[-1]
    )


    minority_ratio = (
        smallest_cluster
        /
        len(cluster_labels)
    )


    # ========================================================
    # SILHOUETTE SCORE
    # ========================================================

    if (
        len(
            np.unique(
                cluster_labels
            )
        )
        >
        1
    ):

        silhouette = silhouette_score(
            reduced,
            cluster_labels
        )

    else:

        silhouette = 0.0


    # ========================================================
    # ACTIVATION CLUSTERING ANOMALY SCORE
    # ========================================================
    #
    # Higher silhouette:
    #   stronger separation
    #
    # Smaller minority cluster:
    #   stronger possible anomaly
    #
    # This is a relative score, not a final verdict.
    # ========================================================

    anomaly_score = (
        max(
            0.0,
            silhouette
        )
        *
        (
            1.0
            -
            minority_ratio
        )
    )


    print(
        f"Silhouette: "
        f"{silhouette:.4f}"
    )


    print(
        f"Minority ratio: "
        f"{minority_ratio:.4f}"
    )


    print(
        f"Anomaly score: "
        f"{anomaly_score:.4f}"
    )


    # ========================================================
    # SAVE CLUSTER LABELS
    # ========================================================

    cluster_file = os.path.join(
        OUTPUT_DIR,
        f"{class_id}_"
        f"{class_name}_clusters.npy"
    )


    np.save(
        cluster_file,
        cluster_labels
    )


    # ========================================================
    # SAVE PCA DATA
    # ========================================================

    pca_file = os.path.join(
        OUTPUT_DIR,
        f"{class_id}_"
        f"{class_name}_pca.npy"
    )


    np.save(
        pca_file,
        reduced
    )


    # ========================================================
    # CREATE CLUSTER PLOT
    # ========================================================

    plt.figure(
        figsize=(7, 6)
    )


    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=cluster_labels,
        alpha=0.7
    )


    plt.xlabel(
        "Principal Component 1"
    )


    plt.ylabel(
        "Principal Component 2"
    )


    plt.title(
        f"Activation Clustering - "
        f"{class_name}\n"
        f"Silhouette={silhouette:.4f}, "
        f"Anomaly={anomaly_score:.4f}"
    )


    plt.tight_layout()


    plot_file = os.path.join(
        PLOT_DIR,
        f"{class_id}_"
        f"{class_name}_clustering.png"
    )


    plt.savefig(
        plot_file,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = {

        "class_id":
            class_id,

        "class_name":
            class_name,

        "cluster_0":
            int(
                cluster_counts[0]
            ),

        "cluster_1":
            int(
                cluster_counts[1]
            ),

        "silhouette":
            float(
                silhouette
            ),

        "minority_ratio":
            float(
                minority_ratio
            ),

        "anomaly_score":
            float(
                anomaly_score
            ),

        "explained_variance":
            float(
                explained_variance
            )
    }


    all_results.append(
        result
    )


# ============================================================
# REMOVE HOOK
# ============================================================

hook.remove()


# ============================================================
# SORT RESULTS
# ============================================================

all_results = sorted(
    all_results,
    key=lambda x:
        x["anomaly_score"],
    reverse=True
)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = os.path.join(
    REPORT_DIR,
    "activation_clustering_report.txt"
)


with open(
    report_file,
    "w"
) as f:

    f.write(
        "ACTIVATION CLUSTERING - CLEAN MODEL\n"
    )

    f.write(
        "=" * 70
        +
        "\n"
    )


    f.write(
        f"Model: "
        f"{MODEL_PATH}\n"
    )


    f.write(
        f"Dataset: CIFAR-10 test\n"
    )


    f.write(
        f"Samples per class: "
        f"{SAMPLES_PER_CLASS}\n"
    )


    f.write(
        f"PCA components: "
        f"{PCA_COMPONENTS}\n"
    )


    f.write(
        f"KMeans clusters: "
        f"{NUM_CLUSTERS}\n\n"
    )


    f.write(
        "Class\t"
        "Cluster0\t"
        "Cluster1\t"
        "Silhouette\t"
        "MinorityRatio\t"
        "AnomalyScore\n"
    )


    f.write(
        "-" * 90
        +
        "\n"
    )


    for result in all_results:

        f.write(
            f"{result['class_name']}\t"
            f"{result['cluster_0']}\t"
            f"{result['cluster_1']}\t"
            f"{result['silhouette']:.4f}\t"
            f"{result['minority_ratio']:.4f}\t"
            f"{result['anomaly_score']:.4f}\n"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")

print("=" * 70)

print(
    "FINAL ACTIVATION CLUSTERING RESULTS"
)

print("=" * 70)


print(
    f"{'Class':<15}"
    f"{'Cluster 0':<12}"
    f"{'Cluster 1':<12}"
    f"{'Silhouette':<14}"
    f"{'Anomaly':<12}"
)


print("-" * 70)


for result in all_results:

    print(
        f"{result['class_name']:<15}"
        f"{result['cluster_0']:<12}"
        f"{result['cluster_1']:<12}"
        f"{result['silhouette']:<14.4f}"
        f"{result['anomaly_score']:<12.4f}"
    )


# ============================================================
# MOST SUSPICIOUS CLASS
# ============================================================

if len(all_results) > 0:

    suspicious = all_results[0]


    print("\n" + "=" * 70)

    print(
        "MOST SUSPICIOUS CLASS"
    )

    print("=" * 70)


    print(
        f"Class: "
        f"{suspicious['class_name']}"
    )


    print(
        f"Anomaly score: "
        f"{suspicious['anomaly_score']:.4f}"
    )


    print(
        f"Silhouette: "
        f"{suspicious['silhouette']:.4f}"
    )


    print(
        f"Clusters: "
        f"{suspicious['cluster_0']} / "
        f"{suspicious['cluster_1']}"
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

print("\nReports saved to:")

print(
    OUTPUT_DIR
)

print(
    PLOT_DIR
)

print(
    REPORT_DIR
)


print(
    "\nActivation clustering completed."
)