import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "outputs/triggers/trigger_sizes.txt"

OUTPUT_FILE = "outputs/reports/anomaly_results.txt"

PLOT_FILE = "outputs/plots/trigger_size_analysis.png"

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
# LOAD TRIGGER SIZES
# ============================================================

print("=" * 60)
print("MAD-BASED BACKDOOR ANOMALY DETECTION")
print("=" * 60)

data = []

with open(INPUT_FILE, "r") as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        parts = line.split(",")

        class_id = int(parts[0])

        class_name = parts[1]

        size = float(parts[2])

        data.append(
            [class_id, class_name, size]
        )


sizes = np.array(
    [row[2] for row in data]
)


# ============================================================
# MEDIAN
# ============================================================

median = np.median(sizes)


# ============================================================
# MEDIAN ABSOLUTE DEVIATION
# ============================================================

absolute_deviation = np.abs(
    sizes - median
)

mad = np.median(
    absolute_deviation
)


# ============================================================
# AVOID ZERO MAD
# ============================================================

if mad < 1e-8:

    mad = 1e-8


# ============================================================
# ROBUST Z-SCORE
# ============================================================

# Smaller trigger size = more suspicious
#
# Robust score:
#
# (median - trigger_size) / MAD
#
# Large positive value means unusually small trigger.

scores = (
    median - sizes
) / mad


# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD = 2.0


# ============================================================
# RESULTS
# ============================================================

print("\nTrigger Size Analysis")
print("-" * 60)

for i, row in enumerate(data):

    class_id = row[0]

    class_name = row[1]

    size = row[2]

    score = scores[i]

    status = (
        "SUSPICIOUS"
        if score > THRESHOLD
        else "Normal"
    )

    print(
        f"{class_name:12s} | "
        f"Size: {size:.6f} | "
        f"Score: {score:.3f} | "
        f"{status}"
    )


# ============================================================
# FIND MOST SUSPICIOUS CLASS
# ============================================================

suspicious_index = np.argmax(scores)

suspicious_class = data[
    suspicious_index
][1]

suspicious_size = data[
    suspicious_index
][2]

suspicious_score = scores[
    suspicious_index
]


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 60)

print(
    f"Median Trigger Size: "
    f"{median:.6f}"
)

print(
    f"MAD: "
    f"{mad:.6f}"
)

print(
    f"Most Suspicious Class: "
    f"{suspicious_class}"
)

print(
    f"Trigger Size: "
    f"{suspicious_size:.6f}"
)

print(
    f"Anomaly Score: "
    f"{suspicious_score:.3f}"
)

print("=" * 60)


if suspicious_score > THRESHOLD:

    print(
        "\n🚨 POTENTIAL BACKDOOR DETECTED"
    )

    print(
        f"Suspicious target class: "
        f"{suspicious_class}"
    )

else:

    print(
        "\n🟢 NO STRONG TRIGGER OUTLIER DETECTED"
    )

    print(
        "The trigger sizes do not show a "
        "strong statistical outlier."
    )


# ============================================================
# SAVE REPORT
# ============================================================

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w"
) as file:

    file.write(
        "MAD-BASED BACKDOOR DETECTION REPORT\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Median Trigger Size: {median:.6f}\n"
    )

    file.write(
        f"MAD: {mad:.6f}\n\n"
    )

    for i, row in enumerate(data):

        status = (
            "SUSPICIOUS"
            if scores[i] > THRESHOLD
            else "Normal"
        )

        file.write(
            f"{row[1]},"
            f"{row[2]:.8f},"
            f"{scores[i]:.4f},"
            f"{status}\n"
        )

    file.write(
        f"\nMost Suspicious Class: "
        f"{suspicious_class}\n"
    )

    file.write(
        f"Anomaly Score: "
        f"{suspicious_score:.4f}\n"
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

os.makedirs(
    "outputs/plots",
    exist_ok=True
)

names = [
    row[1]
    for row in data
]

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    names,
    sizes
)

plt.axhline(
    median,
    linestyle="--",
    label="Median"
)

plt.xlabel(
    "Target Class"
)

plt.ylabel(
    "Optimized Trigger Size"
)

plt.title(
    "Neural Cleanse Trigger Size Analysis"
)

plt.xticks(
    rotation=45
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=150
)

plt.close()


print(
    f"\nReport saved: {OUTPUT_FILE}"
)

print(
    f"Plot saved: {PLOT_FILE}"
)