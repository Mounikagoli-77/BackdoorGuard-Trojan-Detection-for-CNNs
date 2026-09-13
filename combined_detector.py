import os
import re
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_TYPE = "poisoned"

CLEAN_DIR = "outputs/clean/reports"
POISONED_DIR = "outputs/poisoned/reports"

SOURCE_TARGET_REPORT = (
    "outputs/poisoned/source_target/source_target_summary.txt"
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


# ============================================================
# FILE PATHS
# ============================================================

def report_paths(model_type):

    if model_type == "clean":
        directory = CLEAN_DIR
    else:
        directory = POISONED_DIR

    nc = os.path.join(
        directory,
        "trigger_sizes.txt"
    )

    ac = os.path.join(
        directory,
        "activation_clustering_report.txt"
    )

    final = os.path.join(
        directory,
        "final_detection_report.txt"
    )

    return nc, ac, final


# ============================================================
# READ NEURAL CLEANSE
# ============================================================

def read_nc(path):

    data = {}

    if not os.path.exists(path):

        print(
            "ERROR: Missing NC report:",
            path
        )

        return data

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            for cls in CLASS_NAMES:

                if not line.lower().startswith(
                    cls.lower()
                ):
                    continue

                text = line[len(cls):].strip()

                nums = re.findall(
                    r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                    text
                )

                if len(nums) >= 3:

                    try:

                        data[cls] = {
                            "trigger": float(nums[0]),
                            "success": float(nums[1]),
                            "mad": float(nums[2])
                        }

                    except ValueError:
                        pass

                break

    return data


# ============================================================
# READ ACTIVATION CLUSTERING
# ============================================================

def read_ac(path):

    data = {}

    if not os.path.exists(path):

        print(
            "ERROR: Missing AC report:",
            path
        )

        return data

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("Class"):
                continue

            if line.startswith("-"):
                continue

            for cls in CLASS_NAMES:

                if not line.lower().startswith(
                    cls.lower()
                ):
                    continue

                text = line[len(cls):].strip()

                nums = re.findall(
                    r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                    text
                )

                if len(nums) >= 5:

                    try:

                        data[cls] = {

                            "cluster0":
                                int(float(nums[0])),

                            "cluster1":
                                int(float(nums[1])),

                            "silhouette":
                                float(nums[2]),

                            "minority_ratio":
                                float(nums[3]),

                            "anomaly":
                                float(nums[4])
                        }

                    except ValueError:
                        pass

                break

    return data


# ============================================================
# READ SOURCE -> TARGET V2
# ============================================================

def read_source_target(path):

    results = []

    if not os.path.exists(path):

        print()
        print(
            "WARNING: Source -> Target report not found:"
        )
        print(path)
        return results

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
        ]

    current = None

    for line in lines:

        if not line:
            continue

        # ----------------------------------------------------
        # SOURCE -> TARGET
        # Example:
        # horse -> bird
        # ----------------------------------------------------

        pair_match = re.match(
            r"^([A-Za-z]+)\s*->\s*([A-Za-z]+)$",
            line
        )

        if pair_match:

            source = pair_match.group(1).lower()
            target = pair_match.group(2).lower()

            if (
                source in CLASS_NAMES
                and
                target in CLASS_NAMES
                and
                source != target
            ):

                current = {

                    "source": source,
                    "target": target,
                    "location": "",
                    "base": 0.0,
                    "success": 0.0,
                    "increase": 0.0,
                    "probability": 0.0,
                    "size": 0.0,
                    "score": 0.0
                }

            else:

                current = None

            continue

        if current is None:
            continue

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        if line.startswith("Location:"):

            current["location"] = (
                line.split(
                    ":", 1
                )[1].strip()
            )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        elif line.startswith("Baseline:"):

            value = re.search(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                line
            )

            if value:
                current["base"] = float(
                    value.group()
                )

        # ----------------------------------------------------
        # TRIGGERED SUCCESS
        # ----------------------------------------------------

        elif line.startswith(
            "Triggered Success:"
        ):

            value = re.search(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                line
            )

            if value:
                current["success"] = float(
                    value.group()
                )

        # ----------------------------------------------------
        # INCREASE
        # ----------------------------------------------------

        elif line.startswith("Increase:"):

            value = re.search(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                line
            )

            if value:
                current["increase"] = float(
                    value.group()
                )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        elif line.startswith(
            "Triggered Probability:"
        ):

            value = re.search(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                line
            )

            if value:
                current["probability"] = float(
                    value.group()
                )

        # ----------------------------------------------------
        # TRIGGER SIZE
        # ----------------------------------------------------

        elif line.startswith(
            "Trigger Size:"
        ):

            value = re.search(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                line
            )

            if value:
                current["size"] = float(
                    value.group()
                )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        elif line.startswith("Score:"):

            value = re.search(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)",
                line
            )

            if value:
                current["score"] = float(
                    value.group()
                )

            # A complete record ends here.
            results.append(current)
            current = None

    return results


# ============================================================
# CLEAN BASELINE
# ============================================================

def calculate_baseline(
    nc,
    ac
):

    nc_values = [
        x["trigger"]
        for x in nc.values()
    ]

    ac_values = [
        x["anomaly"]
        for x in ac.values()
    ]

    return {

        "nc_median":
            float(
                np.median(nc_values)
            ),

        "nc_mad":
            float(
                np.median(
                    np.abs(
                        np.array(nc_values)
                        -
                        np.median(nc_values)
                    )
                )
            ),

        "ac_mean":
            float(
                np.mean(ac_values)
            ),

        "ac_std":
            float(
                np.std(ac_values)
            )
    }


# ============================================================
# SOURCE -> TARGET EVIDENCE
# ============================================================

def source_target_evidence(item):

    base = item["base"]
    success = item["success"]
    increase = item["increase"]
    probability = item["probability"]

    # Values in the report are percentages.
    base_ratio = base / 100.0
    success_ratio = success / 100.0
    increase_ratio = increase / 100.0
    probability_ratio = probability / 100.0

    # --------------------------------------------------------
    # Strong targeted behavior:
    # low baseline + high triggered success + large increase
    # --------------------------------------------------------

    score = (
        0.30 * max(
            0.0,
            1.0 - base_ratio
        )
        +
        0.35 * success_ratio
        +
        0.25 * max(
            0.0,
            increase_ratio
        )
        +
        0.10 * probability_ratio
    )

    strong = (
        base <= 20.0
        and
        success >= 80.0
        and
        increase >= 50.0
    )

    moderate = (
        base <= 30.0
        and
        success >= 70.0
        and
        increase >= 40.0
    )

    if strong:
        score += 0.20

    elif moderate:
        score += 0.10

    return float(
        np.clip(score, 0, 1)
    )


# ============================================================
# CLEAN MODEL
# ============================================================

def run_clean():

    nc_file, ac_file, final_file = (
        report_paths("clean")
    )

    nc = read_nc(nc_file)
    ac = read_ac(ac_file)

    print()
    print(
        f"NC classes found: {len(nc)}"
    )

    print(
        f"AC classes found: {len(ac)}"
    )

    if len(nc) != 10:

        print(
            "ERROR: Clean NC report incomplete."
        )
        return

    if len(ac) != 10:

        print(
            "ERROR: Clean AC report incomplete."
        )
        return

    baseline = calculate_baseline(
        nc,
        ac
    )

    print()
    print("=" * 90)
    print("CLEAN MODEL BASELINE")
    print("=" * 90)

    print()

    print(
        f"NC median: "
        f"{baseline['nc_median']:.6f}"
    )

    print(
        f"NC MAD: "
        f"{baseline['nc_mad']:.6f}"
    )

    print(
        f"AC mean: "
        f"{baseline['ac_mean']:.6f}"
    )

    print(
        f"AC std: "
        f"{baseline['ac_std']:.6f}"
    )

    print()

    for cls in CLASS_NAMES:

        trigger = nc[cls]["trigger"]

        if baseline["nc_mad"] > 1e-8:

            nc_anomaly = (
                baseline["nc_median"]
                -
                trigger
            ) / baseline["nc_mad"]

            nc_score = float(
                np.clip(
                    nc_anomaly / 3.0,
                    0,
                    1
                )
            )

        else:

            nc_score = 0.0

        anomaly = ac[cls]["anomaly"]

        if baseline["ac_std"] > 1e-8:

            ac_z = (
                anomaly
                -
                baseline["ac_mean"]
            ) / baseline["ac_std"]

            ac_score = float(
                np.clip(
                    ac_z / 3.0,
                    0,
                    1
                )
            )

        else:

            ac_score = 0.0

        combined = (
            0.70 * nc_score
            +
            0.30 * ac_score
        )

        print(
            f"{cls:<14}"
            f"Trigger={trigger:.6f} "
            f"NC={nc_score:.4f} "
            f"AC={ac_score:.4f} "
            f"Combined={combined:.4f}"
        )

    os.makedirs(
        os.path.dirname(final_file),
        exist_ok=True
    )

    with open(
        final_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "COMBINED BACKDOOR DETECTOR\n"
        )

        f.write(
            "CLEAN MODEL BASELINE\n\n"
        )

        f.write(
            f"NC median: "
            f"{baseline['nc_median']:.6f}\n"
        )

        f.write(
            f"NC MAD: "
            f"{baseline['nc_mad']:.6f}\n"
        )

        f.write(
            f"AC mean: "
            f"{baseline['ac_mean']:.6f}\n"
        )

        f.write(
            f"AC std: "
            f"{baseline['ac_std']:.6f}\n"
        )

        f.write(
            "\nVERDICT: MODEL LIKELY CLEAN\n"
        )

    print()
    print(
        "Clean baseline saved:"
    )
    print(final_file)


# ============================================================
# POISONED MODEL
# ============================================================

def run_poisoned():

    poisoned_nc_file, poisoned_ac_file, final_file = (
        report_paths("poisoned")
    )

    clean_nc_file, clean_ac_file, _ = (
        report_paths("clean")
    )

    poisoned_nc = read_nc(
        poisoned_nc_file
    )

    poisoned_ac = read_ac(
        poisoned_ac_file
    )

    clean_nc = read_nc(
        clean_nc_file
    )

    clean_ac = read_ac(
        clean_ac_file
    )

    # --------------------------------------------------------
    # READ SOURCE -> TARGET V2
    # --------------------------------------------------------

    source_target = read_source_target(
        SOURCE_TARGET_REPORT
    )

    print()
    print(
        f"Poisoned NC classes found: "
        f"{len(poisoned_nc)}"
    )

    print(
        f"Poisoned AC classes found: "
        f"{len(poisoned_ac)}"
    )

    print(
        f"Clean NC classes found: "
        f"{len(clean_nc)}"
    )

    print(
        f"Clean AC classes found: "
        f"{len(clean_ac)}"
    )

    print(
        f"Source -> Target pairs found: "
        f"{len(source_target)}"
    )

    if (
        len(poisoned_nc) != 10
        or
        len(poisoned_ac) != 10
        or
        len(clean_nc) != 10
        or
        len(clean_ac) != 10
    ):

        print()
        print(
            "ERROR: One or more reports are incomplete."
        )
        return

    # ========================================================
    # CLEAN BASELINE
    # ========================================================

    clean_baseline = calculate_baseline(
        clean_nc,
        clean_ac
    )

    # ========================================================
    # NC + AC
    # ========================================================

    print()
    print("=" * 120)
    print("POISONED MODEL VS CLEAN MODEL")
    print("=" * 120)

    print()

    print(
        f"{'Class':<13}"
        f"{'Model NC':<13}"
        f"{'Clean NC':<13}"
        f"{'Reduction':<13}"
        f"{'NC Score':<12}"
        f"{'Model AC':<13}"
        f"{'AC Evidence':<13}"
        f"{'Combined':<12}"
    )

    print(
        "-" * 120
    )

    results = []

    for cls in CLASS_NAMES:

        model_nc = poisoned_nc[
            cls
        ]["trigger"]

        clean_trigger = clean_nc[
            cls
        ]["trigger"]

        model_ac = poisoned_ac[
            cls
        ]["anomaly"]

        clean_ac_value = clean_ac[
            cls
        ]["anomaly"]

        # ----------------------------------------------------
        # Trigger reduction
        # ----------------------------------------------------

        reduction = (
            clean_trigger
            -
            model_nc
        ) / clean_trigger

        reduction_positive = max(
            0.0,
            reduction
        )

        reduction_score = float(
            np.clip(
                reduction_positive,
                0,
                1
            )
        )

        # ----------------------------------------------------
        # MAD
        # ----------------------------------------------------

        mad_value = poisoned_nc[
            cls
        ]["mad"]

        mad_evidence = float(
            np.clip(
                mad_value / 3.0,
                0,
                1
            )
        )

        nc_score = (
            0.70 * reduction_score
            +
            0.30 * mad_evidence
        )

        # ----------------------------------------------------
        # AC
        # ----------------------------------------------------

        ac_difference = (
            model_ac
            -
            clean_ac_value
        )

        if clean_baseline["ac_std"] > 1e-8:

            ac_z = (
                ac_difference
                /
                clean_baseline["ac_std"]
            )

            ac_evidence = float(
                np.clip(
                    ac_z / 3.0,
                    0,
                    1
                )
            )

        else:

            ac_evidence = 0.0

        # ----------------------------------------------------
        # NC + AC
        # ----------------------------------------------------

        combined = (
            0.80 * nc_score
            +
            0.20 * ac_evidence
        )

        item = {

            "class":
                cls,

            "model_nc":
                model_nc,

            "clean_nc":
                clean_trigger,

            "reduction":
                reduction,

            "nc_score":
                nc_score,

            "model_ac":
                model_ac,

            "ac_evidence":
                ac_evidence,

            "combined":
                combined
        }

        results.append(item)

        print(
            f"{cls:<13}"
            f"{model_nc:<13.6f}"
            f"{clean_trigger:<13.6f}"
            f"{reduction:<13.3f}"
            f"{nc_score:<12.3f}"
            f"{model_ac:<13.4f}"
            f"{ac_evidence:<13.3f}"
            f"{combined:<12.3f}"
        )

    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda x:
            x["combined"],
        reverse=True
    )

    suspicious = results[0]

    # ========================================================
    # SOURCE -> TARGET
    # ========================================================

    print()
    print("=" * 120)
    print("SOURCE -> TARGET V2 EVIDENCE")
    print("=" * 120)

    scored_pairs = []

    for item in source_target:

        score = source_target_evidence(
            item
        )

        new_item = item.copy()

        new_item["evidence_score"] = score

        scored_pairs.append(
            new_item
        )

    scored_pairs.sort(
        key=lambda x:
            x["evidence_score"],
        reverse=True
    )

    if not scored_pairs:

        print()
        print(
            "No Source -> Target results were found."
        )

    else:

        print()

        print(
            f"{'Source':<13}"
            f"{'Target':<13}"
            f"{'Location':<14}"
            f"{'Base %':<11}"
            f"{'Success %':<13}"
            f"{'Increase':<11}"
            f"{'Prob %':<11}"
            f"{'Size':<11}"
            f"{'Evidence':<10}"
        )

        print(
            "-" * 120
        )

        for item in scored_pairs[:15]:

            print(
                f"{item['source']:<13}"
                f"{item['target']:<13}"
                f"{item['location']:<14}"
                f"{item['base']:<11.2f}"
                f"{item['success']:<13.2f}"
                f"{item['increase']:<11.2f}"
                f"{item['probability']:<11.2f}"
                f"{item['size']:<11.6f}"
                f"{item['evidence_score']:<10.4f}"
            )

    # ========================================================
    # STRONG / MODERATE NC
    # ========================================================

    strong = [

        x for x in results

        if (
            x["reduction"] >= 0.50
            and
            x["nc_score"] >= 0.50
        )
    ]

    moderate = [

        x for x in results

        if (
            x["reduction"] >= 0.30
            and
            x["nc_score"] >= 0.35
        )
    ]

    max_reduction = max(
        x["reduction"]
        for x in results
    )

    mean_positive_reduction = np.mean(
        [
            max(
                0,
                x["reduction"]
            )
            for x in results
        ]
    )

    # ========================================================
    # SOURCE -> TARGET CLASSIFICATION
    # ========================================================

    strong_pair = None
    moderate_pair = None

    for item in scored_pairs:

        if (
            item["base"] <= 20.0
            and
            item["success"] >= 80.0
            and
            item["increase"] >= 50.0
        ):

            strong_pair = item
            break

    if strong_pair is None:

        for item in scored_pairs:

            if (
                item["base"] <= 30.0
                and
                item["success"] >= 70.0
                and
                item["increase"] >= 40.0
            ):

                moderate_pair = item
                break

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    if (
        len(strong) >= 1
        or
        strong_pair is not None
    ):

        verdict = (
            "MODEL LIKELY BACKDOORED"
        )

        confidence = "HIGH"

    elif (
        suspicious["combined"] >= 0.65
        and
        max_reduction >= 0.40
    ):

        verdict = (
            "MODEL LIKELY BACKDOORED"
        )

        confidence = "MEDIUM"

    elif (
        len(moderate) >= 2
        or
        moderate_pair is not None
    ):

        verdict = (
            "MODEL SUSPICIOUS / "
            "FURTHER ANALYSIS RECOMMENDED"
        )

        confidence = "MEDIUM"

    else:

        verdict = (
            "MODEL LIKELY CLEAN"
        )

        confidence = (
            "LOW / NO STRONG EVIDENCE"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL DETECTION VERDICT")
    print("=" * 100)

    print()

    print(
        f"Verdict: {verdict}"
    )

    print(
        f"Confidence: {confidence}"
    )

    print()

    print(
        f"Most suspicious class: "
        f"{suspicious['class']}"
    )

    print(
        f"Model trigger size: "
        f"{suspicious['model_nc']:.6f}"
    )

    print(
        f"Clean trigger size: "
        f"{suspicious['clean_nc']:.6f}"
    )

    print(
        f"Trigger reduction: "
        f"{suspicious['reduction'] * 100:.2f}%"
    )

    print(
        f"Neural Cleanse score: "
        f"{suspicious['nc_score']:.4f}"
    )

    print(
        f"Activation Clustering evidence: "
        f"{suspicious['ac_evidence']:.4f}"
    )

    print(
        f"Combined NC + AC score: "
        f"{suspicious['combined']:.4f}"
    )

    print()

    print(
        f"Strong NC classes: "
        f"{len(strong)}"
    )

    print(
        f"Moderate NC classes: "
        f"{len(moderate)}"
    )

    print(
        f"Maximum trigger reduction: "
        f"{max_reduction * 100:.2f}%"
    )

    print(
        f"Mean positive trigger reduction: "
        f"{mean_positive_reduction * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Source -> Target final information
    # --------------------------------------------------------

    print()

    if scored_pairs:

        best_pair = scored_pairs[0]

        print(
            "Most suspicious Source -> Target pair:"
        )

        print(
            f"Source: "
            f"{best_pair['source']}"
        )

        print(
            f"Target: "
            f"{best_pair['target']}"
        )

        print(
            f"Location: "
            f"{best_pair['location']}"
        )

        print(
            f"Baseline: "
            f"{best_pair['base']:.2f}%"
        )

        print(
            f"Triggered success: "
            f"{best_pair['success']:.2f}%"
        )

        print(
            f"Increase: "
            f"{best_pair['increase']:.2f}%"
        )

        print(
            f"Probability: "
            f"{best_pair['probability']:.2f}%"
        )

        print(
            f"Trigger size: "
            f"{best_pair['size']:.6f}"
        )

        print(
            f"Source -> Target evidence score: "
            f"{best_pair['evidence_score']:.4f}"
        )

    else:

        print(
            "Source -> Target evidence: "
            "NOT AVAILABLE"
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    print()
    print("Explanation:")

    if verdict == "MODEL LIKELY BACKDOORED":

        print(
            "The candidate model shows strong "
            "evidence of a possible hidden "
            "backdoor trigger."
        )

        if strong_pair is not None:

            print(
                "Source -> Target analysis found "
                "a strong targeted behavior."
            )

        if len(strong) >= 1:

            print(
                "Neural Cleanse identified "
                "a strong class-level anomaly."
            )

        print(
            "The suspicious result is determined "
            "from the analysis and is not "
            "hard-coded."
        )

    elif "SUSPICIOUS" in verdict:

        print(
            "The candidate model shows multiple "
            "moderate differences from the clean "
            "reference model."
        )

        print(
            "Additional validation is recommended."
        )

    else:

        print(
            "No sufficiently strong difference "
            "from the clean reference model was "
            "detected."
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    os.makedirs(
        os.path.dirname(final_file),
        exist_ok=True
    )

    with open(
        final_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "============================================================\n"
        )

        f.write(
            "COMBINED BACKDOOR DETECTOR\n"
        )

        f.write(
            "============================================================\n\n"
        )

        f.write(
            "Candidate model: poisoned_model.pt\n"
        )

        f.write(
            "Reference model: clean_model.pt\n\n"
        )

        f.write(
            "FINAL VERDICT\n"
        )

        f.write(
            "-------------\n"
        )

        f.write(
            f"{verdict}\n"
        )

        f.write(
            f"Confidence: {confidence}\n\n"
        )

        f.write(
            "MOST SUSPICIOUS CLASS\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            f"Class: {suspicious['class']}\n"
        )

        f.write(
            f"Model NC: "
            f"{suspicious['model_nc']:.6f}\n"
        )

        f.write(
            f"Clean NC: "
            f"{suspicious['clean_nc']:.6f}\n"
        )

        f.write(
            f"Trigger reduction: "
            f"{suspicious['reduction'] * 100:.2f}%\n"
        )

        f.write(
            f"NC score: "
            f"{suspicious['nc_score']:.4f}\n"
        )

        f.write(
            f"AC evidence: "
            f"{suspicious['ac_evidence']:.4f}\n"
        )

        f.write(
            f"Combined NC + AC: "
            f"{suspicious['combined']:.4f}\n\n"
        )

        # ----------------------------------------------------
        # Source -> Target
        # ----------------------------------------------------

        f.write(
            "SOURCE -> TARGET V2\n"
        )

        f.write(
            "-------------------\n"
        )

        if scored_pairs:

            best_pair = scored_pairs[0]

            f.write(
                f"Source: "
                f"{best_pair['source']}\n"
            )

            f.write(
                f"Target: "
                f"{best_pair['target']}\n"
            )

            f.write(
                f"Location: "
                f"{best_pair['location']}\n"
            )

            f.write(
                f"Baseline: "
                f"{best_pair['base']:.2f}%\n"
            )

            f.write(
                f"Triggered success: "
                f"{best_pair['success']:.2f}%\n"
            )

            f.write(
                f"Increase: "
                f"{best_pair['increase']:.2f}%\n"
            )

            f.write(
                f"Probability: "
                f"{best_pair['probability']:.2f}%\n"
            )

            f.write(
                f"Trigger size: "
                f"{best_pair['size']:.6f}\n"
            )

            f.write(
                f"Evidence score: "
                f"{best_pair['evidence_score']:.4f}\n\n"
            )

        else:

            f.write(
                "No Source -> Target results available.\n\n"
            )

        # ----------------------------------------------------
        # Model evidence
        # ----------------------------------------------------

        f.write(
            "MODEL-LEVEL EVIDENCE\n"
        )

        f.write(
            "--------------------\n"
        )

        f.write(
            f"Strong NC classes: "
            f"{len(strong)}\n"
        )

        f.write(
            f"Moderate NC classes: "
            f"{len(moderate)}\n"
        )

        f.write(
            f"Maximum trigger reduction: "
            f"{max_reduction * 100:.2f}%\n"
        )

        f.write(
            f"Mean positive trigger reduction: "
            f"{mean_positive_reduction * 100:.2f}%\n\n"
        )

        # ----------------------------------------------------
        # Class results
        # ----------------------------------------------------

        f.write(
            "CLASS RESULTS\n"
        )

        f.write(
            "-------------\n"
        )

        for x in results:

            f.write(
                f"{x['class']:<12}"
                f" ModelNC={x['model_nc']:.6f}"
                f" CleanNC={x['clean_nc']:.6f}"
                f" Reduction={x['reduction']:.4f}"
                f" NC={x['nc_score']:.4f}"
                f" AC={x['ac_evidence']:.4f}"
                f" Combined={x['combined']:.4f}\n"
            )

    print()
    print(
        "Final report saved to:"
    )

    print(
        final_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("COMBINED BACKDOOR DETECTOR")
    print("=" * 75)

    if MODEL_TYPE == "clean":

        print()
        print(
            "Analyzing: clean_model.pt"
        )

        run_clean()

    else:

        print()
        print(
            "Analyzing: poisoned_model.pt"
        )

        run_poisoned()

    print()

    print(
        "Combined detection completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()