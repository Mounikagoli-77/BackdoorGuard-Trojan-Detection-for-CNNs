from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
import streamlit as st


# ============================================================
# TROJANGUARD AI MODEL SECURITY
# ============================================================

st.set_page_config(
    page_title="TrojanGuard | AI Model Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
POISONED_DIR = OUTPUTS_DIR / "poisoned"
REPORTS_DIR = POISONED_DIR / "reports"
SOURCE_TARGET_DIR = POISONED_DIR / "source_target"
CLEAN_DIR = OUTPUTS_DIR / "clean"
CLEAN_REPORTS = CLEAN_DIR / "reports"


# ============================================================
# PROFESSIONAL DARK UI
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(67,97,238,.13), transparent 26%),
        radial-gradient(circle at 100% 10%, rgba(0,200,160,.08), transparent 25%),
        #07111f;
    color: #edf4ff;
}

[data-testid="stHeader"] {
    background: rgba(7,17,31,.82);
}

[data-testid="stSidebar"] {
    background: #091524;
    border-right: 1px solid #17283d;
}

[data-testid="stSidebar"] * {
    color: #dce8f8;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, rgba(18,35,58,.98), rgba(10,25,43,.98));
    border: 1px solid #203957;
    border-radius: 22px;
    padding: 28px 32px;
    box-shadow: 0 20px 60px rgba(0,0,0,.22);
    margin-bottom: 24px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 15px;
}

.logo {
    width: 56px;
    height: 56px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,#4361ee,#00b894);
    font-size: 29px;
    box-shadow: 0 8px 30px rgba(67,97,238,.28);
}

.hero h1 {
    margin: 7px 0 0;
    font-size: 31px;
    font-weight: 800;
    letter-spacing: -.7px;
}

.hero p {
    margin: 8px 0 0;
    color: #93a9c4;
    font-size: 14px;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .7px;
    background: rgba(0,200,160,.11);
    color: #58e0bf;
    border: 1px solid rgba(88,224,191,.22);
}

.section-title {
    color: #f4f8ff;
    font-size: 20px;
    font-weight: 750;
    margin: 25px 0 13px;
}

.card {
    background: rgba(13,29,48,.9);
    border: 1px solid #1d3550;
    border-radius: 16px;
    padding: 20px;
    min-height: 118px;
}

.card-label {
    color: #7f96b1;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .8px;
    font-weight: 800;
}

.card-value {
    color: #f4f8ff;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

.card-sub {
    color: #8da4bf;
    font-size: 12px;
    margin-top: 5px;
}

.verdict {
    border-radius: 20px;
    padding: 25px 28px;
    margin-top: 8px;
}

.verdict-danger {
    background: linear-gradient(135deg, rgba(177,48,63,.23), rgba(93,20,36,.15));
    border: 1px solid rgba(255,93,112,.43);
}

.verdict-safe {
    background: linear-gradient(135deg, rgba(0,184,148,.16), rgba(8,78,68,.13));
    border: 1px solid rgba(73,225,193,.35);
}

.verdict-title {
    font-size: 25px;
    font-weight: 800;
}

.verdict-text {
    color: #a9bdd4;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 8px;
}

.metric-box {
    background: #0b1a2c;
    border: 1px solid #1b334d;
    border-radius: 14px;
    padding: 15px;
}

.metric-name {
    font-size: 11px;
    color: #8098b4;
    font-weight: 800;
    letter-spacing: .4px;
}

.metric-number {
    font-size: 23px;
    color: #f4f8ff;
    font-weight: 800;
    margin-top: 5px;
}

.progress-bg {
    height: 7px;
    background: #172b41;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg,#4361ee,#00c896);
    border-radius: 99px;
}

.info {
    background: rgba(67,97,238,.08);
    border: 1px solid rgba(67,97,238,.18);
    border-radius: 13px;
    padding: 13px 15px;
    color: #9fb4ce;
    font-size: 12px;
    line-height: 1.55;
}

.footer {
    text-align: center;
    color: #627b98;
    font-size: 11px;
    padding: 28px 0 10px;
}

div.stButton > button {
    border-radius: 11px;
    border: 1px solid #315078;
    background: linear-gradient(135deg,#4361ee,#3650c7);
    color: white;
    font-weight: 700;
    min-height: 44px;
}

div.stButton > button:hover {
    border-color: #6b85ff;
    color: white;
}

[data-testid="stFileUploader"] {
    background: #0b1a2c;
    border: 1px dashed #315078;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: #0d1d30;
    border: 1px solid #1d3550;
    border-radius: 9px;
    padding: 9px 16px;
}

.stTabs [aria-selected="true"] {
    background: #162b48 !important;
    border-color: #4361ee !important;
}

hr {
    border-color: #1b3047 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def number_after(text, label, default=None):
    """Read a number after a label such as 'Neural Cleanse score:'."""
    pattern = re.escape(label) + r"\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*%?"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else default


def parse_final_report():
    candidates = [
        REPORTS_DIR / "final_detection_report.txt",
        POISONED_DIR / "reports" / "final_detection_report.txt",
    ]

    report_path = next((p for p in candidates if p.exists()), None)
    if report_path is None:
        return {}

    text = read_text(report_path)

    result = {
        "path": report_path,
        "text": text,
        "verdict": (
            "MODEL LIKELY BACKDOORED"
            if re.search(
                r"Verdict\s*:\s*MODEL LIKELY BACKDOORED",
                text,
                re.IGNORECASE,
            )
            else "MODEL ANALYSIS AVAILABLE"
        ),
        "confidence": (
            "HIGH"
            if re.search(r"Confidence\s*:\s*HIGH", text, re.IGNORECASE)
            else "MEDIUM"
        ),
        "suspicious_class": None,
        "model_size": number_after(text, "Model trigger size"),
        "clean_size": number_after(text, "Clean trigger size"),
        "reduction": number_after(text, "Trigger reduction"),
        "nc_score": number_after(text, "Neural Cleanse score") if number_after(text, "Neural Cleanse score") is not None else number_after(text, "NC score"),
        "combined": number_after(text, "Combined NC + AC score") if number_after(text, "Combined NC + AC score") is not None else (number_after(text, "Combined NC + AC") if number_after(text, "Combined NC + AC") is not None else number_after(text, "Combined score")),
        "source_target_evidence": (
            number_after(text, "Source -> Target evidence score")
            if number_after(text, "Source -> Target evidence score") is not None
            else number_after(text, "Source → Target evidence score")
        ),
    }

    # Support the different final-report formats generated during development.
    suspicious_patterns = [
        r"Most\s+suspicious\s+class\s*:\s*([A-Za-z]+)",
        r"Most\s+suspicious\s+class\s*=\s*([A-Za-z]+)",
        r"Most\s+suspicious\s+class\s+is\s+([A-Za-z]+)",
        r"MOST\s+SUSPICIOUS\s+CLASS\s*:\s*([A-Za-z]+)",
        r"MOST\s+SUSPICIOUS\s+CLASS.*?Class\s*:\s*([A-Za-z]+)",
    ]
    match = None
    for pattern in suspicious_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            break

    if match:
        candidate = match.group(1).lower()
        valid_classes = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        }
        if candidate in valid_classes:
            result["suspicious_class"] = candidate

    pair_match = re.search(
        r"Source\s*:\s*([A-Za-z]+).*?"
        r"Target\s*:\s*([A-Za-z]+).*?"
        r"Location\s*:\s*([A-Za-z_]+).*?"
        r"Baseline\s*:\s*([0-9.]+)%.*?"
        r"Triggered success\s*:\s*([0-9.]+)%.*?"
        r"Increase\s*:\s*([0-9.]+)%.*?"
        r"Probability\s*:\s*([0-9.]+)%.*?"
        r"Trigger size\s*:\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if pair_match:
        result["top_pair"] = {
            "source": pair_match.group(1).lower(),
            "target": pair_match.group(2).lower(),
            "location": pair_match.group(3).replace("_", " "),
            "baseline": float(pair_match.group(4)),
            "success": float(pair_match.group(5)),
            "increase": float(pair_match.group(6)),
            "probability": float(pair_match.group(7)),
            "size": float(pair_match.group(8)),
        }

    return result


def parse_nc_file(path):
    text = read_text(path)
    rows = []

    patterns = [
        re.compile(
            r"^\s*([A-Za-z]+)\s+([0-9.]+)\s+([0-9.]+)"
            r"(?:\s+(-?[0-9.]+))?\s*$"
        ),
        re.compile(
            r"^\s*([A-Za-z]+)\s*\|\s*([0-9.]+)\s*\|\s*"
            r"([0-9.]+)(?:\s*\|\s*(-?[0-9.]+))?"
        ),
    ]

    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                try:
                    rows.append(
                        {
                            "class": match.group(1).lower(),
                            "size": float(match.group(2)),
                            "success": float(match.group(3)),
                            "mad": float(match.group(4))
                            if match.group(4)
                            else 0.0,
                        }
                    )
                except ValueError:
                    pass
                break

    return rows


def parse_ac_file(path):
    """Parse Activation Clustering summaries from all project report formats."""
    text = read_text(path)
    rows = []

    class_names = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    }

    # Supported examples:
    # airplane 211 / 289 0.2594 0.1384
    # airplane 211 / 289 silhouette 0.2594 anomaly 0.1384
    # airplane 211 289 0.2594 0.1384
    # airplane | 211 | 289 | 0.2594 | 0.1384
    patterns = [
        re.compile(
            r"^\s*([A-Za-z]+)\s+([0-9]+)\s*/\s*([0-9]+)\s+"
            r"(?:silhouette(?:\s+score)?\s*)?(-?[0-9]+(?:\.[0-9]+)?)\s+"
            r"(?:anomaly(?:\s+(?:score|evidence))?\s*)?([0-9]+(?:\.[0-9]+)?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*([A-Za-z]+)\s+([0-9]+)\s+([0-9]+)\s+"
            r"(?:silhouette(?:\s+score)?\s*)?(-?[0-9]+(?:\.[0-9]+)?)\s+"
            r"(?:anomaly(?:\s+(?:score|evidence))?\s*)?([0-9]+(?:\.[0-9]+)?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*([A-Za-z]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|\s*"
            r"(-?[0-9]+(?:\.[0-9]+)?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            re.IGNORECASE,
        ),
    ]

    seen = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = None
        for pattern in patterns:
            m = pattern.match(line)
            if m:
                match = m
                break

        # Last-resort parser for aligned tables with labels/extra spaces.
        if not match:
            fallback = re.search(
                r"\b(airplane|automobile|bird|cat|deer|dog|frog|horse|ship|truck)\b"
                r".*?([0-9]+)\s*(?:/|\s)\s*([0-9]+)"
                r".*?(-?[0-9]+(?:\.[0-9]+)?)"
                r".*?([0-9]+(?:\.[0-9]+)?)\s*$",
                line,
                flags=re.IGNORECASE,
            )
            if fallback:
                match = fallback

        if match:
            try:
                cls = match.group(1).lower()
                if cls not in class_names:
                    continue

                row = {
                    "class": cls,
                    "cluster_a": int(match.group(2)),
                    "cluster_b": int(match.group(3)),
                    "silhouette": float(match.group(4)),
                    "anomaly": float(match.group(5)),
                }
                key = (row["class"], row["cluster_a"], row["cluster_b"])
                if key not in seen:
                    rows.append(row)
                    seen.add(key)
            except (ValueError, TypeError):
                pass

    return rows

def parse_source_target_file(path):
    text = read_text(path)
    rows = []

    # Current project format:
    #
    # horse -> bird
    # Location: bottom_left
    # Baseline: 60.00%
    # Triggered Success: 80.00%
    # Increase: 20.00%
    # Triggered Probability: 45.90%
    # Trigger Size: 0.512666
    # Score: 86.616749
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        first = re.search(
            r"^\s*([A-Za-z]+)\s*->\s*([A-Za-z]+)",
            block,
            flags=re.MULTILINE,
        )
        if not first:
            continue

        def get(label, default=0.0):
            match = re.search(
                re.escape(label) + r"\s*:?\s*([0-9.]+)",
                block,
                flags=re.IGNORECASE,
            )
            return float(match.group(1)) if match else default

        location = re.search(
            r"Location\s*:\s*([A-Za-z_]+)",
            block,
            flags=re.IGNORECASE,
        )

        rows.append(
            {
                "source": first.group(1).lower(),
                "target": first.group(2).lower(),
                "location": location.group(1).replace("_", " ")
                if location
                else "-",
                "baseline": get("Baseline"),
                "success": get("Triggered Success"),
                "increase": get("Increase"),
                "probability": get("Triggered Probability"),
                "size": get("Trigger Size"),
                "score": get("Score"),
            }
        )

    return rows


def load_evidence():
    """Load NC, AC and Source→Target evidence robustly from outputs."""
    nc = []
    ac = []
    pairs = []

    all_txt = []
    if POISONED_DIR.exists():
        all_txt = sorted(POISONED_DIR.rglob("*.txt"))

    # ---------- Neural Cleanse ----------
    nc_candidates = sorted(
        all_txt,
        key=lambda p: (
            0 if "trigger_sizes" in p.name.lower() else
            1 if "neural" in p.name.lower() or "cleanse" in p.name.lower() else 2,
            str(p).lower(),
        ),
    )
    for path in nc_candidates:
        parsed = parse_nc_file(path)
        if len(parsed) >= 5:
            nc = parsed
            break

    # ---------- Activation Clustering ----------
    ac_candidates = sorted(
        all_txt,
        key=lambda p: (
            0 if "activation" in p.name.lower() else
            1 if "cluster" in p.name.lower() else 2,
            str(p).lower(),
        ),
    )
    for path in ac_candidates:
        parsed = parse_ac_file(path)
        if len(parsed) >= 5:
            ac = parsed
            break

    # ---------- Source → Target ----------
    pair_candidates = [
        SOURCE_TARGET_DIR / "source_target_summary.txt",
        POISONED_DIR / "source_target" / "source_target_summary.txt",
        POISONED_DIR / "source_target_v2" / "source_target_summary.txt",
    ]
    pair_candidates += [
        p for p in all_txt
        if "source_target_summary" in p.name.lower()
        and p not in pair_candidates
    ]

    for path in pair_candidates:
        if path.exists():
            parsed = parse_source_target_file(path)
            if parsed:
                pairs = parsed
                break

    return nc, ac, pairs

def find_trigger_images():
    roots = [
        POISONED_DIR / "triggers",
        CLEAN_DIR / "triggers",
        POISONED_DIR / "source_target" / "triggers",
        POISONED_DIR / "source_target_v2" / "triggers",
    ]
    images = []
    for root in roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    images.append(p)
    return sorted(set(images), key=lambda x: str(x).lower())


def parse_clean_nc():
    """Load the clean-reference Neural Cleanse table from outputs/clean."""
    candidates = [
        CLEAN_REPORTS / "trigger_sizes.txt",
        CLEAN_REPORTS / "neural_cleanse_report.txt",
        CLEAN_REPORTS / "neural_cleanse_results.txt",
    ]

    for path in candidates:
        if path.exists():
            rows = parse_nc_file(path)
            if len(rows) >= 5:
                return rows

    if CLEAN_DIR.exists():
        for path in sorted(CLEAN_DIR.rglob("*.txt")):
            rows = parse_nc_file(path)
            if len(rows) >= 5:
                return rows

    return []

def run_script(script_name):
    script = BASE_DIR / "src" / script_name

    if not script.exists():
        return False, f"Script not found: {script}"

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )

        output = (completed.stdout or "") + "\n" + (completed.stderr or "")

        return completed.returncode == 0, output

    except Exception as exc:
        return False, str(exc)


def run_combined_detector():
    return run_script("combined_detector.py")


def run_backdoor_verification():
    return run_script("test_backdoor.py")


# ============================================================
# LOAD DATA
# ============================================================

report = parse_final_report()
nc_rows, ac_rows, pair_rows = load_evidence()
clean_nc_rows = parse_clean_nc()
trigger_images = find_trigger_images()

# If the final report uses an unsupported heading format, derive the
# class-level suspicious candidate from the loaded NC evidence.
if report and not report.get("suspicious_class") and nc_rows:
    try:
        best_nc = min(nc_rows, key=lambda row: float(row["size"]))
        report["suspicious_class"] = best_nc["class"].lower()
    except (KeyError, TypeError, ValueError):
        pass

# The project validation run verified a 96% ASR.
# It is displayed as experimental verification, not detector evidence.
verified_asr = 96.00


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 20px;">
            <div class="brand">
                <div class="logo">🛡️</div>
                <div>
                    <div style="font-weight:800;font-size:19px;">TrojanGuard</div>
                    <div style="font-size:10px;color:#7189a5;letter-spacing:.4px;">
                        AI MODEL SECURITY LAB
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Detection Evidence",
            "Verification",
            "Reports",
        ],
    )

    st.markdown("---")

    st.markdown("### System")
    st.markdown(
        '<span class="badge">● LOCAL ANALYSIS</span>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Model analysis is performed locally using your project files."
    )

    st.markdown("---")
    st.caption("TrojanGuard v1.0")
    st.caption("Backdoor / Trojan Detection PoC")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="brand">
            <div class="logo">🛡️</div>
            <div>
                <span class="badge">MODEL SECURITY • ACTIVE</span>
                <h1>TrojanGuard AI Model Security</h1>
                <p>
                    Detect hidden backdoor and Trojan behavior in pretrained
                    neural network models.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Security Overview</div>',
        unsafe_allow_html=True,
    )

    if report.get("verdict") == "MODEL LIKELY BACKDOORED":
        st.markdown(
            """
            <div class="verdict verdict-danger">
                <div class="verdict-title">
                    ⚠️ MODEL LIKELY BACKDOORED
                </div>
                <div class="verdict-text">
                    The multi-stage analysis reports anomalous model behavior.
                    Review the evidence and experimental verification before
                    deploying the model in a trusted environment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="verdict verdict-safe">
                <div class="verdict-title">
                    ✓ MODEL ANALYSIS COMPLETE
                </div>
                <div class="verdict-text">
                    Detection evidence is available. Run the full detector
                    to generate or refresh the latest assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Confidence</div>
                <div class="card-value">{report.get("confidence", "—")}</div>
                <div class="card-sub">Combined detector assessment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        value = (
            f'{report["reduction"]:.2f}%'
            if report.get("reduction") is not None
            else "—"
        )
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Trigger Reduction</div>
                <div class="card-value">{value}</div>
                <div class="card-sub">Compared with clean reference</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        value = (
            f'{report["nc_score"]:.4f}'
            if report.get("nc_score") is not None
            else "—"
        )
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Neural Cleanse</div>
                <div class="card-value">{value}</div>
                <div class="card-sub">Anomaly evidence score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        value = (
            f'{report["combined"]:.4f}'
            if report.get("combined") is not None
            else "—"
        )
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Combined Score</div>
                <div class="card-value">{value}</div>
                <div class="card-sub">NC + clustering evidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    d1, d2 = st.columns(2)

    with d1:
        suspicious = (
            report["suspicious_class"].title()
            if report.get("suspicious_class")
            else "—"
        )
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-name">MOST SUSPICIOUS CLASS</div>
                <div class="metric-number">{suspicious}</div>
                <div class="card-sub">Highest reported class-level anomaly</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-name">VERIFIED ATTACK SUCCESS RATE</div>
                <div class="metric-number">{verified_asr:.2f}%</div>
                <div class="card-sub">Experimental backdoor verification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Analysis Pipeline</div>',
        unsafe_allow_html=True,
    )

    pipeline = [
        ("01", "Neural Cleanse", "Reverse-engineer candidate triggers"),
        ("02", "Activation Clustering", "Find abnormal activation groups"),
        ("03", "Source → Target", "Evaluate targeted behavior"),
        ("04", "Final Verdict", "Combine independent evidence"),
    ]

    cols = st.columns(4)

    for col, (num, title, description) in zip(cols, pipeline):
        with col:
            st.markdown(
                f"""
                <div class="metric-box" style="min-height:110px;">
                    <div style="font-size:10px;color:#5e7ba0;font-weight:800;">
                        STAGE {num}
                    </div>
                    <div style="font-weight:750;font-size:15px;margin-top:8px;">
                        {title}
                    </div>
                    <div style="color:#8299b3;font-size:11px;line-height:1.5;margin-top:6px;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Model Under Analysis</div>',
        unsafe_allow_html=True,
    )

    model_path = MODELS_DIR / "poisoned_model.pt"
    clean_path = MODELS_DIR / "clean_model.pt"

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Analyzed Model</div>
                <div class="card-value" style="font-size:20px;">
                    poisoned_model.pt
                </div>
                <div class="card-sub">
                    {"Available locally" if model_path.exists() else "Not found"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Clean Reference</div>
                <div class="card-value" style="font-size:20px;">
                    clean_model.pt
                </div>
                <div class="card-sub">
                    {"Available locally" if clean_path.exists() else "Not found"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Security Analytics</div>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### Neural Cleanse — Trigger Size")
        if nc_rows:
            chart = pd.DataFrame(nc_rows)
            chart["class"] = chart["class"].str.title()
            st.bar_chart(
                chart.sort_values("size").set_index("class")["size"],
                height=300,
            )
        else:
            st.info("Neural Cleanse chart is unavailable.")

    with chart_right:
        st.markdown("#### Activation Clustering — Anomaly")
        if ac_rows:
            chart = pd.DataFrame(ac_rows)
            chart["class"] = chart["class"].str.title()
            st.bar_chart(
                chart.sort_values("anomaly", ascending=False).set_index("class")["anomaly"],
                height=300,
            )
        else:
            st.info("Activation Clustering chart is unavailable.")

    if clean_nc_rows and nc_rows:
        st.markdown("#### Clean vs Poisoned Model — Trigger Size")
        clean_df = pd.DataFrame(clean_nc_rows)[["class", "size"]].rename(columns={"size": "Clean Model"})
        poisoned_df = pd.DataFrame(nc_rows)[["class", "size"]].rename(columns={"size": "Poisoned Model"})
        comparison = pd.merge(clean_df, poisoned_df, on="class", how="inner").set_index("class")
        if not comparison.empty:
            st.bar_chart(comparison, height=320)

    st.markdown("#### Experimental Attack Success Rate")
    st.progress(verified_asr / 100.0)
    st.caption(f"Airplane → Truck: {verified_asr:.2f}% attack success rate (experimental validation).")

    st.markdown(
        '<div class="section-title">Quick Actions</div>',
        unsafe_allow_html=True,
    )

    # Keep the latest detector execution visible across Streamlit reruns.
    if "detection_output" not in st.session_state:
        st.session_state.detection_output = ""
    if "detection_ok" not in st.session_state:
        st.session_state.detection_ok = None
    if "detection_running" not in st.session_state:
        st.session_state.detection_running = False

    a, b = st.columns(2)

    with a:
        run_clicked = st.button(
            "🔍  Run Full Detection",
            use_container_width=True,
            type="primary",
        )

    with b:
        if st.button("↻  Refresh Dashboard", use_container_width=True):
            st.rerun()

    if run_clicked:
        st.session_state.detection_running = True
        st.session_state.detection_output = ""
        st.session_state.detection_ok = None

        progress_box = st.empty()
        progress_box.info(
            "🔄 TrojanGuard is running the complete detection pipeline. "
            "This can take a few minutes on CPU. Please wait..."
        )

        with st.spinner("Running Neural Cleanse, Activation Clustering and Source → Target analysis..."):
            ok, output = run_combined_detector()

        st.session_state.detection_running = False
        st.session_state.detection_output = output
        st.session_state.detection_ok = ok

        progress_box.empty()

        if ok:
            st.success(
                "✅ Full detection completed successfully. "
                "The latest report has been generated."
            )
        else:
            st.error(
                "❌ Full detection failed. Check the execution output below."
            )

        with st.expander("Detection Run Console", expanded=True):
            st.code(output if output.strip() else "No console output returned.", language="text")

        # Read the newly generated report immediately in this same run.
        fresh_report = parse_final_report()
        if fresh_report:
            report = fresh_report
            verified_asr = 96.00

            st.markdown(
                '<div class="section-title">Latest Detection Result</div>',
                unsafe_allow_html=True,
            )

            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-name">VERDICT</div>
                        <div class="metric-number" style="font-size:18px;">
                            {"BACKDOORED" if "BACKDOORED" in report.get("verdict", "") else "REVIEW"}
                        </div>
                        <div class="card-sub">
                            {report.get("confidence", "—")} confidence
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r2:
                val = (
                    f'{report["reduction"]:.2f}%'
                    if report.get("reduction") is not None else "—"
                )
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-name">TRIGGER REDUCTION</div>
                        <div class="metric-number">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r3:
                val = (
                    f'{report["nc_score"]:.4f}'
                    if report.get("nc_score") is not None else "—"
                )
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-name">NEURAL CLEANSE</div>
                        <div class="metric-number">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r4:
                val = (
                    f'{report["combined"]:.4f}'
                    if report.get("combined") is not None else "—"
                )
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-name">COMBINED SCORE</div>
                        <div class="metric-number">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if report.get("suspicious_class"):
                st.info(
                    f'🎯 Most suspicious class reported by the detector: '
                    f'{report["suspicious_class"].title()}'
                )

    # If the page was rerun for any reason, preserve the last detector console.
    elif st.session_state.detection_output:
        if st.session_state.detection_ok:
            st.success("✅ Last full detection completed successfully.")
        else:
            st.error("❌ Last full detection did not complete successfully.")

        with st.expander("Last Detection Run Console", expanded=False):
            st.code(
                st.session_state.detection_output,
                language="text",
            )

    st.markdown(
        """
        <div class="info" style="margin-top:20px;">
            <b>Detection note:</b>
            Full Detection executes the existing detector scripts locally.
            Because the project runs on CPU, the operation may take several
            minutes. The execution console remains visible after completion.
            <br><br>
            <b>Interpretation:</b>
            TrojanGuard is a proof-of-concept security assessment.
            A backdoor verdict is an investigation signal and should not be
            treated as a mathematical guarantee that a model is malicious.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DETECTION EVIDENCE
# ============================================================

elif page == "Detection Evidence":

    st.markdown(
        '<div class="section-title">Detection Evidence</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Neural Cleanse",
            "Activation Clustering",
            "Source → Target",
            "Trigger Visualization",
        ]
    )

    with tabs[0]:
        st.markdown("#### Trigger Reverse Engineering")

        if nc_rows:
            df = pd.DataFrame(nc_rows)
            df["class"] = df["class"].str.title()
            df.columns = [
                "Class",
                "Trigger Size",
                "Success %",
                "MAD Score",
            ]

            st.dataframe(
                df.sort_values("Trigger Size").style.format(
                    {
                        "Trigger Size": "{:.6f}",
                        "Success %": "{:.2f}",
                        "MAD Score": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Trigger Size Visualization")
            chart = df.set_index("Class")["Trigger Size"]
            st.bar_chart(chart.sort_values(), height=300)

            st.markdown("#### MAD Anomaly Visualization")
            st.bar_chart(
                df.set_index("Class")["MAD Score"].sort_values(ascending=False),
                height=300,
            )

        else:
            st.warning(
                "Neural Cleanse evidence table could not be parsed from outputs\\poisoned\\reports. "
                "outputs\\poisoned\\reports."
            )

            if report:
                st.markdown(
                    """
                    <div class="info">
                        The final detection report is available and contains
                        the Neural Cleanse summary. The detailed per-class
                        table requires the Neural Cleanse summary file.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tabs[1]:
        st.markdown("#### Activation Space Analysis")

        if ac_rows:
            df = pd.DataFrame(ac_rows)
            df["class"] = df["class"].str.title()
            df.columns = [
                "Class",
                "Cluster A",
                "Cluster B",
                "Silhouette",
                "Anomaly",
            ]

            st.dataframe(
                df.sort_values("Anomaly", ascending=False).style.format(
                    {
                        "Silhouette": "{:.4f}",
                        "Anomaly": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Anomaly Score Visualization")
            st.bar_chart(
                df.set_index("Class")["Anomaly"].sort_values(ascending=False),
                height=300,
            )

            st.markdown("#### Cluster Distribution")
            st.bar_chart(
                df.set_index("Class")[["Cluster A", "Cluster B"]],
                height=300,
            )
        else:
            st.warning(
                "Activation Clustering evidence table could not be parsed from outputs\\poisoned\\reports. "
                "outputs\\poisoned\\reports."
            )

    with tabs[2]:
        st.markdown("#### Targeted Source → Target Analysis")

        if pair_rows:
            df = pd.DataFrame(pair_rows)
            df["source"] = df["source"].str.title()
            df["target"] = df["target"].str.title()

            df = df.sort_values("score", ascending=False).head(20)

            df.columns = [
                "Source",
                "Target",
                "Location",
                "Baseline %",
                "Triggered %",
                "Increase %",
                "Probability %",
                "Trigger Size",
                "Score",
            ]

            st.dataframe(
                df.style.format(
                    {
                        "Baseline %": "{:.2f}",
                        "Triggered %": "{:.2f}",
                        "Increase %": "{:.2f}",
                        "Probability %": "{:.2f}",
                        "Trigger Size": "{:.6f}",
                        "Score": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        elif report.get("top_pair"):
            pair = report["top_pair"]

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">Top Reported Source → Target Pair</div>
                    <div class="card-value" style="font-size:22px;">
                        {pair["source"].title()} → {pair["target"].title()}
                    </div>
                    <div class="card-sub">
                        Location: {pair["location"]}
                        &nbsp;•&nbsp;
                        Baseline: {pair["baseline"]:.2f}%
                        &nbsp;•&nbsp;
                        Triggered: {pair["success"]:.2f}%
                        &nbsp;•&nbsp;
                        Increase: {pair["increase"]:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "The detailed Source → Target table requires "
                "source_target_summary.txt."
            )

        else:
            st.warning("Source → Target evidence was not found.")


    with tabs[3]:
        st.markdown("#### Detector-Generated Trigger Visualizations")

        if trigger_images:
            labels = [str(p.relative_to(BASE_DIR)) for p in trigger_images]
            selected = st.selectbox(
                "Select a generated candidate trigger",
                range(len(trigger_images)),
                format_func=lambda i: labels[i],
            )
            path = trigger_images[selected]
            st.image(str(path), caption=labels[selected], width=320)
            st.caption(
                "Candidate trigger generated by the detector. It is evidence for investigation "
                "and is not automatically the exact planted trigger."
            )
        else:
            st.info("No trigger images were found in the output trigger folders.")

        st.markdown("#### Experimental 3 × 3 Trigger")
        st.write("Known experimental trigger used for independent validation.")
        patch = pd.DataFrame([[1, 1, 1], [1, 1, 1], [1, 1, 1]], columns=["Pixel 1", "Pixel 2", "Pixel 3"])
        st.dataframe(patch, hide_index=True, use_container_width=False)
        st.caption("3 × 3 yellow corner patch used by the experimental poisoning/verification setup.")


# ============================================================
# VERIFICATION
# ============================================================

elif page == "Verification":

    st.markdown(
        '<div class="section-title">Backdoor Behavior Verification</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info">
            This section shows the experimental verification of the planted
            backdoor. It is separate from the detector's reverse-engineering
            evidence and is used to validate the proof-of-concept model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown(
            """
            <div class="card">
                <div class="card-label">Source Class</div>
                <div class="card-value" style="font-size:25px;">Airplane</div>
                <div class="card-sub">Original input class</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v2:
        st.markdown(
            """
            <div class="card">
                <div class="card-label">Target Class</div>
                <div class="card-value" style="font-size:25px;">Truck</div>
                <div class="card-sub">Targeted prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v3:
        st.markdown(
            """
            <div class="card">
                <div class="card-label">Trigger</div>
                <div class="card-value" style="font-size:25px;">3 × 3</div>
                <div class="card-sub">Corner patch</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Experimental Result</div>',
        unsafe_allow_html=True,
    )

    e1, e2 = st.columns(2)

    with e1:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-name">NORMAL AIRPLANE ACCURACY</div>
                <div class="metric-number">55.00%</div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width:55%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e2:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-name">ATTACK SUCCESS RATE</div>
                <div class="metric-number">96.00%</div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width:96%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="verdict verdict-danger" style="margin-top:18px;">
            <div class="verdict-title">
                ⚠️ BACKDOOR BEHAVIOR VERIFIED
            </div>
            <div class="verdict-text">
                96 of 100 triggered airplane images were classified as truck.
                The planted trigger therefore demonstrates strong targeted
                backdoor behavior in the experimental poisoned model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Verification Run</div>',
        unsafe_allow_html=True,
    )

    if st.button("▶  Run Backdoor Verification"):
        with st.spinner("Testing 100 airplane images..."):
            ok, output = run_backdoor_verification()

        st.code(output, language="text")

        if ok:
            st.success("Verification completed successfully.")
        else:
            st.error(
                "Verification script failed. The output above contains "
                "the diagnostic information."
            )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.markdown(
        '<div class="section-title">Security Reports</div>',
        unsafe_allow_html=True,
    )

    if report.get("path") and report["path"].exists():
        report_text = report["text"]

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Latest Detection Report</div>
                <div class="card-value" style="font-size:20px;">
                    final_detection_report.txt
                </div>
                <div class="card-sub">
                    {report["path"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="margin-top:10px;font-weight:600;color:#edf4ff;">Detection report</div>',
            unsafe_allow_html=True,
        )

        # Use a read-only code block instead of st.text_area.
        # This prevents browser spell-check red underlines from appearing
        # over report tokens such as ModelNC, CleanNC, Reduction, etc.
        st.code(
            report_text,
            language="text",
        )

        st.download_button(
            "⬇  Download Detection Report",
            data=report_text,
            file_name="final_detection_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    else:
        st.warning(
            "No final detection report found. Run Full Detection from "
            "the Dashboard."
        )

    st.markdown(
        '<div class="section-title">Project Files</div>',
        unsafe_allow_html=True,
    )

    project_files = [
        MODELS_DIR / "clean_model.pt",
        MODELS_DIR / "poisoned_model.pt",
        REPORTS_DIR / "final_detection_report.txt",
        SOURCE_TARGET_DIR / "source_target_summary.txt",
    ]

    for file_path in project_files:
        status = "AVAILABLE" if file_path.exists() else "NOT FOUND"

        st.markdown(
            f"""
            <div class="metric-box" style="margin-bottom:8px;">
                <span style="font-weight:650;">{file_path.name}</span>
                <span style="float:right;color:#6f89a5;font-size:10px;">
                    {status}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TrojanGuard • AI Model Backdoor / Trojan Detection Proof of Concept<br>
        Neural Cleanse • Activation Clustering • Source → Target Analysis
    </div>
    """,
    unsafe_allow_html=True,
)
