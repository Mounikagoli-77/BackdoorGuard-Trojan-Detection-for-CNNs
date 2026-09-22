import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TrojanGuard AI Model Security",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# PROJECT PATHS AND CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "poisoned" / "reports"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #071525;
        color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background: #091a2d;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .header,
    .card,
    .info {
        background: #102943;
        border: 1px solid #2a4c70;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .header h1 {
        margin: 0;
        color: #ffffff;
        font-size: 30px;
    }

    .header p {
        color: #a9c9f5;
        margin-bottom: 0;
    }

    .card .label {
        color: #9fc7f7;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .card .value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        margin-top: 14px;
        overflow-wrap: anywhere;
    }

    .card .help {
        color: #8caed3;
        font-size: 12px;
        margin-top: 8px;
    }

    .danger {
        background: #421d2c;
        border: 1px solid #d74c6a;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 20px;
    }

    .danger h2 {
        color: #ffb4c2;
        margin: 0;
    }

    .danger p {
        color: #f2b9c5;
        margin-bottom: 0;
    }

    .success-box {
        background: #103e37;
        border: 1px solid #1a9d78;
        border-radius: 12px;
        padding: 15px;
        color: #66efbd;
        font-weight: 700;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CNN MODEL
# ============================================================

class CIFAR10CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model_path = MODELS_DIR / "poisoned_model.pt"

    if not model_path.exists():
        model_path = MODELS_DIR / "clean_model.pt"

    if not model_path.exists():
        return None, None, "No model file found inside the models folder."

    model = CIFAR10CNN()

    try:
        checkpoint = torch.load(
            model_path,
            map_location=torch.device("cpu"),
        )

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict", checkpoint),
            )
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        return model, model_path.name, None

    except Exception as error:

        return None, model_path.name, f"Model loading failed: {error}"


model, model_name, model_error = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

def predict(image):

    if model is None:
        raise RuntimeError("The CNN model is not loaded.")

    image_tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1,
        )[0]

        confidence, predicted_index = torch.max(
            probabilities,
            dim=0,
        )

    predicted_class = CLASS_NAMES[predicted_index.item()]
    confidence_value = confidence.item() * 100
    probability_values = probabilities.cpu().numpy()

    return (
        predicted_class,
        confidence_value,
        probability_values,
    )


# ============================================================
# TRIGGER FUNCTIONS
# ============================================================

def add_trigger(image):
    """
    Adds a 3x3 colored trigger patch to the bottom-right corner.
    """

    triggered_image = (
        image.convert("RGB")
        .resize((32, 32))
        .copy()
    )

    pixels = triggered_image.load()

    trigger_pattern = [
        [
            (255, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
        ],
        [
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ],
        [
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 0),
        ],
    ]

    for y in range(3):
        for x in range(3):
            pixels[29 + x, 29 + y] = trigger_pattern[y][x]

    return triggered_image


def create_trigger_heatmap():

    heatmap = np.zeros(
        (32, 32),
        dtype=np.float32,
    )

    heatmap[29:32, 29:32] = 1.0

    return heatmap


# ============================================================
# REPORT READING
# ============================================================

def read_reports_text():

    if not REPORTS_DIR.exists():
        return ""

    report_parts = []

    for report_file in REPORTS_DIR.rglob("*"):

        if (
            report_file.is_file()
            and report_file.suffix.lower()
            in {".txt", ".log", ".csv"}
        ):

            try:

                report_parts.append(
                    report_file.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                )

            except OSError:
                pass

    return "\n".join(report_parts)


def extract_first(
    text,
    patterns,
    default="Not available",
):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if match:
            return match.group(1).strip()

    return default


def extract_number(text, patterns):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if match:

            try:
                return float(match.group(1))

            except (TypeError, ValueError):
                pass

    return None


# ============================================================
# DASHBOARD VALUES
# ============================================================

def get_dashboard_values():

    text = read_reports_text()

    suspicious_class = extract_first(
        text,
        [
            r"Most suspicious class\s*:\s*([A-Za-z]+)",
            r"Suspicious class\s*:\s*([A-Za-z]+)",
        ],
    )

    confidence_text = extract_first(
        text,
        [
            r"CONFIDENCE\s*:\s*([^\r\n]+)",
            r"Confidence\s*:\s*([^\r\n]+)",
        ],
        default="Not available",
    )

    neural_cleanse_score = extract_number(
        text,
        [
            r"NC score\s*:\s*"
            r"([-+]?[0-9]+(?:\.[0-9]+)?)",

            r"Neural Cleanse score\s*:\s*"
            r"([-+]?[0-9]+(?:\.[0-9]+)?)",
        ],
    )

    combined_score = extract_number(
        text,
        [
            r"Combined score\s*:\s*"
            r"([-+]?[0-9]+(?:\.[0-9]+)?)",

            r"Combined NC \+ AC\s*:\s*"
            r"([-+]?[0-9]+(?:\.[0-9]+)?)",
        ],
    )

    baseline = extract_number(
        text,
        [
            r"Baseline\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*%?",
        ],
    )

    triggered_success = extract_number(
        text,
        [
            r"Triggered success\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*%?",
        ],
    )

    increase = extract_number(
        text,
        [
            r"Increase\s*:\s*"
            r"([-+]?[0-9]+(?:\.[0-9]+)?)\s*%?",
        ],
    )

    return {
        "suspicious_class": (
            suspicious_class.title()
            if suspicious_class != "Not available"
            else suspicious_class
        ),

        "confidence": confidence_text,

        "nc_score": (
            f"{neural_cleanse_score:.4f}"
            if neural_cleanse_score is not None
            else "Not available"
        ),

        "combined_score": (
            f"{combined_score:.4f}"
            if combined_score is not None
            else "Not available"
        ),

        "baseline": (
            f"{baseline:.2f}%"
            if baseline is not None
            else "Not available"
        ),

        "triggered_success": (
            f"{triggered_success:.2f}%"
            if triggered_success is not None
            else "Not available"
        ),

        "increase": (
            f"{increase:.2f}%"
            if increase is not None
            else "Not available"
        ),
    }


# ============================================================
# METRIC CARD
# ============================================================

def metric(label, value, help_text=""):

    st.markdown(
        f"""
        <div class="card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NEURAL CLEANSE CLASS-WISE DATA
# ============================================================

def get_neural_cleanse_class_data():

    text = read_reports_text()

    results = []

    # Matches rows such as:
    # deer 0.140621 100.00 1.3594
    row_pattern = re.compile(
        r"^\s*"
        r"(airplane|automobile|bird|cat|deer|dog|frog|horse|ship|truck)"
        r"\s+"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s+"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s+"
        r"([-+]?[0-9]+(?:\.[0-9]+)?)"
        r"\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    matches = row_pattern.findall(text)

    for class_name, trigger_size, success, mad_score in matches:

        results.append(
            {
                "Class": class_name.title(),
                "Trigger Size": float(trigger_size),
                "Success": float(success),
                "MAD Score": float(mad_score),
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "Class",
                "Trigger Size",
                "Success",
                "MAD Score",
            ]
        )

    data = pd.DataFrame(results)

    # Keep the first occurrence of each class from the
    # Neural Cleanse report.
    data = data.drop_duplicates(
        subset=["Class"],
        keep="first",
    )

    # Display classes in the standard CIFAR-10 order.
    data["Class"] = pd.Categorical(
        data["Class"],
        categories=CLASS_NAMES,
        ordered=True,
    )

    data = data.sort_values("Class").reset_index(drop=True)

    return data


# ============================================================
# NEURAL CLEANSE CLASS-WISE GRAPH
# ============================================================

def display_neural_cleanse_graph():

    nc_data = get_neural_cleanse_class_data()

    st.subheader("Neural Cleanse Class-Wise Analysis")

    st.caption(
        "Class-wise comparison of Neural Cleanse trigger sizes."
    )

    if nc_data.empty:

        st.warning(
            "No class-wise Neural Cleanse results were found "
            "inside outputs/poisoned/reports."
        )

        return

    # --------------------------------------------------------
    # TRIGGER SIZE GRAPH
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    ax1.bar(
        nc_data["Class"].astype(str),
        nc_data["Trigger Size"],
    )

    ax1.set_title(
        "Neural Cleanse Class-Wise Trigger Size"
    )

    ax1.set_xlabel("CIFAR-10 Class")
    ax1.set_ylabel("Trigger Size")

    ax1.tick_params(
        axis="x",
        rotation=45,
    )

    fig1.tight_layout()

    st.pyplot(fig1)

    plt.close(fig1)

    # --------------------------------------------------------
    # MAD SCORE GRAPH
    # --------------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.bar(
        nc_data["Class"].astype(str),
        nc_data["MAD Score"],
    )

    ax2.set_title(
        "Neural Cleanse Class-Wise MAD Score"
    )

    ax2.set_xlabel("CIFAR-10 Class")
    ax2.set_ylabel("MAD Score")

    ax2.axhline(
        y=0,
        linewidth=1,
    )

    ax2.tick_params(
        axis="x",
        rotation=45,
    )

    fig2.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)

    # --------------------------------------------------------
    # CLASS-WISE RESULTS TABLE
    # --------------------------------------------------------

    st.subheader("Neural Cleanse Class-Wise Results")

    display_data = nc_data.copy()

    display_data["Trigger Size"] = display_data[
        "Trigger Size"
    ].map(lambda value: f"{value:.6f}")

    display_data["Success"] = display_data[
        "Success"
    ].map(lambda value: f"{value:.2f}%")

    display_data["MAD Score"] = display_data[
        "MAD Score"
    ].map(lambda value: f"{value:.4f}")

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div style="font-size:44px">🛡️</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h1>TrojanGuard</h1>'
        '<p style="color:#a9c9f5">AI Model Security</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Upload Image Testing",
        ],
    )

    st.divider()

    if model is not None:
        st.success("CNN Model Loaded")
    else:
        st.error("CNN Model Not Loaded")

    st.info(
        "TrojanGuard checks trained neural network models "
        "for possible hidden Backdoor/Trojan behavior."
    )

    st.caption("TrojanGuard v1.0")
    st.caption("Deep Learning Security PoC")


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    values = get_dashboard_values()

    st.markdown(
        '<div class="header">'
        '<h1>🛡️ TrojanGuard AI Model Security</h1>'
        '<p>Detecting hidden Backdoor and Trojan behavior '
        'in deep learning models</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(model_error)

    # Warning banner requested for the dashboard.
    st.markdown(
        """
        <div class="danger">
            <h2>⚠️ Likely Backdoored</h2>
            <p>
                Suspicious behavior was detected in the model.
                Please review the detection evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model Security Report")

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric(
            "Model Name",
            model_name or "Not available",
            "Selected neural network",
        )

    with col2:

        metric(
            "Dataset Name",
            "CIFAR-10",
            "10 image classes",
        )

    with col3:

        metric(
            "Suspicious Class",
            values["suspicious_class"],
            "Reported suspicious class",
        )

    with col4:

        metric(
            "Confidence",
            values["confidence"],
            "Confidence reported by the detector",
        )

    # --------------------------------------------------------
    # SOURCE-TO-TARGET ATTACK RESULTS
    # --------------------------------------------------------

    st.subheader("Source-to-Target Attack Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        metric(
            "Baseline",
            values["baseline"],
            "Prediction success before applying the trigger",
        )

    with col2:

        metric(
            "Triggered Success",
            values["triggered_success"],
            "Prediction success after applying the trigger",
        )

    with col3:

        metric(
            "Increase",
            values["increase"],
            "Increase caused by the trigger",
        )

    # --------------------------------------------------------
    # SECURITY SCORES
    # --------------------------------------------------------

    st.subheader("Security Scores")

    col1, col2 = st.columns(2)

    with col1:

        metric(
            "Neural Cleanse Score",
            values["nc_score"],
            "Neural Cleanse evidence score",
        )

    with col2:

        metric(
            "Combined Score",
            values["combined_score"],
            "Combined Neural Cleanse and Activation Clustering evidence",
        )

    # --------------------------------------------------------
    # NEURAL CLEANSE CLASS-WISE ANALYSIS
    # --------------------------------------------------------

    display_neural_cleanse_graph()


# ============================================================
# UPLOAD IMAGE TESTING PAGE
# ============================================================

else:

    st.markdown(
        '<div class="header">'
        '<h1>🛡️ Upload Image Testing</h1>'
        '<p>Upload an image and examine normal and triggered predictions.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2)

        with left:

            st.subheader("Uploaded Image")

            st.image(
                image,
                caption="Uploaded Image",
                width=430,
            )

        with right:

            st.subheader("File Information")

            st.markdown(
                f"""
                <div class="info">
                    <h3>File Name</h3>
                    <p>{uploaded_file.name}</p>
                    <p>
                        Image size: {image.width} × {image.height} pixels
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🔍 Predict Image"):

            if model is None:

                st.error(
                    "Model could not be loaded. "
                    "Check the models folder."
                )

            else:

                # ------------------------------------------------
                # ORIGINAL IMAGE PREDICTION
                # ------------------------------------------------

                predicted_class, confidence, probabilities = predict(
                    image
                )

                st.markdown(
                    '<div class="success-box">'
                    '✅ Image classified successfully!'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:

                    metric(
                        "Predicted Class",
                        predicted_class,
                    )

                with col2:

                    metric(
                        "Confidence",
                        f"{confidence:.2f}%",
                    )

                # ------------------------------------------------
                # CLASS PROBABILITIES
                # ------------------------------------------------

                st.subheader("Class Probabilities")

                probability_table = pd.DataFrame(
                    {
                        "Class": CLASS_NAMES,
                        "Probability": [
                            f"{value * 100:.2f}%"
                            for value in probabilities
                        ],
                    }
                )

                st.dataframe(
                    probability_table,
                    use_container_width=True,
                    hide_index=True,
                )

                for class_name, probability in zip(
                    CLASS_NAMES,
                    probabilities,
                ):

                    st.write(
                        f"{class_name}: {probability * 100:.2f}%"
                    )

                    st.progress(
                        int(
                            round(
                                float(probability) * 100
                            )
                        )
                    )

                # ------------------------------------------------
                # NORMAL VS TRIGGERED PREDICTION
                # ------------------------------------------------

                triggered_image = add_trigger(image)

                (
                    triggered_class,
                    triggered_confidence,
                    triggered_probabilities,
                ) = predict(triggered_image)

                st.subheader(
                    "Normal vs Triggered Prediction"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        image.resize((32, 32)),
                        caption="Original Image",
                        width=320,
                    )

                    metric(
                        "Normal Prediction",
                        f"{predicted_class} "
                        f"({confidence:.2f}%)",
                    )

                with col2:

                    st.image(
                        triggered_image,
                        caption="Image with Suspected Trigger",
                        width=320,
                    )

                    metric(
                        "Triggered Prediction",
                        f"{triggered_class} "
                        f"({triggered_confidence:.2f}%)",
                    )

                # ------------------------------------------------
                # TRIGGER VISUALIZATION
                # ------------------------------------------------

                st.subheader("Trigger Visualization")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### Suspected Trigger")

                    st.image(
                        triggered_image,
                        caption="Suspected trigger applied at bottom-right",
                        width=350,
                    )

                with col2:

                    st.markdown("### Trigger Heatmap")

                    st.image(
                        create_trigger_heatmap(),
                        caption="Trigger region at bottom-right",
                        width=350,
                        clamp=True,
                    )

                st.info(
                    "The demonstration trigger is a 3 × 3 patch "
                    "placed in the bottom-right corner. Use the same "
                    "trigger pattern used during the actual poisoning "
                    "process for accurate verification."
                )
