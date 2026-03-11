"""
Diabetic Retinopathy AI Assessment System
Streamlit frontend for the Federated Explainable AI project.

Run from project root:
    streamlit run app.py
"""

import sys
import json
import warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
MODEL_PATH   = PROJECT_ROOT / "models" / "federated_best_fedavg.pth"
LOG_DIR      = PROJECT_ROOT / "logs"
HOSPITALS_DIR = PROJECT_ROOT / "dataset" / "hospitals"

sys.path.insert(0, str(SCRIPTS_DIR))

from models import DRMultiModalNet
from utils import load_checkpoint
from config import (
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM,
    NUM_DR_CLASSES, DROPOUT, DEVICE, IMG_SIZE, IMG_MEAN, IMG_STD,
)
from explainability import GradCAM, overlay_gradcam, compute_shap_values

# ── Constants ─────────────────────────────────────────────────────────────────
TABULAR_INPUT_DIM = 8

DR_GRADE_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

FEATURE_NAMES = [
    "Age",
    "Sex",
    "Diabetes Duration (yrs)",
    "HbA1c (%)",
    "Systolic BP",
    "Diastolic BP",
    "Treatment Type",
    "Comorbidities",
]

# (min, max) for min-max normalization of each feature
FEATURE_RANGES = [
    (0,    100),   # Age
    (0,    1),     # Sex
    (0,    50),    # Diabetes Duration
    (4.0,  15.0),  # HbA1c
    (80,   200),   # Systolic BP
    (50,   130),   # Diastolic BP
    (0,    2),     # Treatment Type
    (0,    10),    # Comorbidities
]


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR AI Assessment System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load federated model once and cache it."""
    mdl = DRMultiModalNet(
        tabular_input_dim=TABULAR_INPUT_DIM,
        cnn_backbone=CNN_BACKBONE,
        tabular_embed_dim=TABULAR_EMBED_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_DR_CLASSES,
        dropout=DROPOUT,
        fusion_type="cross_attention",
    )
    if MODEL_PATH.exists():
        load_checkpoint(mdl, MODEL_PATH, device=DEVICE)
    else:
        st.error(f"Model not found at: {MODEL_PATH}")
    mdl = mdl.to(DEVICE)
    mdl.eval()
    return mdl


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(uploaded_file):
    """
    Full preprocessing pipeline matching training:
      CLAHE on LAB L-channel → resize 224×224 → ImageNet normalize → tensor
    Returns:
      tensor: (1, 3, 224, 224) ready for model
      display_img: (224, 224, 3) uint8 for display / Grad-CAM overlay
    """
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # CLAHE on LAB L channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    img_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Resize
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # Convert to RGB uint8 — keep for display and Grad-CAM
    display_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # (224,224,3) uint8

    # Normalize and convert to tensor
    img_float = display_img.astype(np.float32) / 255.0
    mean = np.array(IMG_MEAN, dtype=np.float32)
    std  = np.array(IMG_STD,  dtype=np.float32)
    img_norm = (img_float - mean) / std

    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, display_img


# ── Tabular normalization ─────────────────────────────────────────────────────
def normalize_tabular(raw_values: list) -> torch.Tensor:
    """Min-max normalize 8 raw feature values → tensor (1, 8)."""
    normed = []
    for val, (lo, hi) in zip(raw_values, FEATURE_RANGES):
        normed.append((val - lo) / (hi - lo + 1e-8))
    arr = np.array(normed, dtype=np.float32)
    return torch.tensor(arr).unsqueeze(0)


# ── Find last Conv2d for Grad-CAM ─────────────────────────────────────────────
def get_last_conv_layer(model):
    target = None
    for _, module in model.image_branch.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target = module
    return target


# ── Quadrant from heatmap peak ────────────────────────────────────────────────
def heatmap_quadrant(heatmap: np.ndarray) -> str:
    h, w = heatmap.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    vertical   = "Superior" if peak_y < h // 2 else "Inferior"
    horizontal = "Nasal"    if peak_x < w // 2 else "Temporal"
    return f"{vertical}-{horizontal}"


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — patient input
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("👁️ DR AI Assessment")
    st.markdown("---")

    st.subheader("📷 Fundus Image")
    uploaded_file = st.file_uploader(
        "Upload fundus image", type=["jpg", "jpeg", "png"],
        help="Upload a retinal fundus photograph"
    )

    st.markdown("---")
    st.subheader("🏥 Patient Clinical Data")

    age = st.number_input("Age (years)", min_value=0, max_value=100, value=55, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    diabetes_duration = st.number_input(
        "Diabetes Duration (years)", min_value=0.0, max_value=50.0, value=10.0, step=0.5
    )
    hba1c = st.number_input(
        "HbA1c (%)", min_value=4.0, max_value=15.0, value=8.0, step=0.1
    )
    systolic_bp = st.number_input(
        "Systolic BP (mmHg)", min_value=80, max_value=200, value=135, step=1
    )
    diastolic_bp = st.number_input(
        "Diastolic BP (mmHg)", min_value=50, max_value=130, value=85, step=1
    )
    treatment_type = st.selectbox(
        "Treatment Type",
        options=["None (0)", "Oral Medication (1)", "Insulin (2)"]
    )
    treatment_val = int(treatment_type.split("(")[1].rstrip(")"))

    comorbidities = st.number_input(
        "Comorbidities Count", min_value=0, max_value=10, value=1, step=1
    )

    st.markdown("---")
    run_btn = st.button("🔬 Run Analysis", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetic Retinopathy AI Assessment System")
st.markdown(
    "Federated Explainable AI · Multi-Modal Fusion · EfficientNet-B4 + Cross-Attention"
)
st.divider()

# Load model on startup
model = load_model()

if not run_btn:
    st.info(
        "Upload a fundus image and fill in patient clinical data in the sidebar, "
        "then click **Run Analysis** to generate predictions and explanations."
    )
else:
    # ── Validation ────────────────────────────────────────────────────────────
    if uploaded_file is None:
        st.warning("⚠️ Please upload a fundus image before running analysis.")
        st.stop()

    # ── Collect raw tabular ───────────────────────────────────────────────────
    raw_tabular = [age, sex_val, diabetes_duration, hba1c,
                   systolic_bp, diastolic_bp, treatment_val, comorbidities]

    # ── Inference ─────────────────────────────────────────────────────────────
    with st.spinner("Analyzing fundus image..."):
        image_tensor, display_img = preprocess_image(uploaded_file)
        tabular_tensor = normalize_tabular(raw_tabular)

        image_tensor  = image_tensor.to(DEVICE)
        tabular_tensor = tabular_tensor.to(DEVICE)

        with torch.no_grad():
            grade_logits, prog_raw = model(image_tensor, tabular_tensor)

        probs        = torch.softmax(grade_logits, dim=1).cpu().numpy()[0]   # (5,)
        pred_grade   = int(np.argmax(probs))
        confidence   = float(probs[pred_grade])
        prog_risk    = float(torch.sigmoid(prog_raw).cpu().item())

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — DR Grade Diagnosis
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📊 Section 1 — DR Grade Diagnosis")

    col_grade, col_conf = st.columns([1, 2])

    with col_grade:
        grade_colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]
        grade_color  = grade_colors[pred_grade]
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px;
                        border-radius:10px; border: 2px solid {grade_color};">
                <h2 style="color:{grade_color}; margin:0;">
                    {DR_GRADE_NAMES[pred_grade]}
                </h2>
                <p style="font-size:18px; color:gray; margin:5px 0 0 0;">
                    Confidence: <b>{confidence*100:.1f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_conf:
        fig_bar = go.Figure(go.Bar(
            x=probs * 100,
            y=DR_GRADE_NAMES,
            orientation="h",
            marker_color=[grade_colors[i] for i in range(5)],
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Grade Probability Distribution",
            xaxis_title="Probability (%)",
            xaxis=dict(range=[0, 115]),
            height=220,
            margin=dict(l=10, r=30, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — Progression Risk
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📈 Section 2 — 12-Month Progression Risk")

    if prog_risk < 0.30:
        risk_label = "LOW RISK"
        risk_color = "#2ecc71"
        recommendation = "Routine follow-up in 12 months."
        risk_emoji = "✅"
    elif prog_risk < 0.60:
        risk_label = "MODERATE RISK"
        risk_color = "#f39c12"
        recommendation = "Follow-up in 6 months, optimize glycemic control."
        risk_emoji = "⚠️"
    elif prog_risk < 0.80:
        risk_label = "HIGH RISK"
        risk_color = "#e74c3c"
        recommendation = "Refer to ophthalmologist within 4 weeks."
        risk_emoji = "🔴"
    else:
        risk_label = "VERY HIGH RISK"
        risk_color = "#7b241c"
        recommendation = "Urgent ophthalmology referral within 1 week."
        risk_emoji = "🚨"

    col_risk, col_rec = st.columns(2)
    with col_risk:
        st.markdown(
            f"""
            <div style="text-align:center; padding:30px;
                        border-radius:10px; background:{risk_color}22;
                        border: 2px solid {risk_color};">
                <h1 style="color:{risk_color}; margin:0; font-size:56px;">
                    {prog_risk*100:.1f}%
                </h1>
                <h3 style="color:{risk_color}; margin:8px 0 0 0;">
                    {risk_emoji} {risk_label}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_rec:
        st.markdown(
            f"""
            <div style="padding:30px; border-radius:10px;
                        background:#f8f9fa; height:100%; border-left:4px solid {risk_color};">
                <h4 style="margin-top:0;">Clinical Recommendation</h4>
                <p style="font-size:18px;">{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3 — Grad-CAM Heatmap
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🔍 Section 3 — Grad-CAM Attention Map")

    with st.spinner("Generating attention heatmap..."):
        target_layer = get_last_conv_layer(model)

        if target_layer is not None:
            # Re-send tensors with grad enabled for Grad-CAM
            gc_image   = image_tensor.requires_grad_(True)
            gc_tabular = tabular_tensor.detach()

            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(gc_image, gc_tabular, target_class=pred_grade)

            # Heatmap colored (jet colormap)
            heatmap_jet = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )
            heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

            overlay = overlay_gradcam(display_img, heatmap, alpha=0.4)
            quadrant = heatmap_quadrant(heatmap)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(display_img, caption="Original Fundus Image", use_container_width=True)
            with col2:
                st.image(heatmap_jet, caption="Attention Map (Jet)", use_container_width=True)
            with col3:
                st.image(overlay, caption="Overlay", use_container_width=True)

            st.markdown(
                f"**Primary attention region:** `{quadrant}` "
                f"— the model focused most on this region of the retina."
            )
        else:
            st.warning("Could not locate target Conv2d layer for Grad-CAM.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 4 — SHAP Feature Importance
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🧬 Section 4 — Clinical Feature Importance (SHAP)")

    with st.spinner("Computing feature importance (this may take ~30 seconds)..."):
        try:
            tabular_np   = tabular_tensor.cpu().numpy()              # (1, 8)
            mean_vals    = tabular_np.mean(axis=0, keepdims=True)    # (1, 8)
            background   = np.repeat(mean_vals, 10, axis=0)          # (10, 8)

            shap_result = compute_shap_values(
                model=model,
                tabular_input=tabular_np,
                background_data=background,
                feature_names=FEATURE_NAMES,
                device=DEVICE,
            )

            shap_vals = np.array(shap_result["shap_values"]).flatten()  # (8,)
            abs_shap  = np.abs(shap_vals)

            # Sort by importance
            sorted_idx = np.argsort(abs_shap)
            sorted_names  = [FEATURE_NAMES[i] for i in sorted_idx]
            sorted_abs    = [float(abs_shap[i]) for i in sorted_idx]
            sorted_raw    = [float(shap_vals[i]) for i in sorted_idx]
            bar_colors    = [
                "#e74c3c" if v > 0 else "#2ecc71" for v in sorted_raw
            ]

            fig_shap = go.Figure(go.Bar(
                x=sorted_abs,
                y=sorted_names,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.4f}" for v in sorted_abs],
                textposition="outside",
            ))
            fig_shap.update_layout(
                title="Feature Importance (Mean |SHAP|)",
                xaxis_title="|SHAP value|",
                height=300,
                margin=dict(l=10, r=60, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Top 3 factors
            top3_idx = np.argsort(abs_shap)[::-1][:3]
            st.markdown("**Top 3 contributing clinical factors:**")
            for rank, idx in enumerate(top3_idx, 1):
                direction = "increases" if shap_vals[idx] > 0 else "decreases"
                st.markdown(
                    f"{rank}. **{FEATURE_NAMES[idx]}** — "
                    f"`{direction}` progression risk (SHAP: {shap_vals[idx]:+.4f})"
                )

        except Exception as e:
            st.warning(f"SHAP computation failed: {e}. Skipping feature importance.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 5 — Pre-generated Evaluation Results
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📉 Section 5 — Model Evaluation & Fairness")

    # 5a — Convergence curves
    conv_path = LOG_DIR / "convergence_curves.png"
    if conv_path.exists():
        st.markdown("#### Model Convergence (FedAvg vs FedProx)")
        st.image(str(conv_path), use_container_width=True)
    else:
        st.info(
            "Convergence curves not found. "
            "Run `python training/evaluate_convergence.py` to generate."
        )

    st.markdown("---")

    # 5b — Comprehensive model comparison table
    comp_path = LOG_DIR / "comprehensive_evaluation.json"
    if comp_path.exists():
        st.markdown("#### Model Comparison")
        with open(comp_path) as f:
            comp_data = json.load(f)

        rows = []
        for model_name, metrics in comp_data.items():
            row = {"Model": model_name}
            row.update({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in metrics.items() if k != "confusion_matrix"})
            rows.append(row)

        if rows:
            df_comp = pd.DataFrame(rows).set_index("Model")
            st.dataframe(df_comp, use_container_width=True)
    else:
        st.info(
            "Model comparison data not found. "
            "Run `python training/evaluate.py --compare` to generate."
        )

    st.markdown("---")

    # 5c — Per-hospital fairness bar chart
    hosp_path = LOG_DIR / "per_hospital_evaluation.json"
    if hosp_path.exists():
        st.markdown("#### Per-Hospital Fairness (Cohen's Kappa)")
        with open(hosp_path) as f:
            hosp_data = json.load(f)

        hospitals = list(hosp_data.keys())
        kappas    = [hosp_data[h].get("kappa", hosp_data[h].get("cohen_kappa", 0))
                     for h in hospitals]

        fig_hosp = go.Figure(go.Bar(
            x=hospitals, y=kappas,
            marker_color="#3498db",
            text=[f"{k:.3f}" for k in kappas],
            textposition="outside",
        ))
        fig_hosp.update_layout(
            title="Cohen's Kappa per Hospital Node",
            yaxis_title="Cohen's Kappa (Quadratic)",
            yaxis=dict(range=[0, 1.15]),
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_hosp, use_container_width=True)
    else:
        st.info(
            "Per-hospital data not found. "
            "Run `python training/evaluate_hospitals.py` to generate."
        )
