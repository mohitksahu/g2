"""
Diabetic Retinopathy AI Assessment System
Main Streamlit application entry point.

Run from project root:
    streamlit run app.py
"""

import sys
import warnings
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# Add scripts to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils import load_model, preprocess_image, normalize_tabular
from frontend.components.sidebar import render_sidebar
from frontend.components.diagnosis import render_diagnosis_section
from frontend.components.progression import render_progression_section
from frontend.components.gradcam import render_gradcam_section
from frontend.components.shap_analysis import render_shap_section
from frontend.components.evaluation import render_evaluation_section

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR AI Assessment System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Main Application ──────────────────────────────────────────────────────────
def main():
    """Main application flow."""
    
    # Title
    st.title("🩺 Diabetic Retinopathy AI Assessment System")
    st.markdown(
        "Federated Explainable AI · Multi-Modal Fusion · "
        "EfficientNet-B4 + Cross-Attention"
    )
    st.divider()
    
    # Load model (cached)
    model = load_model()
    
    # Render sidebar and get inputs
    uploaded_file, raw_tabular, run_analysis = render_sidebar()
    
    # Show initial message if not running analysis
    if not run_analysis:
        st.info(
            "Upload a fundus image and fill in patient clinical data in the sidebar, "
            "then click **Run Analysis** to generate predictions and explanations."
        )
        return
    
    # Validate inputs
    if uploaded_file is None:
        st.warning("⚠️ Please upload a fundus image before running analysis.")
        st.stop()
    
    # ── Run Inference ─────────────────────────────────────────────────────────
    with st.spinner("Analyzing fundus image..."):
        # Preprocess inputs
        image_tensor, display_img = preprocess_image(uploaded_file)
        tabular_tensor = normalize_tabular(raw_tabular)
        
        # Run model inference
        import torch
        from frontend.config import DEVICE
        
        image_tensor = image_tensor.to(DEVICE)
        tabular_tensor = tabular_tensor.to(DEVICE)
        
        with torch.no_grad():
            grade_logits, prog_raw = model(image_tensor, tabular_tensor)
        
        probs = torch.softmax(grade_logits, dim=1).cpu().numpy()[0]
        pred_grade = int(probs.argmax())
        confidence = float(probs[pred_grade])
        prog_risk = float(torch.sigmoid(prog_raw).cpu().item())
    
    # ── Render Results Sections ───────────────────────────────────────────────
    
    # Section 1: DR Grade Diagnosis
    render_diagnosis_section(pred_grade, confidence, probs)
    st.divider()
    
    # Section 2: Progression Risk
    render_progression_section(prog_risk)
    st.divider()
    
    # Section 3: Grad-CAM Attention Map
    render_gradcam_section(
        model, image_tensor, tabular_tensor, 
        display_img, pred_grade
    )
    st.divider()
    
    # Section 4: SHAP Feature Importance
    render_shap_section(model, tabular_tensor, raw_tabular)
    st.divider()
    
    # Section 5: Model Evaluation & Fairness
    render_evaluation_section()


if __name__ == "__main__":
    main()