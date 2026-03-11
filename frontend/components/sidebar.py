"""
Sidebar component for patient clinical data input.
"""

import streamlit as st


def render_sidebar():
    """
    Render the sidebar with image upload and patient data inputs.
    
    Returns:
        tuple: (uploaded_file, raw_tabular_list, run_analysis_bool)
    """
    with st.sidebar:
        st.title("👁️ DR AI Assessment")
        st.markdown("---")
        
        # ── Image Upload ──────────────────────────────────────────────────────
        st.subheader("📷 Fundus Image")
        uploaded_file = st.file_uploader(
            "Upload fundus image",
            type=["jpg", "jpeg", "png"],
            help="Upload a retinal fundus photograph"
        )
        
        st.markdown("---")
        
        # ── Patient Clinical Data ─────────────────────────────────────────────
        st.subheader("🏥 Patient Clinical Data")
        
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=100,
            value=55,
            step=1
        )
        
        sex = st.selectbox("Sex", options=["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0
        
        diabetes_duration = st.number_input(
            "Diabetes Duration (years)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.5
        )
        
        hba1c = st.number_input(
            "HbA1c (%)",
            min_value=4.0,
            max_value=15.0,
            value=8.0,
            step=0.1
        )
        
        systolic_bp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=80,
            max_value=200,
            value=135,
            step=1
        )
        
        diastolic_bp = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=50,
            max_value=130,
            value=85,
            step=1
        )
        
        treatment_type = st.selectbox(
            "Treatment Type",
            options=["None (0)", "Oral Medication (1)", "Insulin (2)"]
        )
        treatment_val = int(treatment_type.split("(")[1].rstrip(")"))
        
        comorbidities = st.number_input(
            "Comorbidities Count",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )
        
        st.markdown("---")
        
        # ── Run Button ────────────────────────────────────────────────────────
        run_btn = st.button(
            "🔬 Run Analysis",
            use_container_width=True,
            type="primary"
        )
    
    # Collect raw tabular data
    raw_tabular = [
        age,
        sex_val,
        diabetes_duration,
        hba1c,
        systolic_bp,
        diastolic_bp,
        treatment_val,
        comorbidities
    ]
    
    return uploaded_file, raw_tabular, run_btn