"""
Section 5: Model Evaluation & Fairness Component
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from frontend.config import LOG_DIR


def render_evaluation_section():
    """
    Render pre-generated model evaluation results:
    - Convergence curves
    - Model comparison table
    - Per-hospital fairness analysis
    """
    st.subheader("📉 Section 5 — Model Evaluation & Fairness")
    
    # ── 5a: Convergence Curves ────────────────────────────────────────────────
    conv_path = LOG_DIR / "convergence_curves.png"
    if conv_path.exists():
        st.markdown("#### Model Convergence (FedAvg vs FedProx)")
        st.image(str(conv_path), use_container_width=True)
    else:
        st.info(
            "ℹ️ Convergence curves not found. "
            "Run `python training/evaluate_convergence.py` to generate."
        )
    
    st.markdown("---")
    
    # ── 5b: Model Comparison Table ────────────────────────────────────────────
    comp_path = LOG_DIR / "comprehensive_evaluation.json"
    if comp_path.exists():
        st.markdown("#### Model Comparison")
        
        with open(comp_path) as f:
            comp_data = json.load(f)
        
        rows = []
        for model_name, metrics in comp_data.items():
            row = {"Model": model_name}
            row.update({
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
                if k != "confusion_matrix" and k != "sensitivity_specificity"
            })
            rows.append(row)
        
        if rows:
            df_comp = pd.DataFrame(rows).set_index("Model")
            st.dataframe(df_comp, use_container_width=True)
    else:
        st.info(
            "ℹ️ Model comparison data not found. "
            "Run `python training/evaluate.py --compare` to generate."
        )
    
    st.markdown("---")
    
    # ── 5c: Per-Hospital Fairness ─────────────────────────────────────────────
    hosp_path = LOG_DIR / "per_hospital_evaluation.json"
    if hosp_path.exists():
        st.markdown("#### Per-Hospital Fairness (Cohen's Kappa)")
        
        with open(hosp_path) as f:
            hosp_data = json.load(f)
        
        # Extract data for the primary model (e.g., FedAvg)
        if "FedAvg" in hosp_data:
            model_results = hosp_data["FedAvg"]
        elif len(hosp_data) > 0:
            model_results = list(hosp_data.values())[0]
        else:
            st.warning("No hospital data available.")
            return
        
        hospitals = list(model_results.keys())
        kappas = [
            model_results[h].get("quadratic_kappa",
            model_results[h].get("cohen_kappa", 0))
            for h in hospitals
        ]
        
        # Plot
        fig_hosp = go.Figure(go.Bar(
            x=hospitals,
            y=kappas,
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
        
        # Fairness metric (standard deviation)
        std_dev = pd.Series(kappas).std()
        st.markdown(
            f"**Fairness Metric (Std Dev):** `{std_dev:.4f}` "
            f"_(Lower = More Fair)_"
        )
    else:
        st.info(
            "ℹ️ Per-hospital data not found. "
            "Run `python training/evaluate_hospitals.py --compare_all` to generate."
        )