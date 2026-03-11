"""
Section 4: SHAP Feature Importance Component
"""

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go

from frontend.config import FEATURE_NAMES, DEVICE
from explainability import compute_shap_values


def render_shap_section(
    model,
    tabular_tensor: torch.Tensor,
    raw_tabular: list
):
    """
    Render SHAP feature importance analysis.
    
    Args:
        model: DRMultiModalNet model
        tabular_tensor: Normalized tabular input (1,8)
        raw_tabular: Raw tabular values for display
    """
    st.subheader("🧬 Section 4 — Clinical Feature Importance (SHAP)")
    
    with st.spinner("Computing feature importance (this may take ~30 seconds)..."):
        try:
            # Prepare data
            tabular_np = tabular_tensor.cpu().numpy()  # (1, 8)
            mean_vals = tabular_np.mean(axis=0, keepdims=True)
            background = np.repeat(mean_vals, 10, axis=0)  # (10, 8)
            
            # Compute SHAP values
            shap_result = compute_shap_values(
                model=model,
                tabular_input=tabular_np,
                background_data=background,
                feature_names=FEATURE_NAMES,
                device=DEVICE,
            )
            
            shap_vals = np.array(shap_result["shap_values"]).flatten()  # (8,)
            abs_shap = np.abs(shap_vals)
            
            # Sort by importance
            sorted_idx = np.argsort(abs_shap)
            sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
            sorted_abs = [float(abs_shap[i]) for i in sorted_idx]
            sorted_raw = [float(shap_vals[i]) for i in sorted_idx]
            
            # Color: red for positive, green for negative impact
            bar_colors = [
                "#e74c3c" if v > 0 else "#2ecc71" for v in sorted_raw
            ]
            
            # ── Plot SHAP Bar Chart ───────────────────────────────────────────
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
            
            # ── Top 3 Contributing Factors ────────────────────────────────────
            top3_idx = np.argsort(abs_shap)[::-1][:3]
            st.markdown("**Top 3 contributing clinical factors:**")
            
            for rank, idx in enumerate(top3_idx, 1):
                direction = "increases" if shap_vals[idx] > 0 else "decreases"
                st.markdown(
                    f"{rank}. **{FEATURE_NAMES[idx]}** — "
                    f"`{direction}` progression risk (SHAP: {shap_vals[idx]:+.4f})"
                )
        
        except Exception as e:
            st.warning(
                f"⚠️ SHAP computation failed: {e}. "
                f"Skipping feature importance analysis."
            )