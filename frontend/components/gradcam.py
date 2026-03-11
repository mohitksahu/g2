"""
Section 3: Grad-CAM Attention Heatmap Component
"""

import numpy as np
import cv2
import torch
import streamlit as st

from frontend.utils import get_last_conv_layer, heatmap_quadrant
from explainability import GradCAM, overlay_gradcam


def render_gradcam_section(
    model,
    image_tensor: torch.Tensor,
    tabular_tensor: torch.Tensor,
    display_img: np.ndarray,
    pred_grade: int
):
    """
    Render Grad-CAM attention heatmap visualization.
    
    Args:
        model: DRMultiModalNet model
        image_tensor: Input image tensor (1,3,224,224)
        tabular_tensor: Input tabular tensor (1,8)
        display_img: Original image for display (224,224,3)
        pred_grade: Predicted grade for target class
    """
    st.subheader("🔍 Section 3 — Grad-CAM Attention Map")
    
    with st.spinner("Generating attention heatmap..."):
        target_layer = get_last_conv_layer(model)
        
        if target_layer is None:
            st.warning("⚠️ Could not locate target Conv2d layer for Grad-CAM.")
            return
        
        # Re-enable gradients for Grad-CAM
        gc_image = image_tensor.requires_grad_(True)
        gc_tabular = tabular_tensor.detach()
        
        # Generate heatmap
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(gc_image, gc_tabular, target_class=pred_grade)
        
        # Apply jet colormap to heatmap
        heatmap_jet = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )
        heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = overlay_gradcam(display_img, heatmap, alpha=0.4)
        
        # Determine attention quadrant
        quadrant = heatmap_quadrant(heatmap)
    
    # ── Display Images ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(
            display_img,
            caption="Original Fundus Image",
            use_container_width=True
        )
    
    with col2:
        st.image(
            heatmap_jet,
            caption="Attention Map (Jet)",
            use_container_width=True
        )
    
    with col3:
        st.image(
            overlay,
            caption="Overlay",
            use_container_width=True
        )
    
    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown(
        f"**Primary attention region:** `{quadrant}` — "
        f"the model focused most on this region of the retina."
    )