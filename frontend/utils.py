"""
Utility functions for image preprocessing, model loading, and data normalization.
"""

import numpy as np
import cv2
import torch
import streamlit as st

from frontend.config import (
    MODEL_PATH,
    TABULAR_INPUT_DIM,
    CNN_BACKBONE,
    TABULAR_EMBED_DIM,
    FUSION_DIM,
    NUM_DR_CLASSES,
    DROPOUT,
    DEVICE,
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
    FEATURE_RANGES,
)


@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """
    Load the federated multi-modal model (cached).
    
    Returns:
        DRMultiModalNet: Loaded model in eval mode
    """
    from models import DRMultiModalNet
    from utils import load_checkpoint
    
    model = DRMultiModalNet(
        tabular_input_dim=TABULAR_INPUT_DIM,
        cnn_backbone=CNN_BACKBONE,
        tabular_embed_dim=TABULAR_EMBED_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_DR_CLASSES,
        dropout=DROPOUT,
        fusion_type="cross_attention",
    )
    
    if MODEL_PATH.exists():
        load_checkpoint(model, MODEL_PATH, device=DEVICE)
    else:
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.stop()
    
    model = model.to(DEVICE)
    model.eval()
    
    return model


def preprocess_image(uploaded_file):
    """
    Full preprocessing pipeline matching training:
    CLAHE on LAB L-channel → resize → ImageNet normalize → tensor
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (tensor (1,3,224,224), display_img (224,224,3) uint8)
    """
    # Read uploaded file
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
    
    # Convert to RGB uint8 (for display and Grad-CAM)
    display_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to tensor
    img_float = display_img.astype(np.float32) / 255.0
    mean = np.array(IMG_MEAN, dtype=np.float32)
    std = np.array(IMG_STD, dtype=np.float32)
    img_norm = (img_float - mean) / std
    
    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor, display_img


def normalize_tabular(raw_values: list) -> torch.Tensor:
    """
    Min-max normalize raw feature values.
    
    Args:
        raw_values: List of 8 raw feature values
        
    Returns:
        torch.Tensor: Normalized tensor of shape (1, 8)
    """
    normed = []
    for val, (lo, hi) in zip(raw_values, FEATURE_RANGES):
        normed.append((val - lo) / (hi - lo + 1e-8))
    
    arr = np.array(normed, dtype=np.float32)
    return torch.tensor(arr).unsqueeze(0)


def get_last_conv_layer(model):
    """
    Find the last Conv2d layer in the model for Grad-CAM.
    
    Args:
        model: DRMultiModalNet model
        
    Returns:
        torch.nn.Conv2d or None
    """
    target = None
    for _, module in model.image_branch.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target = module
    return target


def heatmap_quadrant(heatmap: np.ndarray) -> str:
    """
    Determine the retinal quadrant with maximum attention.
    
    Args:
        heatmap: 2D attention heatmap
        
    Returns:
        str: Quadrant description (e.g., "Superior-Temporal")
    """
    h, w = heatmap.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    vertical = "Superior" if peak_y < h // 2 else "Inferior"
    horizontal = "Nasal" if peak_x < w // 2 else "Temporal"
    
    return f"{vertical}-{horizontal}"