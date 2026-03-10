"""
Explainability module:
  1. Grad-CAM — visual explanation on fundus images (which regions drove the decision)
  2. SHAP      — tabular feature importance (which clinical factors mattered)
  3. Combined clinical output generation
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import IMG_SIZE, IMG_MEAN, IMG_STD, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES


# ──────────────────────────────────────────────
# 1. Grad-CAM for Image Explanations
# ──────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Produces a heatmap showing which image regions contributed most
    to the model's prediction.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: The full DRMultiModalNet or ImageBranch
            target_layer: The CNN layer to compute Grad-CAM on
                          (typically last conv block of EfficientNet)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor,
                 tabular_tensor: torch.Tensor = None,
                 target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            image_tensor: (1, 3, H, W) preprocessed image
            tabular_tensor: (1, tab_dim) tabular features (can be zeros)
            target_class: Class index to explain (None = predicted class)

        Returns:
            heatmap: (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        if tabular_tensor is not None:
            grade_logits, prog_risk = self.model(image_tensor, tabular_tensor)
        else:
            grade_logits = self.model(image_tensor)
            if isinstance(grade_logits, tuple):
                grade_logits = grade_logits[0]

        if target_class is None:
            target_class = grade_logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = grade_logits[0, target_class]
        score.backward(retain_graph=True)

        # Compute Grad-CAM
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()  # (h, w)

        # ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        cam = cam.cpu().numpy()
        cam = np.uint8(255 * cam)

        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray(cam)
        cam_pil = cam_pil.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
        cam = np.array(cam_pil) / 255.0

        return cam


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image: (H, W, 3) original image in [0, 255] uint8
        heatmap: (H, W) Grad-CAM output in [0, 1]
        alpha: Overlay transparency

    Returns:
        (H, W, 3) overlaid image in uint8
    """
    import cv2

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(
            heatmap_colored, (image.shape[1], image.shape[0])
        )

    # Overlay
    overlaid = np.uint8(alpha * heatmap_colored + (1 - alpha) * image)
    return overlaid


# ──────────────────────────────────────────────
# 2. SHAP for Tabular Explanations
# ──────────────────────────────────────────────
def compute_shap_values(
    model: torch.nn.Module,
    tabular_input: np.ndarray,
    background_data: np.ndarray,
    feature_names: list,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Compute SHAP values for tabular features to explain
    which clinical factors contributed most to the prediction.

    MEMORY NOTE: Model is always moved to CPU for SHAP regardless of
    the device argument. SHAP (KernelExplainer) runs hundreds of forward
    passes internally. On a 6GB VRAM GPU with the model already loaded,
    this causes CUDA OOM instantly. CPU with 16GB RAM handles it safely.
    The model is moved back to the original device after SHAP completes.

    Args:
        model: Trained model (will extract tabular branch output)
        tabular_input: (1, num_features) the patient's data
        background_data: (N, num_features) reference samples
        feature_names: List of feature names
        device: original device — model is returned here after SHAP

    Returns:
        dict with 'shap_values', 'feature_names', 'feature_values'
    """
    try:
        import shap
    except ImportError:
        print("[WARN] SHAP not installed. Returning dummy values.")
        return {
            "shap_values": np.zeros_like(tabular_input),
            "feature_names": feature_names,
            "feature_values": tabular_input.flatten(),
        }

    # ── Move model to CPU before SHAP to avoid CUDA OOM ──────────────────────
    # SHAP runs ~2 * num_features * num_background_samples forward passes.
    # Each pass would allocate GPU memory. With 6GB VRAM already used by the
    # loaded model, this exhausts memory. CPU with 16GB RAM handles it safely.
    cpu_device = torch.device("cpu")
    model.to(cpu_device)
    model.eval()

    # Create a wrapper that processes only tabular data through the model
    class TabularWrapper:
        def __init__(self, full_model, img_size=IMG_SIZE):
            self.model = full_model
            self.device = cpu_device          # always CPU — never GPU
            self.img_size = img_size

        def __call__(self, tabular_np):
            self.model.eval()
            with torch.no_grad():
                tab_tensor = torch.tensor(
                    tabular_np, dtype=torch.float32
                ).to(self.device)
                # Dummy zero image — SHAP only explains tabular features
                dummy_batch = torch.zeros(
                    tab_tensor.shape[0], 3, self.img_size, self.img_size
                ).to(self.device)
                grade_logits, prog_risk = self.model(dummy_batch, tab_tensor)
                # Return progression risk as the target for SHAP
                return prog_risk.cpu().numpy()

    wrapper = TabularWrapper(model)

    # Use KernelExplainer (model-agnostic)
    explainer = shap.KernelExplainer(wrapper, background_data)
    shap_values = explainer.shap_values(tabular_input)

    # ── Move model back to original device after SHAP ─────────────────────────
    model.to(device)

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "feature_values": tabular_input.flatten(),
    }


# ──────────────────────────────────────────────
# 3. Clinical Output Generator
# ──────────────────────────────────────────────
DR_GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}

RISK_THRESHOLDS = {
    "low": (0.0, 0.3),
    "moderate": (0.3, 0.6),
    "high": (0.6, 0.8),
    "very_high": (0.8, 1.0),
}

REFERRAL_MAP = {
    "low": "Routine follow-up in 12 months",
    "moderate": "Follow-up in 6 months, optimize glycemic control",
    "high": "Refer to ophthalmologist within 4 weeks",
    "very_high": "Urgent ophthalmology referral within 1 week",
}


def generate_clinical_report(
    grade_logits: torch.Tensor,
    progression_risk: float,
    gradcam_heatmap: np.ndarray = None,
    shap_result: dict = None,
) -> dict:
    """
    Generate a clinician-readable report from model outputs.

    Returns:
        dict with structured clinical output
    """
    # DR Grade
    grade_probs = F.softmax(grade_logits, dim=-1).squeeze().detach().cpu().numpy()
    predicted_grade = int(np.argmax(grade_probs))
    confidence = float(grade_probs[predicted_grade])

    # Risk level
    risk = float(progression_risk)
    risk_level = "low"
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= risk < hi:
            risk_level = level
            break

    report = {
        "dr_grade": predicted_grade,
        "dr_grade_name": DR_GRADE_NAMES.get(predicted_grade, "Unknown"),
        "grade_confidence": round(confidence, 3),
        "grade_probabilities": {
            DR_GRADE_NAMES[i]: round(float(p), 3)
            for i, p in enumerate(grade_probs)
        },
        "progression_risk_12m": round(risk, 3),
        "risk_level": risk_level.upper().replace("_", " "),
        "recommendation": REFERRAL_MAP.get(risk_level, ""),
    }

    # Add top contributing clinical factors from SHAP
    if shap_result is not None:
        sv = shap_result["shap_values"]
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv).flatten()
        names = shap_result["feature_names"]
        values = shap_result["feature_values"]

        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        top_factors = []
        for idx in sorted_idx[:3]:  # Top 3
            top_factors.append({
                "feature": names[idx] if idx < len(names) else f"feature_{idx}",
                "value": round(float(values[idx]), 2) if idx < len(values) else 0,
                "impact": round(float(sv[idx]), 4),
                "direction": "increases risk" if sv[idx] > 0 else "decreases risk",
            })
        report["top_clinical_factors"] = top_factors

    # Add spatial explanation indicator
    if gradcam_heatmap is not None:
        # Find region with highest activation
        h, w = gradcam_heatmap.shape
        max_idx = np.unravel_index(gradcam_heatmap.argmax(), gradcam_heatmap.shape)
        quadrant_y = "superior" if max_idx[0] < h // 2 else "inferior"
        quadrant_x = "nasal" if max_idx[1] < w // 2 else "temporal"
        report["primary_attention_region"] = f"{quadrant_y} {quadrant_x} quadrant"
        report["attention_confidence"] = round(float(gradcam_heatmap.max()), 3)

    return report


def format_report_text(report: dict) -> str:
    """Format clinical report as readable text."""
    lines = [
        "=" * 50,
        "DIABETIC RETINOPATHY ASSESSMENT REPORT",
        "=" * 50,
        "",
        f"DR Grade:              {report['dr_grade_name']} (Grade {report['dr_grade']})",
        f"Grade Confidence:      {report['grade_confidence']:.1%}",
        "",
        f"12-Month Progression Risk: {report['progression_risk_12m']:.1%} -> {report['risk_level']}",
        f"Recommendation:        {report['recommendation']}",
    ]

    if "top_clinical_factors" in report:
        lines.append("")
        lines.append("Key Clinical Drivers:")
        for f in report["top_clinical_factors"]:
            lines.append(
                f"  - {f['feature']} = {f['value']} "
                f"({f['direction']}, impact: {f['impact']:.4f})"
            )

    if "primary_attention_region" in report:
        lines.append("")
        lines.append(f"Primary Image Attention: {report['primary_attention_region']}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)