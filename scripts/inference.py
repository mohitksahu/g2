"""
Single-patient inference pipeline.
Takes a fundus image + optional clinical data → outputs DR assessment with explanations.

Usage:
    python scripts/inference.py --image path/to/fundus.jpg
    python scripts/inference.py --image fundus.jpg --age 58 --diabetes_duration 12 --hba1c 10.2
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    IMG_SIZE, IMG_MEAN, IMG_STD, MODELS_DIR, DATASET_DIR,
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM,
    NUM_DR_CLASSES, DROPOUT, DEVICE,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES,
)
from models import DRMultiModalNet
from preprocess_images import apply_clahe, auto_crop_fundus
from preprocess_tabular import load_preprocessors
from explainability import (
    GradCAM, overlay_gradcam, compute_shap_values,
    generate_clinical_report, format_report_text,
)
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="DR Inference")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to fundus image")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model checkpoint (default: best_model.pth)")
    parser.add_argument("--fusion", type=str, default="cross_attention")

    # Clinical data (optional)
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--diabetes_duration", type=float, default=None)
    parser.add_argument("--hba1c", type=float, default=None)
    parser.add_argument("--systolic_bp", type=float, default=None)
    parser.add_argument("--diastolic_bp", type=float, default=None)
    parser.add_argument("--bmi", type=float, default=None)
    parser.add_argument("--sex", type=str, default=None)
    parser.add_argument("--diabetes_type", type=str, default=None)

    # Output
    parser.add_argument("--save_heatmap", type=str, default=None,
                        help="Save Grad-CAM overlay to this path")
    parser.add_argument("--no_xai", action="store_true",
                        help="Skip explainability (faster)")

    return parser.parse_args()


def preprocess_image_for_inference(image_path: str) -> tuple:
    """
    Preprocess a single fundus image for model inference.
    Returns (image_tensor, original_image_np)
    """
    import cv2

    # Load original
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Preprocess (same pipeline as training)
    img = auto_crop_fundus(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Keep a copy for visualization
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    pil_img = Image.fromarray(original_rgb)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 224, 224)

    return tensor, original_rgb


def build_tabular_input(args, encoders=None, scaler=None) -> tuple:
    """
    Build tabular feature tensor from CLI arguments.
    Returns (tabular_tensor, feature_names, raw_values)
    """
    # Gather raw values
    raw_data = {
        "age": args.age,
        "diabetes_duration": args.diabetes_duration,
        "hba1c": args.hba1c,
        "systolic_bp": args.systolic_bp,
        "diastolic_bp": args.diastolic_bp,
        "bmi": args.bmi,
    }

    feature_names = []
    values = []

    for feat in CONTINUOUS_FEATURES:
        feature_names.append(feat)
        val = raw_data.get(feat)
        if val is not None:
            values.append(float(val))
        else:
            values.append(0.0)  # Mean imputation (scaled 0 = mean)

    # Categorical features (simplified — encode as numeric)
    for feat in CATEGORICAL_FEATURES:
        if feat in ["sex", "diabetes_type"]:
            feature_names.append(feat)
            raw_val = getattr(args, feat, None)
            if raw_val is not None and encoders and feat in encoders:
                le = encoders[feat]
                if raw_val in le.classes_:
                    values.append(float(le.transform([raw_val])[0]))
                else:
                    values.append(0.0)
            else:
                values.append(0.0)

    # Apply scaler if available
    raw_values = values.copy()
    if scaler is not None:
        # Only scale continuous features
        cont_vals = np.array(values[:len(CONTINUOUS_FEATURES)]).reshape(1, -1)
        cont_scaled = scaler.transform(cont_vals).flatten()
        values[:len(CONTINUOUS_FEATURES)] = cont_scaled.tolist()

    tabular_tensor = torch.tensor([values], dtype=torch.float32)
    has_clinical = any(v != 0 for v in raw_values[:len(CONTINUOUS_FEATURES)])

    return tabular_tensor, feature_names, raw_values, has_clinical


def run_inference(args):
    """Main inference pipeline."""
    print(f"\n{'='*50}")
    print("DR ASSESSMENT - INFERENCE")
    print(f"{'='*50}")

    # Determine tab_dim and load model
    tabular_csv = DATASET_DIR / "tabular_processed.csv"
    if tabular_csv.exists():
        from dataset import get_tabular_dim
        tab_dim = get_tabular_dim(tabular_csv)
    else:
        tab_dim = len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FEATURES)

    # Load preprocessors
    preprocessor_dir = DATASET_DIR / "preprocessors"
    encoders, scaler = {}, None
    if preprocessor_dir.exists():
        encoders, scaler = load_preprocessors(preprocessor_dir)

    # Load model
    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    model = DRMultiModalNet(
        tabular_input_dim=tab_dim,
        cnn_backbone=CNN_BACKBONE,
        tabular_embed_dim=TABULAR_EMBED_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_DR_CLASSES,
        dropout=DROPOUT,
        fusion_type=args.fusion,
    )
    load_checkpoint(model, model_path, device=DEVICE)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded: {model_path.name}")

    # Preprocess image
    image_tensor, original_image = preprocess_image_for_inference(args.image)
    image_tensor = image_tensor.to(DEVICE)
    print(f"Image: {args.image}")

    # Build tabular input
    tabular_tensor, feature_names, raw_values, has_clinical = build_tabular_input(
        args, encoders, scaler
    )
    tabular_tensor = tabular_tensor.to(DEVICE)

    if has_clinical:
        print(f"Clinical data provided: {dict(zip(feature_names[:len(CONTINUOUS_FEATURES)], raw_values[:len(CONTINUOUS_FEATURES)]))}")
    else:
        print("No clinical data provided (image-only mode)")

    # Forward pass
    with torch.no_grad():
        grade_logits, prog_risk = model(image_tensor, tabular_tensor)

    # Explainability
    gradcam_heatmap = None
    shap_result = None

    if not args.no_xai:
        # Grad-CAM
        try:
            backbone = model.get_image_backbone()
            # Get last conv layer
            target_layer = None
            for name, module in backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
            if target_layer is not None:
                gradcam = GradCAM(model, target_layer)
                gradcam_heatmap = gradcam.generate(image_tensor, tabular_tensor)
                print("Grad-CAM generated")
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

        # SHAP (if clinical data provided)
        if has_clinical:
            try:
                bg_data = np.zeros((10, len(feature_names)))  # Simple background
                shap_result = compute_shap_values(
                    model, np.array([raw_values[:len(feature_names)]]),
                    bg_data, feature_names, DEVICE,
                )
                print("SHAP values computed")
            except Exception as e:
                print(f"SHAP failed: {e}")

    # Generate clinical report
    report = generate_clinical_report(
        grade_logits, prog_risk.item(),
        gradcam_heatmap, shap_result,
    )

    # Print report
    print(format_report_text(report))

    # Save Grad-CAM overlay
    if gradcam_heatmap is not None and args.save_heatmap:
        overlaid = overlay_gradcam(original_image, gradcam_heatmap)
        from PIL import Image as PILImage
        PILImage.fromarray(overlaid).save(args.save_heatmap)
        print(f"Heatmap saved: {args.save_heatmap}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
