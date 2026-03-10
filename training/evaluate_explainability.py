"""
Comprehensive explainability evaluation.
Generates Grad-CAM heatmaps and SHAP analysis for test samples.

Usage:
    python training/evaluate_explainability.py --num_samples 20
"""

import sys
import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (
    VAL_DIR, PROCESSED_VAL_DIR, MODELS_DIR, LOG_DIR,
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM, NUM_DR_CLASSES, DROPOUT, DEVICE, SEED,
    HOSPITALS_DIR, HOSPITAL_NAMES, IMG_SIZE,
)
from models import DRMultiModalNet
from dataset import DRMultiModalDataset, get_tabular_dim
from explainability import GradCAM, overlay_gradcam, compute_shap_values
from utils import set_seed, load_checkpoint

def generate_gradcam_samples(model, dataloader, num_samples=20):
    """Generate Grad-CAM heatmaps for test samples."""

    output_dir = LOG_DIR / "gradcam_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get target layer
    backbone = model.get_image_backbone()
    target_layer = None
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        print("[ERROR] Could not find Conv2d layer for Grad-CAM")
        return

    gradcam = GradCAM(model, target_layer)

    model.eval()
    count = 0

    print(f"\n{'='*60}")
    print(f"  Generating Grad-CAM Heatmaps")
    print(f"{'='*60}")

    for batch in tqdm(dataloader, desc="  Processing"):
        if count >= num_samples:
            break

        images, tabular, grades, progressions, meta = batch
        images = images.to(DEVICE)
        tabular = tabular.to(DEVICE)

        for i in range(images.size(0)):
            if count >= num_samples:
                break

            img_tensor = images[i:i+1]
            tab_tensor = tabular[i:i+1]
            true_grade = grades[i].item()

            # Generate heatmap
            heatmap = gradcam.generate(img_tensor, tab_tensor)

            # Load original image
            img_path = meta['image_path'][i] if 'image_path' in meta else None
            if img_path and Path(img_path).exists():
                original = cv2.imread(str(img_path))
                original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            else:
                # Use tensor
                img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                original = img_np

            # Overlay
            overlaid = overlay_gradcam(original, heatmap, alpha=0.4)

            # Save
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original)
            axes[0].set_title(f"Original (Grade {true_grade})")
            axes[0].axis('off')

            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis('off')

            axes[2].imshow(overlaid)
            axes[2].set_title("Overlay")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(output_dir / f"gradcam_{count:03d}_grade{true_grade}.png",
                       dpi=150, bbox_inches='tight')
            plt.close()

            count += 1

    print(f"\n[OK] Generated {count} Grad-CAM visualizations")
    print(f"Saved to: {output_dir}")

def analyze_shap_importance(model, dataloader, num_samples=100):
    """Analyze SHAP feature importance across test samples.

    MEMORY NOTE: Background samples capped at 10 and SHAP samples at 5
    to prevent CUDA OOM on 6GB VRAM. compute_shap_values() moves the
    model to CPU automatically for SHAP and returns it to GPU after.
    """

    model.eval()

    # Collect background data
    background_data = []
    tabular_data = []

    for batch in dataloader:
        _, tabular, _, _, _ = batch
        background_data.append(tabular.numpy())
        tabular_data.append(tabular.numpy())
        if len(background_data) * tabular.size(0) >= num_samples:
            break

    background_data = np.vstack(background_data)[:num_samples]
    tabular_data = np.vstack(tabular_data)[:num_samples]

    feature_names = ["age", "diabetes_duration", "hba1c", "systolic_bp", "diastolic_bp", "bmi"]

    print(f"\n{'='*60}")
    print(f"  SHAP Feature Importance Analysis")
    print(f"{'='*60}")

    all_shap_values = []

    # ── CHANGE 1: reduced from 20 to 5 samples to avoid OOM ─────────────────
    # Each sample runs ~2 * features * background_samples forward passes.
    # 5 samples x 8 features x 10 background = 800 passes — safe for 16GB RAM.
    for i in tqdm(range(min(5, len(tabular_data))), desc="  Computing SHAP"):

        # ── CHANGE 2: reduced background from 50 to 10 to avoid OOM ─────────
        # compute_shap_values() moves model to CPU automatically.
        shap_result = compute_shap_values(
            model,
            tabular_data[i:i+1],
            background_data[:10],           # was 50 — reduced to 10
            feature_names,
            DEVICE,
        )

        if shap_result['shap_values'] is not None:
            sv = shap_result['shap_values']
            if isinstance(sv, list):
                sv = sv[0]
            all_shap_values.append(np.array(sv).flatten())

    if all_shap_values:
        all_shap_values = np.vstack(all_shap_values)
        mean_abs_shap = np.abs(all_shap_values).mean(axis=0)

        print(f"\n  Mean Absolute SHAP Values (Feature Importance):")
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        for idx in sorted_idx:
            if idx < len(feature_names):
                print(f"    {feature_names[idx]:<20}: {mean_abs_shap[idx]:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_names)), mean_abs_shap[:len(feature_names)])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Feature Importance for Progression Prediction')
        ax.invert_yaxis()
        plt.tight_layout()

        output_path = LOG_DIR / "shap_feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[OK] SHAP plot saved to: {output_path}")

        # Verify HbA1c and diabetes_duration are top contributors
        hba1c_rank = list(sorted_idx).index(2) + 1 if 2 < len(sorted_idx) else None
        duration_rank = list(sorted_idx).index(1) + 1 if 1 < len(sorted_idx) else None

        print(f"\n  Key Clinical Validation:")
        if hba1c_rank:
            print(f"    HbA1c rank: #{hba1c_rank}")
        if duration_rank:
            print(f"    Diabetes duration rank: #{duration_rank}")

        if hba1c_rank and duration_rank and (hba1c_rank <= 3 or duration_rank <= 3):
            print(f"  [OK] Clinical validity confirmed: HbA1c and/or diabetes_duration in top 3")
        else:
            print(f"  [WARN] Expected risk factors not in top 3")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)

    # Load model
    tab_dim = 6
    for hname in HOSPITAL_NAMES:
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if tab_csv.exists():
            tab_dim = get_tabular_dim(tab_csv)
            break

    model_path = MODELS_DIR / args.model_path
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model = DRMultiModalNet(
        tabular_input_dim=tab_dim,
        cnn_backbone=CNN_BACKBONE,
        tabular_embed_dim=TABULAR_EMBED_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_DR_CLASSES,
        dropout=DROPOUT,
        fusion_type="cross_attention",
    )
    load_checkpoint(model, model_path, device=DEVICE)
    model = model.to(DEVICE)

    # Load validation data with metadata
    val_dataset = DRMultiModalDataset(
        image_dir=PROCESSED_VAL_DIR / "images",
        tabular_csv=VAL_DIR / "tabular.csv",
        return_meta=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Generate Grad-CAM
    generate_gradcam_samples(model, val_loader, args.num_samples)

    # SHAP analysis
    analyze_shap_importance(model, val_loader, num_samples=100)

    print(f"\n{'='*60}")
    print(f"  Explainability Evaluation Complete")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()