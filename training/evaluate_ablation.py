"""
Ablation study evaluation.
Tests image-only, tabular-only, fusion types, and progression head variants.

Usage:
    python training/evaluate_ablation.py
"""

import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (
    VAL_DIR, PROCESSED_VAL_DIR, MODELS_DIR, LOG_DIR,
    BATCH_SIZE, NUM_WORKERS,
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM, NUM_DR_CLASSES, DROPOUT, DEVICE, SEED,
    HOSPITALS_DIR, HOSPITAL_NAMES,
)
from models import DRMultiModalNet, DRImageOnlyNet
from dataset import DRMultiModalDataset, get_tabular_dim
from utils import set_seed, load_checkpoint
from evaluate import evaluate_model_comprehensive

def main():
    set_seed(SEED)
    
    tab_dim = 6
    for hname in HOSPITAL_NAMES:
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if tab_csv.exists():
            tab_dim = get_tabular_dim(tab_csv)
            break
    
    val_dataset = DRMultiModalDataset(
        image_dir=PROCESSED_VAL_DIR / "images",
        tabular_csv=VAL_DIR / "tabular.csv",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    
    ablation_configs = {
        "Full Model (Cross-Attention)": {
            "model_class": DRMultiModalNet,
            "fusion": "cross_attention",
            "image_only": False,
            "filename": "best_model.pth",
        },
        "Concat Fusion": {
            "model_class": DRMultiModalNet,
            "fusion": "concat",
            "image_only": False,
            "filename": "ablation_concat.pth",
        },
        "Image Only": {
            "model_class": DRImageOnlyNet,
            "fusion": None,
            "image_only": True,
            "filename": "ablation_image_only.pth",
        },
    }
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY")
    print(f"{'='*60}")
    
    for name, config in ablation_configs.items():
        model_path = MODELS_DIR / config['filename']
        
        # Try fallback to best_model.pth if specific ablation not found
        if not model_path.exists() and config['filename'] != "best_model.pth":
            print(f"\n[SKIP] {name}: {config['filename']} not found")
            print(f"  To generate: train with --fusion {config['fusion']} or --image_only")
            continue
        
        if not model_path.exists():
            print(f"\n[SKIP] {name}: {config['filename']} not found")
            continue
        
        # Load model
        if config['image_only']:
            model = DRImageOnlyNet(
                cnn_backbone=CNN_BACKBONE,
                num_classes=NUM_DR_CLASSES,
                dropout=DROPOUT,
            )
        else:
            model = DRMultiModalNet(
                tabular_input_dim=tab_dim,
                cnn_backbone=CNN_BACKBONE,
                tabular_embed_dim=TABULAR_EMBED_DIM,
                fusion_dim=FUSION_DIM,
                num_classes=NUM_DR_CLASSES,
                dropout=DROPOUT,
                fusion_type=config['fusion'],
            )
        
        load_checkpoint(model, model_path, device=DEVICE)
        model = model.to(DEVICE)
        
        metrics = evaluate_model_comprehensive(model, val_loader, DEVICE, name)
        results[name] = metrics
        
        print(f"\n  {name}:")
        print(f"    κ:  {metrics['quadratic_kappa']:.4f}")
        print(f"    F1: {metrics['weighted_f1']:.4f}")
        print(f"    Prog AUC: {metrics['progression_auc']:.4f}")
    
    # Comparison table
    if len(results) > 1:
        print(f"\n\n{'='*80}")
        print(f"  ABLATION STUDY SUMMARY")
        print(f"{'='*80}")
        print(f"{'Variant':<35} {'κ':>10} {'F1':>10} {'Prog AUC':>12}")
        print("-" * 80)
        
        for name, m in results.items():
            print(f"{name:<35} {m['quadratic_kappa']:>10.4f} "
                  f"{m['weighted_f1']:>10.4f} {m['progression_auc']:>12.4f}")
        
        print("=" * 80)
    
    # Save
    output_path = LOG_DIR / "ablation_study.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()