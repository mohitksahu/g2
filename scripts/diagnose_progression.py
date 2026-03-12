"""
Quick diagnostic to verify progression head is ignoring tabular features.
Tests model with same image but different tabular inputs.

Usage:
    python scripts/diagnose_progression.py --model federated_best_fedavg.pth
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    MODELS_DIR, DATASET_DIR, HOSPITALS_DIR, HOSPITAL_NAMES,
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM, NUM_DR_CLASSES, DROPOUT, DEVICE,
)
from models import DRMultiModalNet
from dataset import get_tabular_dim
from utils import load_checkpoint
from preprocess_tabular import load_preprocessors

def create_test_tabular_inputs(scaler):
    """Create 5 patient profiles ranging from low to high risk."""
    
    # Format: [age, diabetes_duration, hba1c, systolic_bp, diastolic_bp, bmi, sex, treatment_type]
    
    profiles = {
        "Very Low Risk": [35, 2, 5.8, 110, 70, 22, 0, 0],    # Young, well-controlled
        "Low Risk":      [45, 5, 6.5, 120, 75, 25, 1, 1],    # Moderate control
        "Medium Risk":   [55, 10, 8.0, 130, 82, 28, 0, 2],   # Borderline
        "High Risk":     [62, 15, 10.0, 145, 88, 31, 1, 3],  # Poor control
        "Very High Risk": [68, 20, 12.0, 160, 95, 35, 0, 3], # Severe
    }
    
    processed_profiles = {}
    
    for name, values in profiles.items():
        # Separate continuous and categorical
        continuous = values[:6]  # age, duration, hba1c, sbp, dbp, bmi
        categorical = values[6:]  # sex, treatment_type
        
        # Normalize continuous features
        if scaler is not None:
            continuous_normalized = scaler.transform([continuous])[0].tolist()
        else:
            continuous_normalized = continuous
        
        # Combine
        full_input = continuous_normalized + categorical
        processed_profiles[name] = {
            'raw': values,
            'tensor': torch.tensor([full_input], dtype=torch.float32)
        }
    
    return processed_profiles

def test_tabular_sensitivity(model, profiles, device):
    """Test if model output varies with different tabular inputs."""
    
    model.eval()
    
    # Create a dummy image (same for all tests)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    print(f"\n{'='*80}")
    print("TABULAR FEATURE SENSITIVITY TEST")
    print("(Same image, different clinical profiles)")
    print(f"{'='*80}")
    print(f"\n{'Profile':<18} {'HbA1c':<8} {'Duration':<10} {'Progression':<12} {'Difference'}")
    print("-" * 80)
    
    results = {}
    
    with torch.no_grad():
        for name, data in profiles.items():
            tabular = data['tensor'].to(device)
            grade_logits, prog_pred = model(dummy_image, tabular)
            
            prog_value = prog_pred.item()
            results[name] = {
                'progression': prog_value,
                'hba1c': data['raw'][2],
                'duration': data['raw'][1]
            }
    
    # Print results with differences
    baseline = results["Very Low Risk"]['progression']
    
    for name, res in results.items():
        diff = res['progression'] - baseline
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        
        print(f"{name:<18} {res['hba1c']:<8.1f} {res['duration']:<10.0f} {res['progression']:<12.4f} {diff_str}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("DIAGNOSTIC RESULTS")
    print(f"{'='*80}")
    
    pred_values = [r['progression'] for r in results.values()]
    pred_variance = np.var(pred_values)
    pred_range = max(pred_values) - min(pred_values)
    
    print(f"\n  Prediction Statistics:")
    print(f"    Mean:     {np.mean(pred_values):.4f}")
    print(f"    Variance: {pred_variance:.6f}")
    print(f"    Range:    {pred_range:.4f}")
    print(f"    Min:      {min(pred_values):.4f}")
    print(f"    Max:      {max(pred_values):.4f}")
    
    # Diagnosis
    print(f"\n  Diagnosis:")
    
    if pred_variance < 0.001:
        print(f"    ❌ CRITICAL: Variance < 0.001 - Model ignoring tabular features!")
        status = "FAILED"
    elif pred_variance < 0.005:
        print(f"    ⚠️  WARNING: Low variance - Weak tabular influence")
        status = "WEAK"
    else:
        print(f"    ✅ GOOD: Sufficient variance - Tabular features working")
        status = "WORKING"
    
    if pred_range < 0:
        print(f"    ❌ CRITICAL: Range < 10% - Predictions nearly identical!")
    elif pred_range < 0.30:
        print(f"    ⚠️  WARNING: Range < 30% - Weak discrimination")
    else:
        print(f"    ✅ GOOD: Range > 30% - Strong risk stratification")
    
    # Expected behavior
    print(f"\n  Expected Behavior:")
    print(f"    Variance should be > 0.010")
    print(f"    Range should be > 0.40 (40% difference low to high risk)")
    print(f"    Very Low → Very High should increase by 60-80%")
    
    print(f"\n{'='*80}")
    
    return status, pred_variance, pred_range

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="federated_best_fedavg.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"PROGRESSION HEAD DIAGNOSTIC")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")
    
    # Get tabular dimension
    tab_dim = 6
    for hname in HOSPITAL_NAMES:
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if tab_csv.exists():
            tab_dim = get_tabular_dim(tab_csv)
            break
    
    print(f"Tabular dimension: {tab_dim}")
    
    # Load preprocessors
    preprocessor_dir = DATASET_DIR / "preprocessors"
    scaler = None
    if preprocessor_dir.exists():
        from preprocess_tabular import load_preprocessors
        _, scaler = load_preprocessors(preprocessor_dir)
        print(f"Loaded scaler from: {preprocessor_dir}")
    
    # Load model
    model_path = MODELS_DIR / args.model
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
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
    print(f"Model loaded successfully")
    
    # Create test profiles
    profiles = create_test_tabular_inputs(scaler)
    
    # Run diagnostic
    status, variance, range_val = test_tabular_sensitivity(model, profiles, DEVICE)
    
    # Recommendation
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    
    if status == "FAILED":
        print(f"\n  🔧 ACTION REQUIRED: Fine-tune progression head immediately")
        print(f"     Run: python training/finetune_progression.py --base_model {args.model}")
        print(f"\n  Expected improvements:")
        print(f"     Variance: {variance:.6f} → > 0.010")
        print(f"     Range:    {range_val:.4f} → > 0.40")
    
    elif status == "WEAK":
        print(f"\n  🔧 RECOMMENDED: Fine-tuning will improve performance")
        print(f"     Run: python training/finetune_progression.py --base_model {args.model}")
    
    else:
        print(f"\n  ✅ Model appears to be working correctly")
        print(f"     Fine-tuning may still provide marginal improvements")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()