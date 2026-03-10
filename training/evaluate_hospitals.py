"""
Per-hospital performance evaluation.
Evaluates each hospital's test set separately to measure federated fairness.

Usage:
    python training/evaluate_hospitals.py --model_path models/federated_best_fedavg.pth
    python training/evaluate_hospitals.py --compare_all
"""

import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (
    HOSPITALS_DIR, PROCESSED_HOSPITALS_DIR, MODELS_DIR, LOG_DIR,
    HOSPITAL_NAMES, BATCH_SIZE, NUM_WORKERS,
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM, NUM_DR_CLASSES, DROPOUT, DEVICE, SEED,
)
from models import DRMultiModalNet
from dataset import DRMultiModalDataset, get_tabular_dim
from utils import set_seed, load_checkpoint
from evaluate import evaluate_model_comprehensive

def evaluate_per_hospital(model, hospital_names, tab_dim):
    """Evaluate model on each hospital's test set separately."""
    
    results = {}
    
    for hname in hospital_names:
        test_img_dir = PROCESSED_HOSPITALS_DIR / hname / "test" / "images"
        test_tab_csv = HOSPITALS_DIR / hname / "test" / "tabular.csv"
        
        if not test_img_dir.exists() or not test_tab_csv.exists():
            print(f"[SKIP] {hname}: test data not found")
            continue
        
        test_dataset = DRMultiModalDataset(
            image_dir=test_img_dir,
            tabular_csv=test_tab_csv,
        )
        
        if len(test_dataset) == 0:
            print(f"[SKIP] {hname}: no test samples")
            continue
        
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        
        metrics = evaluate_model_comprehensive(model, test_loader, DEVICE, hname)
        results[hname] = metrics
        
        print(f"\n  {hname}: κ={metrics['quadratic_kappa']:.4f}, "
              f"F1={metrics['weighted_f1']:.4f}, "
              f"Samples={len(test_dataset)}")
    
    return results

def compare_federated_fairness():
    """Compare FedAvg vs FedProx fairness across hospitals."""
    
    tab_dim = 6
    for hname in HOSPITAL_NAMES:
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if tab_csv.exists():
            tab_dim = get_tabular_dim(tab_csv)
            break
    
    algorithms = {
        "FedAvg": "federated_best_fedavg.pth",
        "FedProx": "federated_best_fedprox.pth",
        "Centralized": "best_model.pth",
    }
    
    all_results = {}
    
    for algo_name, filename in algorithms.items():
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"\n[SKIP] {algo_name}: {filename} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"  Evaluating {algo_name}")
        print(f"{'='*60}")
        
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
        
        results = evaluate_per_hospital(model, HOSPITAL_NAMES, tab_dim)
        all_results[algo_name] = results
    
    # Fairness comparison table
    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print(f"  FEDERATED FAIRNESS COMPARISON (Per-Hospital κ)")
        print(f"{'='*80}")
        
        header = f"{'Hospital':<12}"
        for algo in all_results.keys():
            header += f"{algo:>15}"
        print(header)
        print("-" * 80)
        
        for hname in HOSPITAL_NAMES:
            row = f"{hname:<12}"
            for algo, results in all_results.items():
                if hname in results:
                    kappa = results[hname]['quadratic_kappa']
                    row += f"{kappa:>15.4f}"
                else:
                    row += f"{'N/A':>15}"
            print(row)
        
        # Compute std dev (fairness metric)
        print("\n" + "-" * 80)
        row = f"{'Std Dev':<12}"
        for algo, results in all_results.items():
            kappas = [r['quadratic_kappa'] for r in results.values()]
            std = np.std(kappas) if kappas else 0
            row += f"{std:>15.4f}"
        print(row)
        print("=" * 80)
        print("(Lower Std Dev = More Fair)")
    
    # Save results
    results_path = LOG_DIR / "per_hospital_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Saved to: {results_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--compare_all", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)
    
    if args.compare_all:
        compare_federated_fairness()
    else:
        model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "federated_best_fedavg.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return
        
        tab_dim = 6
        for hname in HOSPITAL_NAMES:
            tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
            if tab_csv.exists():
                tab_dim = get_tabular_dim(tab_csv)
                break
        
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
        
        results = evaluate_per_hospital(model, HOSPITAL_NAMES, tab_dim)
        
        print(f"\n{'='*60}")
        print(f"  Per-Hospital Results Summary")
        print(f"{'='*60}")
        for hname, metrics in results.items():
            print(f"{hname}: κ={metrics['quadratic_kappa']:.4f}, "
                  f"F1={metrics['weighted_f1']:.4f}")

if __name__ == "__main__":
    main()