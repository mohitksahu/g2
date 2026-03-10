"""
Enhanced model evaluation with comprehensive metrics.
Includes all classification metrics, federated comparison, and detailed analysis.

Usage:
    python training/evaluate.py --model_path models/best_model.pth
    python training/evaluate.py --compare
    python training/evaluate.py --export_onnx
"""

import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (
    DATASET_DIR, HOSPITALS_DIR, PROCESSED_HOSPITALS_DIR,
    VAL_DIR, PROCESSED_VAL_DIR, MODELS_DIR, LOG_DIR,
    HOSPITAL_NAMES,
    BATCH_SIZE, NUM_WORKERS, CNN_BACKBONE,
    TABULAR_EMBED_DIM, FUSION_DIM, NUM_DR_CLASSES, DROPOUT, DEVICE, SEED,
)
from models import DRMultiModalNet, DRImageOnlyNet
from dataset import DRMultiModalDataset, get_tabular_dim
from utils import set_seed, load_checkpoint, export_to_onnx

def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """
    Compute Quadratic Weighted Kappa (Cohen's Kappa with quadratic weights).
    PyTorch native implementation.
    """
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    
    # Confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    
    # Weight matrix (quadratic)
    weights = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    
    # Normalize confusion matrix
    hist_true = confusion.sum(dim=1)
    hist_pred = confusion.sum(dim=0)
    expected = torch.outer(hist_true, hist_pred) / confusion.sum()
    
    # Compute kappa
    numerator = (weights * confusion).sum()
    denominator = (weights * expected).sum()
    
    if denominator == 0:
        return 0.0
    
    kappa = 1 - (numerator / denominator)
    return kappa.item()

def compute_sensitivity_specificity(y_true, y_pred, num_classes=5):
    """Compute per-class sensitivity and specificity."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    sens_spec = {}
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sens_spec[f"grade_{cls}"] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "support": int(np.sum(y_true == cls))
        }
    
    return sens_spec

def compute_weighted_f1(y_true, y_pred, num_classes=5):
    """Compute weighted F1-score using PyTorch."""
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    
    f1_scores = []
    weights = []
    
    for cls in range(num_classes):
        tp = ((y_true == cls) & (y_pred == cls)).sum().float()
        fp = ((y_true != cls) & (y_pred == cls)).sum().float()
        fn = ((y_true == cls) & (y_pred != cls)).sum().float()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        support = (y_true == cls).sum().item()
        f1_scores.append(f1.item())
        weights.append(support)
    
    weighted_f1 = np.average(f1_scores, weights=weights)
    return weighted_f1, f1_scores

def compute_auc_roc(y_true, y_prob):
    """Compute AUC-ROC for progression prediction using PyTorch."""
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_prob = torch.tensor(y_prob, dtype=torch.float32)
    
    # Sort by predicted probability
    sorted_indices = torch.argsort(y_prob, descending=True)
    y_true_sorted = y_true[sorted_indices]
    
    # Compute TPR and FPR
    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.cumsum(1 - y_true_sorted, dim=0)
    
    total_pos = y_true.sum()
    total_neg = (1 - y_true).sum()
    
    if total_pos == 0 or total_neg == 0:
        return 0.5
    
    tpr = tps / total_pos
    fpr = fps / total_neg
    
    # Compute AUC using trapezoidal rule
    auc = torch.trapz(tpr, fpr).item()
    return abs(auc)

def compute_confusion_matrix(y_true, y_pred, num_classes=5):
    """Compute confusion matrix using PyTorch."""
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    
    return confusion.numpy()

@torch.no_grad()
def evaluate_model_comprehensive(model, dataloader, device, model_name="Model"):
    """
    Comprehensive evaluation with all requested metrics.
    """
    model.eval()
    
    all_grades_true = []
    all_grades_pred = []
    all_grades_prob = []
    all_prog_true = []
    all_prog_pred = []
    
    print(f"\nEvaluating {model_name}...")
    for batch in tqdm(dataloader, desc=f"  {model_name}"):
        images, tabular, grades, progressions = batch[:4]
        images = images.to(device)
        tabular = tabular.to(device)
        
        grade_logits, prog_pred = model(images, tabular)
        
        probs = F.softmax(grade_logits, dim=1).cpu().numpy()
        all_grades_true.extend(grades.numpy())
        all_grades_pred.extend(grade_logits.argmax(dim=1).cpu().numpy())
        all_grades_prob.append(probs)
        all_prog_true.extend(progressions.numpy())
        all_prog_pred.extend(prog_pred.squeeze().cpu().numpy())
    
    all_grades_true = np.array(all_grades_true)
    all_grades_pred = np.array(all_grades_pred)
    all_grades_prob = np.vstack(all_grades_prob)
    all_prog_true = np.array(all_prog_true)
    all_prog_pred = np.array(all_prog_pred)
    
    # Classification metrics
    accuracy = (all_grades_true == all_grades_pred).mean()
    kappa = quadratic_weighted_kappa(all_grades_true, all_grades_pred)
    weighted_f1, per_class_f1 = compute_weighted_f1(all_grades_true, all_grades_pred)
    sens_spec = compute_sensitivity_specificity(all_grades_true, all_grades_pred)
    confusion = compute_confusion_matrix(all_grades_true, all_grades_pred)
    
    # Progression metrics
    prog_true_binary = (all_prog_true > 0.5).astype(int)
    prog_pred_binary = (all_prog_pred > 0.5).astype(int)
    prog_auc = compute_auc_roc(prog_true_binary, all_prog_pred)
    
    metrics = {
        "accuracy": float(accuracy),
        "quadratic_kappa": float(kappa),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": [float(f1) for f1 in per_class_f1],
        "sensitivity_specificity": sens_spec,
        "confusion_matrix": confusion.tolist(),
        "progression_auc": float(prog_auc),
        "progression_accuracy": float((prog_true_binary == prog_pred_binary).mean()),
    }
    
    return metrics

def print_detailed_results(name, metrics):
    """Print comprehensive evaluation results."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    print(f"\n  📊 CLASSIFICATION METRICS:")
    print(f"    Accuracy:              {metrics['accuracy']:.4f}")
    print(f"    Quadratic Kappa (κ):   {metrics['quadratic_kappa']:.4f}")
    print(f"    Weighted F1-Score:     {metrics['weighted_f1']:.4f}")
    
    print(f"\n  📈 PER-CLASS F1-SCORES:")
    for i, f1 in enumerate(metrics['per_class_f1']):
        print(f"    Grade {i}: {f1:.4f}")
    
    print(f"\n  🎯 SENSITIVITY & SPECIFICITY:")
    for grade, vals in metrics['sensitivity_specificity'].items():
        print(f"    {grade}: Sens={vals['sensitivity']:.4f}, "
              f"Spec={vals['specificity']:.4f}, Support={vals['support']}")
    
    print(f"\n  🔮 PROGRESSION PREDICTION:")
    print(f"    AUC-ROC:    {metrics['progression_auc']:.4f}")
    print(f"    Accuracy:   {metrics['progression_accuracy']:.4f}")
    
    print(f"\n  📋 CONFUSION MATRIX:")
    cm = np.array(metrics['confusion_matrix'])
    print("       ", " ".join([f"P{i}" for i in range(5)]))
    for i, row in enumerate(cm):
        print(f"    T{i}: {' '.join([f'{val:3d}' for val in row])}")

def compare_models_comprehensive(test_loader, tab_dim):
    """Compare all trained models with comprehensive metrics."""
    
    model_configs = {
        "Centralized (Best Loss)": ("best_model.pth", "cross_attention", False),
        "Centralized (Best Kappa)": ("best_kappa_model.pth", "cross_attention", False),
        "Federated FedAvg": ("federated_best_fedavg.pth", "cross_attention", False),
        "Federated FedProx": ("federated_best_fedprox.pth", "cross_attention", False),
        "Image-Only Baseline": ("image_only_best.pth", "cross_attention", True),
    }
    
    all_results = {}
    
    for name, (filename, fusion, img_only) in model_configs.items():
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"\n[SKIP] {name}: {filename} not found")
            continue
        
        # Load model
        if img_only:
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
                fusion_type=fusion,
            )
        
        load_checkpoint(model, model_path, device=DEVICE)
        model = model.to(DEVICE)
        
        # Evaluate
        metrics = evaluate_model_comprehensive(model, test_loader, DEVICE, name)
        print_detailed_results(name, metrics)
        
        all_results[name] = metrics
    
    # Comparison table
    if len(all_results) > 1:
        print(f"\n\n{'='*90}")
        print(f"  📊 MODEL COMPARISON SUMMARY")
        print(f"{'='*90}")
        print(f"{'Model':<30} {'Accuracy':>10} {'κ (Quad)':>10} {'F1 (Wtd)':>10} {'Prog AUC':>10}")
        print(f"{'-'*90}")
        
        for name, m in all_results.items():
            print(f"{name:<30} {m['accuracy']:>10.4f} {m['quadratic_kappa']:>10.4f} "
                  f"{m['weighted_f1']:>10.4f} {m['progression_auc']:>10.4f}")
        
        print(f"{'='*90}")
    
    # Save results
    results_path = LOG_DIR / "comprehensive_evaluation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    return all_results

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive DR Model Evaluation")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)
    
    # Get tabular dimension
    tab_dim = 6
    for hname in HOSPITAL_NAMES:
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if tab_csv.exists():
            tab_dim = get_tabular_dim(tab_csv)
            break
    
    # Load test data (using global validation set)
    val_img_dir = PROCESSED_VAL_DIR / "images"
    val_tab_csv = VAL_DIR / "tabular.csv"
    
    test_dataset = DRMultiModalDataset(
        image_dir=val_img_dir,
        tabular_csv=val_tab_csv,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    if args.compare:
        compare_models_comprehensive(test_loader, tab_dim)
    else:
        model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "best_model.pth"
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
        
        metrics = evaluate_model_comprehensive(model, test_loader, DEVICE, model_path.name)
        print_detailed_results(model_path.name, metrics)
        
        if args.export_onnx:
            onnx_path = model_path.with_suffix(".onnx")
            export_to_onnx(model, tab_dim, onnx_path)

if __name__ == "__main__":
    main()