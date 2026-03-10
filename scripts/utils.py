"""
Utility functions:
  - Seeding for reproducibility
  - Metrics computation (accuracy, AUC, kappa, F1)
  - Logging setup
  - Model save/load
  - ONNX export
"""

import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error,
)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    num_classes: int = 5,
) -> dict:
    """
    Compute comprehensive classification metrics for DR grading.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        num_classes: Number of DR classes
    
    Returns:
        dict of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1_val in enumerate(f1_per_class):
        metrics[f"f1_class_{i}"] = f1_val

    # AUC (if probabilities available)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
                metrics["auc_weighted"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
        except ValueError:
            pass  # Not enough classes in batch

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )

    return metrics


def compute_progression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Compute metrics for progression risk prediction.
    
    Args:
        y_true: Ground truth (binary or continuous risk)
        y_pred: Predicted risk scores (0-1)
        threshold: Binarization threshold
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = (y_true >= threshold).astype(int)

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
    }

    # Binary metrics (if applicable)
    if len(np.unique(y_true_binary)) > 1:
        metrics["progression_accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
        metrics["progression_f1"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        try:
            metrics["progression_auc"] = roc_auc_score(y_true_binary, y_pred)
        except ValueError:
            pass

    return metrics


def format_metrics(metrics: dict) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    for key, val in metrics.items():
        if key == "confusion_matrix":
            continue
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
def setup_logger(log_dir: Path, name: str = "training") -> logging.Logger:
    """Setup file + console logger."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(stream=open(sys.stdout.fileno(), 
        mode='w', encoding='utf-8', closefd=False))
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ──────────────────────────────────────────────
# Model Save / Load
# ──────────────────────────────────────────────
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    scheduler=None,
):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, path: Path,
                    optimizer=None, scheduler=None, device="cpu"):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


# ──────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────
def export_to_onnx(
    model: nn.Module,
    tabular_input_dim: int,
    output_path: Path,
    img_size: int = 224,
    device: str = "cpu",
):
    """
    Export model to ONNX format for deployment.
    
    Args:
        model: Trained PyTorch model
        tabular_input_dim: Number of tabular features
        output_path: Where to save .onnx file
        img_size: Input image size
        device: cpu or cuda
    """
    model.eval()
    model = model.to(device)

    dummy_image = torch.randn(1, 3, img_size, img_size).to(device)
    dummy_tabular = torch.randn(1, tabular_input_dim).to(device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_image, dummy_tabular),
        str(output_path),
        input_names=["fundus_image", "clinical_data"],
        output_names=["dr_grade_logits", "progression_risk"],
        dynamic_axes={
            "fundus_image": {0: "batch_size"},
            "clinical_data": {0: "batch_size"},
            "dr_grade_logits": {0: "batch_size"},
            "progression_risk": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"ONNX model exported to: {output_path}")


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────
class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
