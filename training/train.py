"""
Centralized training script (baseline).
Trains the multi-modal DR model on the full training set.

Usage:
    python training/train.py
    python training/train.py --fusion concat --epochs 30
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Add scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (
    DATASET_DIR, HOSPITALS_DIR, PROCESSED_HOSPITALS_DIR,
    VAL_DIR, PROCESSED_VAL_DIR, MODELS_DIR, LOG_DIR,
    HOSPITAL_NAMES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    SCHEDULER_PATIENCE, EARLY_STOP_PATIENCE, NUM_WORKERS,
    LOSS_WEIGHT_GRADE, LOSS_WEIGHT_PROGRESSION,
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM,
    NUM_DR_CLASSES, DROPOUT, DEVICE, SEED,
)
from models import DRMultiModalNet, DRImageOnlyNet
from dataset import DRMultiModalDataset, get_tabular_dim
from utils import (
    set_seed, setup_logger, save_checkpoint, load_checkpoint,
    compute_classification_metrics, compute_progression_metrics,
    format_metrics, EarlyStopping,
)
from torch.utils.data import ConcatDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train DR Multi-Modal Model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--fusion", type=str, default="cross_attention",
                        choices=["cross_attention", "concat"])
    parser.add_argument("--image_only", action="store_true",
                        help="Train image-only baseline (no tabular)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion_grade, criterion_prog,
                    optimizer, device, loss_w_grade, loss_w_prog):
    """Train for one epoch. Returns avg loss."""
    model.train()
    total_loss = 0.0
    num_samples = 0

    all_grades_true = []
    all_grades_pred = []
    all_prog_true = []
    all_prog_pred = []

    for batch in tqdm(dataloader, desc="  Train", leave=False):
        images, tabular, grades, progressions = batch[:4]
        images = images.to(device)
        tabular = tabular.to(device)
        grades = grades.to(device)
        progressions = progressions.to(device)

        optimizer.zero_grad()

        grade_logits, prog_pred = model(images, tabular)

        loss_g = criterion_grade(grade_logits, grades)
        loss_p = criterion_prog(prog_pred.squeeze(), progressions)
        loss = loss_w_grade * loss_g + loss_w_prog * loss_p

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

        # Collect predictions
        all_grades_true.extend(grades.cpu().numpy())
        all_grades_pred.extend(grade_logits.argmax(dim=1).cpu().numpy())
        all_prog_true.extend(progressions.cpu().numpy())
        all_prog_pred.extend(prog_pred.squeeze().detach().cpu().numpy())

    avg_loss = total_loss / max(num_samples, 1)

    # Compute train metrics
    grade_metrics = compute_classification_metrics(
        np.array(all_grades_true), np.array(all_grades_pred)
    )
    prog_metrics = compute_progression_metrics(
        np.array(all_prog_true), np.array(all_prog_pred)
    )

    return avg_loss, grade_metrics, prog_metrics


@torch.no_grad()
def validate(model, dataloader, criterion_grade, criterion_prog,
             device, loss_w_grade, loss_w_prog):
    """Validate model. Returns avg loss and metrics."""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    all_grades_true = []
    all_grades_pred = []
    all_grades_prob = []
    all_prog_true = []
    all_prog_pred = []

    for batch in tqdm(dataloader, desc="  Val", leave=False):
        images, tabular, grades, progressions = batch[:4]
        images = images.to(device)
        tabular = tabular.to(device)
        grades = grades.to(device)
        progressions = progressions.to(device)

        grade_logits, prog_pred = model(images, tabular)

        loss_g = criterion_grade(grade_logits, grades)
        loss_p = criterion_prog(prog_pred.squeeze(), progressions)
        loss = loss_w_grade * loss_g + loss_w_prog * loss_p

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

        probs = torch.softmax(grade_logits, dim=1).cpu().numpy()
        all_grades_true.extend(grades.cpu().numpy())
        all_grades_pred.extend(grade_logits.argmax(dim=1).cpu().numpy())
        all_grades_prob.append(probs)
        all_prog_true.extend(progressions.cpu().numpy())
        all_prog_pred.extend(prog_pred.squeeze().cpu().numpy())

    avg_loss = total_loss / max(num_samples, 1)

    grade_metrics = compute_classification_metrics(
        np.array(all_grades_true),
        np.array(all_grades_pred),
        np.vstack(all_grades_prob) if all_grades_prob else None,
    )
    prog_metrics = compute_progression_metrics(
        np.array(all_prog_true), np.array(all_prog_pred)
    )

    return avg_loss, grade_metrics, prog_metrics


def main():
    args = parse_args()
    set_seed(SEED)

    # Logger
    logger = setup_logger(LOG_DIR, name="centralized_training")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Args: {vars(args)}")

    # Tabular dimension — read from any hospital's tabular.csv
    tab_dim = 6  # default
    for hname in HOSPITAL_NAMES:
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if tab_csv.exists():
            tab_dim = get_tabular_dim(tab_csv)
            break
    logger.info(f"Tabular input dim: {tab_dim}")

    # Centralized baseline: merge ALL hospital train sets into one dataset
    train_datasets = []
    for hname in HOSPITAL_NAMES:
        img_dir = PROCESSED_HOSPITALS_DIR / hname / "train" / "images"
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if img_dir.exists() and tab_csv.exists():
            ds = DRMultiModalDataset(
                image_dir=img_dir,
                tabular_csv=tab_csv,
                transform=DRMultiModalDataset.get_train_transform(),
            )
            train_datasets.append(ds)
            logger.info(f"  {hname}/train: {len(ds)} samples")

    if not train_datasets:
        logger.error("No hospital training data found. Run partition_hospitals.py first.")
        return

    train_dataset = ConcatDataset(train_datasets)

    # Validation set (global)
    val_img_dir = PROCESSED_VAL_DIR / "images"
    val_tab_csv = VAL_DIR / "tabular.csv"
    val_dataset = DRMultiModalDataset(
        image_dir=val_img_dir,
        tabular_csv=val_tab_csv,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Model
    if args.image_only:
        model = DRImageOnlyNet(
            cnn_backbone=CNN_BACKBONE,
            num_classes=NUM_DR_CLASSES,
            dropout=DROPOUT,
        )
        logger.info("Model: Image-Only Baseline")
    else:
        model = DRMultiModalNet(
            tabular_input_dim=tab_dim,
            cnn_backbone=CNN_BACKBONE,
            tabular_embed_dim=TABULAR_EMBED_DIM,
            fusion_dim=FUSION_DIM,
            num_classes=NUM_DR_CLASSES,
            dropout=DROPOUT,
            fusion_type=args.fusion,
        )
        logger.info(f"Model: Multi-Modal with {args.fusion} fusion")

    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # Loss functions
    criterion_grade = nn.CrossEntropyLoss()
    criterion_prog = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=0.5
    )

    # Early stopping
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="min")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(
            model, Path(args.resume), optimizer, scheduler, DEVICE
        )
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    best_kappa = 0.0

    logger.info(f"\n{'='*60}")
    logger.info("Starting centralized training")
    logger.info(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_grade_m, train_prog_m = train_one_epoch(
            model, train_loader, criterion_grade, criterion_prog,
            optimizer, DEVICE, LOSS_WEIGHT_GRADE, LOSS_WEIGHT_PROGRESSION,
        )

        # Validate
        val_loss, val_grade_m, val_prog_m = validate(
            model, val_loader, criterion_grade, criterion_prog,
            DEVICE, LOSS_WEIGHT_GRADE, LOSS_WEIGHT_PROGRESSION,
        )

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"  Train Acc: {train_grade_m['accuracy']:.4f} | "
                     f"Val Acc: {val_grade_m['accuracy']:.4f}")
        logger.info(f"  Val Kappa: {val_grade_m['cohen_kappa']:.4f} | "
                     f"Val F1: {val_grade_m['f1_macro']:.4f}")
        logger.info(f"  LR: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_grade_m,
                MODELS_DIR / "best_model.pth", scheduler,
            )
            logger.info(f"Best model saved (val_loss={val_loss:.4f})")

        if val_grade_m["cohen_kappa"] > best_kappa:
            best_kappa = val_grade_m["cohen_kappa"]
            save_checkpoint(
                model, optimizer, epoch, val_grade_m,
                MODELS_DIR / "best_kappa_model.pth", scheduler,
            )
            logger.info(f"Best kappa model saved (kappa={best_kappa:.4f})")

        # Early stopping
        if early_stop.step(val_loss):
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_grade_m,
        MODELS_DIR / "final_model.pth", scheduler,
    )

    logger.info(f"\n{'='*60}")
    logger.info("Training complete")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation kappa: {best_kappa:.4f}")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
