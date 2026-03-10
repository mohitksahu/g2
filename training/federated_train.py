"""
Federated training script.
Simulates multi-hospital training using FedAvg/FedProx.

Usage:
    python training/federated_train.py
    python training/federated_train.py --num_nodes 4 --rounds 50 --algorithm fedprox
"""

import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (
    DATASET_DIR, HOSPITALS_DIR, PROCESSED_HOSPITALS_DIR,
    VAL_DIR, PROCESSED_VAL_DIR, MODELS_DIR, LOG_DIR,
    HOSPITAL_NAMES,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_FED_NODES, FED_ROUNDS, FED_LOCAL_EPOCHS, FED_ALGORITHM, FED_PROX_MU,
    LOSS_WEIGHT_GRADE, LOSS_WEIGHT_PROGRESSION,
    CNN_BACKBONE, TABULAR_EMBED_DIM, FUSION_DIM,
    NUM_DR_CLASSES, DROPOUT, DEVICE, SEED, NUM_WORKERS,
)
from models import DRMultiModalNet
from dataset import DRMultiModalDataset, get_tabular_dim
from federated import federated_round
from utils import (
    set_seed, setup_logger, save_checkpoint,
    compute_classification_metrics, format_metrics, EarlyStopping,
)

# Training history tracker
training_history = {
    'val_kappa': [],
    'val_loss': [],
    'val_accuracy': [],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Federated DR Training")
    parser.add_argument("--num_nodes", type=int, default=NUM_FED_NODES)
    parser.add_argument("--rounds", type=int, default=FED_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=FED_LOCAL_EPOCHS)
    parser.add_argument("--algorithm", type=str, default=FED_ALGORITHM,
                        choices=["fedavg", "fedprox"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--fusion", type=str, default="cross_attention",
                        choices=["cross_attention", "concat"])
    return parser.parse_args()


@torch.no_grad()
def evaluate_global(model, dataloader, criterion_grade, criterion_prog, device):
    """Evaluate global model on validation set."""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    all_true = []
    all_pred = []
    all_prob = []

    for batch in dataloader:
        images, tabular, grades, progressions = batch[:4]
        images = images.to(device)
        tabular = tabular.to(device)
        grades = grades.to(device)
        progressions = progressions.to(device)

        grade_logits, prog_pred = model(images, tabular)

        loss_g = criterion_grade(grade_logits, grades)
        loss_p = criterion_prog(prog_pred.squeeze(), progressions)
        loss = LOSS_WEIGHT_GRADE * loss_g + LOSS_WEIGHT_PROGRESSION * loss_p

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

        all_true.extend(grades.cpu().numpy())
        all_pred.extend(grade_logits.argmax(dim=1).cpu().numpy())
        all_prob.append(torch.softmax(grade_logits, dim=1).cpu().numpy())

    avg_loss = total_loss / max(num_samples, 1)

    metrics = compute_classification_metrics(
        np.array(all_true),
        np.array(all_pred),
        np.vstack(all_prob) if all_prob else None,
    )

    return avg_loss, metrics


def main():
    args = parse_args()
    set_seed(SEED)

    logger = setup_logger(LOG_DIR, name="federated_training")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Algorithm: {args.algorithm.upper()}")

    # Tabular dimension — read from any hospital's tabular.csv
    tab_dim = 6  # default
    for hname in HOSPITAL_NAMES:
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if tab_csv.exists():
            tab_dim = get_tabular_dim(tab_csv)
            break
    logger.info(f"Tabular input dim: {tab_dim}")

    # Each hospital folder = one federated node
    node_dataloaders = []
    actual_nodes = 0

    for hname in HOSPITAL_NAMES:
        img_dir = PROCESSED_HOSPITALS_DIR / hname / "train" / "images"
        tab_csv = HOSPITALS_DIR / hname / "train" / "tabular.csv"
        if img_dir.exists() and tab_csv.exists():
            node_dataset = DRMultiModalDataset(
                image_dir=img_dir,
                tabular_csv=tab_csv,
                transform=DRMultiModalDataset.get_train_transform(),
            )
            loader = DataLoader(
                node_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            )
            node_dataloaders.append(loader)
            actual_nodes += 1
            logger.info(f"  {hname}: {len(node_dataset)} samples")

    if not node_dataloaders:
        logger.error("No hospital training data found. Run partition_hospitals.py first.")
        return

    logger.info(f"Federated nodes: {actual_nodes}")

    # Validation set (centralized — for global model evaluation)
    val_dataset = DRMultiModalDataset(
        image_dir=PROCESSED_VAL_DIR / "images",
        tabular_csv=VAL_DIR / "tabular.csv",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    logger.info(f"  Validation: {len(val_dataset)} samples")

    # Global model
    global_model = DRMultiModalNet(
        tabular_input_dim=tab_dim,
        cnn_backbone=CNN_BACKBONE,
        tabular_embed_dim=TABULAR_EMBED_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_DR_CLASSES,
        dropout=DROPOUT,
        fusion_type=args.fusion,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"Model params: {total_params:,}")

    # Loss functions
    criterion_grade = nn.CrossEntropyLoss()
    criterion_prog = nn.BCELoss()

    # Early stopping on validation
    early_stop = EarlyStopping(patience=15, mode="min")
    best_val_loss = float("inf")
    best_kappa = 0.0

    # Federated training loop
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Federated Training ({args.algorithm.upper()})")
    logger.info(f"Nodes: {actual_nodes} | Rounds: {args.rounds} | "
                f"Local epochs: {args.local_epochs}")
    logger.info(f"{'='*60}")

    for round_num in range(args.rounds):
        logger.info(f"\n--- Round {round_num+1}/{args.rounds} ---")

        # Execute one federated round
        aggregated_state, node_losses = federated_round(
            global_model=global_model,
            node_dataloaders=node_dataloaders,
            criterion_grade=criterion_grade,
            criterion_prog=criterion_prog,
            device=DEVICE,
            local_epochs=args.local_epochs,
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
            loss_weight_grade=LOSS_WEIGHT_GRADE,
            loss_weight_prog=LOSS_WEIGHT_PROGRESSION,
            algorithm=args.algorithm,
            proximal_mu=FED_PROX_MU if args.algorithm == "fedprox" else 0.0,
        )

        # Update global model
        global_model.load_state_dict(aggregated_state)

        # Log node losses
        for i, nloss in enumerate(node_losses):
            logger.info(f"  Node {i} loss: {nloss:.4f}")

        avg_node_loss = np.mean(node_losses)
        logger.info(f"  Avg node loss: {avg_node_loss:.4f}")

        # Evaluate global model on validation set
        val_loss, val_metrics = evaluate_global(
            global_model, val_loader, criterion_grade, criterion_prog, DEVICE
        )

        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Acc: {val_metrics['accuracy']:.4f} | "
                     f"Val Kappa: {val_metrics['cohen_kappa']:.4f}")

        # Track training history
        training_history['val_kappa'].append(val_metrics['cohen_kappa'])
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_metrics['accuracy'])

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                global_model, None, round_num, val_metrics,
                MODELS_DIR / f"federated_best_{args.algorithm}.pth",
            )
            logger.info(f"Best federated model saved")

        if val_metrics["cohen_kappa"] > best_kappa:
            best_kappa = val_metrics["cohen_kappa"]

        # Early stopping
        if early_stop.step(val_loss):
            logger.info(f"\nEarly stopping at round {round_num+1}")
            break

    # Save final model
    save_checkpoint(
        global_model, None, round_num, val_metrics,
        MODELS_DIR / f"federated_final_{args.algorithm}.pth",
    )

    # Save training history for convergence analysis
    history_path = LOG_DIR / f"{args.algorithm}_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Federated training complete ({args.algorithm.upper()})")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best val kappa: {best_kappa:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
