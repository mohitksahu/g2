"""
Federated Learning utilities:
  - FedAvg aggregation
  - FedProx local training with proximal term
  - Node simulation (partition data into hospital nodes)
  - Communication round orchestration
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from typing import List, Dict


# ──────────────────────────────────────────────
# FedAvg Aggregation
# ──────────────────────────────────────────────
def federated_averaging(
    global_model: nn.Module,
    local_state_dicts: List[OrderedDict],
    sample_counts: List[int],
) -> OrderedDict:
    """
    Weighted Federated Averaging (McMahan et al., 2017).
    
    Args:
        global_model: The global model (used for getting parameter names)
        local_state_dicts: List of state_dicts from each local node
        sample_counts: Number of training samples at each node (for weighting)
    
    Returns:
        Aggregated state_dict for the global model
    """
    total_samples = sum(sample_counts)
    weights = [n / total_samples for n in sample_counts]

    global_state = global_model.state_dict()
    aggregated = OrderedDict()

    for key in global_state.keys():
        aggregated[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for i, local_sd in enumerate(local_state_dicts):
            if key in local_sd:
                aggregated[key] += weights[i] * local_sd[key].float()
            else:
                # Fallback: use global if local doesn't have it (e.g., BN stats)
                aggregated[key] += weights[i] * global_state[key].float()

        # Restore original dtype
        aggregated[key] = aggregated[key].to(global_state[key].dtype)

    return aggregated


# ──────────────────────────────────────────────
# Local Training (per node)
# ──────────────────────────────────────────────
def train_local_node(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_grade: nn.Module,
    criterion_prog: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    loss_weight_grade: float = 0.5,
    loss_weight_prog: float = 0.5,
    global_state_dict: OrderedDict = None,
    proximal_mu: float = 0.0,
) -> tuple:
    """
    Train model locally on one node's data.
    
    Args:
        model: Local copy of the model
        dataloader: Node's training DataLoader
        criterion_grade: Loss for DR grade classification
        criterion_prog: Loss for progression prediction
        optimizer: Optimizer
        device: cuda/cpu
        local_epochs: Number of local training epochs
        loss_weight_grade: Weight for grade loss
        loss_weight_prog: Weight for progression loss
        global_state_dict: Global model params (for FedProx)
        proximal_mu: FedProx regularization strength (0 = FedAvg)
    
    Returns:
        (updated_state_dict, avg_loss, num_samples)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            images, tabular, grades, progressions = batch[:4]
            images = images.to(device)
            tabular = tabular.to(device)
            grades = grades.to(device)
            progressions = progressions.to(device)

            optimizer.zero_grad()

            grade_logits, prog_pred = model(images, tabular)

            # Multi-task loss
            loss_grade = criterion_grade(grade_logits, grades)
            loss_prog = criterion_prog(prog_pred.squeeze(), progressions)
            loss = loss_weight_grade * loss_grade + loss_weight_prog * loss_prog

            # FedProx proximal term
            if proximal_mu > 0 and global_state_dict is not None:
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_state_dict:
                        prox_term += ((param - global_state_dict[name].to(device)) ** 2).sum()
                loss += (proximal_mu / 2) * prox_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        total_loss += epoch_loss

    avg_loss = total_loss / max(total_samples, 1)
    num_samples = len(dataloader.dataset)

    return model.state_dict(), avg_loss, num_samples


# ──────────────────────────────────────────────
# Federated Training Round
# ──────────────────────────────────────────────
def federated_round(
    global_model: nn.Module,
    node_dataloaders: List[DataLoader],
    criterion_grade: nn.Module,
    criterion_prog: nn.Module,
    device: torch.device,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    loss_weight_grade: float = 0.5,
    loss_weight_prog: float = 0.5,
    algorithm: str = "fedavg",
    proximal_mu: float = 0.01,
) -> tuple:
    """
    Execute one federated communication round:
      1. Distribute global model to all nodes
      2. Each node trains locally
      3. Aggregate with FedAvg
    
    Returns:
        (aggregated_state_dict, list_of_node_losses)
    """
    global_state = copy.deepcopy(global_model.state_dict())
    local_state_dicts = []
    sample_counts = []
    node_losses = []

    for node_id, dataloader in enumerate(node_dataloaders):
        # Create local model copy
        local_model = copy.deepcopy(global_model).to(device)

        optimizer = torch.optim.AdamW(
            local_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        mu = proximal_mu if algorithm == "fedprox" else 0.0

        state_dict, avg_loss, num_samples = train_local_node(
            model=local_model,
            dataloader=dataloader,
            criterion_grade=criterion_grade,
            criterion_prog=criterion_prog,
            optimizer=optimizer,
            device=device,
            local_epochs=local_epochs,
            loss_weight_grade=loss_weight_grade,
            loss_weight_prog=loss_weight_prog,
            global_state_dict=global_state if mu > 0 else None,
            proximal_mu=mu,
        )

        local_state_dicts.append(state_dict)
        sample_counts.append(num_samples)
        node_losses.append(avg_loss)

        # Free memory
        del local_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    aggregated = federated_averaging(global_model, local_state_dicts, sample_counts)

    return aggregated, node_losses


# ──────────────────────────────────────────────
# Partition Loader Helper
# ──────────────────────────────────────────────
def create_node_dataloaders(
    dataset,
    partition_csvs: List[str],
    batch_size: int,
    num_workers: int = 2,
) -> List[DataLoader]:
    """
    Create DataLoaders for each federated node from partition CSVs.
    
    Each partition CSV contains the subset of sample indices or IDs
    for that node (created by build_dataset.py).
    """
    import pandas as pd
    from pathlib import Path

    dataloaders = []
    for csv_path in partition_csvs:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  [WARN] Partition file not found: {csv_path}")
            continue

        partition_df = pd.read_csv(csv_path)
        indices = list(range(len(partition_df)))

        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        dataloaders.append(loader)

    return dataloaders
