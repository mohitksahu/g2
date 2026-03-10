"""
Build final dataset:
  1. Merge preprocessed images + tabular data
  2. Stratified train/val/test split (70/15/15)
  3. Save split CSVs to dataset/{train,val,test}/
  4. For federated learning: partition training data into N simulated nodes
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    NUM_FED_NODES, SEED
)


def verify_image_tabular_alignment(tabular_csv: Path, image_dir: Path):
    """
    Check which tabular records have matching preprocessed images.
    Returns DataFrame with only matched records + an 'image_path' column.
    """
    df = pd.read_csv(tabular_csv)

    # Detect ID column
    id_col = None
    for candidate in ["image_id", "patient_id", "id", "filename"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        print("  [WARN] No ID column found. Using row index.")
        df["image_id"] = df.index.astype(str)
        id_col = "image_id"

    # Find matching images
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    all_images = {
        f.stem: f for f in image_dir.rglob("*")
        if f.suffix.lower() in image_extensions
    }

    matched_paths = []
    for _, row in df.iterrows():
        name = str(row[id_col])
        stem = Path(name).stem  # handle if .jpg is included
        if stem in all_images:
            matched_paths.append(str(all_images[stem]))
        elif name in all_images:
            matched_paths.append(str(all_images[name]))
        else:
            matched_paths.append(None)

    df["image_path"] = matched_paths
    matched = df[df["image_path"].notna()].copy()
    unmatched = len(df) - len(matched)

    print(f"  Total tabular records: {len(df)}")
    print(f"  Matched with images:   {len(matched)}")
    print(f"  Unmatched:             {unmatched}")

    return matched


def stratified_split(df: pd.DataFrame, label_col: str = "dr_grade"):
    """
    Stratified split into train (70%), val (15%), test (15%).
    """
    if label_col not in df.columns:
        print(f"  [WARN] Label column '{label_col}' not found. Using random split.")
        train, temp = train_test_split(df, test_size=0.30, random_state=SEED)
        val, test = train_test_split(temp, test_size=0.50, random_state=SEED)
    else:
        train, temp = train_test_split(
            df, test_size=0.30, random_state=SEED,
            stratify=df[label_col]
        )
        val, test = train_test_split(
            temp, test_size=0.50, random_state=SEED,
            stratify=temp[label_col]
        )

    print(f"\n  Split sizes:")
    print(f"    Train: {len(train)}")
    print(f"    Val:   {len(val)}")
    print(f"    Test:  {len(test)}")

    return train, val, test


def create_federated_partitions(train_df: pd.DataFrame, num_nodes: int,
                                 label_col: str = "dr_grade"):
    """
    Partition training data into N non-IID federated nodes to simulate
    real-world hospital data heterogeneity.
    
    Strategy: Dirichlet distribution-based partitioning for realistic
    non-IID splits (commonly used in federated learning research).
    """
    partitions = {}

    if label_col in train_df.columns:
        # Non-IID partitioning using Dirichlet distribution
        alpha = 0.5  # Lower = more heterogeneous across nodes
        labels = train_df[label_col].values
        unique_labels = np.unique(labels)

        node_indices = [[] for _ in range(num_nodes)]

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            np.random.seed(SEED + int(label))
            proportions = np.random.dirichlet([alpha] * num_nodes)

            # Distribute indices according to proportions
            proportions = (proportions * len(label_indices)).astype(int)
            # Fix rounding: give remainder to first node
            proportions[0] += len(label_indices) - proportions.sum()

            start = 0
            for node_id in range(num_nodes):
                end = start + proportions[node_id]
                node_indices[node_id].extend(label_indices[start:end].tolist())
                start = end

        for node_id in range(num_nodes):
            partitions[f"node_{node_id}"] = train_df.iloc[node_indices[node_id]].copy()

    else:
        # Random equal partitioning
        shuffled = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        chunks = np.array_split(shuffled, num_nodes)
        for i, chunk in enumerate(chunks):
            partitions[f"node_{i}"] = chunk

    # Print distribution
    print(f"\n  Federated partitions ({num_nodes} nodes):")
    for name, part_df in partitions.items():
        print(f"    {name}: {len(part_df)} samples", end="")
        if label_col in part_df.columns:
            dist = part_df[label_col].value_counts().sort_index().to_dict()
            print(f"  | Grade dist: {dist}")
        else:
            print()

    return partitions


def save_splits(train_df, val_df, test_df, partitions: dict):
    """Save split CSVs to dataset directories."""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DIR / "split.csv", index=False)
    val_df.to_csv(VAL_DIR / "split.csv", index=False)
    test_df.to_csv(TEST_DIR / "split.csv", index=False)

    # Save federated partitions
    fed_dir = TRAIN_DIR / "federated"
    fed_dir.mkdir(parents=True, exist_ok=True)
    for name, part_df in partitions.items():
        part_df.to_csv(fed_dir / f"{name}.csv", index=False)

    print(f"\n  Saved splits to: {DATASET_DIR}")


def main():
    print(f"\n{'='*60}")
    print(f"Building dataset")
    print(f"{'='*60}")

    tabular_csv = DATASET_DIR / "tabular_processed.csv"
    image_dir = DATASET_DIR / "processed_images"

    if not tabular_csv.exists():
        print(f"  [ERROR] Run preprocess_tabular.py first.")
        print(f"  Expected: {tabular_csv}")
        return

    if not image_dir.exists():
        print(f"  [WARN] No processed images found at {image_dir}")
        print(f"  Run preprocess_images.py first, or building with tabular-only.")

    # Verify alignment
    matched_df = verify_image_tabular_alignment(tabular_csv, image_dir)

    if len(matched_df) == 0:
        print("  [WARN] No image-tabular matches. Using tabular data only.")
        matched_df = pd.read_csv(tabular_csv)

    # Split
    train_df, val_df, test_df = stratified_split(matched_df)

    # Federated partitions
    partitions = create_federated_partitions(train_df, NUM_FED_NODES)

    # Save
    save_splits(train_df, val_df, test_df, partitions)

    print(f"\n{'='*60}")
    print(f"Dataset build complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
