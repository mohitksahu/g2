"""
partition_hospitals.py

Partitions IDRiD images + synthetic tabular data into:
  - 5 simulated hospital nodes (non-IID via Dirichlet)
  - Each hospital: train/test split (80/20)
  - Global validation set (10% held out first)
  - Preprocessed (CLAHE + resize) mirrors same structure

Folder output:
  dataset/
    hospitals/
      H01/ ... H05/
        train/images/ + tabular.csv
        test/images/  + tabular.csv
    val/
      images/ + tabular.csv
    processed/
      hospitals/H01..H05/train|test/images/
      val/images/

Usage:
    python scripts/partition_hospitals.py
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import IMG_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, SEED
from preprocess_images import auto_crop_fundus, apply_clahe

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# Source paths
TABULAR_CSV = DATASET_DIR / "tabular_processed.csv"
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "idrid" / "B. Disease Grading" / "1. Original Images" / "a. Training Set"

NUM_HOSPITALS = 5
VAL_FRACTION = 0.10        # 10% global validation
TEST_FRACTION = 0.20       # 20% of each hospital's data for local test
DIRICHLET_ALPHA = 1.0      # Controls heterogeneity (0.5=very skewed, 1.0=moderate, 10=uniform)
MIN_HOSPITAL_SAMPLES = 15  # Minimum samples per hospital — re-partition if violated


# ─── PREPROCESS ONE IMAGE ─────────────────────────────────────────────────────

def preprocess_and_save(src_path: Path, dst_path: Path):
    """Crop + CLAHE + resize → save to dst."""
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    img = auto_crop_fundus(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), img)
    return True


# ─── COPY RAW IMAGE ───────────────────────────────────────────────────────────

def copy_raw_image(image_id: str, dst_dir: Path) -> bool:
    """Find and copy raw image to destination."""
    for ext in [".jpg", ".jpeg", ".png", ".tif"]:
        src = RAW_IMAGE_DIR / f"{image_id}{ext}"
        if src.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / f"{image_id}{ext}")
            return True
    return False


# ─── DIRICHLET NON-IID PARTITIONING ───────────────────────────────────────────

def dirichlet_partition(df: pd.DataFrame, num_nodes: int,
                        alpha: float, seed: int,
                        min_samples: int = 15) -> list:
    """
    Partition data into non-IID hospital nodes using Dirichlet distribution.
    Each hospital gets a different class distribution — mimics real-world.
    """
    rng = np.random.default_rng(seed)
    labels = df["dr_grade"].values
    unique_labels = np.unique(labels)

    # Retry with increasing alpha until all hospitals have enough samples
    for attempt in range(10):
        node_indices = [[] for _ in range(num_nodes)]

        for label in unique_labels:
            label_idx = np.where(labels == label)[0]
            rng.shuffle(label_idx)

            proportions = rng.dirichlet([alpha] * num_nodes)
            counts = (proportions * len(label_idx)).astype(int)
            # Fix rounding
            counts[0] += len(label_idx) - counts.sum()

            start = 0
            for node in range(num_nodes):
                end = start + counts[node]
                node_indices[node].extend(label_idx[start:end].tolist())
                start = end

        # Check minimum size constraint
        sizes = [len(idx) for idx in node_indices]
        if all(s >= min_samples for s in sizes):
            break
        # Relax alpha and retry
        alpha *= 1.5
        print(f"  [RETRY] Hospital too small (min={min(sizes)}), increasing alpha to {alpha:.2f}")

    return node_indices


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*60}")
    print("HOSPITAL PARTITIONING + PREPROCESSING")
    print(f"{'='*60}")

    # Load tabular
    if not TABULAR_CSV.exists():
        print(f"[ERROR] {TABULAR_CSV} not found. Run generate_synthetic_tabular.py first.")
        return
    df = pd.read_csv(TABULAR_CSV)
    print(f"Loaded {len(df)} samples from tabular CSV")

    # Verify image source
    if not RAW_IMAGE_DIR.exists():
        print(f"[ERROR] Image dir not found: {RAW_IMAGE_DIR}")
        return

    rng = np.random.default_rng(SEED)

    # ─── Step 1: Hold out global validation set (stratified) ──────────────
    train_pool, val_df = train_test_split(
        df, test_size=VAL_FRACTION, random_state=SEED,
        stratify=df["dr_grade"]
    )
    train_pool = train_pool.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"\nGlobal validation: {len(val_df)} samples")
    print(f"Training pool:     {len(train_pool)} samples")

    # ─── Step 2: Partition training pool into hospitals (non-IID) ─────────
    node_indices = dirichlet_partition(train_pool, NUM_HOSPITALS, DIRICHLET_ALPHA, SEED,
                                        min_samples=MIN_HOSPITAL_SAMPLES)

    hospital_dfs = {}
    for h in range(NUM_HOSPITALS):
        hname = f"H{str(h+1).zfill(2)}"
        hdf = train_pool.iloc[node_indices[h]].reset_index(drop=True)
        hospital_dfs[hname] = hdf

    print(f"\nHospital partitions (non-IID, alpha={DIRICHLET_ALPHA}):")
    for hname, hdf in hospital_dfs.items():
        dist = hdf["dr_grade"].value_counts().sort_index().to_dict()
        print(f"  {hname}: {len(hdf):>4} samples | Grade dist: {dist}")

    # ─── Step 3: Split each hospital into train/test ──────────────────────
    hospital_splits = {}
    for hname, hdf in hospital_dfs.items():
        if len(hdf) < 5:
            # Too few samples to split — all go to train
            hospital_splits[hname] = {"train": hdf, "test": pd.DataFrame(columns=hdf.columns)}
            continue

        # Determine if stratified split is safe
        grade_counts = hdf["dr_grade"].value_counts()
        can_stratify = (hdf["dr_grade"].nunique() > 1 and
                        grade_counts.min() >= 2)

        h_train, h_test = train_test_split(
            hdf, test_size=TEST_FRACTION, random_state=SEED,
            stratify=hdf["dr_grade"] if can_stratify else None
        )
        hospital_splits[hname] = {
            "train": h_train.reset_index(drop=True),
            "test": h_test.reset_index(drop=True),
        }

    print(f"\nPer-hospital train/test splits:")
    for hname, splits in hospital_splits.items():
        print(f"  {hname}: train={len(splits['train']):>3}, test={len(splits['test']):>3}")

    # ─── Step 4: Create folder structure + copy images ────────────────────
    print(f"\nCreating folder structure and copying images...")

    hospitals_dir = DATASET_DIR / "hospitals"
    processed_dir = DATASET_DIR / "processed"
    val_raw_dir = DATASET_DIR / "val"
    val_proc_dir = processed_dir / "val"

    # --- Validation set ---
    print(f"\n  [VAL] {len(val_df)} images")
    val_img_dir = val_raw_dir / "images"
    val_proc_img = val_proc_dir / "images"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_proc_img.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="  Val images"):
        img_id = row["image_id"]
        copy_raw_image(img_id, val_img_dir)
        # Preprocess
        for ext in [".jpg", ".jpeg", ".png", ".tif"]:
            src = RAW_IMAGE_DIR / f"{img_id}{ext}"
            if src.exists():
                preprocess_and_save(src, val_proc_img / f"{img_id}.png")
                break

    val_df.to_csv(val_raw_dir / "tabular.csv", index=False)
    val_df.to_csv(val_proc_dir / "tabular.csv", index=False)

    # --- Hospital sets ---
    for hname, splits in hospital_splits.items():
        for split_name in ["train", "test"]:
            split_df = splits[split_name]
            if len(split_df) == 0:
                continue

            # Raw paths
            raw_img_dir = hospitals_dir / hname / split_name / "images"
            raw_img_dir.mkdir(parents=True, exist_ok=True)

            # Processed paths
            proc_img_dir = processed_dir / "hospitals" / hname / split_name / "images"
            proc_img_dir.mkdir(parents=True, exist_ok=True)

            desc = f"  {hname}/{split_name}"
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
                img_id = row["image_id"]
                copy_raw_image(img_id, raw_img_dir)
                for ext in [".jpg", ".jpeg", ".png", ".tif"]:
                    src = RAW_IMAGE_DIR / f"{img_id}{ext}"
                    if src.exists():
                        preprocess_and_save(src, proc_img_dir / f"{img_id}.png")
                        break

            # Save tabular CSV
            split_df.to_csv(hospitals_dir / hname / split_name / "tabular.csv", index=False)
            split_df.to_csv(processed_dir / "hospitals" / hname / split_name / "tabular.csv", index=False)

    # ─── Step 5: Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PARTITION COMPLETE")
    print(f"{'='*60}")
    print(f"\nStructure:")
    print(f"  dataset/")
    print(f"  ├── hospitals/")
    for hname, splits in hospital_splits.items():
        print(f"  │   ├── {hname}/")
        print(f"  │   │   ├── train/ ({len(splits['train'])} images + tabular.csv)")
        print(f"  │   │   └── test/  ({len(splits['test'])} images + tabular.csv)")
    print(f"  ├── val/ ({len(val_df)} images + tabular.csv)")
    print(f"  └── processed/  (CLAHE+resized mirrors)")
    print(f"\nTotal: {len(df)} images distributed across {NUM_HOSPITALS} hospitals + validation")


if __name__ == "__main__":
    main()
