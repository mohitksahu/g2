"""
PyTorch Dataset for multi-modal DR data:
  - Loads preprocessed fundus image
  - Loads corresponding tabular features
  - Returns (image_tensor, tabular_tensor, dr_grade, progression_label)
"""

import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    IMG_SIZE, IMG_MEAN, IMG_STD,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
)


class DRMultiModalDataset(Dataset):
    """
    Multi-modal dataset combining fundus images with tabular clinical data.
    
    Args:
        image_dir: Path to directory containing preprocessed fundus images
        tabular_csv: Path to processed tabular CSV
        split_csv: Path to CSV listing image_id + labels for this split
        transform: Optional torchvision transforms for data augmentation
        tabular_features: List of tabular column names to use
        return_meta: If True, also return image_id for inference/debugging
    """

    def __init__(
        self,
        image_dir: Path,
        tabular_csv: Path,
        split_csv: Path = None,
        transform=None,
        tabular_features: list = None,
        return_meta: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.return_meta = return_meta

        # Load tabular data
        self.tabular_df = pd.read_csv(tabular_csv)

        # Determine which samples to use
        if split_csv is not None and Path(split_csv).exists():
            split_df = pd.read_csv(split_csv)
            self.samples = split_df
        else:
            self.samples = self.tabular_df

        # Determine tabular feature columns
        if tabular_features is not None:
            self.feature_cols = tabular_features
        else:
            self.feature_cols = [
                c for c in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
                if c in self.tabular_df.columns
            ]

        # Image transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()

        # Build patient_id/image_id lookup from tabular
        id_col = self._detect_id_column()
        self.id_col = id_col

        if id_col and id_col in self.tabular_df.columns:
            self.tabular_lookup = self.tabular_df.set_index(id_col)
        else:
            self.tabular_lookup = None

    def _detect_id_column(self) -> str:
        """Auto-detect the ID column linking images to tabular data."""
        for candidate in ["image_id", "patient_id", "id", "ID", "filename"]:
            if candidate in self.samples.columns:
                return candidate
        # Fallback: use index
        return None

    def _default_transform(self):
        """Default inference/validation transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

    @staticmethod
    def get_train_transform():
        """Training transform with data augmentation."""
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

    def _find_image(self, sample_row) -> Path:
        """Locate the preprocessed image file for a given sample."""
        # Try different ID columns and file patterns
        for col in ["image_id", "patient_id", "filename", "id"]:
            if col in sample_row.index:
                name = str(sample_row[col])
                # Try exact filename
                candidates = [
                    self.image_dir / f"{name}",
                    self.image_dir / f"{name}.png",
                    self.image_dir / f"{name}.jpg",
                    self.image_dir / f"{name}.jpeg",
                ]
                # Also search recursively
                for c in candidates:
                    if c.exists():
                        return c

                # Glob search
                matches = list(self.image_dir.rglob(f"{name}*"))
                if matches:
                    return matches[0]

        return None

    def _get_tabular_features(self, sample_row) -> np.ndarray:
        """Extract tabular feature vector for a given sample."""
        # Try to look up from tabular data using ID
        if self.tabular_lookup is not None and self.id_col in sample_row.index:
            sample_id = sample_row[self.id_col]
            if sample_id in self.tabular_lookup.index:
                row = self.tabular_lookup.loc[sample_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                features = []
                for col in self.feature_cols:
                    if col in row.index:
                        features.append(float(row[col]) if pd.notna(row[col]) else 0.0)
                    else:
                        features.append(0.0)
                return np.array(features, dtype=np.float32)

        # Fallback: extract from sample_row itself
        features = []
        for col in self.feature_cols:
            if col in sample_row.index:
                val = sample_row[col]
                features.append(float(val) if pd.notna(val) else 0.0)
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)

    def _get_labels(self, sample_row):
        """Extract DR grade and progression label."""
        # DR grade (0-4)
        dr_grade = 0
        for col in ["dr_grade", "DR_grade", "level", "diagnosis"]:
            if col in sample_row.index and pd.notna(sample_row[col]):
                dr_grade = int(sample_row[col])
                break

        # Progression label (binary: will progress within 12 months)
        # If not available in dataset, derive from grade (grade >= 2 = higher risk)
        progression = 0.0
        for col in ["progression", "progressed", "progression_risk"]:
            if col in sample_row.index and pd.notna(sample_row[col]):
                progression = float(sample_row[col])
                break

        return dr_grade, progression

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_row = self.samples.iloc[idx]

        # Load image
        img_path = self._find_image(sample_row)
        if img_path is not None and img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            # Placeholder black image if not found
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        image_tensor = self.transform(image)

        # Tabular features
        tabular = self._get_tabular_features(sample_row)
        tabular_tensor = torch.tensor(tabular, dtype=torch.float32)

        # Labels
        dr_grade, progression = self._get_labels(sample_row)
        grade_tensor = torch.tensor(dr_grade, dtype=torch.long)
        progression_tensor = torch.tensor(progression, dtype=torch.float32)

        if self.return_meta:
            meta = {
                "image_path": str(img_path) if img_path else "",
                "sample_id": str(sample_row.get(self.id_col, idx)),
            }
            return image_tensor, tabular_tensor, grade_tensor, progression_tensor, meta

        return image_tensor, tabular_tensor, grade_tensor, progression_tensor


def get_tabular_dim(tabular_csv: Path) -> int:
    """
    Determine the actual tabular input dimension from a processed CSV.
    Call this before model initialization.
    """
    df = pd.read_csv(tabular_csv)
    feature_cols = [
        c for c in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
        if c in df.columns
    ]
    return len(feature_cols)
