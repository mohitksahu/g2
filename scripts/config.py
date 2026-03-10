"""
Central configuration for the entire project.
All paths, hyperparameters, and constants in one place.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data directories
DATA_DIR = PROJECT_ROOT / "data"
MBRSET_RAW_DIR = DATA_DIR / "mbrset"
IDRID_RAW_DIR = DATA_DIR / "idrid"

# Dataset directories (hospital-based partitioning)
DATASET_DIR = PROJECT_ROOT / "dataset"
HOSPITALS_DIR = DATASET_DIR / "hospitals"           # Raw hospital folders
PROCESSED_DIR = DATASET_DIR / "processed"            # CLAHE+resized mirrors
PROCESSED_HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
VAL_DIR = DATASET_DIR / "val"                        # Raw global validation
PROCESSED_VAL_DIR = PROCESSED_DIR / "val"            # Processed global validation

# Legacy aliases (kept for backward compatibility)
TRAIN_DIR = HOSPITALS_DIR
TEST_DIR = DATASET_DIR / "test"

# Hospital names
HOSPITAL_NAMES = [f"H{str(i).zfill(2)}" for i in range(1, 6)]  # H01..H05

# Model save directory
MODELS_DIR = PROJECT_ROOT / "models"

# Logs
LOG_DIR = PROJECT_ROOT / "logs"

# ──────────────────────────────────────────────
# IMAGE PREPROCESSING
# ──────────────────────────────────────────────
IMG_SIZE = 224                  # EfficientNet-B4 default input
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ImageNet normalization (EfficientNet pretrained)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# TABULAR FEATURES
# ──────────────────────────────────────────────
# Continuous features expected from clinical data
CONTINUOUS_FEATURES = [
    "age",
    "diabetes_duration",
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
]

# Categorical features (will be one-hot or label encoded)
CATEGORICAL_FEATURES = [
    "sex",
    "diabetes_type",
    "treatment_type",
    "smoking_status",
    "hypertension",
]

# Total tabular input dimension after encoding (updated during preprocessing)
TABULAR_INPUT_DIM = len(CONTINUOUS_FEATURES)  # base; adjusted after encoding

# ──────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────
CNN_BACKBONE = "efficientnet_b4"   # timm model name
CNN_EMBED_DIM = 1792               # EfficientNet-B4 feature dim
TABULAR_EMBED_DIM = 128            # MLP output dim for tabular branch
FUSION_DIM = 256                   # Cross-attention / fused dim
NUM_DR_CLASSES = 5                 # 0-4: No DR, Mild, Moderate, Severe, PDR
DROPOUT = 0.3

# ──────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 4

# Loss weights for multi-task output
LOSS_WEIGHT_GRADE = 0.5           # DR grade classification
LOSS_WEIGHT_PROGRESSION = 0.5    # Progression risk regression

# ──────────────────────────────────────────────
# FEDERATED LEARNING
# ──────────────────────────────────────────────
NUM_FED_NODES = 5                 # Simulated hospital nodes (H01-H05)
FED_ROUNDS = 50                   # Communication rounds
FED_LOCAL_EPOCHS = 5              # Local training epochs per round
FED_ALGORITHM = "fedavg"          # fedavg | fedprox
FED_PROX_MU = 0.01                # FedProx regularization (if used)

# ──────────────────────────────────────────────
# EXPLAINABILITY
# ──────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "features"  # Last conv block of EfficientNet
SHAP_BACKGROUND_SAMPLES = 100

# ──────────────────────────────────────────────
# RANDOM SEED
# ──────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, MBRSET_RAW_DIR, IDRID_RAW_DIR,
              DATASET_DIR, HOSPITALS_DIR, VAL_DIR,
              PROCESSED_DIR, PROCESSED_HOSPITALS_DIR, PROCESSED_VAL_DIR,
              MODELS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_hospital_paths(hospital_name: str) -> dict:
    """Get all paths for a given hospital node."""
    return {
        "train_images": PROCESSED_HOSPITALS_DIR / hospital_name / "train" / "images",
        "train_tabular": HOSPITALS_DIR / hospital_name / "train" / "tabular.csv",
        "test_images": PROCESSED_HOSPITALS_DIR / hospital_name / "test" / "images",
        "test_tabular": HOSPITALS_DIR / hospital_name / "test" / "tabular.csv",
    }
