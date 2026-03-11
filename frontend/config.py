"""
Configuration constants for the Streamlit frontend.
"""

from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODEL_PATH = PROJECT_ROOT / "models" / "federated_best_fedavg.pth"
LOG_DIR = PROJECT_ROOT / "logs"
HOSPITALS_DIR = PROJECT_ROOT / "dataset" / "hospitals"

# ── Model Configuration ───────────────────────────────────────────────────────
TABULAR_INPUT_DIM = 8

# Import from config.py in scripts
import sys
sys.path.insert(0, str(SCRIPTS_DIR))
from config import (
    CNN_BACKBONE,
    TABULAR_EMBED_DIM,
    FUSION_DIM,
    NUM_DR_CLASSES,
    DROPOUT,
    DEVICE,
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
)

# ── DR Grade Labels ───────────────────────────────────────────────────────────
DR_GRADE_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

DR_GRADE_COLORS = [
    "#2ecc71",  # Green - No DR
    "#f39c12",  # Orange - Mild
    "#e67e22",  # Dark Orange - Moderate
    "#e74c3c",  # Red - Severe
    "#8e44ad",  # Purple - Proliferative
]

# ── Clinical Feature Configuration ────────────────────────────────────────────
FEATURE_NAMES = [
    "Age",
    "Sex",
    "Diabetes Duration (yrs)",
    "HbA1c (%)",
    "Systolic BP",
    "Diastolic BP",
    "Treatment Type",
    "Comorbidities",
]

# (min, max) for min-max normalization
FEATURE_RANGES = [
    (0, 100),      # Age
    (0, 1),        # Sex
    (0, 50),       # Diabetes Duration
    (4.0, 15.0),   # HbA1c
    (80, 200),     # Systolic BP
    (50, 130),     # Diastolic BP
    (0, 2),        # Treatment Type
    (0, 10),       # Comorbidities
]

# ── Progression Risk Thresholds ───────────────────────────────────────────────
RISK_THRESHOLDS = {
    "low": 0.30,
    "moderate": 0.60,
    "high": 0.80,
}