# Federated Explainable AI for Diabetic Retinopathy Progression Prediction Using Multi-Modal Fusion

## Overview

This project implements a **federated learning framework** with **explainable AI** for predicting diabetic retinopathy (DR) progression using **multi-modal fusion** of fundus images and clinical tabular data.

### Key Contributions
- **Progression prediction** (not just current-grade classification) — predicts 12-month risk of DR worsening
- **Multi-modal fusion** — combines CNN image features with structured clinical/EHR data via cross-attention
- **Federated learning** — trains across simulated hospital nodes (non-IID Dirichlet partitioning) without sharing raw patient data
- **Explainability** — Grad-CAM heatmaps on images + SHAP values on tabular features for clinician-interpretable outputs

---

## Project Structure

```
g2/
├── data/                           # Raw source data
│   └── idrid/
│       └── B. Disease Grading/     # IDRiD fundus images + labels CSV
├── dataset/                        # Partitioned & processed data
│   ├── hospitals/                  # Raw images per hospital node
│   │   ├── H01/
│   │   │   ├── train/
│   │   │   │   ├── images/         # Original fundus images
│   │   │   │   └── tabular.csv     # Synthetic clinical features
│   │   │   └── test/
│   │   │       ├── images/
│   │   │       └── tabular.csv
│   │   ├── H02/ ... H05/          # Same structure per hospital
│   ├── processed/                  # CLAHE-preprocessed + resized (224×224)
│   │   ├── hospitals/
│   │   │   ├── H01/train/images/   # Preprocessed training images
│   │   │   ├── H01/test/images/
│   │   │   ├── H02/ ... H05/
│   │   └── val/
│   │       └── images/             # Preprocessed validation images
│   └── val/
│       ├── images/                 # Raw validation images
│       └── tabular.csv             # Validation tabular data
├── models/                         # Saved model weights (.pth)
├── logs/                           # Training logs
├── scripts/
│   ├── config.py                   # All configurations, paths, hyperparameters
│   ├── preprocess_images.py        # Image preprocessing (CLAHE, auto-crop, resize)
│   ├── generate_synthetic_tabular.py # Generate synthetic clinical tabular data
│   ├── partition_hospitals.py      # Partition data into hospital nodes (non-IID)
│   ├── dataset.py                  # PyTorch Dataset class (multi-modal)
│   ├── models.py                   # Model architectures (CNN, MLP, Fusion)
│   ├── federated.py                # Federated learning utilities (FedAvg/FedProx)
│   ├── explainability.py           # Grad-CAM + SHAP explanations
│   ├── inference.py                # Single-patient inference pipeline
│   └── utils.py                    # Metrics, logging, checkpointing
├── training/
│   ├── train.py                    # Centralized training (baseline)
│   ├── federated_train.py          # Federated training (FedAvg / FedProx)
│   └── evaluate.py                 # Model evaluation and comparison
├── requirements.txt
└── README.md
```

---

## Dataset

### IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **413 fundus images** with DR severity grades (0-4)
- Source: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- Download the **"B. Disease Grading"** zip file

### Synthetic Clinical Tabular Data
Since paired tabular clinical records are not publicly available with IDRiD, we generate **clinically realistic synthetic tabular data** calibrated to published epidemiological distributions (WESDR, UKPDS studies). Features include:
- `age`, `sex`, `diabetes_duration`, `hba1c`, `systolic_bp`, `diastolic_bp`, `bmi`
- `treatment_type`, `comorbidities_count`, `dme_risk`
- `progression` (binary 12-month progression label)

---

## Complete Setup Guide (From Scratch)

### Prerequisites
- Python 3.9+
- CUDA-capable GPU recommended (training works on CPU but is very slow)

### Step 0: Create Virtual Environment
```bash
cd g2
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Raw Data
1. Download IDRiD **"B. Disease Grading"** from https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
2. Extract the zip file into `data/idrid/` so the structure looks like:
```
data/idrid/B. Disease Grading/
├── 1. Original Images/
│   ├── a. Training Set/     # 413 fundus images (.jpg)
│   └── b. Testing Set/
└── 2. Groundtruths/
    └── a. IDRiD_Disease Grading_Training Labels.csv
```

### Step 3: Generate Synthetic Tabular Data
```bash
python scripts/generate_synthetic_tabular.py
```
This reads the IDRiD labels CSV and generates a paired `tabular.csv` in `data/idrid/` with clinically realistic synthetic features for each image.

### Step 4: Partition Data into Hospital Nodes
```bash
python scripts/partition_hospitals.py
```
This performs:
- **Non-IID Dirichlet partitioning** (alpha=1.0) across 5 simulated hospitals (H01-H05)
- 80/20 train/test split within each hospital (stratified by DR grade)
- 10% global validation holdout
- Copies raw images into `dataset/hospitals/HXX/{train,test}/images/`
- Applies **CLAHE preprocessing** + resize to 224×224 → `dataset/processed/hospitals/HXX/{train,test}/images/`
- Splits tabular data accordingly into per-folder `tabular.csv` files

After this step, the dataset is ready for training.

---

## Training Workflow

### Step 5: Centralized Training (Baseline)
Merges all hospital training data into one dataset and trains normally. This is the **baseline** to compare against federated approaches.

```bash
python training/train.py
```

**Options:**
```bash
python training/train.py --epochs 30 --batch_size 8 --lr 1e-4
python training/train.py --fusion concat        # Use simple concat fusion instead of cross-attention
python training/train.py --image_only           # Image-only ablation (no tabular)
python training/train.py --resume models/best_model.pth  # Resume from checkpoint
```

Saves to `models/`:
- `best_model.pth` — lowest validation loss
- `best_kappa_model.pth` — highest Cohen's Kappa
- `final_model.pth` — last epoch

### Step 6: Federated Training
Each hospital (H01-H05) acts as a **separate federated node**. The global model is trained via FedAvg or FedProx without sharing raw data across nodes.

```bash
# FedAvg (default)
python training/federated_train.py

# FedProx (with proximal regularization)
python training/federated_train.py --algorithm fedprox

# Custom settings
python training/federated_train.py --rounds 30 --local_epochs 3 --lr 1e-4 --batch_size 8
```

Saves to `models/`:
- `federated_best_fedavg.pth` or `federated_best_fedprox.pth`
- `federated_final_fedavg.pth` or `federated_final_fedprox.pth`

### Step 7: Evaluate & Compare Models
```bash
# Evaluate a specific model
python training/evaluate.py --model_path models/best_model.pth

# Compare ALL trained models (centralized vs federated)
python training/evaluate.py --compare

# Export best model to ONNX
python training/evaluate.py --model_path models/best_model.pth --export_onnx
```

The `--compare` flag evaluates every saved checkpoint and prints a side-by-side comparison table with Accuracy, Cohen's Kappa, F1, and AUC.

### Step 8: Single-Patient Inference
```bash
python scripts/inference.py --image path/to/fundus.jpg --age 58 --diabetes_duration 12 --hba1c 10.2
```

---

## Quick Reference: Command Execution Order

```
# One-time setup
pip install -r requirements.txt

# Data pipeline (run once)
python scripts/generate_synthetic_tabular.py
python scripts/partition_hospitals.py

# Training (run either or both)
python training/train.py                          # Centralized baseline
python training/federated_train.py                # Federated FedAvg
python training/federated_train.py --algorithm fedprox  # Federated FedProx

# Evaluation
python training/evaluate.py --compare             # Compare all models
```

---

## Model Architecture

```
Fundus Image ──→ EfficientNet-B4 (pretrained, 70% frozen)
                        │
                        ▼
               Image Embedding (1792-d)
                        │
                 Cross-Attention Fusion ──→ Fused Representation (512-d)
                        │
               Tabular Embedding (128-d)
                        │
                        ▲
Tabular Data ──→ MLP (256→128, BatchNorm + Dropout)


Fused Output:
  ├── Grade Head:       Dense(512→128→5)  → DR Grade (0–4)
  └── Progression Head: Dense(512→64→1)   → 12-month risk score (0–1)

Post-hoc Explainability:
  ├── Grad-CAM heatmap on fundus image (which regions drove the decision)
  └── SHAP values on tabular features (which clinical factors mattered)
```

### DR Grade Labels
| Grade | Meaning            |
|-------|--------------------|
| 0     | No DR              |
| 1     | Mild NPDR          |
| 2     | Moderate NPDR      |
| 3     | Severe NPDR        |
| 4     | Proliferative DR   |

---

## Hospital Partitioning Details

Data is distributed across 5 simulated hospitals using **Dirichlet non-IID sampling** (α=1.0), meaning each hospital has a different class distribution — mimicking real-world scenarios where some hospitals see more severe cases.

| Hospital | Train | Test | Total |
|----------|-------|------|-------|
| H01      | 47    | 12   | 59    |
| H02      | 62    | 16   | 78    |
| H03      | 96    | 25   | 121   |
| H04      | 40    | 10   | 50    |
| H05      | 50    | 13   | 63    |
| **Val**  | —     | —    | **42**|

---

## Key Configuration (scripts/config.py)

| Parameter          | Default           | Description                     |
|--------------------|-------------------|---------------------------------|
| `IMG_SIZE`         | 224               | Input image resolution          |
| `CNN_BACKBONE`     | efficientnet_b4   | Pretrained CNN model            |
| `BATCH_SIZE`       | 16                | Training batch size             |
| `NUM_EPOCHS`       | 50                | Max centralized epochs          |
| `LEARNING_RATE`    | 1e-4              | AdamW learning rate             |
| `NUM_FED_NODES`    | 5                 | Number of federated hospitals   |
| `FED_ROUNDS`       | 50                | Federated communication rounds  |
| `FED_LOCAL_EPOCHS` | 5                 | Local epochs per round          |
| `FED_ALGORITHM`    | fedavg            | fedavg or fedprox               |
| `DROPOUT`          | 0.3               | Dropout rate                    |

---

## Troubleshooting

- **CUDA out of memory**: Reduce `BATCH_SIZE` to 4 or 8 in `scripts/config.py`
- **Slow training on CPU**: EfficientNet-B4 is compute-heavy; use GPU if possible or switch backbone to `efficientnet_b0` in config
- **"No hospital training data found"**: Run `python scripts/partition_hospitals.py` first
- **Import errors**: Make sure you run all commands from the `g2/` root directory

---

## License

Research use only.
