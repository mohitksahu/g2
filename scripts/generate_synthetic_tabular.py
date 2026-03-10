"""
generate_synthetic_tabular.py

Generates clinically realistic synthetic tabular data for any DR image dataset
that has only images + DR grade labels (APTOS, IDRiD, EyePACS, etc.)

Clinical distributions are derived from published epidemiological studies:
- UKPDS (UK Prospective Diabetes Study)
- WESDR (Wisconsin Epidemiologic Study of Diabetic Retinopathy)
- Rema et al. Chennai Urban Rural Epidemiology Study

Output: A CSV matching each image to a synthetic patient profile,
        ready to use directly with dataset.py

Run:
    python scripts/generate_synthetic_tabular.py \
        --input_csv  "data/idrid/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv" \
        --output_csv dataset/tabular_processed.csv \
        --image_col  "Image name" \
        --grade_col  "Retinopathy grade"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── CLINICAL DISTRIBUTIONS PER DR GRADE ─────────────────────────────────────
# Format: { grade: { feature: (mean, std) } }
# Source: WESDR, UKPDS, published meta-analyses on DR risk factors

DISTRIBUTIONS = {
    0: {  # No DR
        "age":               (45.0, 10.0),
        "diabetes_duration": (5.0,  3.0),
        "hba1c":             (6.5,  0.8),
        "systolic_bp":       (120.0, 12.0),
        "diastolic_bp":      (78.0,  8.0),
        "bmi":               (26.0,  4.0),
    },
    1: {  # Mild NPDR
        "age":               (50.0, 10.0),
        "diabetes_duration": (8.0,  4.0),
        "hba1c":             (7.5,  1.0),
        "systolic_bp":       (125.0, 12.0),
        "diastolic_bp":      (80.0,  8.0),
        "bmi":               (27.5,  4.5),
    },
    2: {  # Moderate NPDR
        "age":               (55.0, 10.0),
        "diabetes_duration": (12.0, 5.0),
        "hba1c":             (8.5,  1.2),
        "systolic_bp":       (130.0, 13.0),
        "diastolic_bp":      (83.0,  9.0),
        "bmi":               (28.5,  5.0),
    },
    3: {  # Severe NPDR
        "age":               (58.0, 10.0),
        "diabetes_duration": (15.0, 5.0),
        "hba1c":             (9.5,  1.3),
        "systolic_bp":       (135.0, 13.0),
        "diastolic_bp":      (85.0,  9.0),
        "bmi":               (29.5,  5.0),
    },
    4: {  # Proliferative DR
        "age":               (60.0, 10.0),
        "diabetes_duration": (18.0, 5.0),
        "hba1c":             (10.5, 1.5),
        "systolic_bp":       (140.0, 15.0),
        "diastolic_bp":      (88.0,  10.0),
        "bmi":               (30.5,  5.5),
    },
}

# Sex: % probability of being female per grade
SEX_PROB_FEMALE = {0: 0.50, 1: 0.50, 2: 0.52, 3: 0.55, 4: 0.55}

# Treatment type probabilities per grade
# 0=none, 1=oral medication, 2=insulin, 3=both
TREATMENT_PROBS = {
    0: [0.30, 0.50, 0.15, 0.05],
    1: [0.15, 0.50, 0.25, 0.10],
    2: [0.05, 0.40, 0.35, 0.20],
    3: [0.02, 0.28, 0.40, 0.30],
    4: [0.01, 0.19, 0.45, 0.35],
}


# ─── PROGRESSION LABEL ────────────────────────────────────────────────────────

def compute_progression(row: pd.Series) -> int:
    """
    Simulates 12-month progression risk using clinical rules.
    Based on UKPDS risk engine logic (simplified).
    """
    score = 0
    if row["dr_grade"] >= 2:          score += 2
    if row["dr_grade"] >= 3:          score += 2
    if row["hba1c"] > 9.0:            score += 2
    if row["diabetes_duration"] > 12: score += 1
    if row["systolic_bp"] > 135:      score += 1
    if row["hba1c"] > 8.0:            score += 1
    # Add randomness so boundary cases aren't deterministic
    score += np.random.normal(0, 0.5)
    return int(score >= 4)


# ─── SYNTHETIC GENERATION ─────────────────────────────────────────────────────

def generate_patient_row(grade: int, rng: np.random.Generator) -> dict:
    """Generate one realistic patient profile for a given DR grade."""
    dist = DISTRIBUTIONS[grade]
    row = {}

    row["age"]               = float(np.clip(rng.normal(*dist["age"]),               20, 90))
    row["diabetes_duration"] = float(np.clip(rng.normal(*dist["diabetes_duration"]),  0, 50))
    row["hba1c"]             = float(np.clip(rng.normal(*dist["hba1c"]),              4.0, 16.0))
    row["systolic_bp"]       = float(np.clip(rng.normal(*dist["systolic_bp"]),        80, 200))
    row["diastolic_bp"]      = float(np.clip(rng.normal(*dist["diastolic_bp"]),       50, 130))
    row["bmi"]               = float(np.clip(rng.normal(*dist["bmi"]),               15, 55))

    row["sex"]            = int(rng.random() < SEX_PROB_FEMALE[grade])
    row["treatment_type"] = int(rng.choice([0, 1, 2, 3], p=TREATMENT_PROBS[grade]))

    comorbidity_prob = {0: 0.1, 1: 0.2, 2: 0.35, 3: 0.5, 4: 0.65}
    row["comorbidities_count"] = int(rng.binomial(3, comorbidity_prob[grade]))

    return row


def generate_tabular(input_csv: str, output_csv: str,
                     image_col: str, grade_col: str,
                     seed: int = 42):
    """
    Reads existing image+grade CSV, generates one synthetic patient row
    per image, merges, saves.
    """
    df = pd.read_csv(input_csv)

    # Strip whitespace from column names (IDRiD CSVs have trailing spaces)
    df.columns = df.columns.str.strip()

    print(f"Loaded: {len(df)} rows from {input_csv}")
    print(f"Columns found: {list(df.columns)}")

    if image_col not in df.columns:
        raise ValueError(f"Image column '{image_col}' not found. Available: {list(df.columns)}")
    if grade_col not in df.columns:
        raise ValueError(f"Grade column '{grade_col}' not found. Available: {list(df.columns)}")

    rng = np.random.default_rng(seed)

    synthetic_rows = []
    for _, row in df.iterrows():
        grade = int(row[grade_col])
        grade = np.clip(grade, 0, 4)
        patient = generate_patient_row(grade, rng)
        patient["image_id"] = row[image_col]
        patient["dr_grade"] = grade
        # Preserve DME risk if available
        for col in df.columns:
            if "macular" in col.lower() or "dme" in col.lower():
                patient["dme_risk"] = int(row[col]) if pd.notna(row[col]) else 0
        synthetic_rows.append(patient)

    result_df = pd.DataFrame(synthetic_rows)

    result_df.insert(0, "patient_id", [f"P{str(i).zfill(5)}" for i in range(len(result_df))])

    result_df["progression"] = result_df.apply(compute_progression, axis=1)

    col_order = [
        "patient_id", "image_id", "dr_grade", "progression",
        "age", "sex", "diabetes_duration", "hba1c",
        "systolic_bp", "diastolic_bp", "bmi",
        "treatment_type", "comorbidities_count"
    ]
    if "dme_risk" in result_df.columns:
        col_order.append("dme_risk")
    result_df = result_df[col_order]

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    print(f"\nGenerated tabular data for {len(result_df)} patients")
    print(f"DR Grade distribution:\n{result_df['dr_grade'].value_counts().sort_index()}")
    print(f"Progression rate: {result_df['progression'].mean():.2%}")
    print(f"\nSample row:\n{result_df.iloc[0].to_dict()}")
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data for DR image dataset")
    parser.add_argument("--input_csv",  type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--image_col",  type=str, default="Image name")
    parser.add_argument("--grade_col",  type=str, default="Retinopathy grade")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    generate_tabular(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        image_col=args.image_col,
        grade_col=args.grade_col,
        seed=args.seed,
    )
