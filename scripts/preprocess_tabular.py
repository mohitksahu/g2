"""
Tabular data preprocessing:
  1. Load CSV metadata (patient-level clinical data)
  2. Handle missing values (imputation)
  3. Encode categorical features
  4. Normalize continuous features
  5. Save processed tabular data + fitted encoders/scalers
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MBRSET_RAW_DIR, DATASET_DIR,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
)


def load_raw_tabular(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV metadata."""
    print(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map dataset-specific column names to our standard feature names.
    Adjust this mapping based on the actual CSV headers in your dataset.
    """
    # Example mapping for mBRSET — update when you see actual column names
    column_map = {
        # mBRSET potential mappings (adjust after inspecting actual CSV)
        "patient_id": "patient_id",
        "image_id": "image_id",
        "image_name": "image_id",
        "Age": "age",
        "age": "age",
        "Sex": "sex",
        "sex": "sex",
        "gender": "sex",
        "Gender": "sex",
        "diabetes_duration": "diabetes_duration",
        "DM_duration": "diabetes_duration",
        "dm_duration_years": "diabetes_duration",
        "hba1c": "hba1c",
        "HbA1c": "hba1c",
        "systolic_bp": "systolic_bp",
        "SBP": "systolic_bp",
        "diastolic_bp": "diastolic_bp",
        "DBP": "diastolic_bp",
        "bmi": "bmi",
        "BMI": "bmi",
        "diabetes_type": "diabetes_type",
        "DM_type": "diabetes_type",
        "treatment": "treatment_type",
        "treatment_type": "treatment_type",
        "smoking": "smoking_status",
        "smoking_status": "smoking_status",
        "hypertension": "hypertension",
        "HTN": "hypertension",
        "DR_grade": "dr_grade",
        "dr_grade": "dr_grade",
        "DR_level": "dr_grade",
        "level": "dr_grade",
        "diagnosis": "dr_grade",
    }

    renamed = {}
    for col in df.columns:
        if col in column_map:
            renamed[col] = column_map[col]
    df = df.rename(columns=renamed)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Continuous: median imputation
      - Categorical: mode imputation
    """
    # Continuous
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    if cont_cols:
        imp_cont = SimpleImputer(strategy="median")
        df[cont_cols] = imp_cont.fit_transform(df[cont_cols])

    # Categorical
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True,
                     encoders: dict = None, scaler: StandardScaler = None):
    """
    Encode categorical features (label encoding) and scale continuous features.
    
    Args:
        df: DataFrame with standardized column names
        fit: If True, fit new encoders/scalers. If False, use provided ones.
        encoders: Dict of {col_name: LabelEncoder} (used when fit=False)
        scaler: Fitted StandardScaler (used when fit=False)
    
    Returns:
        df: Transformed DataFrame
        encoders: Dict of fitted LabelEncoders
        scaler: Fitted StandardScaler
    """
    if encoders is None:
        encoders = {}

    # Encode categoricals
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            if col in encoders:
                # Handle unseen labels
                le = encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    # Scale continuous
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    if cont_cols:
        if fit:
            scaler = StandardScaler()
            df[cont_cols] = scaler.fit_transform(df[cont_cols])
        else:
            if scaler is not None:
                df[cont_cols] = scaler.transform(df[cont_cols])

    return df, encoders, scaler


def save_preprocessors(encoders: dict, scaler, output_dir: Path):
    """Save fitted encoders and scaler for inference-time use."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    if scaler is not None:
        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    print(f"  Saved preprocessors to {output_dir}")


def load_preprocessors(preprocessor_dir: Path):
    """Load saved encoders and scaler."""
    encoders = {}
    scaler = None

    enc_path = preprocessor_dir / "label_encoders.pkl"
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)

    scaler_path = preprocessor_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return encoders, scaler


def preprocess_tabular(csv_path: Path, output_dir: Path):
    """
    Full tabular preprocessing pipeline.
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing tabular data")
    print(f"{'='*60}")

    if not csv_path.exists():
        print(f"  [ERROR] CSV not found: {csv_path}")
        print(f"  Place your dataset CSV at: {csv_path}")
        return None

    # Load
    df = load_raw_tabular(csv_path)

    # Standardize column names
    df = standardize_column_names(df)
    print(f"  Standardized columns: {list(df.columns)}")

    # Impute missing
    df = impute_missing(df)
    missing = df.isnull().sum().sum()
    print(f"  Remaining missing values: {missing}")

    # Encode & scale
    df, encoders, scaler = encode_and_scale(df, fit=True)

    # Save processed tabular
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "tabular_processed.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved processed tabular to: {out_csv}")

    # Save preprocessors for inference
    save_preprocessors(encoders, scaler, output_dir / "preprocessors")

    # Print summary stats
    print(f"\n  Feature summary:")
    available_cont = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    print(f"    Continuous features used: {available_cont}")
    print(f"    Categorical features used: {available_cat}")
    print(f"    Total tabular input dim: {len(available_cont) + len(available_cat)}")

    if "dr_grade" in df.columns:
        print(f"\n  DR grade distribution:")
        print(df["dr_grade"].value_counts().sort_index().to_string())

    return df


def main():
    """Run tabular preprocessing."""
    # Primary dataset: mBRSET
    csv_candidates = [
        MBRSET_RAW_DIR / "metadata.csv",
        MBRSET_RAW_DIR / "labels.csv",
        MBRSET_RAW_DIR / "clinical_data.csv",
    ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        # List available CSVs for user guidance
        csvs_found = list(MBRSET_RAW_DIR.rglob("*.csv")) if MBRSET_RAW_DIR.exists() else []
        if csvs_found:
            print(f"Found CSV files in {MBRSET_RAW_DIR}:")
            for c in csvs_found:
                print(f"  {c}")
            csv_path = csvs_found[0]
            print(f"\nUsing: {csv_path}")
        else:
            print(f"No CSV files found in {MBRSET_RAW_DIR}")
            print(f"Please place your dataset CSV in: {MBRSET_RAW_DIR}")
            return

    preprocess_tabular(csv_path, DATASET_DIR)


if __name__ == "__main__":
    main()
