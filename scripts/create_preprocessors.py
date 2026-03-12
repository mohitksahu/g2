# Save as: scripts/create_preprocessors.py

"""
Create scaler.pkl and label_encoders.pkl from existing processed tabular data.
Run: python scripts/create_preprocessors.py
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_DIR, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

def main():
    print(f"\n{'='*60}")
    print("Creating Preprocessors from Existing Data")
    print(f"{'='*60}")
    
    # Load existing processed tabular data
    tabular_csv = DATASET_DIR / "tabular_processed.csv"
    if not tabular_csv.exists():
        print(f"ERROR: {tabular_csv} not found")
        return
    
    df = pd.read_csv(tabular_csv)
    print(f"Loaded {len(df)} rows from {tabular_csv}")
    print(f"Columns: {list(df.columns)}")
    
    # Identify available features
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    print(f"\nContinuous features found: {cont_cols}")
    print(f"Categorical features found: {cat_cols}")
    
    # Fit StandardScaler on continuous features
    scaler = None
    if cont_cols:
        scaler = StandardScaler()
        scaler.fit(df[cont_cols].values)
        print(f"\nScaler statistics:")
        for i, col in enumerate(cont_cols):
            print(f"  {col}: mean={scaler.mean_[i]:.4f}, std={scaler.scale_[i]:.4f}")
    
    # Fit LabelEncoders on categorical features
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str).values)
            encoders[col] = le
            print(f"\n{col} encoder classes: {le.classes_}")
    
    # Save preprocessors
    preprocessor_dir = DATASET_DIR / "preprocessors"
    preprocessor_dir.mkdir(parents=True, exist_ok=True)
    
    with open(preprocessor_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n✓ Saved scaler.pkl")
    
    with open(preprocessor_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print(f"✓ Saved label_encoders.pkl")
    
    print(f"\nPreprocessors saved to: {preprocessor_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()