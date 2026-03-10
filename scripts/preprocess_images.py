"""
Image preprocessing pipeline:
  1. Load raw fundus image
  2. Crop black borders (auto-detect retinal region)
  3. CLAHE enhancement on green channel
  4. Resize to IMG_SIZE x IMG_SIZE
  5. Save processed image to dataset folder
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MBRSET_RAW_DIR, IDRID_RAW_DIR, DATASET_DIR,
    IMG_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID
)


def auto_crop_fundus(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Automatically crop black borders from a fundus image.
    Uses grayscale thresholding to find the retinal disc region.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours and get the largest one (the retina)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add small padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)

    return image[y:y+h, x:x+w]


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on the green channel (best contrast for retinal vessels/lesions),
    then recombine with original R and B channels.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result


def preprocess_single_image(image_path: str, output_path: str) -> bool:
    """
    Full preprocessing pipeline for one fundus image.
    Returns True on success, False on failure.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [WARN] Failed to read: {image_path}")
            return False

        # Step 1: Crop black borders
        img = auto_crop_fundus(img)

        # Step 2: CLAHE enhancement
        img = apply_clahe(img)

        # Step 3: Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Step 4: Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return True

    except Exception as e:
        print(f"  [ERROR] {image_path}: {e}")
        return False


def preprocess_dataset(raw_dir: Path, output_dir: Path, dataset_name: str):
    """
    Process all images in a raw dataset directory.
    Expects images directly in raw_dir or in subdirectories.
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing {dataset_name}")
    print(f"  Source: {raw_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    if not raw_dir.exists():
        print(f"  [SKIP] Directory not found: {raw_dir}")
        return 0

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_files = [
        f for f in raw_dir.rglob("*")
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"  [SKIP] No images found in {raw_dir}")
        return 0

    print(f"  Found {len(image_files)} images")

    processed_dir = output_dir / "processed_images" / dataset_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for img_path in tqdm(image_files, desc=f"  Processing {dataset_name}"):
        # Preserve relative structure
        rel_path = img_path.relative_to(raw_dir)
        out_path = processed_dir / rel_path.with_suffix(".png")

        if preprocess_single_image(img_path, out_path):
            success_count += 1

    print(f"  Processed: {success_count}/{len(image_files)}")
    return success_count


def main():
    """Run preprocessing on all configured datasets."""
    total = 0
    total += preprocess_dataset(MBRSET_RAW_DIR, DATASET_DIR, "mbrset")
    total += preprocess_dataset(IDRID_RAW_DIR, DATASET_DIR, "idrid")

    print(f"\n{'='*60}")
    print(f"Total images preprocessed: {total}")
    print(f"Output directory: {DATASET_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
