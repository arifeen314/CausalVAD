#!/usr/bin/env python3
"""
==============================================================================
Prepare UCF-Crime CLIP Features
==============================================================================
The VadCLIP download organizes features in subfolders by category:
    UCFClipFeatures/
        Abuse/
            Abuse001_x264.npy
        Arrest/
            ...
        Normal_Videos_event/
            ...

This script:
  1. Finds all .npy files across all subfolders
  2. Copies them to a flat directory (data/features/ucf_crime_clip_flat/)
  3. Creates train/test annotation files
  4. Reports feature shapes and statistics

Usage:
    python scripts/prepare_ucf_clip.py
==============================================================================
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

# ─── Configuration ────────────────────────────────────────────────
SOURCE_DIR = PROJECT_ROOT / "data" / "features" / "ucf_crime_clip" / "UCFClipFeatures"
FLAT_DIR = PROJECT_ROOT / "data" / "features" / "ucf_crime_clip_flat"
ANN_DIR = PROJECT_ROOT / "data" / "annotations"

# UCF-Crime anomaly categories
ANOMALY_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]

# Folders that contain NORMAL videos
NORMAL_FOLDERS = [
    "Normal_Videos_event",
    "Training_Normal_Videos_Anomaly",
    "Testing_Normal_Videos_Anomaly",
]

# UCF-Crime standard test split (these video name prefixes go to test set)
# Based on the standard split used by Sultani et al. and RTFM
# Test set: 150 normal + 140 anomaly videos


def find_all_npy_files(source_dir):
    """Recursively find all .npy files and classify them."""
    anomaly_files = []
    normal_files = []
    
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if not f.endswith(".npy"):
                continue
            
            full_path = Path(root) / f
            # Determine if normal or anomaly based on parent folder
            parent_folder = Path(root).name
            
            if parent_folder in NORMAL_FOLDERS or "Normal" in parent_folder:
                normal_files.append((f, full_path, "normal"))
            elif parent_folder in ANOMALY_CATEGORIES:
                anomaly_files.append((f, full_path, parent_folder))
            else:
                # Check if any anomaly category appears in the path
                is_anomaly = False
                for cat in ANOMALY_CATEGORIES:
                    if cat in str(full_path):
                        anomaly_files.append((f, full_path, cat))
                        is_anomaly = True
                        break
                if not is_anomaly:
                    # Default: treat as normal
                    normal_files.append((f, full_path, "normal"))
    
    return anomaly_files, normal_files


def main():
    print("=" * 60)
    print("  Preparing UCF-Crime CLIP Features")
    print("=" * 60)
    
    # ─── Check source exists ─────────────────────────────────────
    if not SOURCE_DIR.exists():
        print(f"\nERROR: Source directory not found: {SOURCE_DIR}")
        print("Make sure you extracted the zip to:")
        print("  data/features/ucf_crime_clip/UCFClipFeatures/")
        sys.exit(1)
    
    # ─── Find all files ──────────────────────────────────────────
    print(f"\nScanning: {SOURCE_DIR}")
    anomaly_files, normal_files = find_all_npy_files(SOURCE_DIR)
    
    print(f"\n  Anomaly videos found: {len(anomaly_files)}")
    print(f"  Normal videos found:  {len(normal_files)}")
    print(f"  Total:                {len(anomaly_files) + len(normal_files)}")
    
    # Count by category
    category_counts = defaultdict(int)
    for _, _, cat in anomaly_files:
        category_counts[cat] += 1
    
    print("\n  Anomaly categories:")
    for cat in sorted(category_counts.keys()):
        print(f"    {cat:<20} {category_counts[cat]:>4} videos")
    
    # ─── Create flat directory ───────────────────────────────────
    FLAT_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Copying to flat directory: {FLAT_DIR}")
    
    all_files = anomaly_files + normal_files
    copied = 0
    skipped = 0
    
    for filename, source_path, category in all_files:
        dest_path = FLAT_DIR / filename
        
        # Handle duplicate filenames by prefixing with category
        if dest_path.exists():
            name_stem = Path(filename).stem
            name_ext = Path(filename).suffix
            dest_path = FLAT_DIR / f"{category}_{name_stem}{name_ext}"
        
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            copied += 1
        else:
            skipped += 1
    
    print(f"  Copied: {copied} files")
    if skipped > 0:
        print(f"  Skipped (already exist): {skipped}")
    
    # ─── Check feature dimensions ────────────────────────────────
    print("\n  Checking feature dimensions...")
    sample_files = list(FLAT_DIR.glob("*.npy"))[:5]
    for sf in sample_files:
        arr = np.load(str(sf))
        print(f"    {sf.name}: shape={arr.shape}, dtype={arr.dtype}")
    
    if len(sample_files) > 0:
        first = np.load(str(sample_files[0]))
        feature_dim = first.shape[-1]
        print(f"\n  Feature dimension: {feature_dim}")
    else:
        print("  WARNING: No .npy files found in flat directory!")
        sys.exit(1)
    
    # ─── Create train/test annotations ───────────────────────────
    # Standard UCF-Crime split: ~810 anomaly train + 800 normal train
    # ~140 anomaly test + 150 normal test
    # We use the "Testing_Normal_Videos_Anomaly" folder for test normals
    
    train_lines = []
    test_lines = []
    
    # Anomaly files: 80% train, 20% test (approximately)
    # Group by category for balanced split
    category_files = defaultdict(list)
    for filename, source_path, category in anomaly_files:
        # Use the flat filename
        flat_name = filename
        if not (FLAT_DIR / flat_name).exists():
            flat_name = f"{category}_{Path(filename).stem}{Path(filename).suffix}"
        
        flat_name_no_ext = Path(flat_name).stem
        category_files[category].append(flat_name_no_ext)
    
    for category, files in category_files.items():
        files.sort()
        split_idx = int(len(files) * 0.8)
        for f in files[:split_idx]:
            train_lines.append(f"{f} 1")
        for f in files[split_idx:]:
            test_lines.append(f"{f} 1")
    
    # Normal files: use folder to determine train vs test
    for filename, source_path, category in normal_files:
        flat_name = filename
        if not (FLAT_DIR / flat_name).exists():
            flat_name = f"{category}_{Path(filename).stem}{Path(filename).suffix}"
        
        flat_name_no_ext = Path(flat_name).stem
        parent = source_path.parent.name
        
        if "Testing" in parent:
            test_lines.append(f"{flat_name_no_ext} 0")
        elif "Training" in parent:
            train_lines.append(f"{flat_name_no_ext} 0")
        else:
            # Normal_Videos_event — split 80/20
            train_lines.append(f"{flat_name_no_ext} 0")
    
    # If test has too few normals, move some from train
    test_normals = [l for l in test_lines if l.endswith(" 0")]
    train_normals = [l for l in train_lines if l.endswith(" 0")]
    
    if len(test_normals) < 50 and len(train_normals) > 100:
        # Move some normals to test
        import random
        random.seed(42)
        random.shuffle(train_normals)
        move_count = min(150, len(train_normals) // 5)
        
        moved = train_normals[:move_count]
        remaining = train_normals[move_count:]
        
        train_lines = [l for l in train_lines if not l.endswith(" 0")] + remaining
        test_lines = test_lines + moved
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(train_lines)
    random.shuffle(test_lines)
    
    # Write annotation files
    train_path = ANN_DIR / "ucf_crime_clip_train.txt"
    test_path = ANN_DIR / "ucf_crime_clip_test.txt"
    
    with open(train_path, "w") as f:
        f.write("# video_name label (0=normal, 1=anomaly)\n")
        f.write("\n".join(train_lines) + "\n")
    
    with open(test_path, "w") as f:
        f.write("# video_name label (0=normal, 1=anomaly)\n")
        f.write("\n".join(test_lines) + "\n")
    
    # ─── Summary ─────────────────────────────────────────────────
    train_anomaly = sum(1 for l in train_lines if l.endswith(" 1"))
    train_normal = sum(1 for l in train_lines if l.endswith(" 0"))
    test_anomaly = sum(1 for l in test_lines if l.endswith(" 1"))
    test_normal = sum(1 for l in test_lines if l.endswith(" 0"))
    
    print(f"\n  Annotation files created:")
    print(f"    Train: {train_path}")
    print(f"      Anomaly: {train_anomaly}, Normal: {train_normal}, Total: {len(train_lines)}")
    print(f"    Test:  {test_path}")
    print(f"      Anomaly: {test_anomaly}, Normal: {test_normal}, Total: {len(test_lines)}")
    
    # ─── Update config note ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  DATA READY!")
    print(f"{'=' * 60}")
    print(f"\n  To train on UCF-Crime CLIP features, run:")
    print(f"    python scripts/train.py --dataset ucf_crime_clip --epochs 50 --batch_size 8 --num_workers 0")
    print(f"\n  Feature directory: {FLAT_DIR}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Total videos:      {len(train_lines) + len(test_lines)}")


if __name__ == "__main__":
    main()
