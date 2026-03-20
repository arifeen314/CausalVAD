#!/usr/bin/env python3
"""
==============================================================================
Merge 10-Crop Features into Single Features Per Video
==============================================================================

The VadCLIP CLIP features have 10 crops per video:
    Abuse001_x264__0.npy
    Abuse001_x264__1.npy
    ...
    Abuse001_x264__9.npy

Each is shape (T, 512). Standard practice is to AVERAGE them.
This gives one (T, 512) feature per video.

This reduces 19,500 files → ~1,950 files (the real UCF-Crime size).

Usage:
    python scripts/merge_crops.py
==============================================================================
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

PROJECT_ROOT = Path(__file__).parent.parent
FLAT_DIR = PROJECT_ROOT / "data" / "features" / "ucf_crime_clip_flat"
MERGED_DIR = PROJECT_ROOT / "data" / "features" / "ucf_crime_merged"
ANN_DIR = PROJECT_ROOT / "data" / "annotations"

# Anomaly categories for labeling
ANOMALY_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def get_base_video_name(filename):
    """
    Extract the base video name by removing the crop suffix.
    
    Examples:
        'Abuse001_x264__0.npy'        → 'Abuse001_x264'
        'Abuse001_x264__9.npy'        → 'Abuse001_x264'
        'Normal_Videos_event001__3.npy' → 'Normal_Videos_event001'
        'normal_Testing_Normal_Videos_Anomaly_001__5.npy' → 'normal_Testing_Normal_Videos_Anomaly_001'
    """
    stem = Path(filename).stem  # Remove .npy
    
    # Match pattern: anything ending with __DIGIT
    match = re.match(r'^(.+)__(\d+)$', stem)
    if match:
        return match.group(1)
    
    # If no crop suffix found, return as is
    return stem


def main():
    print("=" * 60)
    print("  Merging 10-Crop Features → 1 Feature Per Video")
    print("=" * 60)
    
    if not FLAT_DIR.exists():
        print(f"\nERROR: Flat feature directory not found: {FLAT_DIR}")
        print("Run prepare_ucf_clip.py first!")
        sys.exit(1)
    
    # ─── Group files by base video name ──────────────────────────
    print(f"\nScanning: {FLAT_DIR}")
    video_groups = defaultdict(list)
    
    all_files = sorted(FLAT_DIR.glob("*.npy"))
    for f in all_files:
        base_name = get_base_video_name(f.name)
        video_groups[base_name].append(f)
    
    print(f"  Total .npy files:     {len(all_files)}")
    print(f"  Unique videos found:  {len(video_groups)}")
    
    # Show crop distribution
    crop_counts = [len(v) for v in video_groups.values()]
    print(f"  Crops per video:      min={min(crop_counts)}, "
          f"max={max(crop_counts)}, avg={np.mean(crop_counts):.1f}")
    
    # ─── Average crops and save ──────────────────────────────────
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Merging features (averaging crops)...")
    print(f"  Output directory: {MERGED_DIR}")
    
    merged_count = 0
    errors = 0
    
    for base_name, file_list in sorted(video_groups.items()):
        try:
            # Load all crops
            arrays = []
            for f in file_list:
                arr = np.load(str(f)).astype(np.float32)
                arrays.append(arr)
            
            # All crops should have the same shape
            # Average across crops: (10, T, 512) → (T, 512)
            if len(arrays) > 1:
                # Handle potentially different lengths by using shortest
                min_len = min(a.shape[0] for a in arrays)
                trimmed = [a[:min_len] for a in arrays]
                merged = np.mean(trimmed, axis=0)
            else:
                merged = arrays[0]
            
            # Save merged features
            out_path = MERGED_DIR / f"{base_name}.npy"
            np.save(str(out_path), merged.astype(np.float32))
            merged_count += 1
            
        except Exception as e:
            print(f"    ERROR processing {base_name}: {e}")
            errors += 1
    
    print(f"\n  Merged: {merged_count} videos")
    if errors > 0:
        print(f"  Errors: {errors}")
    
    # ─── Check a sample ──────────────────────────────────────────
    sample_files = sorted(MERGED_DIR.glob("*.npy"))[:3]
    print(f"\n  Sample merged features:")
    for sf in sample_files:
        arr = np.load(str(sf))
        print(f"    {sf.name}: shape={arr.shape}, dtype={arr.dtype}")
    
    # ─── Create annotations ──────────────────────────────────────
    print(f"\n  Creating annotation files...")
    
    train_lines = []
    test_lines = []
    
    # Classify each video as normal or anomaly
    all_merged = sorted(MERGED_DIR.glob("*.npy"))
    
    normal_videos = []
    anomaly_videos = []
    
    for f in all_merged:
        name = f.stem
        
        # Check if this is an anomaly video
        is_anomaly = False
        for cat in ANOMALY_CATEGORIES:
            if name.startswith(cat) or name.startswith(f"{cat}_"):
                is_anomaly = True
                break
        
        if is_anomaly:
            anomaly_videos.append(name)
        else:
            normal_videos.append(name)
    
    print(f"  Anomaly videos: {len(anomaly_videos)}")
    print(f"  Normal videos:  {len(normal_videos)}")
    
    # Standard split: ~80% train, ~20% test
    import random
    random.seed(42)
    
    random.shuffle(anomaly_videos)
    random.shuffle(normal_videos)
    
    # Anomaly split
    a_split = int(len(anomaly_videos) * 0.8)
    for v in anomaly_videos[:a_split]:
        train_lines.append(f"{v} 1")
    for v in anomaly_videos[a_split:]:
        test_lines.append(f"{v} 1")
    
    # Normal split
    n_split = int(len(normal_videos) * 0.8)
    for v in normal_videos[:n_split]:
        train_lines.append(f"{v} 0")
    for v in normal_videos[n_split:]:
        test_lines.append(f"{v} 0")
    
    # Shuffle
    random.shuffle(train_lines)
    random.shuffle(test_lines)
    
    # Write
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = ANN_DIR / "ucf_merged_train.txt"
    test_path = ANN_DIR / "ucf_merged_test.txt"
    
    with open(train_path, "w") as f:
        f.write("# video_name label (0=normal, 1=anomaly)\n")
        f.write("\n".join(train_lines) + "\n")
    
    with open(test_path, "w") as f:
        f.write("# video_name label (0=normal, 1=anomaly)\n")
        f.write("\n".join(test_lines) + "\n")
    
    train_a = sum(1 for l in train_lines if l.endswith(" 1"))
    train_n = sum(1 for l in train_lines if l.endswith(" 0"))
    test_a = sum(1 for l in test_lines if l.endswith(" 1"))
    test_n = sum(1 for l in test_lines if l.endswith(" 0"))
    
    print(f"\n  Train: {train_a} anomaly + {train_n} normal = {len(train_lines)} total")
    print(f"  Test:  {test_a} anomaly + {test_n} normal = {len(test_lines)} total")
    print(f"  Saved: {train_path}")
    print(f"         {test_path}")
    
    # ─── Patch train.py for merged dataset ───────────────────────
    train_script = PROJECT_ROOT / "scripts" / "train.py"
    content = train_script.read_text(encoding="utf-8")
    
    if "ucf_merged" not in content:
        # Add new dataset config
        insert_after = '"feature_dim": 512,\n        },'
        # Find the last occurrence before the closing brace
        new_dataset = '''
        "ucf_merged": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_merged"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_merged_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_merged_test.txt"),
            "feature_dim": 512,
        },'''
        
        # Find the right place to insert - after ucf_crime_clip_flat block
        if "ucf_crime_clip_flat" in content:
            marker = '"ucf_crime_clip_flat": {'
            idx = content.find(marker)
            # Find the closing }, of this block
            brace_count = 0
            i = idx
            while i < len(content):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Find the comma after
                        end_idx = content.find(',', i)
                        if end_idx != -1:
                            content = content[:end_idx+1] + new_dataset + content[end_idx+1:]
                        break
                i += 1
        else:
            # Fallback: insert before the closing of datasets dict
            content = content.replace(
                '"feature_dim": 2048,\n        },\n    }',
                '"feature_dim": 2048,\n        },\n' + new_dataset + '\n    }'
            )
        
        # Add to choices
        if "ucf_merged" not in content:
            content = content.replace(
                '"ucf_crime_clip_flat",',
                '"ucf_crime_clip_flat", "ucf_merged",'
            )
        
        train_script.write_text(content, encoding="utf-8")
        print(f"\n  Patched train.py to add 'ucf_merged' dataset option")
    
    # ─── Final instructions ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ALL DONE! Ready to train on real UCF-Crime data.")
    print(f"{'=' * 60}")
    print(f"\n  Run this command:")
    print(f"    python scripts/train.py --dataset ucf_merged --epochs 50 --batch_size 8 --model_dim 256 --num_heads 8 --num_workers 0")
    print(f"\n  Expected training time: 20-40 minutes on your GTX 1650")
    print(f"  Expected AUC: 0.75-0.88 (this is the research result!)")


if __name__ == "__main__":
    main()
