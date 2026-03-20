#!/usr/bin/env python3
"""
==============================================================================
Create Standard UCF-Crime Split from VadCLIP CSV Files
==============================================================================

Maps the official VadCLIP train/test split to our merged features.
This makes results directly comparable to published VadCLIP numbers.

Usage:
    python scripts/create_standard_split.py
==============================================================================
"""

import os
import sys
import csv
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

TRAIN_CSV = PROJECT_ROOT / "data" / "annotations" / "ucf_CLIP_rgb.csv"
TEST_CSV = PROJECT_ROOT / "data" / "annotations" / "ucf_CLIP_rgbtest.csv"
MERGED_DIR = PROJECT_ROOT / "data" / "features" / "ucf_crime_merged"
ANN_DIR = PROJECT_ROOT / "data" / "annotations"

ANOMALY_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def extract_video_name(path_str):
    """
    Extract base video name from VadCLIP CSV path.
    
    Input:  /home/xbgydx/Desktop/UCFClipFeatures/Abuse/Abuse001_x264__0.npy
    Output: Abuse001_x264
    """
    filename = Path(path_str).stem  # Abuse001_x264__0
    # Remove crop suffix __N
    match = re.match(r'^(.+)__(\d+)$', filename)
    if match:
        return match.group(1)
    return filename


def get_label(category_str):
    """Convert category string to binary label."""
    if category_str in ANOMALY_CATEGORIES:
        return 1
    return 0  # Normal, Testing_Normal, Training-Normal, etc.


def main():
    print("=" * 60)
    print("  Creating Standard UCF-Crime Split")
    print("=" * 60)

    # Check files exist
    for f in [TRAIN_CSV, TEST_CSV]:
        if not f.exists():
            print(f"ERROR: {f} not found!")
            print("Download from: https://github.com/nwpu-zxr/VadCLIP/tree/main/list")
            sys.exit(1)

    # Get available merged features
    available = set()
    for f in MERGED_DIR.glob("*.npy"):
        available.add(f.stem)
    print(f"\n  Merged features available: {len(available)}")

    # Parse train CSV
    print(f"\n  Parsing train CSV: {TRAIN_CSV}")
    train_videos = {}  # video_name -> label
    with open(TRAIN_CSV, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            path_str = row[0]
            category = row[1]
            video_name = extract_video_name(path_str)
            label = get_label(category)
            train_videos[video_name] = label

    print(f"  Unique train videos: {len(train_videos)}")
    train_a = sum(1 for v in train_videos.values() if v == 1)
    train_n = sum(1 for v in train_videos.values() if v == 0)
    print(f"    Anomaly: {train_a}, Normal: {train_n}")

    # Parse test CSV
    print(f"\n  Parsing test CSV: {TEST_CSV}")
    test_videos = {}
    with open(TEST_CSV, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            path_str = row[0]
            category = row[1]
            video_name = extract_video_name(path_str)
            label = get_label(category)
            test_videos[video_name] = label

    print(f"  Unique test videos: {len(test_videos)}")
    test_a = sum(1 for v in test_videos.values() if v == 1)
    test_n = sum(1 for v in test_videos.values() if v == 0)
    print(f"    Anomaly: {test_a}, Normal: {test_n}")

    # Match to available merged features
    print(f"\n  Matching to merged features...")

    train_lines = []
    train_missing = 0
    for video_name, label in sorted(train_videos.items()):
        if video_name in available:
            train_lines.append(f"{video_name} {label}")
        else:
            train_missing += 1

    test_lines = []
    test_missing = 0
    for video_name, label in sorted(test_videos.items()):
        if video_name in available:
            test_lines.append(f"{video_name} {label}")
        else:
            test_missing += 1

    print(f"  Train matched: {len(train_lines)} (missing: {train_missing})")
    print(f"  Test matched:  {len(test_lines)} (missing: {test_missing})")

    if train_missing > 0 or test_missing > 0:
        print(f"\n  Note: Missing videos may have different naming conventions.")
        print(f"  Checking for partial matches...")

        # Try fuzzy matching for missing videos
        for video_name, label in sorted(train_videos.items()):
            if video_name not in available:
                # Try without category prefix variations
                for avail_name in available:
                    if video_name.replace("_x264", "") in avail_name or avail_name.replace("_x264", "") in video_name:
                        if not any(video_name in line for line in train_lines):
                            train_lines.append(f"{avail_name} {label}")
                            break

        for video_name, label in sorted(test_videos.items()):
            if video_name not in available:
                for avail_name in available:
                    if video_name.replace("_x264", "") in avail_name or avail_name.replace("_x264", "") in video_name:
                        if not any(video_name in line for line in test_lines):
                            test_lines.append(f"{avail_name} {label}")
                            break

    # Write annotation files
    train_path = ANN_DIR / "ucf_standard_train.txt"
    test_path = ANN_DIR / "ucf_standard_test.txt"

    with open(train_path, "w") as f:
        f.write("# Standard UCF-Crime split (from VadCLIP)\n")
        f.write("\n".join(train_lines) + "\n")

    with open(test_path, "w") as f:
        f.write("# Standard UCF-Crime split (from VadCLIP)\n")
        f.write("\n".join(test_lines) + "\n")

    # Final stats
    final_train_a = sum(1 for l in train_lines if l.endswith(" 1"))
    final_train_n = sum(1 for l in train_lines if l.endswith(" 0"))
    final_test_a = sum(1 for l in test_lines if l.endswith(" 1"))
    final_test_n = sum(1 for l in test_lines if l.endswith(" 0"))

    print(f"\n  STANDARD SPLIT CREATED:")
    print(f"  Train: {train_path}")
    print(f"    Anomaly: {final_train_a}, Normal: {final_train_n}, Total: {len(train_lines)}")
    print(f"  Test:  {test_path}")
    print(f"    Anomaly: {final_test_a}, Normal: {final_test_n}, Total: {len(test_lines)}")

    # Patch train.py to add ucf_standard dataset
    train_script = PROJECT_ROOT / "scripts" / "train.py"
    content = train_script.read_text(encoding="utf-8")

    if "ucf_standard" not in content:
        new_dataset = '''
        "ucf_standard": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_merged"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_standard_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_standard_test.txt"),
            "feature_dim": 512,
        },'''

        # Insert after ucf_merged block
        if "ucf_merged" in content:
            # Find the closing of ucf_merged block
            idx = content.find('"ucf_merged"')
            brace_start = content.find('{', idx)
            depth = 0
            i = brace_start
            while i < len(content):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        comma = content.find(',', i)
                        if comma != -1 and comma - i < 5:
                            content = content[:comma + 1] + new_dataset + content[comma + 1:]
                        break
                i += 1

        # Add to argparse choices
        if '"ucf_standard"' not in content:
            content = content.replace(
                '"ucf_merged",',
                '"ucf_merged", "ucf_standard",'
            )

        train_script.write_text(content, encoding="utf-8")
        print(f"\n  Patched train.py with 'ucf_standard' dataset option")

    print(f"\n{'=' * 60}")
    print(f"  READY! Run:")
    print(f"  python scripts/train.py --dataset ucf_standard --epochs 50 --batch_size 8 --model_dim 256 --num_heads 8 --num_workers 0")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
