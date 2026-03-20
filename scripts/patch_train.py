#!/usr/bin/env python3
"""
Patches train.py to add support for the flattened UCF-Crime CLIP features.
Run once: python scripts/patch_train.py
"""
from pathlib import Path

train_file = Path(__file__).parent.parent / "scripts" / "train.py"
content = train_file.read_text(encoding="utf-8")

# 1. Add ucf_crime_clip_flat dataset config
old_ucf_clip_block = '''        "ucf_crime_clip": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_clip"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_test.txt"),
            "feature_dim": 512,
        },'''

new_ucf_clip_block = '''        "ucf_crime_clip": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_clip"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_test.txt"),
            "feature_dim": 512,
        },
        "ucf_crime_clip_flat": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_clip_flat"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_clip_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_clip_test.txt"),
            "feature_dim": 512,
        },'''

content = content.replace(old_ucf_clip_block, new_ucf_clip_block)

# 2. Add ucf_crime_clip_flat to argparse choices
old_choices = 'choices=["synthetic", "ucf_crime", "ucf_crime_clip",\n                                 "xd_violence", "shanghaitech"]'
new_choices = 'choices=["synthetic", "ucf_crime", "ucf_crime_clip",\n                                 "ucf_crime_clip_flat", "xd_violence", "shanghaitech"]'

content = content.replace(old_choices, new_choices)

# 3. Fix Unicode star character for Windows
content = content.replace('\u2605', '[BEST]')
content = content.replace('\u2014', '-')

train_file.write_text(content, encoding="utf-8")
print("Patched train.py successfully!")
print("  - Added ucf_crime_clip_flat dataset")
print("  - Fixed Unicode characters for Windows")
