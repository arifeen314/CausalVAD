#!/usr/bin/env python3
"""
==============================================================================
CausalVAD — Ablation Study Runner
==============================================================================

Runs all ablation experiments required for the paper's Table 2:

  Experiment                         | Causal Mask | Temporal Decay
  ───────────────────────────────────┼─────────────┼───────────────
  1. CausalVAD (Full)               |     ✓       |      ✓
  2. w/o Causal Mask                 |     ✗       |      ✓
  3. w/o Temporal Decay              |     ✓       |      ✗
  4. w/o Both (Baseline Prompts)     |     ✗       |      ✗
  5. No Prompts (Feature-only)       |    n/a      |     n/a

Usage:
    python scripts/run_ablations.py --dataset synthetic --epochs 30
    python scripts/run_ablations.py --dataset ucf_crime --epochs 50

Results are saved to outputs/ablation_results/
==============================================================================
"""

import os
import sys
import subprocess
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_experiment(name: str, extra_args: list, base_args: dict) -> dict:
    """Run a single training experiment as a subprocess."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {name}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
        "--dataset", base_args["dataset"],
        "--epochs", str(base_args["epochs"]),
        "--batch_size", str(base_args["batch_size"]),
        "--exp_name", f"ablation_{name}_{time.strftime('%m%d_%H%M')}",
    ] + extra_args

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    return {"name": name, "returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser(description="CausalVAD Ablation Studies")
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    base = {"dataset": args.dataset, "epochs": args.epochs, "batch_size": args.batch_size}

    experiments = [
        ("full_model",          []),
        ("no_causal_mask",      ["--no_causal_mask"]),
        ("no_temporal_decay",   ["--no_temporal_decay"]),
        ("no_causal_no_decay",  ["--no_causal_mask", "--no_temporal_decay"]),
        ("no_prompts",          ["--num_prompts", "0", "--no_causal_mask", "--no_temporal_decay"]),
    ]

    results = []
    for name, extra in experiments:
        result = run_experiment(name, extra, base)
        results.append(result)

    # Summary
    out_dir = PROJECT_ROOT / "outputs" / "ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    for r in results:
        status = "SUCCESS" if r["returncode"] == 0 else "FAILED"
        print(f"  {r['name']:<25} {status}")
    print(f"\nCheck individual experiment folders in outputs/ for detailed results.")

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
