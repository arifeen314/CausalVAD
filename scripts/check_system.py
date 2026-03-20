#!/usr/bin/env python3
"""
==============================================================================
CausalVAD — System Diagnostic Script
==============================================================================
Run this FIRST before anything else.
It checks your entire system and tells you exactly what to install.

Usage:
    python check_system.py

You will see a report with ✅ (ready) or ❌ (action needed) for each item.
==============================================================================
"""

import sys
import os
import platform
import shutil
import subprocess
import importlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg):
    print(f"  {Colors.GREEN}✅ {msg}{Colors.END}")


def fail(msg):
    print(f"  {Colors.RED}❌ {msg}{Colors.END}")


def warn(msg):
    print(f"  {Colors.YELLOW}⚠️  {msg}{Colors.END}")


def info(msg):
    print(f"  {Colors.BLUE}ℹ️  {msg}{Colors.END}")


def header(msg):
    width = 60
    print(f"\n{Colors.BOLD}{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}{Colors.END}")


def subheader(msg):
    print(f"\n{Colors.BOLD}  --- {msg} ---{Colors.END}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
issues = []


def check_python():
    header("1. PYTHON VERSION")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    print(f"  Python version: {version_str}")
    print(f"  Executable:     {sys.executable}")

    if v.major == 3 and v.minor >= 9:
        ok(f"Python {version_str} is supported (need 3.9+)")
    elif v.major == 3 and v.minor == 8:
        warn(f"Python {version_str} may work but 3.10+ is recommended")
        issues.append("UPGRADE Python to 3.10+ for best compatibility")
    else:
        fail(f"Python {version_str} is NOT supported — need 3.9+")
        issues.append("INSTALL Python 3.10 from https://www.python.org/downloads/")


def check_os():
    header("2. OPERATING SYSTEM")
    os_name = platform.system()
    os_version = platform.version()
    machine = platform.machine()
    print(f"  OS:       {os_name} {platform.release()}")
    print(f"  Version:  {os_version}")
    print(f"  Arch:     {machine}")

    if os_name in ("Windows", "Linux", "Darwin"):
        ok(f"{os_name} is supported")
    else:
        warn(f"Untested OS: {os_name}")

    if os_name == "Darwin" and "arm" in machine.lower():
        info("Apple Silicon detected — will use MPS acceleration (no CUDA needed)")
    return os_name


def check_gpu():
    header("3. GPU / CUDA")

    # Check NVIDIA GPU via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        print(f"  GPU:     {parts[0]}")
                        print(f"  VRAM:    {parts[1]} MB")
                        print(f"  Driver:  {parts[2]}")
                        vram = int(parts[1])
                        if vram >= 6000:
                            ok(f"GPU has {vram} MB VRAM — sufficient for this project")
                        elif vram >= 4000:
                            warn(f"GPU has {vram} MB VRAM — tight but possible with small batch sizes")
                            issues.append("Consider using CPU mode or reducing batch size")
                        else:
                            warn(f"GPU has {vram} MB VRAM — will use CPU-heavy pipeline")
                            issues.append("Low VRAM — will rely on pre-extracted features + CPU")
            else:
                fail("nvidia-smi found but returned an error")
                issues.append("Check NVIDIA driver installation")
        except Exception as e:
            fail(f"nvidia-smi error: {e}")
    else:
        info("No NVIDIA GPU detected (nvidia-smi not found)")
        info("This is FINE — the project is designed to work on CPU")
        info("We use pre-extracted features so GPU is optional")


def check_pytorch():
    header("4. PYTORCH")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            ok(f"CUDA available — device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            ok("MPS (Apple Silicon) available")
        else:
            info("No GPU acceleration — will use CPU (this is fine)")

        ok("PyTorch is installed and working")
    except ImportError:
        fail("PyTorch is NOT installed")
        issues.append(
            "INSTALL PyTorch: pip install torch torchvision torchaudio\n"
            "         Or with CUDA: visit https://pytorch.org/get-started/locally/"
        )


def check_package(name, import_name=None, min_version=None):
    """Check if a Python package is installed."""
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        ok(f"{name} {version}")
        return True
    except ImportError:
        fail(f"{name} is NOT installed")
        issues.append(f"INSTALL: pip install {name}")
        return False


def check_packages():
    header("5. REQUIRED PYTHON PACKAGES")

    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("transformers", "transformers"),
        ("open_clip_torch", "open_clip"),
        ("scipy", "scipy"),
        ("pyyaml", "yaml"),
    ]

    missing = []
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing.append(pkg_name)

    if missing:
        install_cmd = "pip install " + " ".join(missing)
        info(f"Install all missing packages with:\n         {install_cmd}")


def check_tools():
    header("6. COMMAND-LINE TOOLS")

    # Git
    git = shutil.which("git")
    if git:
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            ok(f"Git: {result.stdout.strip()}")
        except Exception:
            fail("Git found but not working")
    else:
        fail("Git is NOT installed")
        issues.append(
            "INSTALL Git:\n"
            "  Windows: https://git-scm.com/download/win\n"
            "  Mac:     brew install git\n"
            "  Linux:   sudo apt install git"
        )

    # ffmpeg (needed for video frame extraction)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        ok("ffmpeg is installed (needed for video processing)")
    else:
        warn("ffmpeg is NOT installed (needed for frame extraction)")
        issues.append(
            "INSTALL ffmpeg:\n"
            "  Windows: https://ffmpeg.org/download.html (add to PATH)\n"
            "  Mac:     brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


def check_disk_space():
    header("7. DISK SPACE")
    home = Path.home()
    try:
        usage = shutil.disk_usage(str(home))
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        print(f"  Free:  {free_gb:.1f} GB")
        print(f"  Total: {total_gb:.1f} GB")

        if free_gb >= 50:
            ok(f"{free_gb:.1f} GB free — sufficient for datasets + models")
        elif free_gb >= 20:
            warn(f"{free_gb:.1f} GB free — enough but consider cleaning up")
            issues.append("Free up disk space if possible (50 GB recommended)")
        else:
            fail(f"Only {free_gb:.1f} GB free — need at least 20 GB")
            issues.append("CRITICAL: Free up disk space (need 20+ GB for datasets)")
    except Exception as e:
        warn(f"Could not check disk space: {e}")


def check_ram():
    header("8. RAM")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"  Total:     {ram_gb:.1f} GB")
        print(f"  Available: {available_gb:.1f} GB")

        if ram_gb >= 16:
            ok(f"{ram_gb:.1f} GB RAM — excellent")
        elif ram_gb >= 8:
            warn(f"{ram_gb:.1f} GB RAM — will work with smaller batch sizes")
        else:
            fail(f"{ram_gb:.1f} GB RAM — may struggle with feature extraction")
            issues.append("Low RAM — use smaller batch sizes and feature caching")
    except ImportError:
        info("Install psutil for RAM check: pip install psutil")
        info("(Not critical — just informational)")


def print_summary():
    header("SUMMARY")
    if not issues:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 YOUR SYSTEM IS READY!{Colors.END}")
        print(f"  {Colors.GREEN}All checks passed. You can proceed to Step 2.{Colors.END}\n")
    else:
        print(f"\n  {Colors.YELLOW}{Colors.BOLD}📋 ACTION ITEMS ({len(issues)} items):{Colors.END}\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {Colors.YELLOW}{i}. {issue}{Colors.END}\n")
        print(f"  {Colors.BOLD}Fix these items, then run this script again.{Colors.END}\n")


def generate_system_report():
    """Write a machine-readable report for future reference."""
    report_path = Path(__file__).parent.parent / "outputs" / "logs" / "system_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "CausalVAD System Report",
        "=" * 40,
        f"Date: {__import__('datetime').datetime.now().isoformat()}",
        f"Python: {sys.version}",
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Executable: {sys.executable}",
    ]

    try:
        import torch
        lines.append(f"PyTorch: {torch.__version__}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        lines.append("PyTorch: NOT INSTALLED")

    lines.append(f"\nIssues found: {len(issues)}")
    for issue in issues:
        lines.append(f"  - {issue}")

    report_path.write_text("\n".join(lines))
    info(f"Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{Colors.BOLD}{'#' * 60}")
    print(f"#   CausalVAD — System Diagnostic")
    print(f"#   Run this FIRST before setting up the project")
    print(f"{'#' * 60}{Colors.END}")

    check_python()
    os_name = check_os()
    check_gpu()
    check_pytorch()
    check_packages()
    check_tools()
    check_disk_space()
    check_ram()
    generate_system_report()
    print_summary()


if __name__ == "__main__":
    main()
