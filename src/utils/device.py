"""
Device selection utility.
Automatically picks the best available device (CUDA > MPS > CPU).
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the best available compute device.

    Args:
        preference: One of "auto", "cuda", "mps", "cpu".
                    "auto" picks the best available.

    Returns:
        torch.device object ready to use.

    Example:
        >>> device = get_device("auto")
        >>> tensor = torch.randn(3, 3).to(device)
    """
    if preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    elif preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS acceleration")

    elif preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected Apple MPS")
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected CPU (no GPU detected)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def get_device_info() -> dict:
    """Return a dictionary of device information for logging."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    return info
