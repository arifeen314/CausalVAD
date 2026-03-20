"""
Configuration loader.
Loads YAML config files and provides easy attribute access.
"""

import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """
    Nested configuration object with attribute-style access.

    Example:
        >>> cfg = Config.from_yaml("configs/default.yaml")
        >>> print(cfg.model.vlm.name)        # "ViT-B-16"
        >>> print(cfg.training.batch_size)    # 32
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    Config(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def to_dict(self) -> dict:
        """Convert config back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a nested key using dot notation."""
        keys = key.split(".")
        obj = self
        for k in keys:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return default
        return obj

    def __repr__(self):
        return f"Config({self.to_dict()})"


def load_config(
    config_path: str = "configs/default.yaml",
    overrides: Optional[dict] = None
) -> Config:
    """
    Load config with optional overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: Dict of key-value pairs to override.
                   Keys use dot notation: "training.batch_size": 16

    Returns:
        Config object.
    """
    cfg = Config.from_yaml(config_path)

    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return cfg
