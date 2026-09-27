"""Configuration and persistent settings for Point CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "point"
CONFIG_FILE = CONFIG_DIR / "config.json"
BUILTIN_DEFAULT_MODEL = "yolo11n.pt"


def get_default_model() -> str:
    """Resolve the active default model with priority:
    1. POINT_MODEL environment variable
    2. ~/.config/point/config.json setting
    3. Built-in default ('yolo11n.pt')

    Returns:
        The resolved model name or file path string.
    """
    env_model = os.getenv("POINT_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()

    if CONFIG_FILE.is_file():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("model"):
                return str(data["model"]).strip()
        except Exception:
            pass

    return BUILTIN_DEFAULT_MODEL


def set_default_model(model_name: str) -> Path:
    """Save persistent default model to user config.

    Args:
        model_name: Model identifier or checkpoint file path.

    Returns:
        Path to the saved configuration file.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name.strip()}
    CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return CONFIG_FILE
