"""Point: A lightweight command-line object detection tool."""

from point.detector import DEFAULT_MODEL_NAME, Detector, YOLODetector
from point.models import (
    Detection,
    ImageNotFoundError,
    InvalidImageError,
    ModelError,
    PointError,
)
from point.output import (
    DIVIDER,
    MODEL_CATALOG,
    format_model_catalog,
    format_terminal_output,
    normalize_jsonl_filename,
    serialize_jsonl,
    write_jsonl,
)
from point.updater import run_update
from point.version import __version__
from point.visualization import draw_detections

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DIVIDER",
    "Detection",
    "Detector",
    "ImageNotFoundError",
    "InvalidImageError",
    "MODEL_CATALOG",
    "ModelError",
    "PointError",
    "YOLODetector",
    "draw_detections",
    "format_model_catalog",
    "format_terminal_output",
    "normalize_jsonl_filename",
    "run_update",
    "serialize_jsonl",
    "write_jsonl",
    "__version__",
]
