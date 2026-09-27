"""Data models and exception definitions for Point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


class PointError(Exception):
    """Base exception for all Point CLI errors."""


class ImageNotFoundError(PointError, FileNotFoundError):
    """Raised when the specified image file does not exist."""


class InvalidImageError(PointError, ValueError):
    """Raised when the image file cannot be read or identified."""


class ModelError(PointError, RuntimeError):
    """Raised when an object detection model fails to load or infer."""


@dataclass(frozen=True)
class Detection:
    """Represents a single detected object in an image.

    Attributes:
        image: Path to the source image.
        class_name: Object class label (e.g. 'person', 'car').
        class_id: Numerical class identifier.
        confidence: Prediction confidence score between 0.0 and 1.0.
        box: Bounding box coordinates (x1, y1, x2, y2) in pixel space.
    """

    image: str
    class_name: str
    class_id: int
    confidence: float
    box: Sequence[int | float]

    def to_dict(self) -> dict[str, Any]:
        """Convert detection to dictionary representation for JSONL serialization."""
        return {
            "image": self.image,
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "box": [
                int(c) if isinstance(c, float) and c.is_integer() else c
                for c in self.box
            ],
        }
