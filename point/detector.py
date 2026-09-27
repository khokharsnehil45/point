"""Object detection abstractions and backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from point.config import BUILTIN_DEFAULT_MODEL, get_default_model
from point.models import (
    Detection,
    ImageNotFoundError,
    InvalidImageError,
    ModelError,
)

DEFAULT_MODEL_NAME = BUILTIN_DEFAULT_MODEL



def validate_image_path(image_path: str | Path) -> Path:
    """Validate that an image exists and is readable as a valid image file.

    Args:
        image_path: File path to the image.

    Returns:
        Validated Path object.

    Raises:
        ImageNotFoundError: If the file does not exist.
        InvalidImageError: If the file is not a valid image format or is corrupted.
    """
    path = Path(image_path)
    if not path.is_file():
        raise ImageNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not read image: {image_path}") from exc

    return path


class Detector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def detect(self, image_path: str | Path) -> list[Detection]:
        """Perform object detection on the given image path.

        Args:
            image_path: Path to the input image file.

        Returns:
            List of Detection objects.
        """


class YOLODetector(Detector):
    """Ultralytics YOLO detection backend.

    Attributes:
        model_name: Model identifier or local weight path (e.g. 'yolo11n.pt').
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_default_model()
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazily load and cache the YOLO model instance."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
            except Exception as exc:
                raise ModelError(
                    f"Failed to load detection model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def detect(self, image_path: str | Path) -> list[Detection]:
        """Detect objects in the image using the YOLO model.

        Args:
            image_path: Input image file path.

        Returns:
            A list of Detection instances.
        """
        # Validate image existence and integrity before running model
        valid_path = validate_image_path(image_path)
        model = self._get_model()

        try:
            results = model(str(valid_path), verbose=False)
        except Exception as exc:
            raise ModelError(f"Inference failed on '{image_path}': {exc}") from exc

        detections: list[Detection] = []
        source_name = str(image_path)

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            names = getattr(result, "names", {})

            for box in boxes:
                cls_val = box.cls
                if hasattr(cls_val, "item"):
                    cls_id = int(cls_val.item())
                elif hasattr(cls_val, "__getitem__"):
                    cls_id = int(cls_val[0])
                else:
                    cls_id = int(cls_val)

                if isinstance(names, dict):
                    class_name = names.get(cls_id, str(cls_id))
                elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                    class_name = str(names[cls_id])
                else:
                    class_name = str(cls_id)

                conf_val = box.conf
                if hasattr(conf_val, "item"):
                    confidence = float(conf_val.item())
                elif hasattr(conf_val, "__getitem__"):
                    confidence = float(conf_val[0])
                else:
                    confidence = float(conf_val)

                xyxy_val = box.xyxy
                if hasattr(xyxy_val, "tolist"):
                    coords = xyxy_val.tolist()
                else:
                    coords = list(xyxy_val)

                if coords and isinstance(coords[0], (list, tuple)):
                    coords = coords[0]

                x1 = int(round(coords[0]))
                y1 = int(round(coords[1]))
                x2 = int(round(coords[2]))
                y2 = int(round(coords[3]))

                detections.append(
                    Detection(
                        image=source_name,
                        class_name=class_name,
                        class_id=cls_id,
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                    )
                )

        return detections
