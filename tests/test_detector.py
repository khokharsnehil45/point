"""Unit tests for Point detector abstraction, YOLO backend, and image validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from point.detector import (
    DEFAULT_MODEL_NAME,
    Detector,
    YOLODetector,
    validate_image_path,
)
from point.models import (
    ImageNotFoundError,
    InvalidImageError,
    ModelError,
)


def test_detector_cannot_be_instantiated_directly() -> None:
    """Test that abstract Detector base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Detector()  # type: ignore[abstract]


def test_validate_image_path_nonexistent(tmp_path: Path) -> None:
    """Test that nonexistent image path raises ImageNotFoundError."""
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageNotFoundError, match="Image not found"):
        validate_image_path(missing)


def test_validate_image_path_corrupt(tmp_path: Path) -> None:
    """Test that a corrupt non-image file raises InvalidImageError."""
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image file content")
    with pytest.raises(InvalidImageError, match="Could not read image"):
        validate_image_path(corrupt)


def test_validate_image_path_valid(tmp_path: Path) -> None:
    """Test that a valid image passes validation."""
    valid_file = tmp_path / "valid.jpg"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(valid_file)

    result = validate_image_path(valid_file)
    assert result == valid_file


def test_yolo_detector_init() -> None:
    """Test YOLODetector initialization and default model name."""
    detector = YOLODetector()
    assert detector.model_name == DEFAULT_MODEL_NAME
    assert detector._model is None

    custom = YOLODetector(model_name="custom.pt")
    assert custom.model_name == "custom.pt"


def test_yolo_detector_model_caching() -> None:
    """Test that YOLO model instance is cached after first access."""
    detector = YOLODetector()
    mock_yolo_instance = MagicMock()
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.return_value = mock_yolo_instance

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
        m1 = detector._get_model()
        m2 = detector._get_model()
        assert m1 is m2
        mock_ultralytics.YOLO.assert_called_once_with(DEFAULT_MODEL_NAME)


def test_yolo_detector_load_model_error() -> None:
    """Test that failure to load model raises ModelError."""
    detector = YOLODetector(model_name="invalid.pt")
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO.side_effect = RuntimeError("weights corrupted")

    with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
        with pytest.raises(ModelError, match="Failed to load detection model"):
            detector._get_model()


def test_yolo_detector_detect_mocked(tmp_path: Path) -> None:
    """Test YOLODetector detection pipeline with mocked YOLO inference."""
    # Create valid sample image
    image_file = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 100), color="white").save(image_file)

    detector = YOLODetector()

    # Mock ultralytics result object
    mock_box1 = MagicMock()
    mock_box1.cls = [0]
    mock_box1.conf = [0.9412]
    mock_box1.xyxy = [[124.2, 82.1, 318.0, 640.9]]

    mock_box2 = MagicMock()
    mock_box2.cls = [2]
    mock_box2.conf = [0.9100]
    mock_box2.xyxy = [[402.0, 275.0, 781.0, 512.0]]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    mock_result.names = {0: "person", 2: "car"}

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector._model = mock_model

    detections = detector.detect(image_file)

    assert len(detections) == 2
    d1 = detections[0]
    assert d1.class_name == "person"
    assert d1.class_id == 0
    assert pytest.approx(d1.confidence, 0.0001) == 0.9412
    assert d1.box == (124, 82, 318, 641)
    assert d1.image == str(image_file)

    d2 = detections[1]
    assert d2.class_name == "car"
    assert d2.class_id == 2
    assert pytest.approx(d2.confidence, 0.0001) == 0.9100
    assert d2.box == (402, 275, 781, 512)


def test_yolo_detector_detect_empty_results(tmp_path: Path) -> None:
    """Test YOLODetector when no boxes are detected."""
    image_file = tmp_path / "empty.jpg"
    Image.new("RGB", (50, 50), color="black").save(image_file)

    detector = YOLODetector()
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector._model = mock_model

    detections = detector.detect(image_file)
    assert detections == []


def test_yolo_detector_inference_failure(tmp_path: Path) -> None:
    """Test that model inference errors raise ModelError."""
    image_file = tmp_path / "test.jpg"
    Image.new("RGB", (50, 50), color="green").save(image_file)

    detector = YOLODetector()
    mock_model = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
    detector._model = mock_model

    with pytest.raises(ModelError, match="Inference failed"):
        detector.detect(image_file)
