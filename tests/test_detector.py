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


def test_collect_image_paths_directory(tmp_path: Path) -> None:
    """Test collecting image paths from a flat directory."""
    from point.detector import collect_image_paths

    dir_path = tmp_path / "gallery"
    dir_path.mkdir()
    f1 = dir_path / "img1.jpg"
    f2 = dir_path / "img2.png"
    txt = dir_path / "notes.txt"
    Image.new("RGB", (10, 10)).save(f1)
    Image.new("RGB", (10, 10)).save(f2)
    txt.write_text("not an image")

    paths = collect_image_paths(dir_path)
    assert len(paths) == 2
    assert f1 in paths
    assert f2 in paths
    assert txt not in paths


def test_collect_image_paths_recursive(tmp_path: Path) -> None:
    """Test collecting image paths from nested directory structures."""
    from point.detector import collect_image_paths

    dir_path = tmp_path / "dataset"
    sub_dir = dir_path / "val"
    sub_dir.mkdir(parents=True)
    f1 = sub_dir / "val1.webp"
    Image.new("RGB", (10, 10)).save(f1)

    paths = collect_image_paths(dir_path)
    assert len(paths) == 1
    assert paths[0] == f1


def test_collect_image_paths_glob(tmp_path: Path) -> None:
    """Test collecting image paths with glob wildcards."""
    from point.detector import collect_image_paths

    f1 = tmp_path / "flower1.jpg"
    f2 = tmp_path / "flower2.jpg"
    f3 = tmp_path / "tree.png"
    Image.new("RGB", (10, 10)).save(f1)
    Image.new("RGB", (10, 10)).save(f2)
    Image.new("RGB", (10, 10)).save(f3)

    pattern = str(tmp_path / "flower*.jpg")
    paths = collect_image_paths(pattern)
    assert len(paths) == 2
    assert f1 in paths
    assert f2 in paths
    assert f3 not in paths


def test_collect_image_paths_errors(tmp_path: Path) -> None:
    """Test error conditions for collect_image_paths."""
    from point.detector import collect_image_paths

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ImageNotFoundError, match="No valid images found in directory"):
        collect_image_paths(empty_dir)

    with pytest.raises(ImageNotFoundError, match="No images matched pattern"):
        collect_image_paths(str(tmp_path / "nonexistent_*.jpg"))

    with pytest.raises(ImageNotFoundError, match="Image not found"):
        collect_image_paths(tmp_path / "missing.jpg")


def test_yolo_detector_detect_batch_mocked(tmp_path: Path) -> None:
    """Test YOLODetector.detect_batch with multiple images."""
    f1 = tmp_path / "b1.jpg"
    f2 = tmp_path / "b2.png"
    Image.new("RGB", (20, 20)).save(f1)
    Image.new("RGB", (20, 20)).save(f2)

    detector = YOLODetector()

    # Empty inputs
    assert detector.detect_batch([]) == {}

    # Mock ultralytics batch results
    mock_box1 = MagicMock()
    mock_box1.cls = [0]
    mock_box1.conf = [0.95]
    mock_box1.xyxy = [[10.0, 10.0, 50.0, 50.0]]

    r1 = MagicMock()
    r1.boxes = [mock_box1]
    r1.names = {0: "person"}

    r2 = MagicMock()
    r2.boxes = []
    r2.names = {}

    mock_model = MagicMock()
    mock_model.return_value = [r1, r2]
    detector._model = mock_model

    batch_res = detector.detect_batch([f1, f2], batch_size=2)
    assert len(batch_res) == 2
    assert len(batch_res[str(f1)]) == 1
    assert batch_res[str(f1)][0].class_name == "person"
    assert len(batch_res[str(f2)]) == 0

