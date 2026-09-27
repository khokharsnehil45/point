"""Unit tests for Point visualization utilities."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from point.models import Detection
from point.visualization import draw_detections


def test_draw_detections(tmp_path: Path) -> None:
    """Test drawing bounding boxes onto an image."""
    img_path = tmp_path / "base.jpg"
    Image.new("RGB", (200, 200), color="blue").save(img_path)

    detections = [
        Detection(str(img_path), "person", 0, 0.95, (10, 10, 80, 80)),
        Detection(str(img_path), "car", 2, 0.88, (100, 100, 190, 190)),
    ]

    out_path = tmp_path / "annotated.jpg"
    img = draw_detections(img_path, detections, output_path=out_path)

    assert isinstance(img, Image.Image)
    assert out_path.is_file()
    assert img.size == (200, 200)


def test_get_batch_visualize_path(tmp_path: Path) -> None:
    """Test get_batch_visualize_path with default, directory, and file targets."""
    from point.visualization import get_batch_visualize_path

    img = Path("/data/photos/dog.png")

    # Default (no target)
    p_default = get_batch_visualize_path(img, None)
    assert p_default == Path("/data/photos/dog_detected.png")

    # Directory target
    out_dir = tmp_path / "out_folder"
    out_dir.mkdir()
    p_dir = get_batch_visualize_path(img, out_dir)
    assert p_dir == out_dir / "dog_detected.png"

    # Specific file target with index
    p_file = get_batch_visualize_path(img, tmp_path / "result.jpg", index=2)
    assert p_file == tmp_path / "result_3.jpg"
