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
