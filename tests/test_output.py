"""Unit tests for Point output formatting, JSONL serialization, and models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from point.models import Detection
from point.output import (
    DIVIDER,
    format_terminal_output,
    normalize_jsonl_filename,
    serialize_jsonl,
    write_jsonl,
)


def test_detection_creation() -> None:
    """Test creating a Detection dataclass instance."""
    det = Detection(
        image="street.jpg",
        class_name="person",
        class_id=0,
        confidence=0.9412,
        box=(124, 82, 318, 641),
    )
    assert det.image == "street.jpg"
    assert det.class_name == "person"
    assert det.class_id == 0
    assert det.confidence == 0.9412
    assert det.box == (124, 82, 318, 641)


def test_detection_to_dict() -> None:
    """Test converting Detection to dictionary format matching JSONL schema."""
    det = Detection(
        image="street.jpg",
        class_name="car",
        class_id=2,
        confidence=0.91234,
        box=(402, 275, 781, 512),
    )
    d = det.to_dict()
    assert d == {
        "image": "street.jpg",
        "class": "car",
        "class_id": 2,
        "confidence": 0.9123,
        "box": [402, 275, 781, 512],
    }


def test_normalize_jsonl_filename_missing_extension() -> None:
    """Test that missing .jsonl extension is automatically appended."""
    assert normalize_jsonl_filename("detections") == Path("detections.jsonl")
    assert normalize_jsonl_filename("results") == Path("results.jsonl")
    assert normalize_jsonl_filename("dir/my_output") == Path("dir/my_output.jsonl")


def test_normalize_jsonl_filename_existing_extension_not_duplicated() -> None:
    """Test that existing .jsonl extension is not duplicated."""
    assert normalize_jsonl_filename("detections.jsonl") == Path("detections.jsonl")
    assert normalize_jsonl_filename("results.jsonl") == Path("results.jsonl")
    assert normalize_jsonl_filename("dir/out.jsonl") == Path("dir/out.jsonl")


def test_serialize_jsonl_empty() -> None:
    """Test serializing an empty list of detections produces an empty string."""
    assert serialize_jsonl([]) == ""


def test_serialize_jsonl_multiple_detections() -> None:
    """Test serializing multiple detections produces one valid JSON object per line."""
    detections = [
        Detection("street.jpg", "person", 0, 0.94, (124, 82, 318, 641)),
        Detection("street.jpg", "car", 2, 0.91, (402, 275, 781, 512)),
    ]
    raw = serialize_jsonl(detections)
    lines = raw.strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["image"] == "street.jpg"
    assert first["class"] == "person"
    assert first["class_id"] == 0
    assert first["confidence"] == 0.94
    assert first["box"] == [124, 82, 318, 641]

    second = json.loads(lines[1])
    assert second["image"] == "street.jpg"
    assert second["class"] == "car"
    assert second["class_id"] == 2
    assert second["confidence"] == 0.91
    assert second["box"] == [402, 275, 781, 512]


def test_write_jsonl(tmp_path: Path) -> None:
    """Test writing detections to disk in JSONL format."""
    detections = [
        Detection("test.jpg", "bicycle", 1, 0.87, (50, 310, 290, 590)),
    ]
    out_target = tmp_path / "sub" / "output_test"
    written_path = write_jsonl(detections, out_target)

    assert written_path == tmp_path / "sub" / "output_test.jsonl"
    assert written_path.is_file()

    content = written_path.read_text(encoding="utf-8")
    parsed = json.loads(content.strip())
    assert parsed["class"] == "bicycle"
    assert parsed["confidence"] == 0.87
    assert parsed["box"] == [50, 310, 290, 590]


def test_format_terminal_output() -> None:
    """Test terminal formatting matches Point design specification."""
    detections = [
        Detection("street.jpg", "person", 0, 0.9421, (124, 82, 318, 641)),
        Detection("street.jpg", "traffic light", 9, 0.83, (820, 91, 864, 182)),
    ]
    formatted = format_terminal_output("street.jpg", detections)

    assert "POINT" in formatted
    assert DIVIDER in formatted
    assert "Image            : street.jpg" in formatted
    assert "Objects detected : 2" in formatted
    assert "[1]" in formatted
    assert "Class      : person" in formatted
    assert "Confidence : 0.94" in formatted
    assert "Box        : x1=124, y1=82, x2=318, y2=641" in formatted
    assert "[2]" in formatted
    assert "Class      : traffic light" in formatted
    assert "Confidence : 0.83" in formatted
    assert "Box        : x1=820, y1=91, x2=864, y2=182" in formatted


def test_format_terminal_output_empty() -> None:
    """Test terminal formatting when no objects are detected."""
    formatted = format_terminal_output("empty.jpg", [])
    assert "Image            : empty.jpg" in formatted
    assert "Objects detected : 0" in formatted
    assert "No objects detected." in formatted


def test_format_model_catalog() -> None:
    """Test format_model_catalog produces clean pipe-delimited table with current model."""
    from point.output import format_model_catalog

    catalog_str = format_model_catalog("yolo11n.pt")
    assert "POINT MODEL CATALOG" in catalog_str
    assert "Current Active Model : yolo11n.pt" in catalog_str
    assert "yolo11n.pt *" in catalog_str
    assert "yolo11s.pt" in catalog_str
    assert "yolo11m.pt" in catalog_str
    assert "yolo11l.pt" in catalog_str
    assert "yolo11x.pt" in catalog_str
    assert "MODEL" in catalog_str
    assert "PARAMS" in catalog_str
    assert "SPEED" in catalog_str


def test_format_batch_summary() -> None:
    """Test format_batch_summary outputs pipe-bordered header card."""
    from point.output import format_batch_summary

    card = format_batch_summary(total_images=5, total_detections=18)
    assert "POINT BATCH SUMMARY" in card
    assert "Total Images     : 5" in card
    assert "Total Detections : 18" in card


def test_format_batch_terminal_output() -> None:
    """Test format_batch_terminal_output combines batch header with image cards."""
    from point.output import format_batch_terminal_output

    results = {
        "pic1.jpg": [Detection("pic1.jpg", "dog", 16, 0.92, (1, 1, 10, 10))],
        "pic2.jpg": [],
    }
    output = format_batch_terminal_output(results)
    assert "POINT BATCH SUMMARY" in output
    assert "Total Images     : 2" in output
    assert "Total Detections : 1" in output
    assert "Image            : pic1.jpg" in output
    assert "Image            : pic2.jpg" in output
    assert "No objects detected." in output
    assert "Class      : dog" in output


