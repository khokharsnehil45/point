"""Unit tests for Point CLI argument parsing, commands, and error handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from point.cli import create_parser, run
from point.models import (
    Detection,
    ImageNotFoundError,
    InvalidImageError,
    ModelError,
)


def test_parser_required_image_flag() -> None:
    """Test that parser requires -i / --image flag."""
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_short_flags() -> None:
    """Test parser with short flags."""
    parser = create_parser()
    args = parser.parse_args(["-i", "street.jpg", "-show", "-load", "results"])
    assert args.image == "street.jpg"
    assert args.show is True
    assert args.load == "results"
    assert args.model == "yolo11n.pt"


def test_parser_long_flags() -> None:
    """Test parser with standard long flags and custom model."""
    parser = create_parser()
    args = parser.parse_args([
        "--image", "photo.png",
        "--show",
        "--load", "out.jsonl",
        "--model", "custom_weights.pt",
    ])
    assert args.image == "photo.png"
    assert args.show is True
    assert args.load == "out.jsonl"
    assert args.model == "custom_weights.pt"


def test_run_no_output_options(capsys: pytest.CaptureFixture[str]) -> None:
    """Test running Point without -show or -load prints concise summary."""
    mock_detections = [
        Detection("street.jpg", "person", 0, 0.94, (10, 10, 50, 50)),
        Detection("street.jpg", "car", 2, 0.91, (60, 60, 100, 100)),
    ]

    with patch("point.cli.YOLODetector.detect", return_value=mock_detections):
        code = run(["-i", "street.jpg"])
        assert code == 0

        captured = capsys.readouterr()
        assert "Detected 2 object(s)." in captured.out
        assert "Use '-show' to display results or '-load <filename>' to save them." in captured.out


def test_run_show_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Test running Point with -show prints formatted terminal results."""
    mock_detections = [
        Detection("street.jpg", "person", 0, 0.94, (124, 82, 318, 641)),
    ]

    with patch("point.cli.YOLODetector.detect", return_value=mock_detections):
        code = run(["-i", "street.jpg", "-show"])
        assert code == 0

        captured = capsys.readouterr()
        assert "POINT" in captured.out
        assert "Image            : street.jpg" in captured.out
        assert "Objects detected : 1" in captured.out
        assert "Class      : person" in captured.out
        assert "Box        : x1=124, y1=82, x2=318, y2=641" in captured.out


def test_run_load_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test running Point with -load saves JSONL file."""
    mock_detections = [
        Detection("street.jpg", "car", 2, 0.91, (402, 275, 781, 512)),
    ]
    out_file = tmp_path / "detections"

    with patch("point.cli.YOLODetector.detect", return_value=mock_detections):
        code = run(["-i", "street.jpg", "-load", str(out_file)])
        assert code == 0

        expected_jsonl = tmp_path / "detections.jsonl"
        assert expected_jsonl.is_file()

        lines = expected_jsonl.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["class"] == "car"
        assert data["box"] == [402, 275, 781, 512]

        captured = capsys.readouterr()
        assert f"Saved 1 detection(s) to {expected_jsonl}" in captured.out


def test_run_show_and_load_combined(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test combining both -show and -load."""
    mock_detections = [
        Detection("street.jpg", "bicycle", 1, 0.87, (50, 310, 290, 590)),
    ]
    out_file = tmp_path / "detections.jsonl"

    with patch("point.cli.YOLODetector.detect", return_value=mock_detections):
        code = run(["-i", "street.jpg", "-show", "-load", str(out_file)])
        assert code == 0

        # Should show terminal output
        captured = capsys.readouterr()
        assert "POINT" in captured.out
        assert "Class      : bicycle" in captured.out
        assert f"Saved 1 detection(s) to {out_file}" in captured.out

        # Should write JSONL
        assert out_file.is_file()


def test_run_image_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    """Test error handling when image does not exist."""
    code = run(["-i", "does-not-exist.jpg", "-show"])
    assert code == 1

    captured = capsys.readouterr()
    assert "Error: Image not found: does-not-exist.jpg" in captured.err


def test_run_invalid_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test error handling when image file is invalid."""
    bad_img = tmp_path / "corrupt.jpg"
    bad_img.write_bytes(b"invalid image bytes")

    code = run(["-i", str(bad_img), "-show"])
    assert code == 1

    captured = capsys.readouterr()
    assert f"Error: Could not read image: {bad_img}" in captured.err


def test_run_model_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test error handling when model fails."""
    good_img = tmp_path / "test.jpg"
    Image.new("RGB", (20, 20), color="white").save(good_img)

    with patch("point.cli.YOLODetector.detect", side_effect=ModelError("Model weight missing")):
        code = run(["-i", str(good_img), "-show"])
        assert code == 1

        captured = capsys.readouterr()
        assert "Error: Model weight missing" in captured.err


def test_run_visualize_default_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test -visualize without argument creates default _detected.jpg."""
    test_img = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 100), color="white").save(test_img)
    mock_detections = [
        Detection(str(test_img), "person", 0, 0.95, (10, 10, 80, 80)),
    ]

    with patch("point.cli.YOLODetector.detect", return_value=mock_detections):
        code = run(["-i", str(test_img), "-visualize"])
        assert code == 0

        expected_out = tmp_path / "photo_detected.jpg"
        assert expected_out.is_file()

        captured = capsys.readouterr()
        assert f"Saved annotated image to {expected_out}" in captured.out


def test_run_visualize_custom_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test -visualize with custom destination path."""
    test_img = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 100), color="white").save(test_img)
    custom_out = tmp_path / "custom" / "annotated.jpg"
    mock_detections = [
        Detection(str(test_img), "car", 2, 0.89, (20, 20, 90, 90)),
    ]

    with patch("point.cli.YOLODetector.detect", return_value=mock_detections):
        code = run(["-i", str(test_img), "-visualize", str(custom_out)])
        assert code == 0
        assert custom_out.is_file()

        captured = capsys.readouterr()
        assert f"Saved annotated image to {custom_out}" in captured.out


def test_run_visualize_short_flag_o(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test -o shorthand flag for visualization."""
    test_img = tmp_path / "photo.png"
    Image.new("RGB", (80, 80), color="black").save(test_img)
    out_file = tmp_path / "boxed.png"
    mock_detections = [
        Detection(str(test_img), "dog", 16, 0.91, (5, 5, 50, 50)),
    ]

    with patch("point.cli.YOLODetector.detect", return_value=mock_detections):
        code = run(["-i", str(test_img), "-o", str(out_file)])
        assert code == 0
        assert out_file.is_file()

        captured = capsys.readouterr()
        assert f"Saved annotated image to {out_file}" in captured.out


def test_run_model_catalog_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test point -model -catalog prints model catalog table."""
    code = run(["-model", "-catalog"])
    assert code == 0

    captured = capsys.readouterr()
    assert "POINT MODEL CATALOG" in captured.out
    assert "Current Active Model : yolo11n.pt" in captured.out
    assert "yolo11n.pt *" in captured.out
    assert "yolo11s.pt" in captured.out


def test_run_catalog_shorthand(capsys: pytest.CaptureFixture[str]) -> None:
    """Test point -catalog and point -models work directly without image."""
    code1 = run(["-catalog"])
    assert code1 == 0
    captured1 = capsys.readouterr()
    assert "POINT MODEL CATALOG" in captured1.out

    code2 = run(["-models"])
    assert code2 == 0
    captured2 = capsys.readouterr()
    assert "POINT MODEL CATALOG" in captured2.out


def test_run_custom_model_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    """Test point -m custom.pt -catalog reflects custom active model."""
    code = run(["-m", "custom.pt", "-catalog"])
    assert code == 0

    captured = capsys.readouterr()
    assert "Current Active Model : custom.pt" in captured.out


def test_run_set_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test setting persistent default model via -set-model."""
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr("point.config.CONFIG_FILE", cfg_file)
    monkeypatch.setattr("point.config.CONFIG_DIR", tmp_path)

    code = run(["-set-model", "yolo11m.pt"])
    assert code == 0

    captured = capsys.readouterr()
    assert "Default model successfully set to 'yolo11m.pt'" in captured.out
    assert cfg_file.is_file()
    assert "yolo11m.pt" in cfg_file.read_text()


def test_run_update_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test point -update triggers updater."""
    with patch("point.cli.run_update", return_value=0) as mock_updater:
        code = run(["-update"])
        assert code == 0
        mock_updater.assert_called_once()




