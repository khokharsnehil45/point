"""Output formatting and file persistence for Point CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from point.models import Detection

DIVIDER = "======================================================="


def normalize_jsonl_filename(filename: str | Path) -> Path:
    """Ensure a filename has a '.jsonl' extension without duplication.

    Args:
        filename: Destination filename or path string/Path.

    Returns:
        Path object guaranteed to end with '.jsonl'.
    """
    raw_str = str(filename).strip()
    if raw_str.endswith(".jsonl"):
        return Path(raw_str)
    return Path(f"{raw_str}.jsonl")


def serialize_jsonl(detections: Sequence[Detection]) -> str:
    """Serialize a sequence of detections into JSONL formatted text.

    Args:
        detections: Sequence of Detection objects.

    Returns:
        Newline-delimited JSON string where each line is one detection record.
    """
    if not detections:
        return ""
    lines = [json.dumps(d.to_dict()) for d in detections]
    return "\n".join(lines) + "\n"


def write_jsonl(detections: Sequence[Detection], filename: str | Path) -> Path:
    """Write detections to a JSONL file.

    Args:
        detections: Sequence of Detection objects.
        filename: Destination file name or path.

    Returns:
        The resolved Path of the written file.
    """
    target_path = normalize_jsonl_filename(filename)
    if target_path.parent:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as file:
        for det in detections:
            file.write(json.dumps(det.to_dict()) + "\n")

    return target_path


def format_terminal_output(image_name: str, detections: Sequence[Detection]) -> str:
    """Format detection results into a clean, human-readable report using == dividers and | pipes.

    Args:
        image_name: Identifier or path of the analyzed image.
        detections: Sequence of detected objects.

    Returns:
        Formatted string framed in pipes (|) with == horizontal dividers.
    """
    min_width = 55
    inner_lines: list[str] = [
        f"Image            : {image_name}",
        f"Objects detected : {len(detections)}",
    ]

    detection_lines: list[str] = []
    if not detections:
        detection_lines.append("No objects detected.")
    else:
        for idx, det in enumerate(detections, 1):
            if idx > 1:
                detection_lines.append("")
            coords = det.box
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            detection_lines.append(f"[{idx}]")
            detection_lines.append(f"Class      : {det.class_name}")
            detection_lines.append(f"Confidence : {det.confidence:.2f}")
            detection_lines.append(f"Box        : x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    all_content = inner_lines + [line for line in detection_lines if line]
    max_len = max(len(s) for s in all_content) if all_content else 0
    width = max(min_width, max_len + 4)
    inner_width = width - 4

    div = "=" * width
    output_lines: list[str] = [
        div,
        f"|{'POINT'.center(width - 2)}|",
        div,
        f"| {inner_lines[0].ljust(inner_width)} |",
        f"| {inner_lines[1].ljust(inner_width)} |",
        div,
    ]

    for line in detection_lines:
        if not line:
            output_lines.append(f"|{' ' * (width - 2)}|")
        else:
            output_lines.append(f"| {line.ljust(inner_width)} |")

    output_lines.append(div)
    return "\n".join(output_lines)


def format_batch_summary(total_images: int, total_detections: int, width: int = 55) -> str:
    """Format batch summary header into a pipe-bordered card.

    Args:
        total_images: Total number of processed images.
        total_detections: Total number of detected objects across all images.
        width: Total horizontal character width.

    Returns:
        Formatted summary card framed in pipes (|) and == dividers.
    """
    div = "=" * width
    inner_width = width - 4
    lines = [
        div,
        f"|{'POINT BATCH SUMMARY'.center(width - 2)}|",
        div,
        f"| {f'Total Images     : {total_images}'.ljust(inner_width)} |",
        f"| {f'Total Detections : {total_detections}'.ljust(inner_width)} |",
        div,
    ]
    return "\n".join(lines)


def format_batch_terminal_output(results: dict[str, Sequence[Detection]]) -> str:
    """Format batch detection results into human-readable terminal reports.

    Args:
        results: Dictionary mapping image names/paths to sequences of Detections.

    Returns:
        Combined formatted string with summary card and individual cards for each image.
    """
    total_images = len(results)
    total_detections = sum(len(dets) for dets in results.values())

    blocks: list[str] = [format_batch_summary(total_images, total_detections)]
    for image_name, dets in results.items():
        blocks.append(format_terminal_output(image_name, dets))

    return "\n\n".join(blocks)


MODEL_CATALOG = [
    ("yolo11n.pt", "2.6M", "Fastest", "Standard", "Real-time & Edge (Default)"),
    ("yolo11s.pt", "9.4M", "Fast", "Good", "Balanced Desktop"),
    ("yolo11m.pt", "20.1M", "Moderate", "High", "High Precision"),
    ("yolo11l.pt", "25.3M", "Slower", "Higher", "Heavy Workloads"),
    ("yolo11x.pt", "56.9M", "Slowest", "Highest", "Maximum Accuracy"),
]


def format_model_catalog(current_model: str = "yolo11n.pt") -> str:
    """Format available YOLO models and current active model into a pipe-bordered table.

    Args:
        current_model: The currently configured or active model name.

    Returns:
        Formatted catalog card framed in pipes (|) and == dividers.
    """
    width = 72
    inner = width - 4
    div = "=" * width
    subdiv = "|" + "-" * (width - 2) + "|"

    lines = [
        div,
        f"|{'POINT MODEL CATALOG'.center(width - 2)}|",
        div,
        f"| {'Current Active Model : ' + current_model.ljust(inner - 23)} |",
        f"| {'Architecture         : Ultralytics YOLO11'.ljust(inner)} |",
        f"| {'Task                 : Object Detection'.ljust(inner)} |",
        div,
        f"| {'MODEL'.ljust(14)}{'PARAMS'.ljust(10)}{'SPEED'.ljust(12)}{'BEST FOR'.ljust(inner - 36)} |",
        subdiv,
    ]

    for name, params, speed, _acc, desc in MODEL_CATALOG:
        star = " *" if name == current_model else "  "
        entry_name = (name + star).ljust(14)
        lines.append(f"| {entry_name}{params.ljust(10)}{speed.ljust(12)}{desc.ljust(inner - 36)} |")

    lines.append(div)
    lines.append(f"| {'* = Current active model'.ljust(inner)} |")
    lines.append(f"| {'Usage: point -i <image.jpg> -m <model.pt> -show'.ljust(inner)} |")
    lines.append(f"| {'Custom weights are also supported (e.g. -m custom.pt)'.ljust(inner)} |")
    lines.append(div)
    return "\n".join(lines)

