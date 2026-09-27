"""Visualization helpers for image bounding boxes and label overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from point.models import Detection

# Palette of high-visibility colors for multi-class detection
COLOR_PALETTE = [
    "#00FF66",  # Neon Green
    "#00E5FF",  # Bright Cyan
    "#FF9100",  # Vivid Orange
    "#E040FB",  # Bright Purple
    "#FF5252",  # Bright Coral Red
    "#FFEA00",  # Vivid Yellow
    "#76FF03",  # Lime
    "#18FFFF",  # Aqua
]


def get_default_visualize_path(image_path: str | Path) -> Path:
    """Generate default annotated image filename (e.g. test_detected.jpg).

    Args:
        image_path: Input image path.

    Returns:
        Path with '_detected' suffix appended to the base stem.
    """
    p = Path(image_path)
    suffix = p.suffix if p.suffix else ".jpg"
    return p.with_name(f"{p.stem}_detected{suffix}")


def draw_detections(
    image_path: str | Path,
    detections: Sequence[Detection],
    output_path: str | Path | None = None,
    box_color: str | None = None,
    text_color: str = "#000000",
) -> Image.Image:
    """Draw bounding boxes and class labels onto an image.

    Args:
        image_path: Path to the base image.
        detections: Sequence of Detection objects to render.
        output_path: Optional path to save the annotated image.
        box_color: Optional single color for all boxes. If None, uses class-indexed palette.
        text_color: Color string for the label text inside the badge.

    Returns:
        Annotated PIL Image.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det.box

        # Choose color per class or fixed
        color = box_color or COLOR_PALETTE[det.class_id % len(COLOR_PALETTE)]

        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{det.class_name} {det.confidence:.2f}"

        # Text badge above bounding box (or inside top if near image boundary)
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label)
        badge_y1 = max(0, y1 - 18)
        badge_y2 = badge_y1 + (text_bbox[3] - text_bbox[1]) + 4

        draw.rectangle(
            [text_bbox[0] - 2, badge_y1, text_bbox[2] + 4, badge_y2],
            fill=color,
        )
        draw.text((x1 + 1, badge_y1 + 1), label, fill=text_color)

    if output_path:
        out = Path(output_path)
        if out.parent:
            out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)

    return img
