"""Generate a pixel-perfect, presentation-grade showcase screenshot for LinkedIn."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from point.output import format_model_catalog, format_terminal_output
from point.models import Detection


def create_showcase_image(output_path: Path) -> None:
    width, height = 2400, 1350
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)

    # Background subtle dark gradient
    for y in range(height):
        ratio = y / height
        r = int(13 * (1 - ratio) + 8 * ratio)
        g = int(21 * (1 - ratio) + 12 * ratio)
        b = int(38 * (1 - ratio) + 22 * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # Ambient glow accents
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse([200, -100, 1000, 600], fill=(0, 229, 255, 30))
    glow_draw.ellipse([1400, 600, 2200, 1400], fill=(99, 102, 241, 30))
    glow = glow.filter(ImageFilter.GaussianBlur(80))
    img.alpha_composite(glow)
    draw = ImageDraw.Draw(img)

    # Load fonts
    font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    font_regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font_sans_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    title_font = ImageFont.truetype(font_sans_path, 54)
    subtitle_font = ImageFont.truetype(font_sans_path, 26)
    mono_bold = ImageFont.truetype(font_bold_path, 18)
    mono_regular = ImageFont.truetype(font_regular_path, 18)
    badge_font = ImageFont.truetype(font_sans_path, 18)
    install_font = ImageFont.truetype(font_bold_path, 22)

    # Header Bar
    draw.text((100, 60), "POINT", font=title_font, fill=(0, 255, 170, 255))
    draw.text((310, 78), "|  Production-Grade Object Detection CLI", font=subtitle_font, fill=(226, 232, 240, 255))
    draw.text((1700, 80), "github.com/khokharsnehil45/point", font=subtitle_font, fill=(148, 163, 184, 255))

    # Left Window: Terminal
    term_x, term_y = 100, 160
    term_w, term_h = 1260, 1050
    term_radius = 20

    # Terminal shadow & background
    draw.rounded_rectangle([term_x, term_y, term_x + term_w, term_y + term_h], radius=term_radius, fill=(13, 17, 23, 245), outline=(51, 65, 85, 255), width=2)

    # Terminal Title Bar
    draw.rounded_rectangle([term_x, term_y, term_x + term_w, term_y + 55], radius=term_radius, fill=(22, 27, 34, 255))
    draw.rectangle([term_x, term_y + 35, term_x + term_w, term_y + 55], fill=(22, 27, 34, 255))
    draw.line([(term_x, term_y + 55), (term_x + term_w, term_y + 55)], fill=(48, 54, 61, 255), width=1)

    # Traffic Light Buttons
    draw.ellipse([term_x + 25, term_y + 20, term_x + 41, term_y + 36], fill=(255, 95, 86, 255))
    draw.ellipse([term_x + 55, term_y + 20, term_x + 71, term_y + 36], fill=(255, 189, 46, 255))
    draw.ellipse([term_x + 85, term_y + 20, term_x + 101, term_y + 36], fill=(39, 201, 63, 255))

    draw.text((term_x + term_w // 2 - 140, term_y + 16), "bash — point CLI", font=mono_bold, fill=(139, 148, 158, 255))

    # Terminal Body Text
    tx = term_x + 45
    ty = term_y + 75
    line_h = 26

    def draw_prompt(cmd: str):
        nonlocal ty
        draw.text((tx, ty), "user@linux", font=mono_bold, fill=(56, 189, 248, 255))
        draw.text((tx + 120, ty), ":~$", font=mono_bold, fill=(148, 163, 184, 255))
        draw.text((tx + 165, ty), cmd, font=mono_bold, fill=(248, 250, 252, 255))
        ty += line_h + 8

    # Command 1: Detection
    draw_prompt("point -i test.jpg -show --visualize")

    w = 70
    inner = w - 4
    div = "=" * w

    card_lines = [
        div,
        f"|{'POINT'.center(w - 2)}|",
        div,
        f"| {'Image            : test.jpg'.ljust(inner)} |",
        f"| {'Objects detected : 1'.ljust(inner)} |",
        div,
        f"| {'[1]'.ljust(inner)} |",
        f"| {'Class      : car'.ljust(inner)} |",
        f"| {'Confidence : 0.83'.ljust(inner)} |",
        f"| {'Box        : x1=49, y1=163, x2=588, y2=375'.ljust(inner)} |",
        div,
    ]

    for line in card_lines:
        if line.startswith("="):
            draw.text((tx, ty), line, font=mono_regular, fill=(0, 229, 255, 255))
        elif "POINT" in line:
            draw.text((tx, ty), line, font=mono_bold, fill=(0, 255, 170, 255))
        elif "Class" in line or "Objects detected" in line:
            draw.text((tx, ty), line, font=mono_regular, fill=(255, 224, 102, 255))
        elif "Confidence" in line:
            draw.text((tx, ty), line, font=mono_regular, fill=(129, 140, 248, 255))
        elif "Box" in line:
            draw.text((tx, ty), line, font=mono_regular, fill=(74, 222, 128, 255))
        else:
            draw.text((tx, ty), line, font=mono_regular, fill=(226, 232, 240, 255))
        ty += line_h

    ty += 6
    draw.text((tx, ty), "Saved annotated image to test_detected.jpg", font=mono_bold, fill=(52, 211, 153, 255))
    ty += line_h + 14

    # Command 2: Catalog
    draw_prompt("point -model -catalog")

    real_catalog = format_model_catalog("yolo11n.pt")

    for line in real_catalog.split("\n"):
        if line.startswith("="):
            draw.text((tx, ty), line, font=mono_regular, fill=(99, 102, 241, 255))
        elif "MODEL CATALOG" in line:
            draw.text((tx, ty), line, font=mono_bold, fill=(167, 139, 250, 255))
        elif "yolo11n.pt *" in line:
            draw.text((tx, ty), line, font=mono_bold, fill=(52, 211, 153, 255))
        else:
            draw.text((tx, ty), line, font=mono_regular, fill=(203, 213, 225, 255))
        ty += line_h - 2

    # Right Panel: Visual Overlay Preview Card
    right_x, right_y = 1420, 160
    right_w, right_h = 880, 1050

    draw.rounded_rectangle([right_x, right_y, right_x + right_w, right_y + right_h], radius=term_radius, fill=(13, 17, 23, 245), outline=(51, 65, 85, 255), width=2)

    # Card Header
    draw.rounded_rectangle([right_x, right_y, right_x + right_w, right_y + 55], radius=term_radius, fill=(22, 27, 34, 255))
    draw.rectangle([right_x, right_y + 35, right_x + right_w, right_y + 55], fill=(22, 27, 34, 255))
    draw.line([(right_x, right_y + 55), (right_x + right_w, right_y + 55)], fill=(48, 54, 61, 255), width=1)
    draw.text((right_x + 35, right_y + 16), "Visual Detection Output (--visualize)", font=mono_bold, fill=(0, 255, 170, 255))

    # Embed annotated image
    detected_path = Path("/home/kevin/TBuilds/point/test_detected.jpg")
    if detected_path.is_file():
        ann_img = Image.open(detected_path).convert("RGB")
        target_w = right_w - 70
        scale = target_w / ann_img.width
        target_h = int(ann_img.height * scale)
        ann_resized = ann_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        img.paste(ann_resized, (right_x + 35, right_y + 85))
        # Draw frame around image
        draw.rectangle([right_x + 34, right_y + 84, right_x + 36 + target_w, right_y + 86 + target_h], outline=(51, 65, 85, 255), width=2)
        bottom_y = right_y + 85 + target_h + 35
    else:
        bottom_y = right_y + 600

    # JSONL Pipeline Block
    json_box_y = bottom_y
    draw.rounded_rectangle([right_x + 35, json_box_y, right_x + right_w - 35, json_box_y + 115], radius=12, fill=(15, 23, 42, 255), outline=(56, 189, 248, 180), width=1)
    draw.text((right_x + 55, json_box_y + 18), "JSONL Streaming Output Contract (-load):", font=mono_bold, fill=(56, 189, 248, 255))
    json_font = ImageFont.truetype(font_regular_path, 16)
    json_sample = '{"image": "test.jpg", "class": "car", "confidence": 0.83, "box": [49, 163, 588, 375]}'
    draw.text((right_x + 55, json_box_y + 58), json_sample, font=json_font, fill=(226, 232, 240, 255))

    # Feature Badges
    badges_y = json_box_y + 145
    badge_items = [
        ("• Ultralytics YOLO11", (30, 41, 59, 255), (0, 229, 255, 255)),
        ("• One-Line curl Install", (30, 41, 59, 255), (74, 222, 128, 255)),
        ("• Auto-Update (point -update)", (30, 41, 59, 255), (250, 204, 21, 255)),
        ("• 39 Tests (0.15s Mocked)", (30, 41, 59, 255), (192, 132, 252, 255)),
    ]

    bx = right_x + 35
    by = badges_y
    for text, bg_col, text_col in badge_items:
        draw.rounded_rectangle([bx, by, bx + 380, by + 46], radius=10, fill=bg_col, outline=(71, 85, 105, 255), width=1)
        draw.text((bx + 20, by + 12), text, font=badge_font, fill=text_col)
        bx += 410
        if bx > right_x + right_w - 350:
            bx = right_x + 35
            by += 60

    # Bottom Banner: One-Line Install Command
    install_y = 1240
    draw.rounded_rectangle([100, install_y, width - 100, install_y + 65], radius=16, fill=(15, 23, 42, 255), outline=(0, 255, 170, 200), width=2)
    draw.text((130, install_y + 18), "Install with one command :", font=install_font, fill=(0, 255, 170, 255))
    install_cmd = "curl -sSL https://raw.githubusercontent.com/khokharsnehil45/point/main/install.sh | bash"
    draw.text((490, install_y + 18), install_cmd, font=install_font, fill=(248, 250, 252, 255))

    img.save(output_path, "PNG")
    print(f"Refined screenshot successfully saved to: {output_path}")


if __name__ == "__main__":
    out = Path("/home/kevin/TBuilds/point/point_showcase.png")
    create_showcase_image(out)
