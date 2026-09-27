"""Command-line interface for Point."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from point.detector import DEFAULT_MODEL_NAME, YOLODetector, collect_image_paths
from point.models import (
    ImageNotFoundError,
    InvalidImageError,
    ModelError,
    PointError,
)
from point.config import get_default_model, set_default_model
from point.output import (
    format_batch_terminal_output,
    format_model_catalog,
    format_terminal_output,
    write_jsonl,
)
from point.updater import run_update
from point.visualization import (
    draw_detections,
    get_batch_visualize_path,
    get_default_visualize_path,
)
from pathlib import Path
from typing import Any


class PointArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser requiring -i/--image unless viewing or setting model or updating."""

    def parse_args(
        self, args: Sequence[str] | None = None, namespace: Any = None
    ) -> argparse.Namespace:
        raw_args = list(args) if args is not None else sys.argv[1:]
        parsed = super().parse_args(args, namespace)

        # If user requested model/catalog/setting/update without an image
        if not parsed.image:
            if (
                getattr(parsed, "catalog", False)
                or getattr(parsed, "set_model", None)
                or getattr(parsed, "update", False)
                or "-update" in raw_args
                or "--update" in raw_args
                or "-model" in raw_args
                or "--models" in raw_args
                or "-models" in raw_args
                or "-catalog" in raw_args
                or "--catalog" in raw_args
                or "-set-model" in raw_args
                or "--set-model" in raw_args
            ):
                if not getattr(parsed, "set_model", None) and not getattr(parsed, "update", False):
                    parsed.catalog = True
                return parsed
            self.error("the following arguments are required: -i/--image")
        return parsed


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for Point."""
    parser = PointArgumentParser(
        prog="point",
        description="Point: Lightweight command-line object detection tool.",
    )
    parser.add_argument(
        "-i",
        "--image",
        required=False,
        nargs="+",
        help="Input image path(s), directory, or glob pattern.",
    )
    parser.add_argument(
        "-show",
        "--show",
        action="store_true",
        help="Print detection results to the terminal.",
    )
    parser.add_argument(
        "-load",
        "--load",
        metavar="FILENAME",
        help="Save detection results to a JSONL file (e.g. -load detections).",
    )
    parser.add_argument(
        "-visualize",
        "--visualize",
        nargs="?",
        const="",
        metavar="OUT_PATH",
        help="Save an annotated image with drawn bounding boxes and labels (e.g. -visualize or -visualize out.jpg).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="visualize_output",
        metavar="OUT_PATH",
        help="Custom output image path for visualization (shorthand for -visualize OUT_PATH).",
    )
    parser.add_argument(
        "-model",
        "-m",
        "--model",
        default=get_default_model(),
        nargs="?",
        const=get_default_model(),
        help=f"Detection model name or weight path (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "-set-model",
        "--set-model",
        metavar="MODEL_NAME",
        help="Set a persistent default model (e.g. -set-model yolo11s.pt).",
    )
    parser.add_argument(
        "-catalog",
        "--catalog",
        "-models",
        "--models",
        dest="catalog",
        action="store_true",
        help="View available model catalog and current active model information.",
    )
    parser.add_argument(
        "-update",
        "--update",
        dest="update",
        action="store_true",
        help="Update Point CLI to the latest version from GitHub.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="point 0.1.0",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Execute the Point CLI with the given command line arguments.

    Args:
        argv: Optional argument sequence (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if getattr(args, "update", False):
        return run_update()

    if getattr(args, "set_model", None):
        cfg_file = set_default_model(args.set_model)
        print(f"Default model successfully set to '{args.set_model}' (saved to {cfg_file}).")
        return 0

    active_model = args.model or get_default_model()

    if getattr(args, "catalog", False):
        print(format_model_catalog(current_model=active_model))
        return 0

    raw_images = args.image if isinstance(args.image, (list, tuple)) else [args.image]

    # Check if single item that is not a directory and has no glob characters
    is_simple_single = (
        len(raw_images) == 1
        and not Path(raw_images[0]).is_dir()
        and not any(c in str(raw_images[0]) for c in ("*", "?", "["))
    )

    try:
        detector = YOLODetector(model_name=active_model)
        if is_simple_single:
            single_image = raw_images[0]
            detections = detector.detect(single_image)
            all_results = {str(single_image): detections}
            image_paths = [Path(single_image)]
        else:
            image_paths = collect_image_paths(raw_images)
            if len(image_paths) == 1:
                detections = detector.detect(image_paths[0])
                all_results = {str(image_paths[0]): detections}
            else:
                all_results = detector.detect_batch(image_paths)
    except ImageNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except InvalidImageError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except ModelError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except PointError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Error: Unexpected error: {exc}\n")
        return 1

    visualize_requested = (args.visualize is not None) or bool(args.visualize_output)
    visualize_target: str | None = None
    if args.visualize_output:
        visualize_target = args.visualize_output
    elif args.visualize is not None and args.visualize.strip():
        visualize_target = args.visualize.strip()

    is_batch = len(image_paths) > 1
    total_detections = sum(len(dets) for dets in all_results.values())
    all_detections = [det for dets in all_results.values() for det in dets]

    # Neither -show nor -load nor visualize specified: output concise summary
    if not args.show and not args.load and not visualize_requested:
        if is_batch:
            print(f"Processed {len(image_paths)} image(s), detected {total_detections} object(s) in total.")
        else:
            print(f"Detected {total_detections} object(s).")
        print("Use '-show' to display results or '-load <filename>' to save them.")
        return 0

    if args.show:
        if is_batch:
            print(format_batch_terminal_output(all_results))
        else:
            single_key = str(image_paths[0])
            print(format_terminal_output(single_key, all_results[single_key]))

    if args.load:
        saved_path = write_jsonl(all_detections, args.load)
        if is_batch:
            print(f"Saved {total_detections} detection(s) from {len(image_paths)} image(s) to {saved_path}")
        else:
            print(f"Saved {total_detections} detection(s) to {saved_path}")

    if visualize_requested:
        if is_batch:
            saved_count = 0
            for idx, p in enumerate(image_paths):
                out_path = get_batch_visualize_path(p, visualize_target, index=idx)
                draw_detections(p, all_results[str(p)], output_path=out_path)
                saved_count += 1
            if visualize_target and (
                Path(visualize_target).is_dir()
                or str(visualize_target).endswith("/")
                or Path(visualize_target).suffix.lower() not in (
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"
                )
            ):
                print(f"Saved {saved_count} annotated image(s) to {visualize_target}")
            else:
                print(f"Saved {saved_count} annotated image(s)")
        else:
            single_p = image_paths[0]
            out_path = Path(visualize_target) if visualize_target else get_default_visualize_path(single_p)
            draw_detections(single_p, all_results[str(single_p)], output_path=out_path)
            print(f"Saved annotated image to {out_path}")

    return 0


def main() -> None:
    """Entry point for the console script."""
    sys.exit(run())


if __name__ == "__main__":
    main()
