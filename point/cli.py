"""Command-line interface for Point."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from point.detector import DEFAULT_MODEL_NAME, YOLODetector
from point.models import (
    ImageNotFoundError,
    InvalidImageError,
    ModelError,
    PointError,
)
from point.config import get_default_model, set_default_model
from point.output import format_model_catalog, format_terminal_output, write_jsonl
from point.updater import run_update
from point.visualization import draw_detections, get_default_visualize_path
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
        help="Input image path.",
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

    try:
        detector = YOLODetector(model_name=active_model)
        detections = detector.detect(args.image)
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

    visualize_path = None
    if args.visualize_output:
        visualize_path = Path(args.visualize_output)
    elif args.visualize is not None:
        if args.visualize.strip():
            visualize_path = Path(args.visualize.strip())
        else:
            visualize_path = get_default_visualize_path(args.image)

    # Neither -show nor -load nor visualize specified: output concise summary
    if not args.show and not args.load and visualize_path is None:
        print(f"Detected {len(detections)} object(s).")
        print("Use '-show' to display results or '-load <filename>' to save them.")
        return 0

    if args.show:
        print(format_terminal_output(args.image, detections))

    if args.load:
        saved_path = write_jsonl(detections, args.load)
        print(f"Saved {len(detections)} detection(s) to {saved_path}")

    if visualize_path is not None:
        draw_detections(args.image, detections, output_path=visualize_path)
        print(f"Saved annotated image to {visualize_path}")

    return 0


def main() -> None:
    """Entry point for the console script."""
    sys.exit(run())


if __name__ == "__main__":
    main()
