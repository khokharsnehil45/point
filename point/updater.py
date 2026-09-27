"""Self-update utilities for Point CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen
import json

from point.output import DIVIDER
from point.version import __version__

GITHUB_REPO = "khokharsnehil45/point"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"


def check_latest_commit() -> str | None:
    """Fetch the latest commit SHA from the GitHub main branch."""
    try:
        req = Request(API_URL, headers={"User-Agent": "point-cli-updater"})
        with urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("sha", "")[:7]
    except Exception:
        pass
    return None


def run_update() -> int:
    """Perform self-update of Point CLI.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    width = 60
    inner = width - 4
    div = "=" * width

    print(div)
    print(f"|{'POINT AUTO-UPDATE'.center(width - 2)}|")
    print(div)
    print(f"| {'Current Version : v' + __version__.ljust(inner - 19)} |")
    print(f"| {'Source Repo     : https://github.com/' + GITHUB_REPO.ljust(inner - 18)} |")
    print(div)

    # Check if running within a local git repository
    current_dir = Path(__file__).resolve().parent.parent
    git_dir = current_dir / ".git"

    if git_dir.is_dir():
        print(f"| {'Detected local repository checkout.'.ljust(inner)} |")
        print(f"| {'Pulling latest commits from git...'.ljust(inner)} |")
        try:
            pull_proc = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(current_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if pull_proc.returncode != 0:
                print(f"| {'Git pull output: ' + pull_proc.stderr.strip()[:inner - 17].ljust(inner - 17)} |")
            else:
                out_msg = pull_proc.stdout.strip().replace("\n", " ")
                print(f"| {out_msg[:inner].ljust(inner)} |")

            # Re-install package
            print(f"| {'Reinstalling package dependencies...'.ljust(inner)} |")
            install_proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=str(current_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if install_proc.returncode == 0:
                print(f"| {'Point updated successfully!'.ljust(inner)} |")
            else:
                print(f"| {'Notice: pip install exited with code ' + str(install_proc.returncode).ljust(inner - 38)} |")
        except Exception as exc:
            print(f"| {'Update failed: ' + str(exc)[:inner - 15].ljust(inner - 15)} |")
            print(div)
            return 1
    else:
        # Remote pip upgrade
        print(f"| {'Updating via remote git package...'.ljust(inner)} |")
        pkg_url = f"git+https://github.com/{GITHUB_REPO}.git"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", pkg_url],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                print(f"| {'Point updated to latest version!'.ljust(inner)} |")
            else:
                print(f"| {'Could not upgrade package automatically.'.ljust(inner)} |")
                print(f"| {'Run manually: pip install --upgrade ' + pkg_url[:inner - 36].ljust(inner - 36)} |")
        except Exception as exc:
            print(f"| {'Update check failed: ' + str(exc)[:inner - 22].ljust(inner - 22)} |")
            print(div)
            return 1

    print(div)
    return 0
