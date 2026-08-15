"""Recursive camera-folder scanning. Yields paths; never loads pixels."""

from __future__ import annotations

from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_allowed_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def iter_image_files(folder: Path) -> list[Path]:
    """Return image paths under ``folder`` (recursive), sorted for resume stability."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {root}")

    found: list[Path] = []
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
            if not is_allowed_image(path):
                continue
            resolved = path.resolve()
            found.append(resolved)
        except OSError:
            continue
    found.sort(key=lambda item: str(item).lower())
    return found
