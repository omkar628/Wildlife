from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_jpeg(path: Path, color: tuple[int, int, int] = (40, 80, 40), size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    image.save(path, format="JPEG", quality=90)
    return path
