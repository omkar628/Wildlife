"""Save tiger crops next to detections. Never overwrite another crop."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from backend.detector.parser import ParsedDetection


def padded_box(
    detection: ParsedDetection,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    pad_x = detection.bbox_width * padding
    pad_y = detection.bbox_height * padding
    left = max(0, int(detection.bbox_x - pad_x))
    top = max(0, int(detection.bbox_y - pad_y))
    right = min(image_width, int(detection.bbox_x + detection.bbox_width + pad_x))
    bottom = min(image_height, int(detection.bbox_y + detection.bbox_height + pad_y))
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def crop_destination(crops_dir: Path, image_id: int, detection_id: int) -> Path:
    folder = Path(crops_dir) / f"image_{image_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"detection_{detection_id}.jpg"


def save_tiger_crop(
    source_path: Path,
    destination: Path,
    detection: ParsedDetection,
    *,
    padding: float,
    jpeg_quality: int,
) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        box = padded_box(detection, image.width, image.height, padding)
        cropped = image.crop(box)
        cropped.save(destination, format="JPEG", quality=jpeg_quality)
    return destination
