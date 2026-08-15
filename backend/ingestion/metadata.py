"""Read image size and capture time. Prefer EXIF; fall back to filesystem."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags, UnidentifiedImageError


EXIF_DATETIME = 306
EXIF_IFD_POINTER = 0x8769
DATETIME_ORIGINAL = 36867
DATETIME_DIGITIZED = 36868


def _parse_exif_datetime(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                return parsed.isoformat()
            return parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def _filesystem_mtime(path: Path) -> str | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def extract_image_metadata(path: Path) -> dict[str, Any]:
    """Return width/height and a timestamp with an explicit source.

    ``timestamp_source`` is one of: ``exif``, ``filesystem``, ``unknown``.
    Filesystem time is a fallback only — it is not treated as capture time.
    """
    result: dict[str, Any] = {
        "width": None,
        "height": None,
        "timestamp": None,
        "timestamp_source": "unknown",
    }
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            result["width"] = int(image.width)
            result["height"] = int(image.height)
            exif = image.getexif()
            if exif:
                candidates = [exif.get(EXIF_DATETIME)]
                try:
                    ifd = exif.get_ifd(EXIF_IFD_POINTER)
                except Exception:
                    ifd = {}
                if ifd:
                    candidates.insert(0, ifd.get(DATETIME_ORIGINAL))
                    candidates.insert(1, ifd.get(DATETIME_DIGITIZED))
                # Named lookup as a last EXIF attempt (Pillow version differences).
                named = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
                candidates.extend(
                    [
                        named.get("DateTimeOriginal"),
                        named.get("DateTimeDigitized"),
                        named.get("DateTime"),
                    ]
                )
                for candidate in candidates:
                    parsed = _parse_exif_datetime(candidate)
                    if parsed:
                        result["timestamp"] = parsed
                        result["timestamp_source"] = "exif"
                        break
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Unreadable image: {image_path.name}: {exc}") from exc

    if result["timestamp"] is None:
        fallback = _filesystem_mtime(image_path)
        if fallback:
            result["timestamp"] = fallback
            result["timestamp_source"] = "filesystem"
    return result
