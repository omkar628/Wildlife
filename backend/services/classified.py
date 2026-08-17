"""Copy processed detections into data/classified without touching originals.

Directory layout:

    data/classified/
      tiger/T001/...
      tiger/unidentified/...
      prey/...
      rival/...
      human/...
      blank/...

Copies are JPEG files named:

    {animal_or_tiger_id}_{camera_id}_{timestamp}_{source_id}.jpg
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image

from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import DetectionRepository, ObservationRepository

logger = logging.getLogger(__name__)

CLASS_FOLDERS = ("tiger", "prey", "rival", "human", "blank")
UNIDENTIFIED = "unidentified"


def sanitize_token(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    cleaned = re.sub(r"[^\w.\-]+", "_", text)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def format_stamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_time"
    compact = (
        text.replace("T", "_")
        .replace(" ", "_")
        .replace("-", "")
        .replace(":", "")
        .replace("Z", "")
    )
    compact = re.sub(r"[^0-9_]", "", compact)
    compact = compact.strip("_")
    if "+" in text:
        compact = compact.split("+", 1)[0]
    return compact[:15] if compact else "unknown_time"


def classified_filename(
    prefix: str,
    camera_id: Any,
    timestamp: Any,
    source_id: int,
) -> str:
    return (
        f"{sanitize_token(prefix)}_{sanitize_token(camera_id, 'Camera')}_"
        f"{format_stamp(timestamp)}_{int(source_id)}.jpg"
    )


def classified_dir(root: Path, class_name: str, tiger_id: str | None = None) -> Path:
    name = (class_name or "blank").strip().lower()
    if name not in CLASS_FOLDERS:
        name = "blank"
    if name == "tiger":
        folder = sanitize_token(tiger_id, UNIDENTIFIED) if tiger_id else UNIDENTIFIED
        return Path(root) / "tiger" / folder
    return Path(root) / name


def write_jpeg_copy(source: Path, destination: Path, *, jpeg_quality: int = 95) -> Path:
    """Write a JPEG copy. Never modify or replace the source file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=jpeg_quality)
    return destination


class ClassifiedStore:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.root = Path(settings.classified_dir)
        self.detections = DetectionRepository(db)
        self.observations = ObservationRepository(db)

    def store_detection(
        self,
        *,
        source_path: str | Path,
        class_name: str,
        camera_id: str | None,
        timestamp: str | None,
        detection_id: int,
        observation_id: int | None = None,
        tiger_id: str | None = None,
        image_id: int | None = None,
    ) -> Path | None:
        if not source_path:
            return None
        source = Path(source_path)
        if not source.is_file():
            logger.warning("Classified copy skipped; original missing: %s", source)
            return None
        name = (class_name or "").strip().lower()
        if name not in {"tiger", "prey", "rival", "human"}:
            name = "blank"
        prefix = tiger_id if (name == "tiger" and tiger_id) else name
        source_id = observation_id if observation_id is not None else detection_id
        dest_dir = classified_dir(self.root, name, tiger_id)
        destination = dest_dir / classified_filename(prefix, camera_id, timestamp, source_id)
        try:
            saved = write_jpeg_copy(
                source,
                destination,
                jpeg_quality=self.settings.crop_jpeg_quality,
            )
        except Exception:
            logger.exception("Failed to copy classified image for detection %s", detection_id)
            return None
        path_text = str(saved)
        self.detections.set_classified_path(detection_id, path_text)
        if observation_id is not None:
            self.observations.set_classified_path(observation_id, path_text)
        return saved

    def store_blank(
        self,
        *,
        source_path: str | Path,
        camera_id: str | None,
        timestamp: str | None,
        image_id: int,
    ) -> Path | None:
        source = Path(source_path)
        if not source.is_file():
            return None
        destination = classified_dir(self.root, "blank") / classified_filename(
            "blank", camera_id, timestamp, image_id
        )
        try:
            return write_jpeg_copy(
                source,
                destination,
                jpeg_quality=self.settings.crop_jpeg_quality,
            )
        except Exception:
            logger.exception("Failed to copy blank classified image for image %s", image_id)
            return None

    def relocate_tiger(
        self,
        *,
        observation_id: int,
        tiger_id: str,
    ) -> Path | None:
        """Move an unidentified tiger copy under tiger/{T00X}/. Originals stay put."""
        observation = self.observations.get_joined(observation_id)
        if observation is None:
            return None
        identity = (tiger_id or "").strip()
        if not identity:
            return None
        source_path = observation.get("classified_path") or observation.get("original_path")
        if not source_path:
            return None
        source = Path(str(source_path))
        if not source.is_file():
            original = observation.get("original_path")
            if not original or not Path(str(original)).is_file():
                return None
            source = Path(str(original))
        camera_id = observation.get("camera_id")
        timestamp = observation.get("timestamp") or observation.get("image_timestamp")
        detection_id = int(observation["detection_id"])
        dest_dir = classified_dir(self.root, "tiger", identity)
        destination = dest_dir / classified_filename(
            identity, camera_id, timestamp, observation_id
        )
        try:
            saved = write_jpeg_copy(
                source,
                destination,
                jpeg_quality=self.settings.crop_jpeg_quality,
            )
        except Exception:
            logger.exception(
                "Failed to relocate classified tiger image for observation %s",
                observation_id,
            )
            return None
        path_text = str(saved)
        self.detections.set_classified_path(detection_id, path_text)
        self.observations.set_classified_path(observation_id, path_text)
        old = observation.get("classified_path")
        if old and Path(str(old)) != saved:
            old_path = Path(str(old))
            try:
                if old_path.is_file() and _is_under(old_path, self.root / "tiger" / UNIDENTIFIED):
                    old_path.unlink()
            except OSError:
                logger.warning("Could not remove old unidentified copy %s", old_path)
        return saved


def _is_under(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False
