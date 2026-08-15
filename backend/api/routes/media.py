"""Serve original images and tiger crops by database id only.

Clients never pass filesystem paths, which avoids path traversal.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from backend.api.deps import db_dep
from backend.database.connection import Database
from backend.database.repositories import ImageRepository, ObservationRepository

router = APIRouter(prefix="/media", tags=["media"])

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _file_response(path: Path) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media file is not on disk.")
    suffix = path.suffix.lower()
    if suffix not in _MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported media type.")
    return FileResponse(path, media_type=_MEDIA_TYPES[suffix], filename=path.name)


@router.get("/images/{image_id}")
def get_image_file(image_id: int, db: Database = Depends(db_dep)) -> FileResponse:
    image = ImageRepository(db).get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    return _file_response(Path(image["original_path"]))


@router.get("/crops/{observation_id}")
def get_crop_file(observation_id: int, db: Database = Depends(db_dep)) -> FileResponse:
    observation = ObservationRepository(db).get(observation_id)
    if observation is None or not observation.get("crop_path"):
        raise HTTPException(status_code=404, detail="Crop not found.")
    return _file_response(Path(observation["crop_path"]))
