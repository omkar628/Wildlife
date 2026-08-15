from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import db_dep
from backend.database.connection import Database
from backend.database.repositories import DetectionRepository, ImageRepository

router = APIRouter(prefix="/images", tags=["images"])


@router.get("")
def list_images(
    camera_id: str | None = Query(default=None),
    limit: int = Query(default=40, ge=1, le=200),
    db: Database = Depends(db_dep),
) -> dict:
    images = ImageRepository(db).list_recent(limit=limit, camera_id=camera_id)
    detections = DetectionRepository(db)
    payload = []
    for image in images:
        boxes = detections.list_for_image(int(image["image_id"]))
        payload.append({**image, "detections": boxes, "detection_count": len(boxes)})
    return {"images": payload}


@router.get("/{image_id}")
def get_image(image_id: int, db: Database = Depends(db_dep)) -> dict:
    image = ImageRepository(db).get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    boxes = DetectionRepository(db).list_for_image(image_id)
    return {"image": image, "detections": boxes}
