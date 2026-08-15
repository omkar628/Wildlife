from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import db_dep
from backend.database.connection import Database
from backend.database.repositories import DetectionRepository, ObservationRepository

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("")
def list_detections(
    class_name: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Database = Depends(db_dep),
) -> dict:
    items = DetectionRepository(db).list_recent(limit=limit, class_name=class_name)
    return {"detections": items}


@router.get("/{detection_id}")
def get_detection(detection_id: int, db: Database = Depends(db_dep)) -> dict:
    item = DetectionRepository(db).get(detection_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Detection not found.")
    observation = ObservationRepository(db).get_by_detection(detection_id)
    return {"detection": item, "observation": observation}
