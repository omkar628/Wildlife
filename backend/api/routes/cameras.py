from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import db_dep
from backend.api.schemas import CameraUpsertRequest
from backend.database.connection import Database
from backend.database.repositories import CameraRepository

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("")
def list_cameras(db: Database = Depends(db_dep)) -> dict:
    return {"cameras": CameraRepository(db).list_all()}


@router.post("")
def upsert_camera(payload: CameraUpsertRequest, db: Database = Depends(db_dep)) -> dict:
    metadata = json.dumps(payload.metadata) if payload.metadata is not None else None
    camera = CameraRepository(db).upsert(
        payload.camera_id,
        latitude=payload.latitude,
        longitude=payload.longitude,
        elevation=payload.elevation,
        habitat=payload.habitat,
        metadata=metadata,
    )
    return camera


@router.get("/{camera_id}")
def get_camera(camera_id: str, db: Database = Depends(db_dep)) -> dict:
    camera = CameraRepository(db).get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return camera
