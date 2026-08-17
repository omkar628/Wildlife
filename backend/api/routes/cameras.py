from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import db_dep
from backend.api.schemas import CameraCreateRequest, CameraEnabledRequest, CameraUpdateRequest
from backend.database.connection import Database
from backend.database.repositories import CameraRepository

router = APIRouter(prefix="/cameras", tags=["cameras"])


def _metadata_json(metadata: dict | None) -> str | None:
    if metadata is None:
        return None
    return json.dumps(metadata)


def _http_camera_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        message = str(exc)
        status = 409 if "already exists" in message.lower() or "cannot delete" in message.lower() else 400
        return HTTPException(status_code=status, detail=message)
    return HTTPException(status_code=400, detail=str(exc))


@router.get("")
def list_cameras(db: Database = Depends(db_dep)) -> dict:
    cameras = CameraRepository(db).list_with_stats()
    return {"cameras": cameras}


@router.post("")
def create_camera(payload: CameraCreateRequest, db: Database = Depends(db_dep)) -> dict:
    try:
        camera = CameraRepository(db).create(
            payload.camera_id,
            name=payload.name,
            latitude=payload.latitude,
            longitude=payload.longitude,
            elevation=payload.elevation,
            habitat=payload.habitat,
            metadata=_metadata_json(payload.metadata),
            enabled=payload.enabled,
        )
    except (ValueError, KeyError) as exc:
        raise _http_camera_error(exc) from exc
    return camera


@router.get("/{camera_id}")
def get_camera(camera_id: str, db: Database = Depends(db_dep)) -> dict:
    cameras = CameraRepository(db)
    camera = cameras.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    stats = next(
        (item for item in cameras.list_with_stats() if item["camera_id"] == camera_id),
        camera,
    )
    return stats


@router.put("/{camera_id}")
def update_camera(
    camera_id: str,
    payload: CameraUpdateRequest,
    db: Database = Depends(db_dep),
) -> dict:
    provided = payload.model_dump(exclude_unset=True)
    try:
        camera = CameraRepository(db).update(
            camera_id,
            name=provided.get("name") if "name" in provided else None,
            latitude=provided["latitude"] if "latitude" in provided else ...,
            longitude=provided["longitude"] if "longitude" in provided else ...,
            elevation=provided["elevation"] if "elevation" in provided else ...,
            habitat=provided["habitat"] if "habitat" in provided else ...,
            metadata=_metadata_json(provided["metadata"]) if "metadata" in provided else ...,
            enabled=provided.get("enabled"),
            new_camera_id=provided.get("camera_id"),
        )
    except (ValueError, KeyError) as exc:
        raise _http_camera_error(exc) from exc
    return camera


@router.patch("/{camera_id}")
def patch_camera(
    camera_id: str,
    payload: CameraEnabledRequest,
    db: Database = Depends(db_dep),
) -> dict:
    try:
        return CameraRepository(db).set_enabled(camera_id, payload.enabled)
    except (ValueError, KeyError) as exc:
        raise _http_camera_error(exc) from exc


@router.delete("/{camera_id}")
def delete_camera(camera_id: str, db: Database = Depends(db_dep)) -> dict:
    try:
        CameraRepository(db).delete(camera_id)
    except (ValueError, KeyError) as exc:
        raise _http_camera_error(exc) from exc
    return {"ok": True, "camera_id": camera_id}
