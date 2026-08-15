from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import db_dep, settings_dep
from backend.api.schemas import SettingsUpdateRequest
from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import SettingsRepository

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
def get_runtime_settings(settings: Settings = Depends(settings_dep), db: Database = Depends(db_dep)) -> dict:
    stored = SettingsRepository(db)
    stored_auto = stored.get("confidence_auto_accept")
    stored_min = stored.get("confidence_detect_min")
    return {
        "confidence_auto_accept": float(stored_auto) if stored_auto else settings.confidence_auto_accept,
        "confidence_detect_min": float(stored_min) if stored_min else settings.confidence_detect_min,
        "detector_batch_size": settings.detector_batch_size,
        "detector_imgsz": settings.detector_imgsz,
        "detector_device": settings.detector_device,
        "class_map": settings.class_map,
        "detector_model_path": str(settings.detector_model_path),
        "database_path": str(settings.database_path),
        "crops_dir": str(settings.crops_dir),
    }


@router.put("")
def update_runtime_settings(
    payload: SettingsUpdateRequest,
    settings: Settings = Depends(settings_dep),
    db: Database = Depends(db_dep),
) -> dict:
    stored = SettingsRepository(db)
    if payload.confidence_auto_accept is not None:
        settings.confidence_auto_accept = payload.confidence_auto_accept
        stored.set("confidence_auto_accept", str(payload.confidence_auto_accept))
    if payload.confidence_detect_min is not None:
        if payload.confidence_detect_min > settings.confidence_auto_accept:
            raise HTTPException(
                status_code=400,
                detail="detect_min cannot be greater than auto_accept.",
            )
        settings.confidence_detect_min = payload.confidence_detect_min
        stored.set("confidence_detect_min", str(payload.confidence_detect_min))
    return get_runtime_settings(settings=settings, db=db)
