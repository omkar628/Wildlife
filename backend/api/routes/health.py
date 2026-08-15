from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import get_detector, get_reid_adapter, settings_dep
from backend.config import Settings
from backend.detector.service import DetectorService
from backend.reid.adapter import UnavailableReIDAdapter

router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    settings: Settings = Depends(settings_dep),
    detector: DetectorService = Depends(get_detector),
    reid: UnavailableReIDAdapter = Depends(get_reid_adapter),
) -> dict:
    return {
        "status": "ok",
        "offline": True,
        "detector": {
            "available": detector.available(),
            "path": str(settings.detector_model_path),
            "device": detector.device,
        },
        "reid": reid.status(),
        "database": str(settings.database_path),
        "confidence": {
            "auto_accept": settings.confidence_auto_accept,
            "detect_min": settings.confidence_detect_min,
        },
    }
