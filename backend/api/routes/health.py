from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import (
    get_detector,
    get_gnn_service,
    get_identity_service,
    get_reid_adapter,
    get_reid_encoder,
    settings_dep,
)
from backend.config import Settings
from backend.detector.service import DetectorService
from backend.reid.adapter import UnavailableReIDAdapter
from backend.reid.identity import LocalIdentityService
from backend.reid.megadescriptor import MegaDescriptorEncoder
from backend.services.gnn_service import GNNService

router = APIRouter(tags=["health"])


def _gnn_status(gnn: GNNService) -> dict:
    try:
        return gnn.status()
    except Exception as exc:
        return {
            "loaded": False,
            "device": "cpu",
            "path": None,
            "reason": str(exc),
        }


@router.get("/health")
def health(
    settings: Settings = Depends(settings_dep),
    detector: DetectorService = Depends(get_detector),
    reid: UnavailableReIDAdapter = Depends(get_reid_adapter),
    encoder: MegaDescriptorEncoder = Depends(get_reid_encoder),
    identity: LocalIdentityService = Depends(get_identity_service),
    gnn: GNNService = Depends(get_gnn_service),
) -> dict:
    encoder_status = encoder.status()
    reid_status = {
        **encoder_status,
        "atrw_assets": reid.status(),
        "local_identity": identity.status(),
        "match_threshold": settings.reid_match_threshold,
        "review_threshold": settings.reid_review_threshold,
    }
    return {
        "status": "ok",
        "offline": True,
        "detector": {
            "available": detector.available(),
            "path": str(settings.detector_model_path),
            "device": detector.device,
        },
        "reid": reid_status,
        "gnn": _gnn_status(gnn),
        "database": str(settings.database_path),
        "confidence": {
            "auto_accept": settings.confidence_auto_accept,
            "detect_min": settings.confidence_detect_min,
        },
    }
