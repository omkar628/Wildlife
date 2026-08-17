from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import (
    db_dep,
    get_alert_service,
    get_identity_service,
    get_review_service,
    settings_dep,
)
from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import (
    DetectionRepository,
    ImageRepository,
    JobRepository,
    TigerRepository,
)
from backend.reid.identity import LocalIdentityService
from backend.review.service import ReviewService
from backend.services.alerts import AlertService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("")
def dashboard(
    db: Database = Depends(db_dep),
    settings: Settings = Depends(settings_dep),
    reviews: ReviewService = Depends(get_review_service),
    identity: LocalIdentityService = Depends(get_identity_service),
    alerts: AlertService = Depends(get_alert_service),
) -> dict:
    class_counts = DetectionRepository(db).class_counts()
    status_counts = ImageRepository(db).count_by_status()
    jobs = JobRepository(db).list_recent(8)
    tigers = TigerRepository(db).list_all()
    image_total = sum(status_counts.values())
    detection_total = sum(class_counts.values())
    return {
        "images": {
            "total": image_total,
            "by_status": status_counts,
        },
        "detections": {
            "total": detection_total,
            "by_class": class_counts,
            "tiger": class_counts.get("tiger", 0),
            "prey": class_counts.get("prey", 0),
            "rival": class_counts.get("rival", 0),
            "human": class_counts.get("human", 0),
        },
        "review": {
            "pending": reviews.pending_count(),
            "unidentified_tigers": identity.unidentified_count(),
        },
        "tigers": {
            "known": len(tigers),
        },
        "alerts": alerts.summary(),
        "recent_jobs": jobs,
        "confidence_auto_accept": settings.confidence_auto_accept,
    }
