from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import get_alert_service, get_gnn_service
from backend.api.schemas import AlertFilterRequest, AlertSyncRequest
from backend.services.alerts import AlertService
from backend.services.gnn_service import GNNService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts(
    unread: bool | None = Query(default=None),
    alert_type: str | None = Query(default=None),
    tiger_id: str | None = Query(default=None),
    camera_id: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    since_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    alerts: AlertService = Depends(get_alert_service),
) -> dict:
    items = alerts.list_alerts(
        unread=unread,
        alert_type=alert_type,
        tiger_id=tiger_id,
        camera_id=camera_id,
        severity=severity,
        since_id=since_id,
        limit=limit,
    )
    return {"alerts": items, "count": len(items)}


@router.get("/summary")
def alert_summary(alerts: AlertService = Depends(get_alert_service)) -> dict:
    return alerts.summary()


@router.post("/sync")
def sync_alerts(
    payload: AlertSyncRequest | None = None,
    alerts: AlertService = Depends(get_alert_service),
    gnn: GNNService = Depends(get_gnn_service),
) -> dict:
    include_gnn = bool(payload.include_gnn) if payload else False
    created = alerts.sync_from_database()
    created["predictions"] = alerts.sync_predictions(gnn) if include_gnn else 0
    return {"created": created, "summary": alerts.summary()}


@router.post("/read-all")
def mark_all_read(
    payload: AlertFilterRequest | None = None,
    alerts: AlertService = Depends(get_alert_service),
) -> dict:
    updated = alerts.mark_all_read(
        alert_type=payload.alert_type if payload else None,
        tiger_id=payload.tiger_id if payload else None,
        camera_id=payload.camera_id if payload else None,
    )
    return {"updated": updated, "summary": alerts.summary()}


@router.post("/clear")
def clear_filtered(
    payload: AlertFilterRequest | None = None,
    alerts: AlertService = Depends(get_alert_service),
) -> dict:
    updated = alerts.clear_filtered(
        alert_type=payload.alert_type if payload else None,
        tiger_id=payload.tiger_id if payload else None,
        camera_id=payload.camera_id if payload else None,
    )
    return {"updated": updated, "summary": alerts.summary()}


@router.post("/{alert_id}/read")
def mark_read(
    alert_id: int,
    alerts: AlertService = Depends(get_alert_service),
) -> dict:
    row = alerts.mark_read(alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return row


@router.post("/{alert_id}/clear")
def clear_alert(
    alert_id: int,
    alerts: AlertService = Depends(get_alert_service),
) -> dict:
    row = alerts.clear(alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return row
