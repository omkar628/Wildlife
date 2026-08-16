from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_identity_service
from backend.api.schemas import IdentityAssignRequest
from backend.reid.identity import LocalIdentityService

router = APIRouter(prefix="/observations", tags=["observations"])


@router.get("/unidentified")
def unidentified_observations(
    identity: LocalIdentityService = Depends(get_identity_service),
) -> dict:
    items = identity.unidentified()
    return {
        "observations": items,
        "pending": identity.unidentified_count(),
        "tigers": identity.gallery.catalog(),
        "next_tiger_id": identity.next_local_id(),
    }


@router.post("/{observation_id}/identity")
def assign_observation_identity(
    observation_id: int,
    payload: IdentityAssignRequest,
    identity: LocalIdentityService = Depends(get_identity_service),
) -> dict:
    try:
        return identity.assign(
            observation_id,
            action=payload.action,
            tiger_id=payload.tiger_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
