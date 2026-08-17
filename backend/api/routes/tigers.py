from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import db_dep, get_graph_service, get_identity_service
from backend.database.connection import Database
from backend.database.repositories import TigerRepository
from backend.graph.builder import GraphService
from backend.reid.identity import LocalIdentityService

router = APIRouter(prefix="/tigers", tags=["tigers"])


@router.get("")
def list_tigers(
    db: Database = Depends(db_dep),
    identity: LocalIdentityService = Depends(get_identity_service),
) -> dict:
    return {
        "tigers": TigerRepository(db).list_all(),
        "catalog": identity.gallery.catalog(),
        "next_tiger_id": identity.next_local_id(),
    }


@router.get("/{tiger_id}")
def get_tiger(
    tiger_id: str,
    db: Database = Depends(db_dep),
    graph: GraphService = Depends(get_graph_service),
    identity: LocalIdentityService = Depends(get_identity_service),
) -> dict:
    tiger = TigerRepository(db).get(tiger_id)
    if tiger is None:
        raise HTTPException(status_code=404, detail="Tiger not found.")
    history = [event.to_dict() for event in graph.get_tiger_history(tiger_id)]
    route = graph.build_tiger_route(tiger_id)
    activity = route.get("activity_area") or {}
    cameras_visited = route.get("visited_stations") or []
    return {
        "tiger": {
            **tiger,
            "observation_count": route.get("observation_count", len(history)),
            "last_camera": route.get("last_observed_station"),
            "last_seen": route.get("last_observed_timestamp") or tiger.get("last_seen"),
        },
        "history": history,
        "references": identity.gallery.references_for(tiger_id),
        "last_camera": route.get("last_observed_station"),
        "last_seen": route.get("last_observed_timestamp") or tiger.get("last_seen"),
        "cameras_visited": cameras_visited,
        "observation_count": route.get("observation_count", len(history)),
        "most_frequent_camera": route.get("most_frequent_camera"),
        "most_frequent_count": route.get("most_frequent_count") or 0,
        "activity_area": activity,
        "current_station": route.get("current_station"),
    }
