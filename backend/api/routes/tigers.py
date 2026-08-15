from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import db_dep, get_graph_service
from backend.database.connection import Database
from backend.database.repositories import TigerRepository
from backend.graph.builder import GraphService

router = APIRouter(prefix="/tigers", tags=["tigers"])


@router.get("")
def list_tigers(db: Database = Depends(db_dep)) -> dict:
    return {"tigers": TigerRepository(db).list_all()}


@router.get("/{tiger_id}")
def get_tiger(
    tiger_id: str,
    db: Database = Depends(db_dep),
    graph: GraphService = Depends(get_graph_service),
) -> dict:
    tiger = TigerRepository(db).get(tiger_id)
    if tiger is None:
        raise HTTPException(status_code=404, detail="Tiger not found.")
    history = [event.to_dict() for event in graph.get_tiger_history(tiger_id)]
    return {"tiger": tiger, "history": history}
