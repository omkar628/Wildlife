from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.deps import get_graph_service, get_reid_adapter
from backend.graph.builder import GraphService
from backend.reid.adapter import UnavailableReIDAdapter

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("")
def get_graph(graph: GraphService = Depends(get_graph_service)) -> dict:
    return graph.export_payload()


@router.get("/cameras")
def camera_graph(graph: GraphService = Depends(get_graph_service)) -> dict:
    return graph.build_camera_graph().to_dict()


@router.get("/observations")
def observation_graph(
    tiger_id: str | None = Query(default=None),
    graph: GraphService = Depends(get_graph_service),
) -> dict:
    return graph.build_tiger_observation_graph(tiger_id=tiger_id).to_dict()


@router.get("/reid-status")
def reid_status(reid: UnavailableReIDAdapter = Depends(get_reid_adapter)) -> dict:
    return reid.status()
