from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.deps import get_gnn_service, get_graph_service, get_reid_adapter
from backend.graph.builder import GraphService
from backend.reid.adapter import UnavailableReIDAdapter
from backend.services.gnn_service import GNNService

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("")
def get_graph(
    graph: GraphService = Depends(get_graph_service),
    gnn: GNNService = Depends(get_gnn_service),
) -> dict:
    payload = graph.export_payload()
    try:
        payload["gnn"] = gnn.graph_payload()
    except Exception as exc:
        payload["gnn"] = {
            "implemented": True,
            "loaded": False,
            "reason": str(exc),
            "predictions": [],
        }
    return payload


@router.get("/cameras")
def camera_graph(graph: GraphService = Depends(get_graph_service)) -> dict:
    return graph.build_camera_graph().to_dict()


@router.get("/observations")
def observation_graph(
    tiger_id: str | None = Query(default=None),
    graph: GraphService = Depends(get_graph_service),
) -> dict:
    return graph.build_tiger_observation_graph(tiger_id=tiger_id).to_dict()


@router.get("/predictions")
def gnn_predictions(
    tiger_id: str | None = Query(default=None),
    gnn: GNNService = Depends(get_gnn_service),
) -> dict:
    return gnn.predict_for_tiger(tiger_id)


@router.get("/tigers/{tiger_id}/route")
def tiger_route(
    tiger_id: str,
    graph: GraphService = Depends(get_graph_service),
    gnn: GNNService = Depends(get_gnn_service),
) -> dict:
    """Observed route from SQLite plus next-station ranking from the existing GNN."""
    try:
        prediction = gnn.predict_for_tiger(tiger_id)
    except Exception as exc:
        prediction = {
            "available": False,
            "reason": "Prediction unavailable — insufficient data.",
            "detail": str(exc),
            "tiger_id": tiger_id,
        }
    return graph.build_tiger_route(tiger_id, prediction)


@router.get("/reid-status")
def reid_status(reid: UnavailableReIDAdapter = Depends(get_reid_adapter)) -> dict:
    return reid.status()
