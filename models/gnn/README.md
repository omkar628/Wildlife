# GNN weights

Place `gnn_model_v3_optimized_best.pt` here (or leave it at the project root).
The app auto-discovers either location.

Runtime inference lives in `backend/services/gnn_service.py` using the
V3.1 `DistanceAwareGNN` copy in `backend/services/gnn_architecture.py`.
Do not change:

- YOLO detector
- image ingestion
- SQLite schema
- frontend routing

The GNN should consume structured observation/camera graph JSON and return
events such as:

```json
{
  "tiger_id": "T017",
  "camera_id": "C02",
  "timestamp": "2026-08-15T10:30:00",
  "confidence": 0.91
}
```
