# Future GNN

This directory is intentionally empty.

When the GNN is ready, place it here (for example `models/gnn/`) and wire it
to `backend.graph.builder.GraphService`. Do not change:

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
