# Graph module

Graph construction lives in `backend/graph/`.

This folder is reserved for exported graph artifacts (JSON, later GNN inputs).
Nothing is written here during the first milestone.

GNN inference lives in `backend/services/gnn_service.py`. It consumes:

- `build_camera_graph()`
- `build_tiger_observation_graph()`
- `get_tiger_history()`
- `get_camera_connections()`

Do not put GNN weights or training code in this folder.
