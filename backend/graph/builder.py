"""Build camera / observation graphs from SQLite.

No GNN lives here. A future models/gnn package should call these functions
and return movement predictions as structured events.
"""

from __future__ import annotations

from collections import defaultdict

from backend.database.connection import Database
from backend.database.repositories import CameraRepository, ObservationRepository
from backend.graph.types import (
    CameraEdge,
    CameraGraph,
    CameraNode,
    ObservationEvent,
    TigerObservationGraph,
)


class GraphService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.cameras = CameraRepository(db)
        self.observations = ObservationRepository(db)

    def build_camera_graph(self) -> CameraGraph:
        nodes = self._camera_nodes()
        edges = self._transition_edges()
        return CameraGraph(nodes=nodes, edges=edges)

    def build_tiger_observation_graph(self, tiger_id: str | None = None) -> TigerObservationGraph:
        events = self._events(tiger_id=tiger_id)
        return TigerObservationGraph(
            events=events,
            cameras=self._camera_nodes(),
            edges=self._transition_edges(tiger_id=tiger_id),
        )

    def get_tiger_history(self, tiger_id: str) -> list[ObservationEvent]:
        return self._events(tiger_id=tiger_id)

    def get_camera_connections(self) -> list[CameraEdge]:
        return self._transition_edges()

    def _camera_nodes(self) -> list[CameraNode]:
        cameras = self.cameras.list_all()
        image_counts = {
            row["camera_id"]: int(row["n"])
            for row in self.db.fetchall(
                """
                SELECT camera_id, COUNT(*) AS n
                FROM images
                WHERE camera_id IS NOT NULL
                GROUP BY camera_id
                """
            )
        }
        observation_counts = {
            row["camera_id"]: int(row["n"])
            for row in self.db.fetchall(
                """
                SELECT i.camera_id AS camera_id, COUNT(*) AS n
                FROM tiger_observations o
                JOIN detections d ON d.detection_id = o.detection_id
                JOIN images i ON i.image_id = d.image_id
                WHERE i.camera_id IS NOT NULL
                GROUP BY i.camera_id
                """
            )
        }
        return [
            CameraNode(
                camera_id=str(camera["camera_id"]),
                latitude=camera.get("latitude"),
                longitude=camera.get("longitude"),
                elevation=camera.get("elevation"),
                habitat=camera.get("habitat"),
                observation_count=observation_counts.get(camera["camera_id"], 0),
                image_count=image_counts.get(camera["camera_id"], 0),
            )
            for camera in cameras
        ]

    def _events(self, tiger_id: str | None = None) -> list[ObservationEvent]:
        if tiger_id:
            rows = self.observations.list_for_tiger(tiger_id)
        else:
            rows = self.observations.list_all()
        events: list[ObservationEvent] = []
        for row in rows:
            events.append(
                ObservationEvent(
                    tiger_id=row.get("tiger_id"),
                    camera_id=row.get("camera_id"),
                    timestamp=row.get("timestamp") or row.get("image_timestamp"),
                    confidence=row.get("reid_confidence") if row.get("reid_confidence") is not None else row.get("confidence"),
                    detection_id=int(row["detection_id"]),
                    observation_id=int(row["observation_id"]),
                    class_name=row.get("final_class_name") or row.get("class_name"),
                    crop_path=row.get("crop_path"),
                )
            )
        return events

    def _transition_edges(self, tiger_id: str | None = None) -> list[CameraEdge]:
        """Consecutive observations of the same identified tiger become edges.

        Unidentified observations (tiger_id IS NULL) cannot form movement edges.
        That is intentional — the GNN should not invent identity.
        """
        if tiger_id:
            rows = self.observations.list_for_tiger(tiger_id)
        else:
            rows = [
                row
                for row in self.observations.list_all()
                if row.get("tiger_id")
            ]

        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if not row.get("tiger_id") or not row.get("camera_id"):
                continue
            grouped[str(row["tiger_id"])].append(row)

        aggregates: dict[tuple[str, str, str], CameraEdge] = {}
        for identity, items in grouped.items():
            items.sort(key=lambda item: item.get("timestamp") or item.get("image_timestamp") or item.get("created_at") or "")
            for previous, current in zip(items, items[1:]):
                source = previous.get("camera_id")
                target = current.get("camera_id")
                if not source or not target or source == target:
                    continue
                key = (str(source), str(target), identity)
                stamp = current.get("timestamp") or current.get("image_timestamp")
                if key not in aggregates:
                    aggregates[key] = CameraEdge(
                        source=str(source),
                        target=str(target),
                        tiger_id=identity,
                        weight=1,
                        first_timestamp=stamp,
                        last_timestamp=stamp,
                    )
                else:
                    aggregates[key].weight += 1
                    aggregates[key].last_timestamp = stamp
        return list(aggregates.values())

    def export_payload(self) -> dict:
        """Stable JSON the frontend (and later the GNN) can consume."""
        camera_graph = self.build_camera_graph()
        observation_graph = self.build_tiger_observation_graph()
        return {
            "camera_graph": camera_graph.to_dict(),
            "observation_graph": observation_graph.to_dict(),
            "gnn": {
                "implemented": False,
                "expected_input": "camera_graph + observation_graph",
                "expected_output_example": {
                    "tiger_id": "T017",
                    "camera_id": "C02",
                    "timestamp": "2026-08-15T10:30:00",
                    "confidence": 0.91,
                    "kind": "prediction",
                },
            },
        }
