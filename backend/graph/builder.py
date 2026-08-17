"""Build camera / observation graphs from SQLite.

No GNN lives here. A future models/gnn package should call these functions
and return movement predictions as structured events.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from backend.database.connection import Database
from backend.database.repositories import CameraRepository, ObservationRepository
from backend.graph.types import (
    CameraEdge,
    CameraGraph,
    CameraNode,
    ObservationEvent,
    TigerObservationGraph,
)

HOME_RANGE_LABEL = "Estimated home range"
NOT_ENOUGH_OBSERVATIONS = "Not enough observations"
PREDICTION_UNAVAILABLE = "Prediction unavailable — insufficient data."

_ZONE_ALIASES = {
    "core": "core",
    "core_zone": "core",
    "core-zone": "core",
    "core zone": "core",
    "buffer": "buffer",
    "buffer_zone": "buffer",
    "buffer-zone": "buffer",
    "buffer zone": "buffer",
    "village": "village-adjacent",
    "village_adjacent": "village-adjacent",
    "village-adjacent": "village-adjacent",
    "village adjacent": "village-adjacent",
    "fringe": "village-adjacent",
}


def parse_camera_metadata(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def infer_zone_type(habitat: Any, metadata: Any) -> str | None:
    """Return a recorded zone type only. Never invent one from location."""
    meta = parse_camera_metadata(metadata)
    for key in ("zone_type", "zone", "management_zone"):
        value = meta.get(key)
        mapped = _map_zone_text(value)
        if mapped:
            return mapped
    mapped = _map_zone_text(habitat)
    if mapped:
        return mapped
    if isinstance(habitat, str):
        text = habitat.strip().lower()
        if "village" in text:
            return "village-adjacent"
        if "buffer" in text:
            return "buffer"
        if "core" in text:
            return "core"
    return None


def _map_zone_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _ZONE_ALIASES.get(value.strip().lower())


def finite_coord(latitude: Any, longitude: Any) -> tuple[float, float] | None:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return lat, lon


def occupancy_level(value: int, maximum: int) -> str:
    if value <= 0 or maximum <= 0:
        return "none"
    ratio = value / maximum
    if ratio <= 1 / 3:
        return "low"
    if ratio <= 2 / 3:
        return "medium"
    return "high"


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain. Returns vertices in counterclockwise order."""
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def human_prediction_summary(camera_id: str, score: float | None) -> str:
    if score is None or not math.isfinite(float(score)):
        return f"Likely next station is {camera_id}."
    percent = max(0, min(100, int(round(float(score) * 100))))
    return f"Likely next station is {camera_id} ({percent}% model confidence)."


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

    def camera_lookup(self) -> dict[str, dict[str, Any]]:
        return {str(camera["camera_id"]): camera for camera in self.cameras.list_all()}

    def camera_coord_payload(self, camera: dict[str, Any] | None, camera_id: str) -> dict[str, Any]:
        if camera is None:
            return {
                "camera_id": camera_id,
                "latitude": None,
                "longitude": None,
                "registered": False,
                "missing_coordinates": True,
                "zone_type": None,
                "habitat": None,
            }
        coords = finite_coord(camera.get("latitude"), camera.get("longitude"))
        return {
            "camera_id": str(camera["camera_id"]),
            "latitude": coords[0] if coords else None,
            "longitude": coords[1] if coords else None,
            "registered": True,
            "missing_coordinates": coords is None,
            "zone_type": infer_zone_type(camera.get("habitat"), camera.get("metadata")),
            "habitat": camera.get("habitat"),
        }

    def build_observed_route(self, tiger_id: str) -> list[dict[str, Any]]:
        """Chronological identified observations joined to registered camera coordinates."""
        cameras = self.camera_lookup()
        route: list[dict[str, Any]] = []
        for event in self.get_tiger_history(tiger_id):
            camera_id = event.camera_id
            if not camera_id:
                continue
            camera = cameras.get(str(camera_id))
            coords = self.camera_coord_payload(camera, str(camera_id))
            route.append(
                {
                    "camera_id": str(camera_id),
                    "timestamp": event.timestamp,
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                    "confidence": event.confidence,
                    "observation_id": event.observation_id,
                    "detection_id": event.detection_id,
                    "tiger_id": event.tiger_id or tiger_id,
                    "missing_coordinates": coords["missing_coordinates"],
                    "registered": coords["registered"],
                    "zone_type": coords["zone_type"],
                    "habitat": coords["habitat"],
                }
            )
        return route

    def station_occupancy(self, tiger_id: str | None = None) -> dict[str, Any]:
        """Occupancy from stored detections and tiger observations only."""
        cameras = self.cameras.list_all()
        all_species = {
            str(row["camera_id"]): int(row["n"])
            for row in self.db.fetchall(
                """
                SELECT i.camera_id AS camera_id, COUNT(*) AS n
                FROM detections d
                JOIN images i ON i.image_id = d.image_id
                WHERE i.camera_id IS NOT NULL
                  AND d.review_status != 'ignored'
                GROUP BY i.camera_id
                """
            )
        }
        tiger_captures = {
            str(row["camera_id"]): int(row["n"])
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
        unique_tigers = {
            str(row["camera_id"]): int(row["n"])
            for row in self.db.fetchall(
                """
                SELECT i.camera_id AS camera_id, COUNT(DISTINCT o.tiger_id) AS n
                FROM tiger_observations o
                JOIN detections d ON d.detection_id = o.detection_id
                JOIN images i ON i.image_id = d.image_id
                WHERE i.camera_id IS NOT NULL
                  AND o.tiger_id IS NOT NULL
                GROUP BY i.camera_id
                """
            )
        }
        selected_captures: dict[str, int] = {}
        if tiger_id:
            selected_captures = {
                str(row["camera_id"]): int(row["n"])
                for row in self.db.fetchall(
                    """
                    SELECT i.camera_id AS camera_id, COUNT(*) AS n
                    FROM tiger_observations o
                    JOIN detections d ON d.detection_id = o.detection_id
                    JOIN images i ON i.image_id = d.image_id
                    WHERE i.camera_id IS NOT NULL
                      AND o.tiger_id = ?
                    GROUP BY i.camera_id
                    """,
                    (tiger_id,),
                )
            }

        latest: dict[str, dict[str, Any]] = {}
        first_last: dict[str, dict[str, Any]] = {}
        for row in self.db.fetchall(
            """
            SELECT i.camera_id AS camera_id,
                   o.tiger_id AS tiger_id,
                   COALESCE(o.timestamp, i.timestamp, o.created_at) AS ts
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE i.camera_id IS NOT NULL
            ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at)
            """
        ):
            camera_id = str(row["camera_id"])
            stamp = row["ts"]
            bucket = first_last.setdefault(camera_id, {"first": stamp, "last": stamp, "count": 0})
            bucket["last"] = stamp
            bucket["count"] += 1
            if row["tiger_id"]:
                latest[camera_id] = {
                    "tiger_id": str(row["tiger_id"]),
                    "timestamp": stamp,
                }

        max_all = max(all_species.values(), default=0)
        max_tiger = max(tiger_captures.values(), default=0)
        max_selected = max(selected_captures.values(), default=0)

        stations: list[dict[str, Any]] = []
        for camera in cameras:
            camera_id = str(camera["camera_id"])
            coords = self.camera_coord_payload(camera, camera_id)
            span = first_last.get(camera_id)
            frequency = None
            span_days = None
            if span and span["first"] and span["last"] and span["count"] > 1:
                span_days = _timestamp_span_days(span["first"], span["last"])
                if span_days and span_days > 0:
                    frequency = span["count"] / span_days
            stations.append(
                {
                    **coords,
                    "all_species_detections": all_species.get(camera_id, 0),
                    "tiger_captures": tiger_captures.get(camera_id, 0),
                    "unique_tigers": unique_tigers.get(camera_id, 0),
                    "selected_tiger_captures": selected_captures.get(camera_id, 0),
                    "latest_tiger_id": latest.get(camera_id, {}).get("tiger_id"),
                    "latest_tiger_timestamp": latest.get(camera_id, {}).get("timestamp"),
                    "occupancy_level_all_species": occupancy_level(
                        all_species.get(camera_id, 0), max_all
                    ),
                    "occupancy_level_tiger": occupancy_level(
                        tiger_captures.get(camera_id, 0), max_tiger
                    ),
                    "occupancy_level_selected_tiger": occupancy_level(
                        selected_captures.get(camera_id, 0), max_selected
                    ),
                    "capture_frequency_per_day": frequency,
                    "capture_span_days": span_days,
                }
            )

        return {
            "stations": stations,
            "supported_modes": ["all_species", "tiger", "selected_tiger"],
            "selected_tiger_id": tiger_id,
        }

    def estimate_home_range(self, observed_route: list[dict[str, Any]]) -> dict[str, Any]:
        """Convex hull of registered camera coordinates this tiger actually visited."""
        unique_coords: dict[tuple[float, float], str] = {}
        for stop in observed_route:
            coords = finite_coord(stop.get("latitude"), stop.get("longitude"))
            if coords is None:
                continue
            unique_coords.setdefault(coords, str(stop["camera_id"]))
        points = list(unique_coords.keys())
        hull = convex_hull(points)
        if len(hull) < 3:
            return {
                "available": False,
                "reason": NOT_ENOUGH_OBSERVATIONS,
                "label": HOME_RANGE_LABEL,
                "polygon": [],
                "point_count": len(points),
                "unique_stations": len(unique_coords),
            }
        return {
            "available": True,
            "reason": None,
            "label": HOME_RANGE_LABEL,
            "polygon": [
                {
                    "latitude": lat,
                    "longitude": lon,
                    "camera_id": unique_coords[(lat, lon)],
                }
                for lat, lon in hull
            ],
            "point_count": len(points),
            "unique_stations": len(unique_coords),
        }

    def attach_prediction_coordinates(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Join GNN camera IDs to registered coordinates. Never invent locations."""
        cameras = self.camera_lookup()
        attached = dict(prediction)
        available = bool(prediction.get("available"))
        if not available:
            attached.setdefault("reason", prediction.get("reason") or PREDICTION_UNAVAILABLE)
            attached["ranked_candidates"] = []
            attached["summary"] = PREDICTION_UNAVAILABLE
            predicted = prediction.get("predicted_camera_id")
            if predicted:
                coords = self.camera_coord_payload(cameras.get(str(predicted)), str(predicted))
                attached["latitude"] = coords["latitude"]
                attached["longitude"] = coords["longitude"]
            return attached

        ranked: list[dict[str, Any]] = []
        for item in prediction.get("ranked_candidates") or []:
            camera_id = str(item.get("camera_id") or "")
            if not camera_id:
                continue
            coords = self.camera_coord_payload(cameras.get(camera_id), camera_id)
            score = item.get("confidence", item.get("score"))
            try:
                score_f = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_f = None
            ranked.append(
                {
                    "rank": item.get("rank"),
                    "camera_id": camera_id,
                    "score": score_f,
                    "confidence": score_f,
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                    "registered": coords["registered"],
                    "missing_coordinates": coords["missing_coordinates"],
                    "zone_type": coords["zone_type"],
                }
            )

        predicted_id = prediction.get("predicted_camera_id")
        predicted_coords = (
            self.camera_coord_payload(cameras.get(str(predicted_id)), str(predicted_id))
            if predicted_id
            else {
                "latitude": None,
                "longitude": None,
                "missing_coordinates": True,
                "registered": False,
            }
        )
        score = prediction.get("confidence")
        attached.update(
            {
                "ranked_candidates": ranked,
                "latitude": predicted_coords["latitude"],
                "longitude": predicted_coords["longitude"],
                "missing_coordinates": predicted_coords["missing_coordinates"],
                "registered": predicted_coords["registered"],
                "summary": human_prediction_summary(str(predicted_id), score)
                if predicted_id
                else PREDICTION_UNAVAILABLE,
            }
        )
        return attached

    def build_tiger_route(self, tiger_id: str, prediction: dict[str, Any] | None = None) -> dict[str, Any]:
        observed = self.build_observed_route(tiger_id)
        current = observed[-1] if observed else None
        visited: list[str] = []
        for stop in observed:
            if stop["camera_id"] not in visited:
                visited.append(stop["camera_id"])
        raw_prediction = prediction or {
            "available": False,
            "reason": PREDICTION_UNAVAILABLE,
            "tiger_id": tiger_id,
        }
        return {
            "tiger_id": tiger_id,
            "observed_route": observed,
            "current_station": current,
            "predictions": self.attach_prediction_coordinates(raw_prediction),
            "occupancy": self.station_occupancy(tiger_id=tiger_id),
            "home_range": self.estimate_home_range(observed),
            "visited_stations": visited,
            "observation_count": len(observed),
            "last_observed_station": current["camera_id"] if current else None,
            "last_observed_timestamp": current["timestamp"] if current else None,
        }

    def export_payload(self) -> dict:
        """Stable JSON the frontend (and later the GNN) can consume."""
        camera_graph = self.build_camera_graph()
        observation_graph = self.build_tiger_observation_graph()
        return {
            "camera_graph": camera_graph.to_dict(),
            "observation_graph": observation_graph.to_dict(),
            "occupancy": self.station_occupancy(),
            "gnn": {
                "implemented": True,
                "expected_input": "identified tiger history + live camera graph",
                "expected_output_example": {
                    "tiger_id": "T017",
                    "camera_id": "C02",
                    "timestamp": "2026-08-15T10:30:00",
                    "confidence": 0.91,
                    "kind": "prediction",
                },
            },
        }


def _parse_stamp(value: Any):
    from datetime import datetime, timezone

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _timestamp_span_days(first: Any, last: Any) -> float | None:
    start = _parse_stamp(first)
    end = _parse_stamp(last)
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    if seconds <= 0:
        return None
    return seconds / 86400.0
