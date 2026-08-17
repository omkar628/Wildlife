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
from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    ObservationRepository,
)
from backend.graph.types import (
    CameraEdge,
    CameraGraph,
    CameraNode,
    ObservationEvent,
    TigerObservationGraph,
)

HOME_RANGE_LABEL = "Estimated home range"
ACTIVITY_AREA_LABEL = "Observed Activity Area"
NOT_ENOUGH_OBSERVATIONS = "Not enough observations"
PREDICTION_UNAVAILABLE = "Prediction unavailable — insufficient data."
OCCUPANCY_LABEL = "Observed activity / occupancy"
MOVEMENT_CLASSES = ("tiger", "prey", "rival", "human")
EARTH_RADIUS_KM = 6371.0

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


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * math.asin(math.sqrt(min(1.0, a)))


def camera_distance_km(
    source: dict[str, Any] | None,
    target: dict[str, Any] | None,
) -> float | None:
    if source is None or target is None:
        return None
    start = finite_coord(source.get("latitude"), source.get("longitude"))
    end = finite_coord(target.get("latitude"), target.get("longitude"))
    if start is None or end is None:
        return None
    return haversine_km(start[0], start[1], end[0], end[1])


def _in_time_range(stamp: Any, time_from: str | None, time_to: str | None) -> bool:
    if not time_from and not time_to:
        return True
    text = str(stamp).strip() if stamp is not None else ""
    if not text:
        return False
    if time_from and text < time_from:
        return False
    if time_to and text > time_to:
        return False
    return True


def finite_coord(latitude: Any, longitude: Any) -> tuple[float, float] | None:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
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
        self.detections = DetectionRepository(db)

    def build_camera_graph(
        self,
        tiger_id: str | None = None,
        animal_class: str | None = None,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> CameraGraph:
        nodes = self._camera_nodes()
        if animal_class and animal_class != "tiger":
            edges: list[CameraEdge] = []
        else:
            edges = self._transition_edges(
                tiger_id=tiger_id,
                camera_id=camera_id,
                time_from=time_from,
                time_to=time_to,
            )
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
        class_counts: dict[str, dict[str, int]] = {}
        for row in self.db.fetchall(
            """
            SELECT i.camera_id AS camera_id,
                   LOWER(COALESCE(d.final_class_name, d.class_name)) AS class_name,
                   COUNT(*) AS n
            FROM detections d
            JOIN images i ON i.image_id = d.image_id
            WHERE i.camera_id IS NOT NULL
              AND d.review_status != 'ignored'
            GROUP BY i.camera_id, LOWER(COALESCE(d.final_class_name, d.class_name))
            """
        ):
            camera_id = str(row["camera_id"])
            bucket = class_counts.setdefault(
                camera_id, {"tiger": 0, "prey": 0, "rival": 0, "human": 0}
            )
            name = str(row["class_name"] or "")
            if name in bucket:
                bucket[name] += int(row["n"])

        nodes: list[CameraNode] = []
        for camera in cameras:
            camera_id = str(camera["camera_id"])
            counts = class_counts.get(
                camera_id, {"tiger": 0, "prey": 0, "rival": 0, "human": 0}
            )
            enabled = camera.get("enabled") not in {0, "0", False}
            nodes.append(
                CameraNode(
                    camera_id=camera_id,
                    name=camera.get("name") or camera_id,
                    latitude=camera.get("latitude"),
                    longitude=camera.get("longitude"),
                    elevation=camera.get("elevation"),
                    habitat=camera.get("habitat"),
                    observation_count=observation_counts.get(camera["camera_id"], 0),
                    image_count=image_counts.get(camera["camera_id"], 0),
                    enabled=enabled,
                    status="enabled" if enabled else "disabled",
                    tiger_count=counts["tiger"],
                    prey_count=counts["prey"],
                    rival_count=counts["rival"],
                    human_count=counts["human"],
                )
            )
        return nodes

    def _events(self, tiger_id: str | None = None) -> list[ObservationEvent]:
        if tiger_id:
            rows = self.observations.list_for_tiger(tiger_id)
        else:
            rows = self.observations.list_all()
        events: list[ObservationEvent] = []
        for row in rows:
            image_id = row.get("image_id")
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
                    image_id=int(image_id) if image_id is not None else None,
                    reid_confidence=row.get("reid_confidence"),
                    embedding_available=bool(row.get("embedding_available")),
                    bbox_x=row.get("bbox_x"),
                    bbox_y=row.get("bbox_y"),
                    bbox_width=row.get("bbox_width"),
                    bbox_height=row.get("bbox_height"),
                    filename=row.get("filename"),
                )
            )
        return events

    def _transition_edges(
        self,
        tiger_id: str | None = None,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> list[CameraEdge]:
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
            stamp = row.get("timestamp") or row.get("image_timestamp") or row.get("created_at")
            if not _in_time_range(stamp, time_from, time_to):
                continue
            grouped[str(row["tiger_id"])].append(row)

        cameras = self.camera_lookup()
        aggregates: dict[tuple[str, str, str], CameraEdge] = {}
        for identity, items in grouped.items():
            items.sort(key=lambda item: item.get("timestamp") or item.get("image_timestamp") or item.get("created_at") or "")
            for previous, current in zip(items, items[1:]):
                source = previous.get("camera_id")
                target = current.get("camera_id")
                if not source or not target or source == target:
                    continue
                if camera_id and camera_id not in {source, target}:
                    continue
                key = (str(source), str(target), identity)
                stamp = current.get("timestamp") or current.get("image_timestamp")
                prev_obs = previous.get("observation_id")
                curr_obs = current.get("observation_id")
                prev_det = previous.get("detection_id")
                curr_det = current.get("detection_id")
                confidence = current.get("reid_confidence")
                if confidence is None:
                    confidence = current.get("confidence")
                if key not in aggregates:
                    aggregates[key] = CameraEdge(
                        source=str(source),
                        target=str(target),
                        tiger_id=identity,
                        weight=1,
                        first_timestamp=stamp,
                        last_timestamp=stamp,
                        animal_class="tiger",
                        identity=identity,
                        distance_km=camera_distance_km(cameras.get(str(source)), cameras.get(str(target))),
                        confidence=float(confidence) if confidence is not None else None,
                        observation_ids=[int(x) for x in (prev_obs, curr_obs) if x is not None],
                        detection_ids=[int(x) for x in (prev_det, curr_det) if x is not None],
                        source_observation_id=int(prev_obs) if prev_obs is not None else None,
                        destination_observation_id=int(curr_obs) if curr_obs is not None else None,
                        kind="observed",
                    )
                else:
                    edge = aggregates[key]
                    edge.weight += 1
                    edge.last_timestamp = stamp
                    if curr_obs is not None:
                        edge.observation_ids.append(int(curr_obs))
                    if curr_det is not None:
                        edge.detection_ids.append(int(curr_det))
                    if confidence is not None:
                        edge.confidence = float(confidence)
        return list(aggregates.values())

    def _class_transition_edges(
        self,
        animal_class: str,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> list[CameraEdge]:
        """Class-level chronological edges. Not individual identities except tiger."""
        class_name = animal_class.strip().lower()
        if class_name not in MOVEMENT_CLASSES or class_name == "tiger":
            return []
        rows = self.detections.list_for_movement(
            animal_class=class_name,
            time_from=time_from,
            time_to=time_to,
        )
        if not rows:
            return []
        cameras = self.camera_lookup()
        items = sorted(
            rows,
            key=lambda item: (item.get("timestamp") or "", int(item.get("detection_id") or 0)),
        )
        aggregates: dict[tuple[str, str, str], CameraEdge] = {}
        for previous, current in zip(items, items[1:]):
            source = previous.get("camera_id")
            target = current.get("camera_id")
            if not source or not target or source == target:
                continue
            if camera_id and camera_id not in {source, target}:
                continue
            key = (str(source), str(target), class_name)
            stamp = current.get("timestamp")
            prev_det = previous.get("detection_id")
            curr_det = current.get("detection_id")
            prev_obs = previous.get("observation_id")
            curr_obs = current.get("observation_id")
            confidence = current.get("confidence")
            if key not in aggregates:
                aggregates[key] = CameraEdge(
                    source=str(source),
                    target=str(target),
                    tiger_id=None,
                    weight=1,
                    first_timestamp=stamp,
                    last_timestamp=stamp,
                    animal_class=class_name,
                    identity=class_name,
                    distance_km=camera_distance_km(cameras.get(str(source)), cameras.get(str(target))),
                    confidence=float(confidence) if confidence is not None else None,
                    observation_ids=[int(x) for x in (prev_obs, curr_obs) if x is not None],
                    detection_ids=[int(x) for x in (prev_det, curr_det) if x is not None],
                    source_observation_id=int(prev_obs) if prev_obs is not None else None,
                    destination_observation_id=int(curr_obs) if curr_obs is not None else None,
                    kind="observed",
                )
            else:
                edge = aggregates[key]
                edge.weight += 1
                edge.last_timestamp = stamp
                if curr_obs is not None:
                    edge.observation_ids.append(int(curr_obs))
                if curr_det is not None:
                    edge.detection_ids.append(int(curr_det))
                if confidence is not None:
                    edge.confidence = float(confidence)
        return list(aggregates.values())

    def build_movement_edges(
        self,
        tiger_id: str | None = None,
        animal_class: str | None = None,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> list[CameraEdge]:
        """Observed movement only. Predictions are never stored here."""
        wanted = (animal_class or "").strip().lower() or None
        edges: list[CameraEdge] = []
        if wanted in {None, "tiger"}:
            edges.extend(
                self._transition_edges(
                    tiger_id=tiger_id,
                    camera_id=camera_id,
                    time_from=time_from,
                    time_to=time_to,
                )
            )
        classes = MOVEMENT_CLASSES if wanted is None else (wanted,)
        for class_name in classes:
            if class_name == "tiger":
                continue
            edges.extend(
                self._class_transition_edges(
                    class_name,
                    camera_id=camera_id,
                    time_from=time_from,
                    time_to=time_to,
                )
            )
        return edges

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
                    "reid_confidence": event.reid_confidence,
                    "observation_id": event.observation_id,
                    "detection_id": event.detection_id,
                    "image_id": event.image_id,
                    "tiger_id": event.tiger_id or tiger_id,
                    "class_name": event.class_name,
                    "crop_path": event.crop_path,
                    "filename": event.filename,
                    "embedding_available": event.embedding_available,
                    "bbox_x": event.bbox_x,
                    "bbox_y": event.bbox_y,
                    "bbox_width": event.bbox_width,
                    "bbox_height": event.bbox_height,
                    "missing_coordinates": coords["missing_coordinates"],
                    "registered": coords["registered"],
                    "zone_type": coords["zone_type"],
                    "habitat": coords["habitat"],
                }
            )
        return route

    def station_occupancy(
        self,
        tiger_id: str | None = None,
        animal_class: str | None = None,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> dict[str, Any]:
        """Observed activity from stored detections. Not a statistical occupancy model."""
        cameras = self.cameras.list_all()
        time_clauses = []
        time_params: list[Any] = []
        if time_from:
            time_clauses.append("COALESCE(i.timestamp, d.created_at) >= ?")
            time_params.append(time_from)
        if time_to:
            time_clauses.append("COALESCE(i.timestamp, d.created_at) <= ?")
            time_params.append(time_to)
        time_sql = (" AND " + " AND ".join(time_clauses)) if time_clauses else ""
        camera_sql = " AND i.camera_id = ?" if camera_id else ""
        camera_params = [camera_id] if camera_id else []

        all_species = {
            str(row["camera_id"]): int(row["n"])
            for row in self.db.fetchall(
                f"""
                SELECT i.camera_id AS camera_id, COUNT(*) AS n
                FROM detections d
                JOIN images i ON i.image_id = d.image_id
                WHERE i.camera_id IS NOT NULL
                  AND d.review_status != 'ignored'
                  {time_sql}{camera_sql}
                GROUP BY i.camera_id
                """,
                (*time_params, *camera_params),
            )
        }
        class_counts: dict[str, dict[str, int]] = {}
        for row in self.db.fetchall(
            f"""
            SELECT i.camera_id AS camera_id,
                   LOWER(COALESCE(d.final_class_name, d.class_name)) AS class_name,
                   COUNT(*) AS n
            FROM detections d
            JOIN images i ON i.image_id = d.image_id
            WHERE i.camera_id IS NOT NULL
              AND d.review_status != 'ignored'
              {time_sql}{camera_sql}
            GROUP BY i.camera_id, LOWER(COALESCE(d.final_class_name, d.class_name))
            """,
            (*time_params, *camera_params),
        ):
            cid = str(row["camera_id"])
            bucket = class_counts.setdefault(
                cid, {"tiger": 0, "prey": 0, "rival": 0, "human": 0}
            )
            name = str(row["class_name"] or "")
            if name in bucket:
                bucket[name] += int(row["n"])
        tiger_obs_time = []
        tiger_obs_params: list[Any] = []
        if time_from:
            tiger_obs_time.append("COALESCE(o.timestamp, i.timestamp, o.created_at) >= ?")
            tiger_obs_params.append(time_from)
        if time_to:
            tiger_obs_time.append("COALESCE(o.timestamp, i.timestamp, o.created_at) <= ?")
            tiger_obs_params.append(time_to)
        tiger_time_sql = (" AND " + " AND ".join(tiger_obs_time)) if tiger_obs_time else ""
        tiger_camera_sql = " AND i.camera_id = ?" if camera_id else ""
        tiger_camera_params = [camera_id] if camera_id else []
        tiger_captures = {
            str(row["camera_id"]): int(row["n"])
            for row in self.db.fetchall(
                f"""
                SELECT i.camera_id AS camera_id, COUNT(*) AS n
                FROM tiger_observations o
                JOIN detections d ON d.detection_id = o.detection_id
                JOIN images i ON i.image_id = d.image_id
                WHERE i.camera_id IS NOT NULL
                  {tiger_time_sql}{tiger_camera_sql}
                GROUP BY i.camera_id
                """,
                (*tiger_obs_params, *tiger_camera_params),
            )
        }
        unique_tigers = {
            str(row["camera_id"]): int(row["n"])
            for row in self.db.fetchall(
                f"""
                SELECT i.camera_id AS camera_id, COUNT(DISTINCT o.tiger_id) AS n
                FROM tiger_observations o
                JOIN detections d ON d.detection_id = o.detection_id
                JOIN images i ON i.image_id = d.image_id
                WHERE i.camera_id IS NOT NULL
                  AND o.tiger_id IS NOT NULL
                  {tiger_time_sql}{tiger_camera_sql}
                GROUP BY i.camera_id
                """,
                (*tiger_obs_params, *tiger_camera_params),
            )
        }
        selected_captures: dict[str, int] = {}
        if tiger_id:
            selected_captures = {
                str(row["camera_id"]): int(row["n"])
                for row in self.db.fetchall(
                    f"""
                    SELECT i.camera_id AS camera_id, COUNT(*) AS n
                    FROM tiger_observations o
                    JOIN detections d ON d.detection_id = o.detection_id
                    JOIN images i ON i.image_id = d.image_id
                    WHERE i.camera_id IS NOT NULL
                      AND o.tiger_id = ?
                      {tiger_time_sql}{tiger_camera_sql}
                    GROUP BY i.camera_id
                    """,
                    (tiger_id, *tiger_obs_params, *tiger_camera_params),
                )
            }

        latest: dict[str, dict[str, Any]] = {}
        first_last: dict[str, dict[str, Any]] = {}
        for row in self.db.fetchall(
            f"""
            SELECT i.camera_id AS camera_id,
                   o.tiger_id AS tiger_id,
                   COALESCE(o.timestamp, i.timestamp, o.created_at) AS ts
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE i.camera_id IS NOT NULL
              {tiger_time_sql}{tiger_camera_sql}
            ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at)
            """,
            (*tiger_obs_params, *tiger_camera_params),
        ):
            obs_camera_id = str(row["camera_id"])
            stamp = row["ts"]
            bucket = first_last.setdefault(obs_camera_id, {"first": stamp, "last": stamp, "count": 0})
            bucket["last"] = stamp
            bucket["count"] += 1
            if row["tiger_id"]:
                latest[obs_camera_id] = {
                    "tiger_id": str(row["tiger_id"]),
                    "timestamp": stamp,
                }

        max_all = max(all_species.values(), default=0)
        max_tiger = max(tiger_captures.values(), default=0)
        max_selected = max(selected_captures.values(), default=0)
        max_prey = max((counts.get("prey", 0) for counts in class_counts.values()), default=0)
        max_rival = max((counts.get("rival", 0) for counts in class_counts.values()), default=0)
        max_human = max((counts.get("human", 0) for counts in class_counts.values()), default=0)

        stations: list[dict[str, Any]] = []
        for camera in cameras:
            cid = str(camera["camera_id"])
            if camera_id and cid != camera_id:
                continue
            coords = self.camera_coord_payload(camera, cid)
            span = first_last.get(cid)
            frequency = None
            span_days = None
            if span and span["first"] and span["last"] and span["count"] > 1:
                span_days = _timestamp_span_days(span["first"], span["last"])
                if span_days and span_days > 0:
                    frequency = span["count"] / span_days
            counts = class_counts.get(cid, {"tiger": 0, "prey": 0, "rival": 0, "human": 0})
            enabled = camera.get("enabled") not in {0, "0", False}
            stations.append(
                {
                    **coords,
                    "name": camera.get("name") or cid,
                    "enabled": enabled,
                    "status": "enabled" if enabled else "disabled",
                    "all_species_detections": all_species.get(cid, 0),
                    "tiger_captures": tiger_captures.get(cid, 0),
                    "tiger_detections": counts["tiger"],
                    "prey_detections": counts["prey"],
                    "rival_detections": counts["rival"],
                    "human_detections": counts["human"],
                    "unique_tigers": unique_tigers.get(cid, 0),
                    "selected_tiger_captures": selected_captures.get(cid, 0),
                    "latest_tiger_id": latest.get(cid, {}).get("tiger_id"),
                    "latest_tiger_timestamp": latest.get(cid, {}).get("timestamp"),
                    "occupancy_level_all_species": occupancy_level(
                        all_species.get(cid, 0), max_all
                    ),
                    "occupancy_level_tiger": occupancy_level(
                        tiger_captures.get(cid, 0), max_tiger
                    ),
                    "occupancy_level_prey": occupancy_level(counts["prey"], max_prey),
                    "occupancy_level_rival": occupancy_level(counts["rival"], max_rival),
                    "occupancy_level_human": occupancy_level(counts["human"], max_human),
                    "occupancy_level_selected_tiger": occupancy_level(
                        selected_captures.get(cid, 0), max_selected
                    ),
                    "capture_frequency_per_day": frequency,
                    "capture_span_days": span_days,
                }
            )

        return {
            "label": OCCUPANCY_LABEL,
            "stations": stations,
            "supported_modes": [
                "all_species",
                "tiger",
                "prey",
                "rival",
                "human",
                "selected_tiger",
            ],
            "selected_tiger_id": tiger_id,
            "animal_class": animal_class,
            "camera_id": camera_id,
            "time_from": time_from,
            "time_to": time_to,
        }

    def build_activity_area(
        self,
        tiger_id: str,
        observed_route: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Per-camera observation intensity. Not a territory or home range."""
        route = observed_route if observed_route is not None else self.build_observed_route(tiger_id)
        cameras = self.camera_lookup()
        counts: dict[str, int] = {}
        last_seen: dict[str, str | None] = {}
        for stop in route:
            camera_id = str(stop["camera_id"])
            counts[camera_id] = counts.get(camera_id, 0) + 1
            last_seen[camera_id] = stop.get("timestamp")
        maximum = max(counts.values(), default=0)
        ranked: list[dict[str, Any]] = []
        for camera_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            coords = self.camera_coord_payload(cameras.get(camera_id), camera_id)
            ranked.append(
                {
                    "camera_id": camera_id,
                    "observation_count": count,
                    "intensity": (count / maximum) if maximum else 0.0,
                    "last_seen": last_seen.get(camera_id),
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                    "missing_coordinates": coords["missing_coordinates"],
                    "registered": coords["registered"],
                }
            )
        strongest = ranked[0] if ranked else None
        hull = self.estimate_home_range(route)
        return {
            "label": ACTIVITY_AREA_LABEL,
            "tiger_id": tiger_id,
            "cameras": ranked,
            "strongest_camera": strongest["camera_id"] if strongest else None,
            "strongest_count": strongest["observation_count"] if strongest else 0,
            "region": {
                "available": hull["available"],
                "reason": hull["reason"],
                "label": ACTIVITY_AREA_LABEL,
                "polygon": hull["polygon"],
                "point_count": hull["point_count"],
                "unique_stations": hull["unique_stations"],
            },
        }

    def build_wildlife_entities(
        self,
        tiger_id: str | None = None,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> dict[str, Any]:
        """Individual tigers plus class-level prey/rival/human camera nodes."""
        cameras = self.camera_lookup()
        tigers: list[dict[str, Any]] = []
        wanted = (tiger_id or "").strip() or None
        for tiger in self.db.fetchall("SELECT tiger_id FROM tigers ORDER BY tiger_id"):
            identity = str(tiger["tiger_id"])
            if wanted and identity != wanted:
                continue
            full_route = self.build_observed_route(identity)
            route = full_route
            if camera_id:
                route = [stop for stop in route if stop.get("camera_id") == camera_id]
            if time_from or time_to:
                route = [
                    stop
                    for stop in route
                    if _in_time_range(stop.get("timestamp"), time_from, time_to)
                ]
            last = route[-1] if route else None
            activity = self.build_activity_area(identity, full_route)
            last_camera = last["camera_id"] if last else None
            coords = (
                self.camera_coord_payload(cameras.get(str(last_camera)), str(last_camera))
                if last_camera
                else {
                    "latitude": None,
                    "longitude": None,
                    "missing_coordinates": True,
                    "registered": False,
                }
            )
            tigers.append(
                {
                    "tiger_id": identity,
                    "last_camera": last_camera,
                    "last_seen": last["timestamp"] if last else None,
                    "observation_count": len(full_route),
                    "cameras_visited": list(
                        dict.fromkeys(stop["camera_id"] for stop in full_route)
                    ),
                    "most_frequent_camera": activity.get("strongest_camera"),
                    "most_frequent_count": activity.get("strongest_count") or 0,
                    "latitude": coords.get("latitude"),
                    "longitude": coords.get("longitude"),
                    "missing_coordinates": coords.get("missing_coordinates", True),
                    "registered": coords.get("registered", False),
                    "activity_area": activity,
                }
            )

        class_nodes: dict[str, list[dict[str, Any]]] = {
            "prey": [],
            "rival": [],
            "human": [],
        }
        for class_name in class_nodes:
            rows = self.detections.list_for_movement(
                animal_class=class_name,
                camera_id=camera_id,
                time_from=time_from,
                time_to=time_to,
            )
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                cid = row.get("camera_id")
                if not cid:
                    continue
                grouped[str(cid)].append(row)
            for cid, items in grouped.items():
                items.sort(key=lambda item: item.get("timestamp") or "")
                latest = items[-1]
                coords = self.camera_coord_payload(cameras.get(cid), cid)
                class_nodes[class_name].append(
                    {
                        "animal_class": class_name,
                        "camera_id": cid,
                        "detection_count": len(items),
                        "last_seen": latest.get("timestamp"),
                        "confidence": latest.get("confidence"),
                        "detection_id": latest.get("detection_id"),
                        "image_id": latest.get("image_id"),
                        "latitude": coords["latitude"],
                        "longitude": coords["longitude"],
                        "missing_coordinates": coords["missing_coordinates"],
                        "registered": coords["registered"],
                    }
                )

        return {
            "tigers": tigers,
            "prey": class_nodes["prey"],
            "rival": class_nodes["rival"],
            "human": class_nodes["human"],
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
        activity = self.build_activity_area(tiger_id, observed)
        return {
            "tiger_id": tiger_id,
            "observed_route": observed,
            "current_station": current,
            "predictions": self.attach_prediction_coordinates(raw_prediction),
            "occupancy": self.station_occupancy(tiger_id=tiger_id),
            "home_range": self.estimate_home_range(observed),
            "activity_area": activity,
            "visited_stations": visited,
            "observation_count": len(observed),
            "last_observed_station": current["camera_id"] if current else None,
            "last_observed_timestamp": current["timestamp"] if current else None,
            "most_frequent_camera": activity.get("strongest_camera"),
            "most_frequent_count": activity.get("strongest_count") or 0,
        }

    def export_payload(
        self,
        tiger_id: str | None = None,
        animal_class: str | None = None,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> dict:
        """Stable JSON the frontend (and later the GNN) can consume."""
        camera_graph = self.build_camera_graph(
            tiger_id=tiger_id,
            animal_class=animal_class,
            camera_id=camera_id,
            time_from=time_from,
            time_to=time_to,
        )
        observation_graph = self.build_tiger_observation_graph(tiger_id=tiger_id)
        movement_edges = self.build_movement_edges(
            tiger_id=tiger_id,
            animal_class=animal_class,
            camera_id=camera_id,
            time_from=time_from,
            time_to=time_to,
        )
        return {
            "camera_graph": camera_graph.to_dict(),
            "observation_graph": observation_graph.to_dict(),
            "movement_edges": [edge.to_dict() for edge in movement_edges],
            "occupancy": self.station_occupancy(
                tiger_id=tiger_id,
                animal_class=animal_class,
                camera_id=camera_id,
                time_from=time_from,
                time_to=time_to,
            ),
            "wildlife_entities": self.build_wildlife_entities(
                tiger_id=tiger_id,
                camera_id=camera_id,
                time_from=time_from,
                time_to=time_to,
            ),
            "filters": {
                "tiger_id": tiger_id,
                "animal_class": animal_class,
                "camera_id": camera_id,
                "time_from": time_from,
                "time_to": time_to,
            },
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
