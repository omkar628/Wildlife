"""Generate in-app alerts from real detections, identities, and GNN output.

Alerts are persisted with a unique event_key so the same observation,
identity, or prediction cannot notify twice.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import (
    AlertRepository,
    CameraRepository,
    DetectionRepository,
    ObservationRepository,
    TigerRepository,
)
from backend.graph.builder import GraphService

logger = logging.getLogger(__name__)

CLASS_ALERTS = {
    "tiger": ("tiger_detected", "TIGER ALERT", "warning"),
    "prey": ("prey_detected", "PREY ALERT", "info"),
    "rival": ("rival_detected", "RIVAL ALERT", "warning"),
    "human": ("human_detected", "HUMAN ALERT", "critical"),
}


def _stamp_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown time"
    parsed = None
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.strftime("%d %b %Y, %H:%M")


def _percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{max(0, min(100, int(round(number * 100))))}%"


def _location_label(camera: dict[str, Any] | None, camera_id: str | None) -> str | None:
    if not camera_id:
        return None
    if camera is None:
        return str(camera_id)
    lat = camera.get("latitude")
    lon = camera.get("longitude")
    try:
        if lat is not None and lon is not None:
            return f"{camera_id} ({float(lat):.5f}, {float(lon):.5f})"
    except (TypeError, ValueError):
        pass
    return str(camera_id)


class AlertService:
    def __init__(self, db: Database, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings
        self.alerts = AlertRepository(db)
        self.detections = DetectionRepository(db)
        self.observations = ObservationRepository(db)
        self.tigers = TigerRepository(db)
        self.cameras = CameraRepository(db)
        self.graph = GraphService(db)

    @property
    def repeat_threshold(self) -> int:
        if self.settings is None:
            return 3
        return int(self.settings.alert_repeat_threshold)

    @property
    def unusual_min(self) -> int:
        if self.settings is None:
            return 4
        return int(self.settings.alert_unusual_min_detections)

    @property
    def confidence_floor(self) -> float:
        if self.settings is None:
            return 0.60
        return float(self.settings.confidence_auto_accept)

    def list_alerts(self, **filters: Any) -> list[dict[str, Any]]:
        return [self._serialize(row) for row in self.alerts.list_filtered(**filters)]

    def summary(self) -> dict[str, Any]:
        payload = self.alerts.summary()
        payload["recent"] = [self._serialize(row) for row in payload.get("recent") or []]
        return payload

    def mark_read(self, alert_id: int) -> dict[str, Any] | None:
        row = self.alerts.mark_read(alert_id)
        return self._serialize(row) if row else None

    def mark_all_read(self, **filters: Any) -> int:
        return self.alerts.mark_all_read(**filters)

    def clear(self, alert_id: int) -> dict[str, Any] | None:
        row = self.alerts.clear(alert_id)
        return self._serialize(row) if row else None

    def clear_filtered(self, **filters: Any) -> int:
        return self.alerts.clear_filtered(**filters)

    def from_detection(
        self,
        detection: dict[str, Any],
        *,
        tiger_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Alert for an accepted, above-threshold detection only."""
        if int(detection.get("accepted") or 0) != 1:
            return None
        class_name = str(
            detection.get("final_class_name") or detection.get("class_name") or ""
        ).strip().lower()
        if class_name not in CLASS_ALERTS:
            return None
        try:
            confidence = float(detection.get("confidence"))
        except (TypeError, ValueError):
            return None
        if confidence < self.confidence_floor:
            return None
        detection_id = int(detection["detection_id"])
        camera_id = detection.get("camera_id")
        timestamp = detection.get("timestamp") or detection.get("created_at")
        identity = (tiger_id or detection.get("tiger_id") or None)
        spec = CLASS_ALERTS[class_name]
        subject = identity or class_name.capitalize()
        camera_label = camera_id or "an unregistered camera"
        explanation = (
            f"{subject} detected at {camera_label}. "
            f"Confidence: {_percent(confidence)}. {_stamp_label(timestamp)}."
        )
        if identity and class_name == "tiger":
            explanation = (
                f"{identity} detected at {camera_label}\n"
                f"Confidence: {_percent(confidence)}\n"
                f"{_stamp_label(timestamp)}"
            )
        camera = self.cameras.get(str(camera_id)) if camera_id else None
        created = self.alerts.insert(
            alert_type=spec[0],
            severity=spec[2],
            title=spec[1],
            explanation=explanation,
            event_key=f"detection:{detection_id}:{class_name}",
            animal_class=class_name,
            tiger_id=identity,
            camera_id=str(camera_id) if camera_id else None,
            confidence=confidence,
            timestamp=str(timestamp) if timestamp else None,
            location=_location_label(camera, str(camera_id) if camera_id else None),
            source_table="detections",
            source_id=detection_id,
            observation_id=detection.get("observation_id"),
            detection_id=detection_id,
            image_id=detection.get("image_id"),
        )
        if created is None:
            return None
        return self.alerts.get(created)

    def from_identity(
        self,
        observation: dict[str, Any],
        tiger_id: str,
        *,
        created: bool = False,
    ) -> list[dict[str, Any]]:
        produced: list[dict[str, Any]] = []
        identity = (tiger_id or "").strip()
        if not identity:
            return produced
        detection = dict(observation)
        detection["tiger_id"] = identity
        detection["accepted"] = 1
        detection["final_class_name"] = "tiger"
        alert = self.from_detection(detection, tiger_id=identity)
        if alert:
            produced.append(alert)
        camera_id = observation.get("camera_id")
        timestamp = observation.get("timestamp") or observation.get("image_timestamp")
        camera = self.cameras.get(str(camera_id)) if camera_id else None
        if created or self._is_first_observation(identity, int(observation["observation_id"])):
            created_id = self.alerts.insert(
                alert_type="new_tiger",
                severity="warning",
                title="NEW TIGER IDENTITY",
                explanation=(
                    f"New local tiger {identity} confirmed"
                    + (f" at {camera_id}" if camera_id else "")
                    + f". {_stamp_label(timestamp)}"
                ),
                event_key=f"new_tiger:{identity}",
                animal_class="tiger",
                tiger_id=identity,
                camera_id=str(camera_id) if camera_id else None,
                confidence=observation.get("reid_confidence") or observation.get("confidence"),
                timestamp=str(timestamp) if timestamp else None,
                location=_location_label(camera, str(camera_id) if camera_id else None),
                source_table="tiger_observations",
                source_id=int(observation["observation_id"]),
                observation_id=int(observation["observation_id"]),
                detection_id=observation.get("detection_id"),
                image_id=observation.get("image_id"),
            )
            if created_id:
                found = self.alerts.get(created_id)
                if found:
                    produced.append(found)
        repeat = self._maybe_repeat(identity, str(camera_id) if camera_id else None)
        if repeat:
            produced.append(repeat)
        return produced

    def from_prediction(self, prediction: dict[str, Any]) -> dict[str, Any] | None:
        if not prediction.get("available"):
            return None
        tiger_id = str(prediction.get("tiger_id") or "").strip()
        predicted = str(prediction.get("predicted_camera_id") or "").strip()
        if not tiger_id or not predicted:
            return None
        history = prediction.get("history") or []
        last_obs = None
        last_camera = None
        if history:
            last = history[-1]
            last_obs = last.get("observation_id")
            last_camera = last.get("camera_id")
        if last_obs is None:
            route = self.graph.build_observed_route(tiger_id)
            if route:
                last_obs = route[-1].get("observation_id")
                last_camera = route[-1].get("camera_id")
        if last_obs is None:
            return None
        confidence = prediction.get("confidence")
        camera = self.cameras.get(predicted)
        created = self.alerts.insert(
            alert_type="gnn_prediction",
            severity="info",
            title="MOVEMENT PREDICTION",
            explanation=(
                f"{tiger_id} is predicted to move toward {predicted}\n"
                f"GNN confidence: {_percent(confidence)}"
                + (f"\nLast observed: {last_camera}" if last_camera else "")
            ),
            event_key=f"gnn:{tiger_id}:{last_obs}:{predicted}",
            animal_class="tiger",
            tiger_id=tiger_id,
            camera_id=predicted,
            confidence=float(confidence) if confidence is not None else None,
            timestamp=prediction.get("prediction_timestamp"),
            location=_location_label(camera, predicted),
            source_table="gnn",
            source_id=int(last_obs),
            observation_id=int(last_obs),
            metadata=json.dumps(
                {
                    "last_camera": last_camera,
                    "predicted_camera_id": predicted,
                    "kind": "prediction",
                }
            ),
        )
        return self.alerts.get(created) if created else None

    def sync_from_database(self) -> dict[str, int]:
        """Backfill alerts from stored detections/identities. Never invents events."""
        created = {
            "detections": 0,
            "identities": 0,
            "repeats": 0,
            "unusual": 0,
        }
        for row in self.detections.list_for_movement():
            alert = self.from_detection(row, tiger_id=row.get("tiger_id"))
            if alert:
                created["detections"] += 1
        seen: set[str] = set()
        for row in self.observations.list_identified():
            tiger_id = str(row.get("tiger_id") or "")
            if not tiger_id or tiger_id in seen:
                continue
            seen.add(tiger_id)
            produced = self.from_identity(row, tiger_id, created=True)
            created["identities"] += sum(1 for item in produced if item.get("alert_type") == "new_tiger")
        created["repeats"] = self.sync_repeats()
        created["unusual"] = self.sync_unusual()
        return created

    def sync_repeats(self) -> int:
        created = 0
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.observations.list_identified():
            tiger_id = row.get("tiger_id")
            camera_id = row.get("camera_id")
            if not tiger_id or not camera_id:
                continue
            grouped[(str(tiger_id), str(camera_id))].append(row)
        for (tiger_id, camera_id), items in grouped.items():
            if len(items) < self.repeat_threshold:
                continue
            items.sort(key=lambda item: item.get("timestamp") or item.get("image_timestamp") or "")
            latest = items[-1]
            camera = self.cameras.get(camera_id)
            alert_id = self.alerts.insert(
                alert_type="tiger_repeat",
                severity="info",
                title="TIGER REPEAT",
                explanation=(
                    f"{tiger_id} repeatedly appearing at {camera_id} "
                    f"({len(items)} observations). Last seen {_stamp_label(latest.get('timestamp') or latest.get('image_timestamp'))}."
                ),
                event_key=f"tiger_repeat:{tiger_id}:{camera_id}",
                animal_class="tiger",
                tiger_id=tiger_id,
                camera_id=camera_id,
                confidence=latest.get("reid_confidence") or latest.get("confidence"),
                timestamp=latest.get("timestamp") or latest.get("image_timestamp"),
                location=_location_label(camera, camera_id),
                source_table="tiger_observations",
                source_id=int(latest["observation_id"]),
                observation_id=int(latest["observation_id"]),
                detection_id=latest.get("detection_id"),
                image_id=latest.get("image_id"),
                metadata=json.dumps({"observation_count": len(items)}),
            )
            if alert_id:
                created += 1
        return created

    def sync_unusual(self) -> int:
        """Unusual activity from stored class counts. Requires mixed, high volume."""
        created = 0
        cameras = self.cameras.list_all()
        occupancy = self.graph.station_occupancy()
        by_id = {item["camera_id"]: item for item in occupancy["stations"]}
        for camera in cameras:
            camera_id = str(camera["camera_id"])
            station = by_id.get(camera_id)
            if not station:
                continue
            total = int(station.get("all_species_detections") or 0)
            if total < self.unusual_min:
                continue
            classes = {
                name: int(station.get(f"{name}_detections") or 0)
                for name in ("tiger", "prey", "rival", "human")
            }
            active = [name for name, count in classes.items() if count > 0]
            if len(active) < 2:
                continue
            created_id = self.alerts.insert(
                alert_type="unusual_activity",
                severity="warning",
                title="UNUSUAL ACTIVITY",
                explanation=(
                    f"Unusual wildlife activity at {camera_id}: "
                    f"{total} accepted detections across {', '.join(active)}."
                ),
                event_key=f"unusual:{camera_id}",
                animal_class=None,
                camera_id=camera_id,
                timestamp=station.get("latest_tiger_timestamp"),
                location=_location_label(camera, camera_id),
                source_table="detections",
                metadata=json.dumps({"counts": classes, "total": total}),
            )
            if created_id:
                created += 1
        return created

    def sync_predictions(self, gnn: Any) -> int:
        created = 0
        if gnn is None:
            return 0
        for tiger in self.tigers.list_all():
            try:
                prediction = gnn.predict_for_tiger(str(tiger["tiger_id"]))
            except Exception:
                logger.exception("GNN prediction alert failed for %s", tiger.get("tiger_id"))
                continue
            if self.from_prediction(prediction):
                created += 1
        return created

    def after_job(self, gnn: Any | None = None) -> dict[str, int]:
        counts = {
            "detections": 0,
            "identities": 0,
            "repeats": self.sync_repeats(),
            "unusual": self.sync_unusual(),
            "predictions": self.sync_predictions(gnn) if gnn is not None else 0,
        }
        return counts

    def _is_first_observation(self, tiger_id: str, observation_id: int) -> bool:
        rows = self.observations.list_for_tiger(tiger_id)
        if not rows:
            return True
        first = min(int(row["observation_id"]) for row in rows)
        return first == observation_id

    def _maybe_repeat(self, tiger_id: str, camera_id: str | None) -> dict[str, Any] | None:
        if not camera_id:
            return None
        rows = [
            row
            for row in self.observations.list_for_tiger(tiger_id)
            if row.get("camera_id") == camera_id
        ]
        if len(rows) < self.repeat_threshold:
            return None
        self.sync_repeats()
        found = self.alerts.get_by_event_key(f"tiger_repeat:{tiger_id}:{camera_id}")
        return found

    def _serialize(self, row: dict[str, Any] | None) -> dict[str, Any]:
        if row is None:
            return {}
        payload = dict(row)
        payload["read"] = bool(row.get("read"))
        payload["cleared"] = bool(row.get("cleared"))
        payload["stamp_label"] = _stamp_label(row.get("timestamp") or row.get("created_at"))
        metadata = row.get("metadata")
        if isinstance(metadata, str) and metadata:
            try:
                payload["metadata"] = json.loads(metadata)
            except (TypeError, ValueError):
                payload["metadata"] = {}
        return payload
