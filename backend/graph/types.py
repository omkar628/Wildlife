"""Graph-ready structures for the frontend and a future GNN.

The backend emits events. The frontend owns animation.
The GNN, when it arrives, should consume these structures rather than
calling YOLO or reading images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ObservationEvent:
    tiger_id: str | None
    camera_id: str | None
    timestamp: str | None
    confidence: float | None
    detection_id: int
    observation_id: int
    class_name: str | None = None
    crop_path: str | None = None
    image_id: int | None = None
    reid_confidence: float | None = None
    embedding_available: bool = False
    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None
    filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraNode:
    camera_id: str
    latitude: float | None
    longitude: float | None
    elevation: float | None
    habitat: str | None
    observation_count: int
    image_count: int
    name: str | None = None
    enabled: bool = True
    status: str = "enabled"
    tiger_count: int = 0
    prey_count: int = 0
    rival_count: int = 0
    human_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraEdge:
    source: str
    target: str
    tiger_id: str | None
    weight: int
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    animal_class: str | None = None
    identity: str | None = None
    distance_km: float | None = None
    confidence: float | None = None
    observation_ids: list[int] = field(default_factory=list)
    detection_ids: list[int] = field(default_factory=list)
    source_observation_id: int | None = None
    destination_observation_id: int | None = None
    kind: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraGraph:
    nodes: list[CameraNode] = field(default_factory=list)
    edges: list[CameraEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass
class TigerObservationGraph:
    events: list[ObservationEvent] = field(default_factory=list)
    cameras: list[CameraNode] = field(default_factory=list)
    edges: list[CameraEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "cameras": [node.to_dict() for node in self.cameras],
            "edges": [edge.to_dict() for edge in self.edges],
        }
