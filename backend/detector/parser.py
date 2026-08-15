"""Convert YOLO outputs into plain detection records.

This module does not load the model. Tests can feed synthetic boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ParsedDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        if values and isinstance(values[0], (list, tuple)):
            return [float(item) for item in values[0]]
        return [float(item) for item in values]
    return [float(values)]


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return x1, y1, width, height


def parse_yolo_result(
    boxes_xyxy: Iterable[Iterable[float]],
    class_ids: Iterable[Any],
    confidences: Iterable[Any],
    class_map: Mapping[int, str],
) -> list[ParsedDetection]:
    parsed: list[ParsedDetection] = []
    for xyxy, raw_class, raw_conf in zip(boxes_xyxy, class_ids, confidences):
        coords = [float(item) for item in xyxy]
        if len(coords) < 4:
            continue
        x, y, w, h = xyxy_to_xywh(coords[0], coords[1], coords[2], coords[3])
        class_id = int(raw_class)
        class_name = class_map.get(class_id, f"class_{class_id}")
        parsed.append(
            ParsedDetection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(raw_conf),
                bbox_x=x,
                bbox_y=y,
                bbox_width=w,
                bbox_height=h,
            )
        )
    return parsed


def parse_ultralytics_result(result: Any, class_map: Mapping[int, str]) -> list[ParsedDetection]:
    """Accept an Ultralytics ``Results`` object or a dict with the same fields."""
    if result is None:
        return []
    if isinstance(result, dict):
        boxes = result.get("xyxy") or result.get("boxes") or []
        class_ids = result.get("cls") or result.get("class_ids") or []
        confidences = result.get("conf") or result.get("confidences") or []
        return parse_yolo_result(boxes, class_ids, confidences, class_map)

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    if xyxy is None:
        return []
    xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
    cls_list = cls.tolist() if cls is not None and hasattr(cls, "tolist") else (list(cls) if cls is not None else [])
    conf_list = (
        conf.tolist() if conf is not None and hasattr(conf, "tolist") else (list(conf) if conf is not None else [])
    )
    return parse_yolo_result(xyxy_list, cls_list, conf_list, class_map)
