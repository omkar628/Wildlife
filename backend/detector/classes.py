"""Canonical detector classes for the trained YOLO11n model."""

from __future__ import annotations

CLASS_ID_TO_NAME: dict[int, str] = {
    0: "tiger",
    1: "prey",
    2: "rival",
    3: "human",
}

CLASS_NAME_TO_ID: dict[str, int] = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}

REVIEW_CHOICES = ("tiger", "prey", "rival", "human", "other", "ignore")
