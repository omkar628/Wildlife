"""Confidence filtering. Thresholds come from settings, never hard-coded."""

from __future__ import annotations

from dataclasses import dataclass

from backend.detector.parser import ParsedDetection


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    accepted: bool
    needs_review: bool
    reason: str


def filter_detection(
    detection: ParsedDetection,
    *,
    auto_accept: float,
    detect_min: float,
) -> FilterDecision:
    """Classify a detection against configurable thresholds.

    - confidence < detect_min     → drop
    - detect_min <= conf < auto   → keep + human review
    - confidence >= auto_accept   → keep + auto-accept
    """
    if detect_min < 0 or auto_accept < 0:
        raise ValueError("Confidence thresholds must be non-negative.")
    if detect_min > 1 or auto_accept > 1:
        raise ValueError("Confidence thresholds must be at most 1.0.")

    confidence = detection.confidence
    if confidence < detect_min:
        return FilterDecision(keep=False, accepted=False, needs_review=False, reason="below_detect_min")
    if confidence < auto_accept:
        return FilterDecision(keep=True, accepted=False, needs_review=True, reason="below_auto_accept")
    return FilterDecision(keep=True, accepted=True, needs_review=False, reason="auto_accepted")
