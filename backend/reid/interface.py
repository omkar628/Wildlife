"""Tiger Re-ID contract.

The trained assets exist, but this repository does not contain the original
inference code (model class, preprocessing, FAISS search threshold).

Do not invent those details. Callers should depend on this interface so a
real backend can be dropped in later without touching YOLO, SQLite, or the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ReIDMatch:
    tiger_id: str
    score: float
    gallery_index: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ReIDResult:
    available: bool
    tiger_id: str | None
    confidence: float | None
    matches: list[ReIDMatch] = field(default_factory=list)
    reason: str = ""
    matched: bool = False
    needs_review: bool = True
    decision: str = ""
    suggested_tiger_id: str | None = None


@runtime_checkable
class TigerReIDBackend(Protocol):
    def is_available(self) -> bool:
        ...

    def status(self) -> dict:
        ...

    def identify_crop(self, crop_path: Path) -> ReIDResult:
        ...
