"""Local field-tiger gallery.

Confirmed observations (tiger_id set by a human) are the only references.
The encoder is disabled. ATRW FAISS / metadata are never consulted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.database.connection import Database
from backend.database.repositories import ObservationRepository, TigerRepository
from backend.reid.interface import ReIDResult


ENCODER_DISABLED = (
    "Local encoder is disabled. Assign field tiger IDs by human confirmation."
)


class LocalTigerGallery:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.observations = ObservationRepository(db)
        self.tigers = TigerRepository(db)

    def encoder_enabled(self) -> bool:
        return False

    def identify_crop(self, crop_path: Path) -> ReIDResult:
        return ReIDResult(
            available=False,
            tiger_id=None,
            confidence=None,
            matches=[],
            reason=ENCODER_DISABLED,
        )

    def references_for(self, tiger_id: str, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.observations.list_for_tiger(tiger_id)
        refs: list[dict[str, Any]] = []
        for row in rows:
            if not row.get("crop_path"):
                continue
            refs.append(_reference_payload(row))
            if len(refs) >= limit:
                break
        return refs

    def catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for tiger in self.tigers.list_all():
            tiger_id = str(tiger["tiger_id"])
            refs = self.references_for(tiger_id, limit=4)
            catalog.append(
                {
                    "tiger_id": tiger_id,
                    "first_seen": tiger.get("first_seen"),
                    "last_seen": tiger.get("last_seen"),
                    "observation_count": tiger.get("observation_count"),
                    "references": refs,
                }
            )
        return catalog


def _reference_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_id": int(row["observation_id"]),
        "detection_id": int(row["detection_id"]),
        "tiger_id": row.get("tiger_id"),
        "camera_id": row.get("camera_id"),
        "timestamp": row.get("timestamp") or row.get("image_timestamp"),
        "crop_path": row.get("crop_path"),
        "filename": row.get("filename"),
        "human_verified": bool(row.get("human_verified")),
    }
