"""Human-confirmed local field-tiger identity.

Creates and assigns T001, T002, ... IDs. Never invents an identity,
never writes ATRW gallery IDs, and never runs the disabled encoder.
"""

from __future__ import annotations

import re
from typing import Any

from backend.database.connection import Database
from backend.database.repositories import (
    ObservationRepository,
    TigerRepository,
)
from backend.reid.gallery import ENCODER_DISABLED, LocalTigerGallery

LOCAL_ID_PATTERN = re.compile(r"^T(\d{3,})$")
ATRW_NUMERIC = re.compile(r"^\d+$")


class LocalIdentityService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.observations = ObservationRepository(db)
        self.tigers = TigerRepository(db)
        self.gallery = LocalTigerGallery(db)

    def status(self) -> dict[str, Any]:
        return {
            "mode": "human_confirmation",
            "encoder_enabled": False,
            "encoder_reason": ENCODER_DISABLED,
            "assigns_atrw_ids": False,
            "id_namespace": "T001+",
        }

    def unidentified(self, limit: int = 40) -> list[dict[str, Any]]:
        return self.observations.list_unidentified(limit=limit)

    def unidentified_count(self) -> int:
        return self.observations.unidentified_count()

    def next_local_id(self) -> str:
        highest = 0
        for tiger in self.tigers.list_all():
            match = LOCAL_ID_PATTERN.fullmatch(str(tiger["tiger_id"]))
            if match:
                highest = max(highest, int(match.group(1)))
        return f"T{highest + 1:03d}"

    def assign(
        self,
        observation_id: int,
        *,
        action: str,
        tiger_id: str | None = None,
    ) -> dict[str, Any]:
        observation = self.observations.get_joined(observation_id)
        if observation is None:
            raise KeyError(f"Observation {observation_id} was not found.")

        choice = action.strip().lower()
        if choice == "create":
            assigned = self.next_local_id()
        elif choice == "assign":
            assigned = self._require_existing_local_id(tiger_id)
        else:
            raise ValueError("action must be 'assign' or 'create'.")

        seen_at = observation.get("timestamp") or observation.get("image_timestamp")
        self.tigers.upsert_seen(assigned, seen_at)
        self.observations.set_identity(observation_id, assigned, None)
        self.observations.mark_human_verified(observation_id, True)

        updated = self.observations.get_joined(observation_id)
        assert updated is not None
        return {
            "observation": updated,
            "tiger_id": assigned,
            "created": choice == "create",
            "references": self.gallery.references_for(assigned),
        }

    def _require_existing_local_id(self, tiger_id: str | None) -> str:
        if tiger_id is None or not str(tiger_id).strip():
            raise ValueError("tiger_id is required when action is 'assign'.")
        identity = str(tiger_id).strip()
        if ATRW_NUMERIC.fullmatch(identity):
            raise ValueError(
                "ATRW gallery IDs cannot be assigned to field tigers. "
                "Use a local ID such as T001."
            )
        if not LOCAL_ID_PATTERN.fullmatch(identity):
            raise ValueError("Field tiger IDs must look like T001, T002, …")
        if self.tigers.get(identity) is None:
            raise KeyError(
                f"{identity} is not a known field tiger. Create a new tiger first."
            )
        return identity
