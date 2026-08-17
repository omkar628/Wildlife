"""Human-confirmed local field-tiger identity.

Creates and assigns T001, T002, ... IDs. Never invents an identity and
never writes ATRW gallery IDs. MegaDescriptor may auto-assign only when
the configured high-confidence rule is satisfied.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import (
    ObservationRepository,
    SuggestionRepository,
    TigerRepository,
)
from backend.reid.gallery import ENCODER_DISABLED, LocalTigerGallery
from backend.reid.matching import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MIN_MARGIN,
    DEFAULT_REVIEW_THRESHOLD,
    decide_match,
    is_atrw_numeric_id,
    is_local_tiger_id,
)

logger = logging.getLogger(__name__)


class LocalIdentityService:
    def __init__(
        self,
        db: Database,
        settings: Settings | None = None,
        encoder: Any | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.encoder = encoder
        match_threshold = (
            settings.reid_match_threshold if settings is not None else DEFAULT_MATCH_THRESHOLD
        )
        review_threshold = (
            settings.reid_review_threshold if settings is not None else DEFAULT_REVIEW_THRESHOLD
        )
        min_margin = settings.reid_min_margin if settings is not None else DEFAULT_MIN_MARGIN
        model_name = settings.reid_model_id if settings is not None else "BVRA/MegaDescriptor-S-224"
        self.observations = ObservationRepository(db)
        self.tigers = TigerRepository(db)
        self.suggestions = SuggestionRepository(db)
        self.gallery = LocalTigerGallery(
            db,
            encoder=encoder,
            match_threshold=match_threshold,
            review_threshold=review_threshold,
            min_margin=min_margin,
            model_name=model_name,
        )
        self.match_threshold = match_threshold
        self.review_threshold = review_threshold
        self.min_margin = min_margin
        self._rematching = False

    def status(self) -> dict[str, Any]:
        encoder_on = self.gallery.encoder_enabled()
        return {
            "mode": "megadescriptor_local" if encoder_on else "human_confirmation",
            "encoder_enabled": encoder_on,
            "encoder_reason": None if encoder_on else ENCODER_DISABLED,
            "assigns_atrw_ids": False,
            "id_namespace": "T001+",
            "match_threshold": self.match_threshold,
            "review_threshold": self.review_threshold,
            "min_margin": self.min_margin,
            "threshold_note": (
                "Auto-assign existing T00X above reid_match_threshold when the "
                "top-1 / top-2 gap is at least reid_min_margin. Auto-create a "
                "new T00X only for accepted tigers with no close gallery match. "
                "Uncertain cases stay in Human Review."
            ),
        }

    def unidentified(self, limit: int = 40) -> list[dict[str, Any]]:
        self.rematch_unidentified(write_identity=True)
        rows = self.observations.list_unidentified(limit=limit)
        enriched: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            suggestion = self.suggestions.get(int(row["observation_id"]))
            payload["reid"] = _suggestion_payload(suggestion)
            enriched.append(payload)
        return enriched

    def unidentified_count(self) -> int:
        return self.observations.unidentified_count()

    def next_local_id(self) -> str:
        highest = 0
        for tiger in self.tigers.list_all():
            if is_local_tiger_id(str(tiger["tiger_id"])):
                match = str(tiger["tiger_id"])
                highest = max(highest, int(match[1:]))
        return f"T{highest + 1:03d}"

    def rematch_unidentified(self, *, write_identity: bool = True) -> None:
        """Re-score pending accepted-tiger crops against the current gallery."""
        if self._rematching:
            return
        self._rematching = True
        try:
            self.gallery.ensure_reference_embeddings()
            for row in self.observations.list_unidentified(limit=200):
                try:
                    self.identify_new_observation(
                        int(row["observation_id"]),
                        write_identity=write_identity,
                    )
                except Exception:
                    logger.exception(
                        "Re-match failed for observation %s",
                        row.get("observation_id"),
                    )
        finally:
            self._rematching = False

    def identify_new_observation(
        self,
        observation_id: int,
        *,
        write_identity: bool = True,
    ) -> dict[str, Any]:
        """Run MegaDescriptor after a crop exists. Failures leave tiger_id NULL."""
        observation = self.observations.get_joined(observation_id)
        if observation is None:
            raise KeyError(f"Observation {observation_id} was not found.")
        if observation.get("tiger_id"):
            return {
                "observation_id": observation_id,
                "matched": True,
                "tiger_id": observation.get("tiger_id"),
                "needs_review": False,
                "reason": "Observation already has a local tiger ID.",
            }

        self.gallery.ensure_reference_embeddings()
        crop_path = observation.get("crop_path")
        try:
            vector = self.gallery.embed_and_store(observation_id, crop_path)
        except Exception as exc:
            decision = {
                "matched": False,
                "tiger_id": None,
                "similarity": None,
                "candidates": [],
                "needs_review": True,
                "decision": "unknown",
                "suggested_tiger_id": None,
                "reason": f"Re-ID inference failed: {exc}",
            }
            self.suggestions.upsert(observation_id, decision)
            return decision

        if vector is None:
            decision = {
                "matched": False,
                "tiger_id": None,
                "similarity": None,
                "candidates": [],
                "needs_review": True,
                "decision": "unknown",
                "suggested_tiger_id": None,
                "reason": ENCODER_DISABLED if not self.gallery.encoder_enabled() else "No crop available for Re-ID.",
            }
            self.suggestions.upsert(observation_id, decision)
            return decision

        result = decide_match(
            vector,
            self.gallery.embeddings.list_identified(),
            match_threshold=self.match_threshold,
            review_threshold=self.review_threshold,
            min_margin=self.min_margin,
        )
        payload = result.to_dict()
        if result.suggested_tiger_id and is_atrw_numeric_id(result.suggested_tiger_id):
            payload["matched"] = False
            payload["tiger_id"] = None
            payload["needs_review"] = True
            payload["decision"] = "unknown"
            payload["suggested_tiger_id"] = None
            payload["reason"] = "ATRW gallery IDs cannot be assigned to field tigers."
            self.suggestions.upsert(observation_id, payload)
            return payload

        accepted_tiger = _detection_is_accepted_tiger(observation)
        if write_identity and accepted_tiger and result.decision == "accept":
            if result.tiger_id and is_local_tiger_id(result.tiger_id):
                self._commit_identity(observation, result.tiger_id, result.similarity, crop_path)
                payload["matched"] = True
                payload["tiger_id"] = result.tiger_id
                payload["needs_review"] = False
                self.suggestions.upsert(observation_id, payload)
                self.rematch_unidentified(write_identity=True)
                return payload
        if write_identity and accepted_tiger and result.decision == "new":
            assigned = self.next_local_id()
            self._commit_identity(observation, assigned, result.similarity, crop_path)
            payload["matched"] = True
            payload["tiger_id"] = assigned
            payload["suggested_tiger_id"] = assigned
            payload["needs_review"] = False
            payload["decision"] = "create"
            payload["reason"] = f"Created new local tiger {assigned}."
            self.suggestions.upsert(observation_id, payload)
            self.rematch_unidentified(write_identity=True)
            return payload

        payload["tiger_id"] = None
        payload["matched"] = False
        payload["needs_review"] = True
        if not accepted_tiger and result.decision in {"accept", "new"}:
            payload["decision"] = "review"
            payload["reason"] = (
                "Tiger class is still in review; identity is not assigned yet."
            )
        self.suggestions.upsert(observation_id, payload)
        return payload

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
        if choice == "keep":
            self.suggestions.mark_deferred(observation_id)
            return {
                "observation": observation,
                "tiger_id": None,
                "created": False,
                "kept_unidentified": True,
                "references": [],
            }
        if choice == "create":
            assigned = self.next_local_id()
        elif choice == "assign":
            assigned = self._require_existing_local_id(tiger_id)
        else:
            raise ValueError("action must be 'assign', 'create', or 'keep'.")

        seen_at = observation.get("timestamp") or observation.get("image_timestamp")
        self._commit_identity(observation, assigned, None, observation.get("crop_path"))
        self.observations.mark_human_verified(observation_id, True)
        self.rematch_unidentified(write_identity=True)
        self.suggestions.upsert(
            observation_id,
            {
                "matched": True,
                "suggested_tiger_id": assigned,
                "similarity": None,
                "needs_review": False,
                "decision": "human",
                "candidates": [],
                "reason": "Assigned by a reviewer.",
                "deferred": False,
            },
        )

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
        if is_atrw_numeric_id(identity):
            raise ValueError(
                "ATRW gallery IDs cannot be assigned to field tigers. "
                "Use a local ID such as T001."
            )
        if not is_local_tiger_id(identity):
            raise ValueError("Field tiger IDs must look like T001, T002, …")
        if self.tigers.get(identity) is None:
            raise KeyError(
                f"{identity} is not a known field tiger. Create a new tiger first."
            )
        return identity

    def _commit_identity(
        self,
        observation: dict[str, Any],
        tiger_id: str,
        similarity: float | None,
        crop_path: Any,
    ) -> None:
        seen_at = observation.get("timestamp") or observation.get("image_timestamp")
        self.tigers.upsert_seen(tiger_id, seen_at)
        self.observations.set_identity(int(observation["observation_id"]), tiger_id, similarity)
        self.gallery.attach_to_tiger(int(observation["observation_id"]), tiger_id, crop_path)


def _detection_is_accepted_tiger(observation: dict[str, Any]) -> bool:
    """Auto-assign only after YOLO class is accepted or human-confirmed as tiger."""
    if int(observation.get("accepted") or 0) != 1:
        return False
    name = observation.get("final_class_name") or observation.get("class_name")
    return str(name or "").strip().lower() == "tiger"


def _suggestion_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    candidates = row.get("candidates")
    if isinstance(candidates, str):
        try:
            candidates = json.loads(candidates)
        except (TypeError, ValueError):
            candidates = []
    return {
        "matched": bool(row.get("matched")),
        "tiger_id": row.get("suggested_tiger_id") if row.get("matched") else None,
        "suggested_tiger_id": row.get("suggested_tiger_id"),
        "similarity": row.get("similarity"),
        "needs_review": bool(row.get("needs_review", 1)),
        "decision": row.get("decision"),
        "candidates": candidates or [],
        "reason": row.get("reason"),
        "deferred": bool(row.get("deferred")),
    }
