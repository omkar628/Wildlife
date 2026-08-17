"""Local field-tiger gallery.

Confirmed and high-confidence local IDs (T001…) own embeddings here.
The ATRW FAISS index and numeric gallery IDs are never consulted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import PROJECT_ROOT
from backend.database.connection import Database
from backend.database.repositories import (
    EmbeddingRepository,
    ObservationRepository,
    SuggestionRepository,
    TigerRepository,
)
from backend.reid.interface import ReIDResult
from backend.reid.matching import decide_match, decode_embedding, is_local_tiger_id
from backend.reid.megadescriptor import DEFAULT_MODEL_ID, l2_normalize

logger = logging.getLogger(__name__)

ENCODER_DISABLED = (
    "Local encoder is disabled. Assign field tiger IDs by human confirmation."
)


class LocalTigerGallery:
    def __init__(
        self,
        db: Database,
        encoder: Any | None = None,
        *,
        match_threshold: float = 0.90,
        review_threshold: float = 0.55,
        min_margin: float = 0.05,
        model_name: str = DEFAULT_MODEL_ID,
    ) -> None:
        self.db = db
        self.observations = ObservationRepository(db)
        self.tigers = TigerRepository(db)
        self.embeddings = EmbeddingRepository(db)
        self.suggestions = SuggestionRepository(db)
        self.encoder = encoder
        self.match_threshold = match_threshold
        self.review_threshold = review_threshold
        self.min_margin = min_margin
        self.model_name = model_name

    def encoder_enabled(self) -> bool:
        return bool(self.encoder is not None and getattr(self.encoder, "is_available", lambda: False)())

    def identify_crop(self, crop_path: Path) -> ReIDResult:
        if not self.encoder_enabled():
            return ReIDResult(
                available=False,
                tiger_id=None,
                confidence=None,
                matches=[],
                reason=ENCODER_DISABLED,
                matched=False,
                needs_review=True,
                decision="unknown",
            )
        try:
            vector = self.encoder.embed_crop(Path(crop_path))
        except Exception as exc:
            return ReIDResult(
                available=False,
                tiger_id=None,
                confidence=None,
                matches=[],
                reason=f"MegaDescriptor inference failed: {exc}",
                matched=False,
                needs_review=True,
                decision="unknown",
            )
        self.ensure_reference_embeddings()
        decision = decide_match(
            vector,
            self.embeddings.list_identified(),
            match_threshold=self.match_threshold,
            review_threshold=self.review_threshold,
            min_margin=self.min_margin,
        )
        return decision.to_result(available=True)

    def resolve_crop(self, crop_path: Path | str | None) -> Path | None:
        if crop_path is None or str(crop_path).strip() == "":
            return None
        path = Path(crop_path)
        if path.is_file():
            return path
        if not path.is_absolute():
            alt = PROJECT_ROOT / path
            if alt.is_file():
                return alt
        return None

    def embed_and_store(self, observation_id: int, crop_path: Path | str | None) -> np.ndarray | None:
        existing = self.embeddings.get_by_observation(observation_id)
        if existing is not None:
            loaded = decode_embedding(existing["vector"])
            if loaded is not None:
                return loaded
        if not self.encoder_enabled():
            return None
        path = self.resolve_crop(crop_path)
        if path is None:
            logger.warning("No crop on disk for observation %s (%s)", observation_id, crop_path)
            return None
        vector = l2_normalize(self.encoder.embed_crop(path))
        self.embeddings.upsert(
            observation_id,
            np.ascontiguousarray(vector, dtype=np.float32).tobytes(),
            int(vector.shape[0]),
            self.model_name,
            tiger_id=None,
        )
        return vector

    def attach_to_tiger(self, observation_id: int, tiger_id: str, crop_path: Path | str | None = None) -> None:
        """Persist this observation's embedding as a local reference for tiger_id."""
        if not is_local_tiger_id(tiger_id):
            logger.warning("Refusing to attach non-local tiger id %s", tiger_id)
            return
        try:
            self.embed_and_store(observation_id, crop_path)
        except Exception:
            logger.exception(
                "Failed to embed observation %s for %s; identity is still saved",
                observation_id,
                tiger_id,
            )
        self.embeddings.set_tiger_id(observation_id, tiger_id)

    def ensure_reference_embeddings(self) -> int:
        """Embed identified tigers that have a crop but no stored vector."""
        if not self.encoder_enabled():
            return 0
        stored = 0
        for row in self.observations.list_identified():
            tiger_id = str(row.get("tiger_id") or "").strip()
            if not is_local_tiger_id(tiger_id):
                continue
            observation_id = int(row["observation_id"])
            try:
                self.attach_to_tiger(observation_id, tiger_id, row.get("crop_path"))
            except Exception:
                logger.exception("Could not backfill embedding for observation %s", observation_id)
                continue
            if self.embeddings.get_by_observation(observation_id) is not None:
                stored += 1
        return stored

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
                    "embedding_count": self.embeddings.count_for_tiger(tiger_id),
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
