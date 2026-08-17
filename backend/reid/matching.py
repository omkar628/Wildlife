"""Local-gallery matching. Never assigns ATRW IDs or empty-gallery guesses."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.reid.interface import ReIDMatch, ReIDResult
from backend.reid.megadescriptor import l2_normalize


def decode_embedding(raw: Any) -> np.ndarray | None:
    """Load a stored float32 embedding. Accepts bytes, memoryview, or arrays."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        array = np.asarray(raw, dtype=np.float32).reshape(-1)
    else:
        try:
            array = np.frombuffer(bytes(raw), dtype=np.float32).copy()
        except (TypeError, ValueError):
            return None
    if array.size == 0:
        return None
    return l2_normalize(array)

LOCAL_ID_PATTERN = re.compile(r"^T(\d{3,})$")
ATRW_NUMERIC = re.compile(r"^\d+$")

# Provisional field-safety defaults. See config/settings.yaml.
# ATRW same-individual cosine reached 0.32 while different-individual
# cosine reached 0.70, so 0.70 is not a safe auto-accept on that benchmark.
DEFAULT_MATCH_THRESHOLD = 0.90
DEFAULT_REVIEW_THRESHOLD = 0.55
# Minimum cosine gap between top-1 and top-2 before auto-assign.
DEFAULT_MIN_MARGIN = 0.05


def is_local_tiger_id(tiger_id: str | None) -> bool:
    if tiger_id is None:
        return False
    return bool(LOCAL_ID_PATTERN.fullmatch(str(tiger_id).strip()))


def is_atrw_numeric_id(tiger_id: str | None) -> bool:
    if tiger_id is None:
        return False
    return bool(ATRW_NUMERIC.fullmatch(str(tiger_id).strip()))


@dataclass
class RankedTiger:
    tiger_id: str
    score: float
    support: int
    mean_score: float


@dataclass
class MatchDecision:
    matched: bool
    tiger_id: str | None
    similarity: float | None
    needs_review: bool
    decision: str
    suggested_tiger_id: str | None
    reason: str
    candidates: list[RankedTiger] = field(default_factory=list)

    def to_result(self, available: bool = True) -> ReIDResult:
        matches = [
            ReIDMatch(
                tiger_id=item.tiger_id,
                score=item.score,
                extra={"support": item.support, "mean_score": item.mean_score},
            )
            for item in self.candidates
        ]
        return ReIDResult(
            available=available,
            tiger_id=self.tiger_id if self.matched else None,
            confidence=self.similarity,
            matches=matches,
            reason=self.reason,
            matched=self.matched,
            needs_review=self.needs_review,
            decision=self.decision,
            suggested_tiger_id=self.suggested_tiger_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "tiger_id": self.tiger_id if self.matched else None,
            "similarity": self.similarity,
            "candidates": [
                {
                    "tiger_id": item.tiger_id,
                    "similarity": item.score,
                    "support": item.support,
                    "mean_similarity": item.mean_score,
                }
                for item in self.candidates
            ],
            "needs_review": self.needs_review,
            "decision": self.decision,
            "suggested_tiger_id": self.suggested_tiger_id,
            "reason": self.reason,
        }


def rank_local_tigers(
    query: np.ndarray,
    gallery: list[dict[str, Any]],
) -> list[RankedTiger]:
    """Aggregate image-level cosine scores by local tiger_id.

    Per-image argmax is not used for identity. Each tiger is scored by the
    best cosine among its stored embeddings, with mean score kept as context.
    """
    query_vec = l2_normalize(query)
    buckets: dict[str, list[float]] = {}
    for item in gallery:
        tiger_id = str(item.get("tiger_id") or "").strip()
        if not is_local_tiger_id(tiger_id) or is_atrw_numeric_id(tiger_id):
            continue
        gallery_vec = decode_embedding(item.get("vector"))
        if gallery_vec is None or gallery_vec.size != query_vec.size:
            continue
        score = float(np.clip(np.dot(query_vec, gallery_vec), -1.0, 1.0))
        if not math.isfinite(score):
            continue
        buckets.setdefault(tiger_id, []).append(score)

    ranked: list[RankedTiger] = []
    for tiger_id, scores in buckets.items():
        ranked.append(
            RankedTiger(
                tiger_id=tiger_id,
                score=float(max(scores)),
                support=len(scores),
                mean_score=float(sum(scores) / len(scores)),
            )
        )
    ranked.sort(key=lambda item: (-item.score, -item.support, item.tiger_id))
    return ranked


def decide_match(
    query: np.ndarray,
    gallery: list[dict[str, Any]],
    *,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> MatchDecision:
    ranked = rank_local_tigers(query, gallery)
    if not ranked:
        return MatchDecision(
            matched=False,
            tiger_id=None,
            similarity=None,
            needs_review=True,
            decision="new",
            suggested_tiger_id=None,
            reason="Local gallery has no usable embeddings; this may be a new tiger.",
            candidates=[],
        )

    top = ranked[0]
    if not is_local_tiger_id(top.tiger_id) or is_atrw_numeric_id(top.tiger_id):
        return MatchDecision(
            matched=False,
            tiger_id=None,
            similarity=top.score,
            needs_review=True,
            decision="unknown",
            suggested_tiger_id=None,
            reason="Top candidate is not a local T001+ field ID.",
            candidates=ranked,
        )

    second_score = ranked[1].score if len(ranked) > 1 else None
    margin = None if second_score is None else top.score - second_score

    if top.score >= match_threshold:
        if second_score is not None and margin < min_margin:
            return MatchDecision(
                matched=False,
                tiger_id=None,
                similarity=top.score,
                needs_review=True,
                decision="review",
                suggested_tiger_id=top.tiger_id,
                reason=(
                    f"{top.tiger_id} and {ranked[1].tiger_id} are too close "
                    f"(margin {margin:.3f} < {min_margin:.3f})."
                ),
                candidates=ranked,
            )
        return MatchDecision(
            matched=True,
            tiger_id=top.tiger_id,
            similarity=top.score,
            needs_review=False,
            decision="accept",
            suggested_tiger_id=top.tiger_id,
            reason=f"High-confidence local match to {top.tiger_id}.",
            candidates=ranked,
        )
    if top.score >= review_threshold:
        return MatchDecision(
            matched=False,
            tiger_id=None,
            similarity=top.score,
            needs_review=True,
            decision="review",
            suggested_tiger_id=top.tiger_id,
            reason=f"Uncertain match; {top.tiger_id} needs human review.",
            candidates=ranked,
        )
    return MatchDecision(
        matched=False,
        tiger_id=None,
        similarity=top.score,
        needs_review=True,
        decision="new",
        suggested_tiger_id=None,
        reason="Similarity is below the review threshold; this may be a new tiger.",
        candidates=ranked,
    )
