from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_review_service
from backend.api.schemas import ReviewDecisionRequest
from backend.review.service import ReviewService

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("")
def list_reviews(reviews: ReviewService = Depends(get_review_service)) -> dict:
    return {"reviews": reviews.queue(), "pending": reviews.pending_count()}


@router.post("/{review_id}/decide")
def decide_review(
    review_id: int,
    payload: ReviewDecisionRequest,
    reviews: ReviewService = Depends(get_review_service),
) -> dict:
    try:
        return reviews.decide(review_id, payload.human_class)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
