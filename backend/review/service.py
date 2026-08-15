"""Human review decisions. Original images are never modified."""

from __future__ import annotations

from pathlib import Path

from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import (
    DetectionRepository,
    ObservationRepository,
    ReviewRepository,
)
from backend.detector.parser import ParsedDetection
from backend.services.crops import save_tiger_crop


VALID_DECISIONS = {"tiger", "prey", "rival", "human", "other", "ignore"}


class ReviewService:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.reviews = ReviewRepository(db)
        self.detections = DetectionRepository(db)
        self.observations = ObservationRepository(db)

    def queue(self, limit: int = 40) -> list[dict]:
        return self.reviews.pending(limit=limit)

    def pending_count(self) -> int:
        return self.reviews.pending_count()

    def decide(self, review_id: int, human_class: str) -> dict:
        choice = human_class.strip().lower()
        if choice not in VALID_DECISIONS:
            raise ValueError(f"Invalid review decision: {human_class}")

        review = self.reviews.get(review_id)
        if review is None:
            raise KeyError(f"Review {review_id} was not found.")
        if review["status"] != "pending":
            raise ValueError(f"Review {review_id} is already {review['status']}.")

        detection = self.detections.get(int(review["detection_id"]))
        if detection is None:
            raise KeyError(f"Detection {review['detection_id']} was not found.")

        if choice == "ignore":
            self.reviews.decide(review_id, None, "ignored")
            self.detections.update_review_decision(
                int(review["detection_id"]),
                None,
                None,
                "ignored",
                accepted=False,
            )
            return {"review_id": review_id, "status": "ignored", "human_class": None}

        class_id = self.settings.class_id(choice)
        if class_id is None:
            class_id = 99
        self.reviews.decide(review_id, choice, "reviewed")
        self.detections.update_review_decision(
            int(review["detection_id"]),
            class_id,
            choice,
            "reviewed",
            accepted=True,
        )

        if choice == "tiger":
            self._ensure_tiger_observation(detection)

        updated = self.reviews.get(review_id)
        assert updated is not None
        return updated

    def _ensure_tiger_observation(self, detection: dict) -> None:
        existing = self.observations.get_by_detection(int(detection["detection_id"]))
        if existing is not None:
            return
        parsed = ParsedDetection(
            class_id=int(detection["class_id"]),
            class_name=str(detection["class_name"]),
            confidence=float(detection["confidence"]),
            bbox_x=float(detection["bbox_x"]),
            bbox_y=float(detection["bbox_y"]),
            bbox_width=float(detection["bbox_width"]),
            bbox_height=float(detection["bbox_height"]),
        )
        crop_path = None
        source = Path(detection["original_path"])
        if source.is_file():
            from backend.services.crops import crop_destination

            destination = crop_destination(
                self.settings.crops_dir,
                int(detection["image_id"]),
                int(detection["detection_id"]),
            )
            saved = save_tiger_crop(
                source,
                destination,
                parsed,
                padding=self.settings.crop_padding,
                jpeg_quality=self.settings.crop_jpeg_quality,
            )
            crop_path = str(saved)
        self.observations.create(
            detection_id=int(detection["detection_id"]),
            tiger_id=None,
            reid_confidence=None,
            crop_path=crop_path,
            timestamp=detection.get("timestamp"),
            human_verified=True,
        )
