"""Human review decisions. Original images are never modified."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


VALID_DECISIONS = {"tiger", "prey", "rival", "human", "other", "ignore"}


class ReviewService:
    def __init__(self, db: Database, settings: Settings, identity=None) -> None:
        self.db = db
        self.settings = settings
        self.identity = identity
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
            self._retract_tiger_observation(int(review["detection_id"]))
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
        else:
            self._retract_tiger_observation(int(review["detection_id"]))
            self._archive_accepted(detection, choice)

        updated = self.reviews.get(review_id)
        assert updated is not None
        return updated

    def _retract_tiger_observation(self, detection_id: int) -> None:
        """A non-tiger class decision must not leave a tiger ID or GNN event."""
        self.observations.delete_for_detection(detection_id)

    def _ensure_tiger_observation(self, detection: dict) -> None:
        existing = self.observations.get_by_detection(int(detection["detection_id"]))
        if existing is not None:
            observation_id = int(existing["observation_id"])
            self.observations.mark_human_verified(observation_id, True)
            self._run_identity(observation_id)
            self._archive_accepted(detection, "tiger")
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
        observation_id = self.observations.create(
            detection_id=int(detection["detection_id"]),
            tiger_id=None,
            reid_confidence=None,
            crop_path=crop_path,
            timestamp=detection.get("timestamp"),
            human_verified=True,
        )
        self._run_identity(observation_id)
        self._archive_accepted(detection, "tiger")

    def _archive_accepted(self, detection: dict, class_name: str) -> None:
        try:
            from backend.services.alerts import AlertService
            from backend.services.classified import ClassifiedStore

            observation = self.observations.get_by_detection(int(detection["detection_id"]))
            tiger_id = observation.get("tiger_id") if observation else None
            ClassifiedStore(self.db, self.settings).store_detection(
                source_path=detection.get("original_path"),
                class_name=class_name,
                camera_id=detection.get("camera_id"),
                timestamp=detection.get("timestamp"),
                detection_id=int(detection["detection_id"]),
                observation_id=int(observation["observation_id"]) if observation else None,
                tiger_id=tiger_id,
                image_id=detection.get("image_id"),
            )
            AlertService(self.db, self.settings).from_detection(
                {
                    **detection,
                    "accepted": 1,
                    "final_class_name": class_name,
                    "class_name": class_name,
                    "observation_id": observation.get("observation_id") if observation else None,
                    "tiger_id": tiger_id,
                },
                tiger_id=tiger_id,
            )
        except Exception:
            logger.exception(
                "Review archive/alert failed for detection %s",
                detection.get("detection_id"),
            )

    def _run_identity(self, observation_id: int) -> None:
        if self.identity is None:
            return
        try:
            self.identity.identify_new_observation(observation_id)
        except Exception:
            logger.exception(
                "Re-ID failed for observation %s after review; leaving unidentified",
                observation_id,
            )
