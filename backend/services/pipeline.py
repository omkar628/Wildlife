"""Folder import pipeline: scan → hash → YOLO → filter → SQLite → crops.

Runs in a background thread so the FastAPI process can keep serving the UI.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    ErrorRepository,
    ImageRepository,
    JobRepository,
    ObservationRepository,
    ReviewRepository,
    utc_now,
)
from backend.detector.parser import ParsedDetection
from backend.detector.service import DetectorService
from backend.ingestion.hasher import hash_file
from backend.ingestion.metadata import extract_image_metadata
from backend.ingestion.scanner import iter_image_files
from backend.services.confidence import filter_detection
from backend.services.crops import crop_destination, save_tiger_crop

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        detector: DetectorService,
        identity: Any | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.detector = detector
        self.identity = identity
        self.jobs = JobRepository(db)
        self.images = ImageRepository(db)
        self.detections = DetectionRepository(db)
        self.reviews = ReviewRepository(db)
        self.observations = ObservationRepository(db)
        self.cameras = CameraRepository(db)
        self.errors = ErrorRepository(db)
        self._threads: dict[int, threading.Thread] = {}
        self._cancel = set[int]()
        self._lock = threading.Lock()

    def start_import(
        self,
        folder_path: str,
        camera_id: str,
        latitude: float | None = None,
        longitude: float | None = None,
        elevation: float | None = None,
        habitat: str | None = None,
    ) -> dict:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a folder: {folder}")

        camera_id = camera_id.strip()
        if not camera_id:
            raise ValueError("camera_id is required.")

        self.cameras.upsert(
            camera_id,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            habitat=habitat,
        )
        job_id = self.jobs.create(
            folder_path=str(folder.resolve()),
            camera_id=camera_id,
            confidence_threshold=self.settings.confidence_auto_accept,
        )
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, folder.resolve(), camera_id),
            name=f"import-job-{job_id}",
            daemon=True,
        )
        with self._lock:
            self._threads[job_id] = thread
        thread.start()
        job = self.jobs.get(job_id)
        assert job is not None
        return job

    def cancel(self, job_id: int) -> None:
        self._cancel.add(job_id)

    def _run_job(self, job_id: int, folder: Path, camera_id: str) -> None:
        started = time.perf_counter()
        self.jobs.update(job_id, status="running", started_at=utc_now())
        try:
            paths = iter_image_files(folder)
            self.jobs.update(job_id, total_images=len(paths))
            logger.info("Job %s: found %s images in %s", job_id, len(paths), folder)

            batch_size = self.settings.detector_batch_size
            pending_batch: list[tuple[int, Path]] = []

            for path in paths:
                if job_id in self._cancel:
                    self.jobs.update(job_id, status="cancelled", finished_at=utc_now())
                    return
                try:
                    prepared = self._register_image(job_id, path, camera_id)
                except Exception as exc:
                    logger.exception("Failed to register %s", path)
                    self.errors.add(job_id, str(path), str(exc))
                    self.jobs.increment(job_id, failed=1)
                    continue

                if prepared is None:
                    continue
                pending_batch.append(prepared)
                if len(pending_batch) >= batch_size:
                    self._process_batch(job_id, pending_batch)
                    pending_batch = []

            if pending_batch:
                self._process_batch(job_id, pending_batch)

            elapsed = time.perf_counter() - started
            self.jobs.update(
                job_id,
                status="completed",
                finished_at=utc_now(),
                error_message=f"elapsed_seconds={elapsed:.2f}",
            )
            logger.info("Job %s completed in %.2fs", job_id, elapsed)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self.jobs.update(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error_message=str(exc),
            )

    def _register_image(
        self,
        job_id: int,
        path: Path,
        camera_id: str,
    ) -> tuple[int, Path] | None:
        file_hash = hash_file(path)
        existing = self.images.find_by_hash(file_hash)
        if existing is not None:
            status = existing["processing_status"]
            if status == "completed":
                self.jobs.increment(job_id, duplicates=1)
                return None
            if status == "failed":
                self.images.set_status(int(existing["image_id"]), "pending")
                return int(existing["image_id"]), path
            if status in {"pending", "processing"}:
                return int(existing["image_id"]), path
            self.jobs.increment(job_id, skipped=1)
            return None

        try:
            meta = extract_image_metadata(path)
        except ValueError as exc:
            self.errors.add(job_id, str(path), str(exc))
            self.jobs.increment(job_id, failed=1)
            return None

        image_id = self.images.create(
            file_hash=file_hash,
            original_path=str(path),
            filename=path.name,
            camera_id=camera_id,
            timestamp=meta["timestamp"],
            timestamp_source=meta["timestamp_source"],
            width=meta["width"],
            height=meta["height"],
            job_id=job_id,
            status="pending",
        )
        return image_id, path

    def _process_batch(self, job_id: int, batch: list[tuple[int, Path]]) -> None:
        image_ids = [item[0] for item in batch]
        paths = [item[1] for item in batch]
        for image_id in image_ids:
            self.images.set_status(image_id, "processing")

        try:
            predictions = self.detector.predict_paths(paths)
        except Exception as exc:
            logger.exception("Detector batch failed")
            for image_id, path in batch:
                self.images.set_status(image_id, "failed", str(exc))
                self.errors.add(job_id, str(path), f"detector: {exc}")
                self.jobs.increment(job_id, failed=1)
            return

        if len(predictions) != len(batch):
            logger.error(
                "Detector returned %s results for %s images",
                len(predictions),
                len(batch),
            )

        paired = min(len(batch), len(predictions))
        for index in range(paired):
            image_id, path = batch[index]
            detections = predictions[index]
            try:
                self._store_detections(job_id, image_id, path, detections)
                self.images.set_status(image_id, "completed")
                self.jobs.increment(job_id, processed=1)
            except Exception as exc:
                logger.exception("Failed storing detections for %s", path)
                self.images.set_status(image_id, "failed", str(exc))
                self.errors.add(job_id, str(path), str(exc))
                self.jobs.increment(job_id, failed=1)

        for image_id, path in batch[paired:]:
            message = "Detector returned fewer results than images in the batch."
            self.images.set_status(image_id, "failed", message)
            self.errors.add(job_id, str(path), f"detector: {message}")
            self.jobs.increment(job_id, failed=1)

    def _store_detections(
        self,
        job_id: int,
        image_id: int,
        path: Path,
        detections: list[ParsedDetection],
    ) -> None:
        image = self.images.get(image_id)
        timestamp = image.get("timestamp") if image else None
        auto_accept = self.settings.confidence_auto_accept
        detect_min = self.settings.confidence_detect_min

        for detection in detections:
            decision = filter_detection(
                detection,
                auto_accept=auto_accept,
                detect_min=detect_min,
            )
            if not decision.keep:
                continue

            review_status = "pending" if decision.needs_review else "none"
            final_class_id = detection.class_id if decision.accepted else None
            final_class_name = detection.class_name if decision.accepted else None
            detection_id = self.detections.create(
                image_id=image_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox_x=detection.bbox_x,
                bbox_y=detection.bbox_y,
                bbox_width=detection.bbox_width,
                bbox_height=detection.bbox_height,
                accepted=decision.accepted,
                review_status=review_status,
                final_class_id=final_class_id,
                final_class_name=final_class_name,
            )

            increment: dict[str, int] = {}
            if decision.accepted:
                key = f"{detection.class_name}_count"
                if key in {
                    "tiger_count",
                    "prey_count",
                    "rival_count",
                    "human_count",
                    "other_count",
                }:
                    increment[key] = 1
            if decision.needs_review:
                increment["low_confidence_count"] = 1
                increment["review_count"] = 1
                self.reviews.create(
                    detection_id=detection_id,
                    predicted_class=detection.class_name,
                    predicted_confidence=detection.confidence,
                )
            if increment:
                self.jobs.increment(job_id, **increment)

            is_tiger = detection.class_name == "tiger" and (
                decision.accepted or decision.needs_review
            )
            if is_tiger:
                destination = crop_destination(self.settings.crops_dir, image_id, detection_id)
                saved = save_tiger_crop(
                    path,
                    destination,
                    detection,
                    padding=self.settings.crop_padding,
                    jpeg_quality=self.settings.crop_jpeg_quality,
                )
                observation_id = self.observations.create(
                    detection_id=detection_id,
                    tiger_id=None,
                    reid_confidence=None,
                    crop_path=str(saved),
                    timestamp=timestamp,
                    human_verified=False,
                )
                self._identify_observation(observation_id)

    def _identify_observation(self, observation_id: int) -> None:
        """Best-effort Re-ID. Failures leave the observation unidentified."""
        if self.identity is None or not getattr(self.settings, "reid_enabled", True):
            return
        try:
            self.identity.identify_new_observation(observation_id)
        except Exception:
            logger.exception(
                "Re-ID failed for observation %s; leaving unidentified",
                observation_id,
            )
