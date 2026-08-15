"""Data access layer. Keep SQL here so services stay testable."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.database.connection import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def row_to_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class CameraRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(
        self,
        camera_id: str,
        latitude: float | None = None,
        longitude: float | None = None,
        elevation: float | None = None,
        habitat: str | None = None,
        metadata: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get(camera_id)
        if existing is None:
            self.db.execute(
                """
                INSERT INTO cameras(camera_id, latitude, longitude, elevation, habitat, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (camera_id, latitude, longitude, elevation, habitat, metadata, utc_now()),
            )
        else:
            self.db.execute(
                """
                UPDATE cameras
                SET latitude = COALESCE(?, latitude),
                    longitude = COALESCE(?, longitude),
                    elevation = COALESCE(?, elevation),
                    habitat = COALESCE(?, habitat),
                    metadata = COALESCE(?, metadata)
                WHERE camera_id = ?
                """,
                (latitude, longitude, elevation, habitat, metadata, camera_id),
            )
        found = self.get(camera_id)
        assert found is not None
        return found

    def get(self, camera_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        return row_to_dict(row) if row else None

    def list_all(self) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in self.db.fetchall("SELECT * FROM cameras ORDER BY camera_id")]


class JobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, folder_path: str, camera_id: str, confidence_threshold: float) -> int:
        return self.db.execute_returning_id(
            """
            INSERT INTO import_jobs(folder_path, camera_id, status, confidence_threshold, created_at)
            VALUES (?, ?, 'queued', ?, ?)
            """,
            (folder_path, camera_id, confidence_threshold, utc_now()),
        )

    def get(self, job_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,))
        return row_to_dict(row) if row else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM import_jobs ORDER BY job_id DESC LIMIT ?",
            (limit,),
        )
        return [row_to_dict(row) for row in rows]

    def update(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [job_id]
        self.db.execute(f"UPDATE import_jobs SET {assignments} WHERE job_id = ?", values)

    def increment(self, job_id: int, **deltas: int) -> None:
        if not deltas:
            return
        assignments = ", ".join(f"{key} = {key} + ?" for key in deltas)
        values = list(deltas.values()) + [job_id]
        self.db.execute(f"UPDATE import_jobs SET {assignments} WHERE job_id = ?", values)


class ImageRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def find_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM images WHERE file_hash = ?", (file_hash,))
        return row_to_dict(row) if row else None

    def create(
        self,
        file_hash: str,
        original_path: str,
        filename: str,
        camera_id: str | None,
        timestamp: str | None,
        timestamp_source: str | None,
        width: int | None,
        height: int | None,
        job_id: int | None,
        status: str = "pending",
    ) -> int:
        return self.db.execute_returning_id(
            """
            INSERT INTO images(
                file_hash, original_path, filename, camera_id, timestamp,
                timestamp_source, created_at, processing_status, width, height, job_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_hash,
                original_path,
                filename,
                camera_id,
                timestamp,
                timestamp_source,
                utc_now(),
                status,
                width,
                height,
                job_id,
            ),
        )

    def set_status(
        self,
        image_id: int,
        status: str,
        error_message: str | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE images
            SET processing_status = ?, error_message = ?
            WHERE image_id = ?
            """,
            (status, error_message, image_id),
        )

    def get(self, image_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM images WHERE image_id = ?", (image_id,))
        return row_to_dict(row) if row else None

    def list_recent(self, limit: int = 40, camera_id: str | None = None) -> list[dict[str, Any]]:
        if camera_id:
            rows = self.db.fetchall(
                """
                SELECT * FROM images
                WHERE camera_id = ?
                ORDER BY image_id DESC
                LIMIT ?
                """,
                (camera_id, limit),
            )
        else:
            rows = self.db.fetchall(
                "SELECT * FROM images ORDER BY image_id DESC LIMIT ?",
                (limit,),
            )
        return [row_to_dict(row) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self.db.fetchall(
            "SELECT processing_status AS status, COUNT(*) AS n FROM images GROUP BY processing_status"
        )
        return {str(row["status"]): int(row["n"]) for row in rows}


class DetectionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        image_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox_x: float,
        bbox_y: float,
        bbox_width: float,
        bbox_height: float,
        accepted: bool,
        review_status: str,
        final_class_id: int | None,
        final_class_name: str | None,
    ) -> int:
        return self.db.execute_returning_id(
            """
            INSERT INTO detections(
                image_id, class_id, class_name, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height,
                accepted, review_status, final_class_id, final_class_name, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                class_id,
                class_name,
                confidence,
                bbox_x,
                bbox_y,
                bbox_width,
                bbox_height,
                1 if accepted else 0,
                review_status,
                final_class_id,
                final_class_name,
                utc_now(),
            ),
        )

    def get(self, detection_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone(
            """
            SELECT d.*, i.original_path, i.filename, i.camera_id, i.timestamp,
                   i.width AS image_width, i.height AS image_height
            FROM detections d
            JOIN images i ON i.image_id = d.image_id
            WHERE d.detection_id = ?
            """,
            (detection_id,),
        )
        return row_to_dict(row) if row else None

    def list_for_image(self, image_id: int) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM detections WHERE image_id = ? ORDER BY detection_id",
            (image_id,),
        )
        return [row_to_dict(row) for row in rows]

    def list_recent(self, limit: int = 50, class_name: str | None = None) -> list[dict[str, Any]]:
        if class_name:
            rows = self.db.fetchall(
                """
                SELECT d.*, i.original_path, i.filename, i.camera_id, i.timestamp
                FROM detections d
                JOIN images i ON i.image_id = d.image_id
                WHERE COALESCE(d.final_class_name, d.class_name) = ?
                ORDER BY d.detection_id DESC
                LIMIT ?
                """,
                (class_name, limit),
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT d.*, i.original_path, i.filename, i.camera_id, i.timestamp
                FROM detections d
                JOIN images i ON i.image_id = d.image_id
                ORDER BY d.detection_id DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [row_to_dict(row) for row in rows]

    def update_review_decision(
        self,
        detection_id: int,
        final_class_id: int | None,
        final_class_name: str | None,
        review_status: str,
        accepted: bool,
    ) -> None:
        self.db.execute(
            """
            UPDATE detections
            SET final_class_id = ?, final_class_name = ?, review_status = ?, accepted = ?
            WHERE detection_id = ?
            """,
            (
                final_class_id,
                final_class_name,
                review_status,
                1 if accepted else 0,
                detection_id,
            ),
        )

    def class_counts(self) -> dict[str, int]:
        rows = self.db.fetchall(
            """
            SELECT COALESCE(final_class_name, class_name) AS name, COUNT(*) AS n
            FROM detections
            WHERE review_status != 'ignored'
            GROUP BY COALESCE(final_class_name, class_name)
            """
        )
        return {str(row["name"]): int(row["n"]) for row in rows}


class ReviewRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        detection_id: int,
        predicted_class: str,
        predicted_confidence: float,
    ) -> int:
        return self.db.execute_returning_id(
            """
            INSERT INTO reviews(detection_id, predicted_class, predicted_confidence, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
            """,
            (detection_id, predicted_class, predicted_confidence, utc_now()),
        )

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT r.*, d.class_id, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                   d.image_id, i.original_path, i.filename, i.camera_id, i.timestamp,
                   i.width AS image_width, i.height AS image_height
            FROM reviews r
            JOIN detections d ON d.detection_id = r.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE r.status = 'pending'
            ORDER BY r.review_id
            LIMIT ?
            """,
            (limit,),
        )
        return [row_to_dict(row) for row in rows]

    def get(self, review_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM reviews WHERE review_id = ?", (review_id,))
        return row_to_dict(row) if row else None

    def get_by_detection(self, detection_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM reviews WHERE detection_id = ?", (detection_id,))
        return row_to_dict(row) if row else None

    def decide(self, review_id: int, human_class: str | None, status: str) -> None:
        self.db.execute(
            """
            UPDATE reviews
            SET human_class = ?, status = ?, reviewed_at = ?
            WHERE review_id = ?
            """,
            (human_class, status, utc_now(), review_id),
        )

    def pending_count(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) AS n FROM reviews WHERE status = 'pending'")
        return int(row["n"]) if row else 0


class ObservationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        detection_id: int,
        tiger_id: str | None,
        reid_confidence: float | None,
        crop_path: str | None,
        timestamp: str | None,
        human_verified: bool = False,
    ) -> int:
        return self.db.execute_returning_id(
            """
            INSERT INTO tiger_observations(
                detection_id, tiger_id, reid_confidence, crop_path,
                human_verified, timestamp, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detection_id,
                tiger_id,
                reid_confidence,
                crop_path,
                1 if human_verified else 0,
                timestamp,
                utc_now(),
            ),
        )

    def get_by_detection(self, detection_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone(
            "SELECT * FROM tiger_observations WHERE detection_id = ?",
            (detection_id,),
        )
        return row_to_dict(row) if row else None

    def get(self, observation_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone(
            "SELECT * FROM tiger_observations WHERE observation_id = ?",
            (observation_id,),
        )
        return row_to_dict(row) if row else None

    def set_crop_path(self, observation_id: int, crop_path: str) -> None:
        self.db.execute(
            "UPDATE tiger_observations SET crop_path = ? WHERE observation_id = ?",
            (crop_path, observation_id),
        )

    def set_identity(
        self,
        observation_id: int,
        tiger_id: str | None,
        reid_confidence: float | None,
    ) -> None:
        self.db.execute(
            """
            UPDATE tiger_observations
            SET tiger_id = ?, reid_confidence = ?
            WHERE observation_id = ?
            """,
            (tiger_id, reid_confidence, observation_id),
        )

    def list_all(self) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT o.*, d.confidence, d.class_name, d.final_class_name,
                   i.camera_id, i.original_path, i.filename, i.timestamp AS image_timestamp
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at)
            """
        )
        return [row_to_dict(row) for row in rows]

    def list_for_tiger(self, tiger_id: str) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT o.*, d.confidence, i.camera_id, i.original_path, i.filename
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE o.tiger_id = ?
            ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at)
            """,
            (tiger_id,),
        )
        return [row_to_dict(row) for row in rows]


class TigerRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_seen(self, tiger_id: str, seen_at: str | None) -> None:
        existing = self.db.fetchone("SELECT * FROM tigers WHERE tiger_id = ?", (tiger_id,))
        now = utc_now()
        if existing is None:
            self.db.execute(
                """
                INSERT INTO tigers(tiger_id, first_seen, last_seen, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tiger_id, seen_at or now, seen_at or now, now),
            )
            return
        self.db.execute(
            """
            UPDATE tigers
            SET first_seen = CASE
                    WHEN first_seen IS NULL OR (? IS NOT NULL AND ? < first_seen) THEN ?
                    ELSE first_seen
                END,
                last_seen = CASE
                    WHEN last_seen IS NULL OR (? IS NOT NULL AND ? > last_seen) THEN ?
                    ELSE last_seen
                END
            WHERE tiger_id = ?
            """,
            (seen_at, seen_at, seen_at, seen_at, seen_at, seen_at, tiger_id),
        )

    def list_all(self) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT t.*,
                   (SELECT COUNT(*) FROM tiger_observations o WHERE o.tiger_id = t.tiger_id) AS observation_count
            FROM tigers t
            ORDER BY t.tiger_id
            """
        )
        return [row_to_dict(row) for row in rows]

    def get(self, tiger_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM tigers WHERE tiger_id = ?", (tiger_id,))
        return row_to_dict(row) if row else None


class SettingsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, key: str) -> str | None:
        row = self.db.fetchone("SELECT value FROM app_settings WHERE key = ?", (key,))
        return str(row["value"]) if row else None

    def set(self, key: str, value: str) -> None:
        self.db.execute(
            """
            INSERT INTO app_settings(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, utc_now()),
        )


class ErrorRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, job_id: int | None, original_path: str, error_message: str) -> None:
        self.db.execute(
            """
            INSERT INTO image_errors(job_id, original_path, error_message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, original_path, error_message, utc_now()),
        )

    def list_for_job(self, job_id: int) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM image_errors WHERE job_id = ? ORDER BY error_id",
            (job_id,),
        )
        return [row_to_dict(row) for row in rows]
