"""Data access layer. Keep SQL here so services stay testable."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.database.connection import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def coordinates_are_valid(latitude: Any, longitude: Any) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def row_to_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def normalize_camera_id(camera_id: str) -> str:
    cleaned = (camera_id or "").strip()
    if not cleaned:
        raise ValueError("camera_id is required.")
    if len(cleaned) > 64:
        raise ValueError("camera_id must be 64 characters or fewer.")
    allowed = []
    for char in cleaned:
        if char.isalnum() or char in "._-":
            allowed.append(char)
        elif char in {" ", "\t"}:
            allowed.append("_")
        else:
            raise ValueError(
                "camera_id may contain letters, numbers, '.', '_' and '-' only."
            )
    normalized = "".join(allowed).strip("._-")
    if not normalized:
        raise ValueError("camera_id is required.")
    return normalized


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
        name: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        camera_id = normalize_camera_id(camera_id)
        existing = self.get(camera_id)
        if existing is None:
            self.db.execute(
                """
                INSERT INTO cameras(
                    camera_id, name, latitude, longitude, elevation,
                    habitat, metadata, enabled, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    (name or camera_id).strip() or camera_id,
                    latitude,
                    longitude,
                    elevation,
                    habitat,
                    metadata,
                    1 if enabled is None or enabled else 0,
                    utc_now(),
                ),
            )
        else:
            self.db.execute(
                """
                UPDATE cameras
                SET name = COALESCE(?, name),
                    latitude = COALESCE(?, latitude),
                    longitude = COALESCE(?, longitude),
                    elevation = COALESCE(?, elevation),
                    habitat = COALESCE(?, habitat),
                    metadata = COALESCE(?, metadata),
                    enabled = COALESCE(?, enabled)
                WHERE camera_id = ?
                """,
                (
                    name.strip() if isinstance(name, str) and name.strip() else None,
                    latitude,
                    longitude,
                    elevation,
                    habitat,
                    metadata,
                    None if enabled is None else (1 if enabled else 0),
                    camera_id,
                ),
            )
        found = self.get(camera_id)
        assert found is not None
        return found

    def create(
        self,
        camera_id: str,
        name: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        elevation: float | None = None,
        habitat: str | None = None,
        metadata: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        camera_id = normalize_camera_id(camera_id)
        if self.get(camera_id) is not None:
            raise ValueError(f"Camera ID already exists: {camera_id}")
        return self.upsert(
            camera_id,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            habitat=habitat,
            metadata=metadata,
            name=name,
            enabled=enabled,
        )

    def update(
        self,
        camera_id: str,
        *,
        name: str | None = None,
        latitude: Any = ...,
        longitude: Any = ...,
        elevation: Any = ...,
        habitat: Any = ...,
        metadata: Any = ...,
        enabled: bool | None = None,
        new_camera_id: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get(camera_id)
        if existing is None:
            raise KeyError(f"Camera not found: {camera_id}")

        target_id = camera_id
        if new_camera_id is not None:
            target_id = normalize_camera_id(new_camera_id)
            if target_id != camera_id:
                if self.get(target_id) is not None:
                    raise ValueError(f"Camera ID already exists: {target_id}")
                self._rename(camera_id, target_id)
                existing = self.get(target_id)
                assert existing is not None

        assignments: list[str] = []
        values: list[Any] = []
        if name is not None:
            assignments.append("name = ?")
            values.append(name.strip() or target_id)
        if latitude is not ...:
            assignments.append("latitude = ?")
            values.append(latitude)
        if longitude is not ...:
            assignments.append("longitude = ?")
            values.append(longitude)
        if elevation is not ...:
            assignments.append("elevation = ?")
            values.append(elevation)
        if habitat is not ...:
            assignments.append("habitat = ?")
            values.append(habitat)
        if metadata is not ...:
            assignments.append("metadata = ?")
            values.append(metadata)
        if enabled is not None:
            assignments.append("enabled = ?")
            values.append(1 if enabled else 0)
        if assignments:
            values.append(target_id)
            self.db.execute(
                f"UPDATE cameras SET {', '.join(assignments)} WHERE camera_id = ?",
                values,
            )
        found = self.get(target_id)
        assert found is not None
        return found

    def _rename(self, old_id: str, new_id: str) -> None:
        """Change a camera primary key and rewrite foreign keys."""
        row = self.get(old_id)
        if row is None:
            raise KeyError(f"Camera not found: {old_id}")
        self.db.execute(
            """
            INSERT INTO cameras(
                camera_id, name, latitude, longitude, elevation,
                habitat, metadata, enabled, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                row.get("name") or new_id,
                row.get("latitude"),
                row.get("longitude"),
                row.get("elevation"),
                row.get("habitat"),
                row.get("metadata"),
                0 if row.get("enabled") in {0, "0", False} else 1,
                row.get("created_at") or utc_now(),
            ),
        )
        self.db.execute(
            "UPDATE images SET camera_id = ? WHERE camera_id = ?",
            (new_id, old_id),
        )
        self.db.execute(
            "UPDATE import_jobs SET camera_id = ? WHERE camera_id = ?",
            (new_id, old_id),
        )
        self.db.execute("DELETE FROM cameras WHERE camera_id = ?", (old_id,))

    def set_enabled(self, camera_id: str, enabled: bool) -> dict[str, Any]:
        existing = self.get(camera_id)
        if existing is None:
            raise KeyError(f"Camera not found: {camera_id}")
        self.db.execute(
            "UPDATE cameras SET enabled = ? WHERE camera_id = ?",
            (1 if enabled else 0, camera_id),
        )
        found = self.get(camera_id)
        assert found is not None
        return found

    def delete(self, camera_id: str) -> None:
        existing = self.get(camera_id)
        if existing is None:
            raise KeyError(f"Camera not found: {camera_id}")
        image_row = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM images WHERE camera_id = ?",
            (camera_id,),
        )
        image_count = int(image_row["n"]) if image_row else 0
        if image_count > 0:
            raise ValueError(
                f"Cannot delete {camera_id}: {image_count} images are linked. "
                "Disable the camera instead."
            )
        self.db.execute(
            """
            DELETE FROM image_errors
            WHERE job_id IN (SELECT job_id FROM import_jobs WHERE camera_id = ?)
            """,
            (camera_id,),
        )
        self.db.execute("DELETE FROM import_jobs WHERE camera_id = ?", (camera_id,))
        self.db.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))

    def get(self, camera_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        return row_to_dict(row) if row else None

    def list_all(self) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in self.db.fetchall("SELECT * FROM cameras ORDER BY camera_id")]

    def list_with_stats(self) -> list[dict[str, Any]]:
        cameras = self.list_all()
        if not cameras:
            return []
        image_counts = {
            str(row["camera_id"]): int(row["n"])
            for row in self.db.fetchall(
                """
                SELECT camera_id, COUNT(*) AS n
                FROM images
                WHERE camera_id IS NOT NULL
                GROUP BY camera_id
                """
            )
        }
        class_counts: dict[str, dict[str, int]] = {}
        for row in self.db.fetchall(
            """
            SELECT i.camera_id AS camera_id,
                   LOWER(COALESCE(d.final_class_name, d.class_name)) AS class_name,
                   COUNT(*) AS n
            FROM detections d
            JOIN images i ON i.image_id = d.image_id
            WHERE i.camera_id IS NOT NULL
              AND d.review_status != 'ignored'
            GROUP BY i.camera_id, LOWER(COALESCE(d.final_class_name, d.class_name))
            """
        ):
            camera_id = str(row["camera_id"])
            bucket = class_counts.setdefault(
                camera_id, {"tiger": 0, "prey": 0, "rival": 0, "human": 0, "other": 0}
            )
            name = str(row["class_name"] or "other")
            if name not in bucket:
                name = "other"
            bucket[name] += int(row["n"])
        observation_counts = {
            str(row["camera_id"]): int(row["n"])
            for row in self.db.fetchall(
                """
                SELECT i.camera_id AS camera_id, COUNT(*) AS n
                FROM detections d
                JOIN images i ON i.image_id = d.image_id
                WHERE i.camera_id IS NOT NULL
                  AND d.review_status != 'ignored'
                GROUP BY i.camera_id
                """
            )
        }
        out: list[dict[str, Any]] = []
        for camera in cameras:
            camera_id = str(camera["camera_id"])
            counts = class_counts.get(
                camera_id, {"tiger": 0, "prey": 0, "rival": 0, "human": 0, "other": 0}
            )
            enabled = camera.get("enabled") not in {0, "0", False}
            payload = dict(camera)
            payload.update(
                {
                    "name": camera.get("name") or camera_id,
                    "enabled": enabled,
                    "status": "enabled" if enabled else "disabled",
                    "image_count": image_counts.get(camera_id, 0),
                    "observation_count": observation_counts.get(camera_id, 0),
                    "tiger_count": counts["tiger"],
                    "prey_count": counts["prey"],
                    "rival_count": counts["rival"],
                    "human_count": counts["human"],
                    "missing_coordinates": not coordinates_are_valid(
                        camera.get("latitude"), camera.get("longitude")
                    ),
                }
            )
            out.append(payload)
        return out


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

    def set_classified_path(self, detection_id: int, classified_path: str) -> None:
        self.db.execute(
            "UPDATE detections SET classified_path = ? WHERE detection_id = ?",
            (classified_path, detection_id),
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

    def list_for_movement(
        self,
        animal_class: str | None = None,
        camera_id: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Accepted, non-ignored detections joined to camera and timestamp."""
        clauses = [
            "d.review_status != 'ignored'",
            "d.accepted = 1",
            "i.camera_id IS NOT NULL",
        ]
        params: list[Any] = []
        if animal_class:
            clauses.append("LOWER(COALESCE(d.final_class_name, d.class_name)) = ?")
            params.append(animal_class.strip().lower())
        if camera_id:
            clauses.append("i.camera_id = ?")
            params.append(camera_id)
        if time_from:
            clauses.append("COALESCE(i.timestamp, d.created_at) >= ?")
            params.append(time_from)
        if time_to:
            clauses.append("COALESCE(i.timestamp, d.created_at) <= ?")
            params.append(time_to)
        where = " AND ".join(clauses)
        rows = self.db.fetchall(
            f"""
            SELECT d.detection_id, d.image_id, d.confidence,
                   d.class_name, d.final_class_name,
                   d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                   i.camera_id, i.filename, i.original_path,
                   COALESCE(i.timestamp, d.created_at) AS timestamp,
                   o.observation_id, o.tiger_id, o.reid_confidence, o.crop_path
            FROM detections d
            JOIN images i ON i.image_id = d.image_id
            LEFT JOIN tiger_observations o ON o.detection_id = d.detection_id
            WHERE {where}
            ORDER BY COALESCE(i.timestamp, d.created_at), d.detection_id
            """,
            params,
        )
        return [row_to_dict(row) for row in rows]


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

    def get_joined(self, observation_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone(
            """
            SELECT o.*, d.confidence, d.class_name, d.final_class_name,
                   d.accepted, d.review_status,
                   d.image_id, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                   i.camera_id, i.original_path, i.filename,
                   i.timestamp AS image_timestamp,
                   i.width AS image_width, i.height AS image_height
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE o.observation_id = ?
            """,
            (observation_id,),
        )
        return row_to_dict(row) if row else None

    def list_unidentified(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT o.*, d.confidence, d.class_name, d.final_class_name,
                   d.image_id, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                   i.camera_id, i.original_path, i.filename,
                   i.timestamp AS image_timestamp,
                   i.width AS image_width, i.height AS image_height
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE o.tiger_id IS NULL
              AND d.accepted = 1
              AND LOWER(COALESCE(d.final_class_name, d.class_name)) = 'tiger'
            ORDER BY COALESCE(
                (SELECT s.deferred FROM reid_suggestions s WHERE s.observation_id = o.observation_id),
                0
            ), o.observation_id
            LIMIT ?
            """,
            (limit,),
        )
        return [row_to_dict(row) for row in rows]

    def unidentified_count(self) -> int:
        row = self.db.fetchone(
            """
            SELECT COUNT(*) AS n
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            WHERE o.tiger_id IS NULL
              AND d.accepted = 1
              AND LOWER(COALESCE(d.final_class_name, d.class_name)) = 'tiger'
            """
        )
        return int(row["n"]) if row else 0

    def mark_human_verified(self, observation_id: int, verified: bool = True) -> None:
        self.db.execute(
            "UPDATE tiger_observations SET human_verified = ? WHERE observation_id = ?",
            (1 if verified else 0, observation_id),
        )

    def set_crop_path(self, observation_id: int, crop_path: str) -> None:
        self.db.execute(
            "UPDATE tiger_observations SET crop_path = ? WHERE observation_id = ?",
            (crop_path, observation_id),
        )

    def set_classified_path(self, observation_id: int, classified_path: str) -> None:
        self.db.execute(
            "UPDATE tiger_observations SET classified_path = ? WHERE observation_id = ?",
            (classified_path, observation_id),
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
                   d.image_id, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                   i.camera_id, i.original_path, i.filename,
                   i.timestamp AS image_timestamp,
                   EXISTS(
                       SELECT 1 FROM tiger_embeddings e
                       WHERE e.observation_id = o.observation_id
                   ) AS embedding_available
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at)
            """
        )
        return [row_to_dict(row) for row in rows]

    def list_identified(self) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT o.*, d.confidence, d.class_name, d.final_class_name,
                   d.image_id, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                   i.camera_id, i.original_path, i.filename,
                   i.timestamp AS image_timestamp,
                   EXISTS(
                       SELECT 1 FROM tiger_embeddings e
                       WHERE e.observation_id = o.observation_id
                   ) AS embedding_available
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE o.tiger_id IS NOT NULL
            ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at)
            """
        )
        return [row_to_dict(row) for row in rows]

    def list_for_tiger(self, tiger_id: str) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT o.*, d.confidence, d.class_name, d.final_class_name,
                   d.image_id, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
                   i.camera_id, i.original_path, i.filename,
                   i.timestamp AS image_timestamp,
                   EXISTS(
                       SELECT 1 FROM tiger_embeddings e
                       WHERE e.observation_id = o.observation_id
                   ) AS embedding_available
            FROM tiger_observations o
            JOIN detections d ON d.detection_id = o.detection_id
            JOIN images i ON i.image_id = d.image_id
            WHERE o.tiger_id = ?
            ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at)
            """,
            (tiger_id,),
        )
        return [row_to_dict(row) for row in rows]

    def delete_for_detection(self, detection_id: int) -> bool:
        """Remove the observation and its Re-ID rows. Crop files stay on disk."""
        existing = self.get_by_detection(detection_id)
        if existing is None:
            return False
        observation_id = int(existing["observation_id"])
        self.db.execute(
            "DELETE FROM reid_suggestions WHERE observation_id = ?",
            (observation_id,),
        )
        self.db.execute(
            "DELETE FROM tiger_embeddings WHERE observation_id = ?",
            (observation_id,),
        )
        self.db.execute(
            "DELETE FROM tiger_observations WHERE observation_id = ?",
            (observation_id,),
        )
        return True


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
                   (SELECT COUNT(*) FROM tiger_observations o WHERE o.tiger_id = t.tiger_id) AS observation_count,
                   (
                       SELECT i.camera_id
                       FROM tiger_observations o
                       JOIN detections d ON d.detection_id = o.detection_id
                       JOIN images i ON i.image_id = d.image_id
                       WHERE o.tiger_id = t.tiger_id
                       ORDER BY COALESCE(o.timestamp, i.timestamp, o.created_at) DESC
                       LIMIT 1
                   ) AS last_camera
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


class EmbeddingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(
        self,
        observation_id: int,
        vector: bytes,
        dim: int,
        model: str,
        tiger_id: str | None = None,
    ) -> int:
        existing = self.get_by_observation(observation_id)
        if existing is None:
            return self.db.execute_returning_id(
                """
                INSERT INTO tiger_embeddings(
                    observation_id, tiger_id, vector, dim, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (observation_id, tiger_id, vector, dim, model, utc_now()),
            )
        self.db.execute(
            """
            UPDATE tiger_embeddings
            SET tiger_id = COALESCE(?, tiger_id)
            WHERE observation_id = ?
            """,
            (tiger_id, observation_id),
        )
        return int(existing["embedding_id"])

    def set_tiger_id(self, observation_id: int, tiger_id: str | None) -> None:
        self.db.execute(
            "UPDATE tiger_embeddings SET tiger_id = ? WHERE observation_id = ?",
            (tiger_id, observation_id),
        )

    def get_by_observation(self, observation_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone(
            "SELECT * FROM tiger_embeddings WHERE observation_id = ?",
            (observation_id,),
        )
        return row_to_dict(row) if row else None

    def list_for_tiger(self, tiger_id: str) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT * FROM tiger_embeddings
            WHERE tiger_id = ?
            ORDER BY embedding_id
            """,
            (tiger_id,),
        )
        return [row_to_dict(row) for row in rows]

    def list_identified(self) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT * FROM tiger_embeddings
            WHERE tiger_id IS NOT NULL
            ORDER BY embedding_id
            """
        )
        return [row_to_dict(row) for row in rows]

    def count_for_tiger(self, tiger_id: str) -> int:
        row = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM tiger_embeddings WHERE tiger_id = ?",
            (tiger_id,),
        )
        return int(row["n"]) if row else 0


class SuggestionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, observation_id: int, payload: dict[str, Any]) -> None:
        existing = self.get(observation_id)
        candidates = payload.get("candidates")
        if not isinstance(candidates, str):
            candidates = json.dumps(candidates or [])
        values = (
            1 if payload.get("matched") else 0,
            payload.get("suggested_tiger_id"),
            payload.get("similarity"),
            1 if payload.get("needs_review", True) else 0,
            payload.get("decision") or "unknown",
            candidates,
            payload.get("reason"),
            1 if payload.get("deferred") else 0,
            utc_now(),
            observation_id,
        )
        if existing is None:
            self.db.execute(
                """
                INSERT INTO reid_suggestions(
                    matched, suggested_tiger_id, similarity, needs_review,
                    decision, candidates, reason, deferred, created_at, observation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return
        self.db.execute(
            """
            UPDATE reid_suggestions
            SET matched = ?, suggested_tiger_id = ?, similarity = ?, needs_review = ?,
                decision = ?, candidates = ?, reason = ?, deferred = ?, created_at = ?
            WHERE observation_id = ?
            """,
            values,
        )

    def get(self, observation_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone(
            "SELECT * FROM reid_suggestions WHERE observation_id = ?",
            (observation_id,),
        )
        return row_to_dict(row) if row else None

    def mark_deferred(self, observation_id: int) -> None:
        existing = self.get(observation_id)
        if existing is None:
            self.upsert(
                observation_id,
                {
                    "matched": False,
                    "needs_review": True,
                    "decision": "unknown",
                    "candidates": [],
                    "reason": "Kept unidentified by a reviewer.",
                    "deferred": True,
                },
            )
            return
        self.db.execute(
            "UPDATE reid_suggestions SET deferred = 1 WHERE observation_id = ?",
            (observation_id,),
        )


class AlertRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(
        self,
        *,
        alert_type: str,
        severity: str,
        title: str,
        explanation: str,
        event_key: str,
        animal_class: str | None = None,
        tiger_id: str | None = None,
        camera_id: str | None = None,
        confidence: float | None = None,
        timestamp: str | None = None,
        location: str | None = None,
        source_table: str | None = None,
        source_id: int | None = None,
        observation_id: int | None = None,
        detection_id: int | None = None,
        image_id: int | None = None,
        metadata: str | None = None,
    ) -> int | None:
        existing = self.get_by_event_key(event_key)
        if existing is not None:
            return None
        return self.db.execute_returning_id(
            """
            INSERT INTO alerts(
                alert_type, severity, title, explanation, animal_class, tiger_id,
                camera_id, confidence, timestamp, location, event_key, source_table,
                source_id, observation_id, detection_id, image_id, read, cleared,
                created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
            """,
            (
                alert_type,
                severity,
                title,
                explanation,
                animal_class,
                tiger_id,
                camera_id,
                confidence,
                timestamp,
                location,
                event_key,
                source_table,
                source_id,
                observation_id,
                detection_id,
                image_id,
                utc_now(),
                metadata,
            ),
        )

    def get(self, alert_id: int) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        return row_to_dict(row) if row else None

    def get_by_event_key(self, event_key: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM alerts WHERE event_key = ?", (event_key,))
        return row_to_dict(row) if row else None

    def list_filtered(
        self,
        *,
        unread: bool | None = None,
        cleared: bool = False,
        alert_type: str | None = None,
        tiger_id: str | None = None,
        camera_id: str | None = None,
        severity: str | None = None,
        since_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["cleared = ?"]
        params: list[Any] = [1 if cleared else 0]
        if unread is True:
            clauses.append("read = 0")
        elif unread is False:
            clauses.append("read = 1")
        if alert_type:
            clauses.append("alert_type = ?")
            params.append(alert_type)
        if tiger_id:
            clauses.append("tiger_id = ?")
            params.append(tiger_id)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if since_id is not None:
            clauses.append("alert_id > ?")
            params.append(since_id)
        where = " AND ".join(clauses)
        rows = self.db.fetchall(
            f"""
            SELECT * FROM alerts
            WHERE {where}
            ORDER BY alert_id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [row_to_dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        unread_row = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM alerts WHERE read = 0 AND cleared = 0"
        )
        critical_row = self.db.fetchone(
            """
            SELECT COUNT(*) AS n FROM alerts
            WHERE read = 0 AND cleared = 0 AND severity = 'critical'
            """
        )
        recent = self.list_filtered(limit=8)
        return {
            "unread": int(unread_row["n"]) if unread_row else 0,
            "critical": int(critical_row["n"]) if critical_row else 0,
            "recent": recent,
        }

    def mark_read(self, alert_id: int) -> dict[str, Any] | None:
        existing = self.get(alert_id)
        if existing is None:
            return None
        self.db.execute("UPDATE alerts SET read = 1 WHERE alert_id = ?", (alert_id,))
        return self.get(alert_id)

    def mark_all_read(
        self,
        *,
        alert_type: str | None = None,
        tiger_id: str | None = None,
        camera_id: str | None = None,
    ) -> int:
        clauses = ["cleared = 0", "read = 0"]
        params: list[Any] = []
        if alert_type:
            clauses.append("alert_type = ?")
            params.append(alert_type)
        if tiger_id:
            clauses.append("tiger_id = ?")
            params.append(tiger_id)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        where = " AND ".join(clauses)
        cursor = self.db.execute(f"UPDATE alerts SET read = 1 WHERE {where}", params)
        return int(cursor.rowcount or 0)

    def clear(self, alert_id: int) -> dict[str, Any] | None:
        existing = self.get(alert_id)
        if existing is None:
            return None
        self.db.execute(
            "UPDATE alerts SET cleared = 1, read = 1 WHERE alert_id = ?",
            (alert_id,),
        )
        return self.get(alert_id)

    def clear_filtered(
        self,
        *,
        alert_type: str | None = None,
        tiger_id: str | None = None,
        camera_id: str | None = None,
    ) -> int:
        clauses = ["cleared = 0"]
        params: list[Any] = []
        if alert_type:
            clauses.append("alert_type = ?")
            params.append(alert_type)
        if tiger_id:
            clauses.append("tiger_id = ?")
            params.append(tiger_id)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        where = " AND ".join(clauses)
        cursor = self.db.execute(
            f"UPDATE alerts SET cleared = 1, read = 1 WHERE {where}",
            params,
        )
        return int(cursor.rowcount or 0)


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
