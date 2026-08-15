"""SQLite schema for Wildlife Intelligence.

Designed so camera / tiger observations can later be exported as a graph
without changing table names or foreign keys.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    latitude REAL,
    longitude REAL,
    elevation REAL,
    habitat TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tigers (
    tiger_id TEXT PRIMARY KEY,
    first_seen TEXT,
    last_seen TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_path TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    status TEXT NOT NULL,
    total_images INTEGER NOT NULL DEFAULT 0,
    processed INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    tiger_count INTEGER NOT NULL DEFAULT 0,
    prey_count INTEGER NOT NULL DEFAULT 0,
    rival_count INTEGER NOT NULL DEFAULT 0,
    human_count INTEGER NOT NULL DEFAULT 0,
    other_count INTEGER NOT NULL DEFAULT 0,
    low_confidence_count INTEGER NOT NULL DEFAULT 0,
    review_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    error_message TEXT,
    confidence_threshold REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL UNIQUE,
    original_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    camera_id TEXT,
    timestamp TEXT,
    timestamp_source TEXT,
    created_at TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    error_message TEXT,
    width INTEGER,
    height INTEGER,
    job_id INTEGER,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY (job_id) REFERENCES import_jobs(job_id)
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_width REAL NOT NULL,
    bbox_height REAL NOT NULL,
    final_class_id INTEGER,
    final_class_name TEXT,
    accepted INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'none',
    created_at TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(image_id)
);

CREATE TABLE IF NOT EXISTS tiger_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL UNIQUE,
    tiger_id TEXT,
    reid_confidence REAL,
    crop_path TEXT,
    human_verified INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id),
    FOREIGN KEY (tiger_id) REFERENCES tigers(tiger_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL UNIQUE,
    predicted_class TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    human_class TEXT,
    status TEXT NOT NULL,
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id)
);

CREATE TABLE IF NOT EXISTS image_errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    original_path TEXT NOT NULL,
    error_message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES import_jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_images_camera ON images(camera_id);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(processing_status);
CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(timestamp);
CREATE INDEX IF NOT EXISTS idx_images_job ON images(job_id);

CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id);
CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_detections_final_class ON detections(final_class_name);
CREATE INDEX IF NOT EXISTS idx_detections_review ON detections(review_status);
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence);

CREATE INDEX IF NOT EXISTS idx_obs_tiger ON tiger_observations(tiger_id);
CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON tiger_observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_obs_detection ON tiger_observations(detection_id);

CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status);
CREATE INDEX IF NOT EXISTS idx_reviews_detection ON reviews(detection_id);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON import_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_camera ON import_jobs(camera_id);

CREATE INDEX IF NOT EXISTS idx_errors_job ON image_errors(job_id);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    connection.commit()
