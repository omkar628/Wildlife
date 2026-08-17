"""Classified copies never rename or replace original camera-trap files."""

from __future__ import annotations

from pathlib import Path

from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    ImageRepository,
    ObservationRepository,
    TigerRepository,
)
from backend.services.classified import ClassifiedStore, classified_filename
from tests.image_helpers import make_jpeg


def test_classified_filename_format():
    name = classified_filename("T001", "Camera_03", "2026-08-17T14:32:01", 152)
    assert name == "T001_Camera_03_20260817_143201_152.jpg"
    prey = classified_filename("prey", "Camera_01", "2026-08-17T14:32:01", 153)
    assert prey == "prey_Camera_01_20260817_143201_153.jpg"


def test_store_and_relocate_leaves_original_untouched(db, tmp_settings, tmp_path):
    original = tmp_path / "import" / "Camera_01" / "trap.jpg"
    make_jpeg(original)
    before = original.read_bytes()
    CameraRepository(db).create("Camera_01", latitude=21.65, longitude=79.24)
    image_id = ImageRepository(db).create(
        "hash-1",
        str(original),
        "trap.jpg",
        "Camera_01",
        "2026-08-17T14:32:01",
        "exif",
        64,
        48,
        None,
    )
    detection_id = DetectionRepository(db).create(
        image_id, 0, "tiger", 0.94, 1, 1, 20, 20, True, "none", 0, "tiger"
    )
    observation_id = ObservationRepository(db).create(
        detection_id, None, None, None, "2026-08-17T14:32:01"
    )

    store = ClassifiedStore(db, tmp_settings)
    unidentified = store.store_detection(
        source_path=original,
        class_name="tiger",
        camera_id="Camera_01",
        timestamp="2026-08-17T14:32:01",
        detection_id=detection_id,
        observation_id=observation_id,
    )
    assert unidentified is not None
    assert unidentified.parent.name == "unidentified"
    assert original.read_bytes() == before

    TigerRepository(db).upsert_seen("T001", "2026-08-17T14:32:01")
    ObservationRepository(db).set_identity(observation_id, "T001", 0.91)
    relocated = store.relocate_tiger(observation_id=observation_id, tiger_id="T001")
    assert relocated is not None
    assert relocated.parent.name == "T001"
    assert "T001_Camera_01_20260817_143201" in relocated.name
    assert original.read_bytes() == before
    assert original.name == "trap.jpg"
    assert not unidentified.exists()

    prey_src = tmp_path / "import" / "Camera_01" / "deer.jpg"
    make_jpeg(prey_src)
    prey_image = ImageRepository(db).create(
        "hash-2", str(prey_src), "deer.jpg", "Camera_01", "2026-08-17T15:00:00", "exif", 64, 48, None
    )
    prey_det = DetectionRepository(db).create(
        prey_image, 1, "prey", 0.88, 1, 1, 20, 20, True, "none", 1, "prey"
    )
    prey_copy = store.store_detection(
        source_path=prey_src,
        class_name="prey",
        camera_id="Camera_01",
        timestamp="2026-08-17T15:00:00",
        detection_id=prey_det,
    )
    assert prey_copy is not None
    assert prey_copy.parent.name == "prey"
    assert prey_copy.name.startswith("prey_Camera_01_")
    assert prey_src.is_file()
