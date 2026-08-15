from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    ImageRepository,
    JobRepository,
    ObservationRepository,
    ReviewRepository,
)
from backend.detector.parser import ParsedDetection
from backend.services.pipeline import PipelineService
from tests.image_helpers import make_jpeg
from tests.fakes import FakeDetector


def _pipeline(db, settings, detector=None) -> PipelineService:
    return PipelineService(db, settings, detector or FakeDetector())


def test_pipeline_inserts_detections_and_review_items(db, tmp_settings, tmp_path):
    folder = tmp_path / "C01"
    make_jpeg(folder / "scene.jpg")
    CameraRepository(db).upsert("C01")
    jobs = JobRepository(db)
    job_id = jobs.create(str(folder), "C01", tmp_settings.confidence_auto_accept)
    _pipeline(db, tmp_settings)._run_job(job_id, folder, "C01")

    job = jobs.get(job_id)
    assert job["status"] == "completed"
    assert job["processed"] == 1
    assert job["tiger_count"] == 1
    assert job["low_confidence_count"] == 1
    assert job["review_count"] == 1

    detections = DetectionRepository(db).list_recent(20)
    names = sorted(item["class_name"] for item in detections)
    assert names == ["prey", "tiger"]
    tiger = next(item for item in detections if item["class_name"] == "tiger")
    assert tiger["accepted"] == 1
    prey = next(item for item in detections if item["class_name"] == "prey")
    assert prey["review_status"] == "pending"
    assert ReviewRepository(db).pending_count() == 1
    observation = ObservationRepository(db).get_by_detection(tiger["detection_id"])
    assert observation is not None
    assert observation["crop_path"]
    assert observation["tiger_id"] is None


def test_corrupt_image_does_not_stop_pipeline(db, tmp_settings, tmp_path):
    folder = tmp_path / "C02"
    make_jpeg(folder / "good.jpg")
    (folder / "bad.jpg").write_bytes(b"not a jpeg")
    CameraRepository(db).upsert("C02")
    jobs = JobRepository(db)
    job_id = jobs.create(str(folder), "C02", 0.6)
    _pipeline(db, tmp_settings)._run_job(job_id, folder, "C02")
    job = jobs.get(job_id)
    assert job["status"] == "completed"
    assert job["processed"] == 1
    assert job["failed"] == 1


def test_duplicate_image_is_not_reinserted(db, tmp_settings, tmp_path):
    folder = tmp_path / "C03"
    make_jpeg(folder / "same.jpg", (15, 15, 15))
    CameraRepository(db).upsert("C03")
    service = _pipeline(db, tmp_settings)
    jobs = JobRepository(db)
    first = jobs.create(str(folder), "C03", 0.6)
    service._run_job(first, folder, "C03")
    second = jobs.create(str(folder), "C03", 0.6)
    service._run_job(second, folder, "C03")
    assert jobs.get(second)["duplicates"] == 1
    assert jobs.get(second)["processed"] == 0
    count = db.fetchone("SELECT COUNT(*) AS n FROM images")
    assert int(count["n"]) == 1


def test_resume_retries_failed_image(db, tmp_settings, tmp_path):
    folder = tmp_path / "C04"
    path = make_jpeg(folder / "retry.jpg")
    CameraRepository(db).upsert("C04")
    images = ImageRepository(db)
    from backend.ingestion.hasher import hash_file

    image_id = images.create(
        file_hash=hash_file(path),
        original_path=str(path),
        filename=path.name,
        camera_id="C04",
        timestamp=None,
        timestamp_source="unknown",
        width=64,
        height=48,
        job_id=None,
        status="failed",
    )
    jobs = JobRepository(db)
    job_id = jobs.create(str(folder), "C04", 0.6)
    _pipeline(db, tmp_settings)._run_job(job_id, folder, "C04")
    assert images.get(image_id)["processing_status"] == "completed"
    assert jobs.get(job_id)["processed"] == 1


def test_empty_detections_still_marks_image_completed(db, tmp_settings, tmp_path):
    folder = tmp_path / "C05"
    make_jpeg(folder / "empty.jpg")
    CameraRepository(db).upsert("C05")
    detector = FakeDetector(mapping={"empty.jpg": []})
    jobs = JobRepository(db)
    job_id = jobs.create(str(folder), "C05", 0.6)
    _pipeline(db, tmp_settings, detector)._run_job(job_id, folder, "C05")
    rows = ImageRepository(db).list_recent()
    assert rows[0]["processing_status"] == "completed"
    assert DetectionRepository(db).list_recent() == []
