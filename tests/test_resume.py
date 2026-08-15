from backend.database.repositories import CameraRepository, ImageRepository, JobRepository
from backend.ingestion.hasher import hash_file
from backend.services.pipeline import PipelineService
from tests.image_helpers import make_jpeg
from tests.fakes import FakeDetector


def test_pending_image_is_picked_up_on_rerun(db, tmp_settings, tmp_path):
    folder = tmp_path / "site"
    first = make_jpeg(folder / "one.jpg", (1, 2, 3))
    second = make_jpeg(folder / "two.jpg", (9, 8, 7))
    CameraRepository(db).upsert("C09")
    images = ImageRepository(db)
    images.create(
        file_hash=hash_file(first),
        original_path=str(first),
        filename=first.name,
        camera_id="C09",
        timestamp=None,
        timestamp_source="unknown",
        width=64,
        height=48,
        job_id=None,
        status="completed",
    )
    images.create(
        file_hash=hash_file(second),
        original_path=str(second),
        filename=second.name,
        camera_id="C09",
        timestamp=None,
        timestamp_source="unknown",
        width=64,
        height=48,
        job_id=None,
        status="pending",
    )
    jobs = JobRepository(db)
    job_id = jobs.create(str(folder), "C09", 0.6)
    PipelineService(db, tmp_settings, FakeDetector())._run_job(job_id, folder, "C09")
    job = jobs.get(job_id)
    assert job["duplicates"] == 1
    assert job["processed"] == 1
    pending = [
        row
        for row in images.list_recent(10)
        if row["filename"] == "two.jpg"
    ]
    assert pending[0]["processing_status"] == "completed"
