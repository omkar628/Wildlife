from backend.database.repositories import CameraRepository, ReviewRepository
from backend.review.service import ReviewService
from backend.services.pipeline import PipelineService
from tests.image_helpers import make_jpeg
from tests.fakes import FakeDetector


def test_review_decision_updates_detection(db, tmp_settings, tmp_path):
    folder = tmp_path / "rev"
    make_jpeg(folder / "x.jpg")
    CameraRepository(db).upsert("C01")
    from backend.database.repositories import JobRepository

    job_id = JobRepository(db).create(str(folder), "C01", 0.6)
    PipelineService(db, tmp_settings, FakeDetector())._run_job(job_id, folder, "C01")
    pending = ReviewRepository(db).pending()
    assert pending
    review_id = int(pending[0]["review_id"])
    service = ReviewService(db, tmp_settings)
    result = service.decide(review_id, "prey")
    assert result["status"] == "reviewed"
    assert result["human_class"] == "prey"
    assert service.pending_count() == 0
