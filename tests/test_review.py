from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    ObservationRepository,
    ReviewRepository,
    TigerRepository,
)
from backend.detector.parser import ParsedDetection
from backend.reid.identity import LocalIdentityService
from backend.review.service import ReviewService
from backend.services.pipeline import PipelineService
from tests.image_helpers import make_jpeg
from tests.fakes import FakeDetector, FakeEncoder


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


def test_reclassifying_tiger_as_prey_removes_observation(db, tmp_settings, tmp_path):
    folder = tmp_path / "rev-tiger"
    make_jpeg(folder / "maybe.jpg")
    CameraRepository(db).upsert("C01")
    from backend.database.repositories import JobRepository

    detector = FakeDetector(
        {
            "maybe.jpg": [ParsedDetection(0, "tiger", 0.40, 8, 8, 30, 24)],
        }
    )
    job_id = JobRepository(db).create(str(folder), "C01", 0.6)
    PipelineService(db, tmp_settings, detector)._run_job(job_id, folder, "C01")
    detection = next(
        item for item in DetectionRepository(db).list_recent(10) if item["class_name"] == "tiger"
    )
    assert ObservationRepository(db).get_by_detection(detection["detection_id"]) is not None
    pending = ReviewRepository(db).pending()
    assert pending
    ReviewService(db, tmp_settings).decide(int(pending[0]["review_id"]), "prey")
    assert ObservationRepository(db).get_by_detection(detection["detection_id"]) is None
    assert ObservationRepository(db).list_all() == []


def test_confirming_tiger_keeps_observation_and_can_identify(db, tmp_settings, tmp_path):
    folder = tmp_path / "rev-keep"
    crop_name = "maybe.jpg"
    make_jpeg(folder / crop_name)
    CameraRepository(db).upsert("C01", latitude=21.6, longitude=79.2)
    from backend.database.repositories import JobRepository

    detector = FakeDetector(
        {
            crop_name: [ParsedDetection(0, "tiger", 0.40, 8, 8, 30, 24)],
        }
    )
    identity = LocalIdentityService(db, tmp_settings, FakeEncoder())
    job_id = JobRepository(db).create(str(folder), "C01", 0.6)
    PipelineService(db, tmp_settings, detector, identity=identity)._run_job(job_id, folder, "C01")
    pending = ReviewRepository(db).pending()
    ReviewService(db, tmp_settings, identity=identity).decide(int(pending[0]["review_id"]), "tiger")
    observations = ObservationRepository(db).list_all()
    assert len(observations) == 1
    assert observations[0]["human_verified"] == 1
    assert observations[0]["tiger_id"] == "T001"
    assert TigerRepository(db).get("T001") is not None


def test_unaccepted_tiger_stays_in_class_review_not_identity_queue(db, tmp_settings, tmp_path):
    folder = tmp_path / "rev-class"
    make_jpeg(folder / "maybe.jpg")
    CameraRepository(db).upsert("C01")
    from backend.database.repositories import JobRepository

    detector = FakeDetector(
        {
            "maybe.jpg": [ParsedDetection(0, "tiger", 0.40, 8, 8, 30, 24)],
        }
    )
    identity = LocalIdentityService(db, tmp_settings, FakeEncoder())
    job_id = JobRepository(db).create(str(folder), "C01", 0.6)
    PipelineService(db, tmp_settings, detector, identity=identity)._run_job(job_id, folder, "C01")
    pending = ReviewRepository(db).pending()
    assert pending
    assert pending[0]["predicted_class"] == "tiger"
    assert identity.unidentified() == []
    assert identity.unidentified_count() == 0
