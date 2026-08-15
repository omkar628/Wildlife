from backend.database.repositories import CameraRepository, ImageRepository
from backend.ingestion.hasher import hash_file
from tests.image_helpers import make_jpeg


def test_hash_lookup_finds_existing_image(db, tmp_path):
    path = make_jpeg(tmp_path / "cam" / "a.jpg")
    digest = hash_file(path)
    CameraRepository(db).upsert("C01")
    images = ImageRepository(db)
    images.create(
        file_hash=digest,
        original_path=str(path),
        filename=path.name,
        camera_id="C01",
        timestamp=None,
        timestamp_source="filesystem",
        width=64,
        height=48,
        job_id=None,
        status="completed",
    )
    found = images.find_by_hash(digest)
    assert found is not None
    assert found["filename"] == "a.jpg"
    assert images.find_by_hash("0" * 64) is None
