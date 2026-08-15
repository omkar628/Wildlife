from backend.database.repositories import CameraRepository, ImageRepository, utc_now
from backend.database.schema import SCHEMA_VERSION


def test_schema_initializes(db):
    row = db.fetchone("SELECT value FROM schema_meta WHERE key = 'version'")
    assert row is not None
    assert int(row["value"]) == SCHEMA_VERSION


def test_camera_and_image_insert(db):
    cameras = CameraRepository(db)
    images = ImageRepository(db)
    cameras.upsert("C01", latitude=12.3, longitude=77.1, habitat="dry deciduous")
    image_id = images.create(
        file_hash="abc123",
        original_path=r"D:\CameraTrap\C01\img.jpg",
        filename="img.jpg",
        camera_id="C01",
        timestamp=utc_now(),
        timestamp_source="exif",
        width=100,
        height=80,
        job_id=None,
    )
    stored = images.get(image_id)
    assert stored is not None
    assert stored["camera_id"] == "C01"
    assert stored["file_hash"] == "abc123"
    assert cameras.get("C01")["habitat"] == "dry deciduous"


def test_foreign_keys_enabled(db):
    images = ImageRepository(db)
    try:
        images.create(
            file_hash="orphan",
            original_path="x.jpg",
            filename="x.jpg",
            camera_id="MISSING",
            timestamp=None,
            timestamp_source=None,
            width=1,
            height=1,
            job_id=None,
        )
        raised = False
    except Exception:
        raised = True
    assert raised
