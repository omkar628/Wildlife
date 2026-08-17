"""Alerts come from stored detections/identities/GNN, never invented events."""

from __future__ import annotations

from pathlib import Path

from backend.api.deps import get_pipeline
from backend.config import get_settings
from backend.database.connection import get_database
from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    ImageRepository,
    ObservationRepository,
    TigerRepository,
)
from backend.main import create_app
from backend.services.alerts import AlertService
from backend.services.pipeline import PipelineService
from tests.fakes import FakeDetector
from tests.image_helpers import make_jpeg
from tests.test_api import ApiClient
from tests.test_gnn_inference import _seed_identified_history


def _client(tmp_settings) -> ApiClient:
    app = create_app()
    pipeline = PipelineService(get_database(), get_settings(), FakeDetector())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return ApiClient(app)


def _seed_camera(db, camera_id, lat=21.65, lon=79.24):
    return CameraRepository(db).create(camera_id, latitude=lat, longitude=lon)


def _seed_detection(
    db,
    *,
    camera_id,
    stamp,
    file_hash,
    filename,
    class_name="tiger",
    tiger_id=None,
    confidence=0.91,
    accepted=True,
):
    images = ImageRepository(db)
    detections = DetectionRepository(db)
    image_id = images.create(
        file_hash, filename, filename, camera_id, stamp, "exif", 64, 48, None
    )
    class_id = {"tiger": 0, "prey": 1, "rival": 2, "human": 3}.get(class_name, 4)
    detection_id = detections.create(
        image_id,
        class_id,
        class_name,
        confidence,
        1,
        1,
        20,
        20,
        accepted,
        "none",
        class_id if accepted else None,
        class_name if accepted else None,
    )
    observation_id = None
    if class_name == "tiger":
        observation_id = ObservationRepository(db).create(
            detection_id, tiger_id, 0.88, None, stamp
        )
    return {
        "detection_id": detection_id,
        "observation_id": observation_id,
        "image_id": image_id,
        "camera_id": camera_id,
        "timestamp": stamp,
        "class_name": class_name,
        "final_class_name": class_name if accepted else None,
        "confidence": confidence,
        "accepted": 1 if accepted else 0,
        "tiger_id": tiger_id,
    }


def test_no_alert_below_confidence_threshold(db, tmp_settings):
    _seed_camera(db, "Camera_01")
    row = _seed_detection(
        db,
        camera_id="Camera_01",
        stamp="2026-08-17T14:32:01",
        file_hash="low",
        filename="low.jpg",
        class_name="prey",
        confidence=0.22,
        accepted=False,
    )
    service = AlertService(db, tmp_settings)
    assert service.from_detection(row) is None
    assert service.list_alerts() == []


def test_class_alerts_and_dedup(db, tmp_settings):
    _seed_camera(db, "Camera_03")
    TigerRepository(db).upsert_seen("T001", "2026-08-17T14:32:01")
    tiger = _seed_detection(
        db,
        camera_id="Camera_03",
        stamp="2026-08-17T14:32:01",
        file_hash="t1",
        filename="t1.jpg",
        tiger_id="T001",
        confidence=0.94,
    )
    service = AlertService(db, tmp_settings)
    first = service.from_detection(tiger, tiger_id="T001")
    second = service.from_detection(tiger, tiger_id="T001")
    assert first is not None
    assert second is None
    assert first["alert_type"] == "tiger_detected"
    assert first["tiger_id"] == "T001"
    assert first["camera_id"] == "Camera_03"
    assert first["event_key"] == f"detection:{tiger['detection_id']}:tiger"

    human = _seed_detection(
        db,
        camera_id="Camera_03",
        stamp="2026-08-17T15:00:00",
        file_hash="h1",
        filename="h1.jpg",
        class_name="human",
        confidence=0.88,
    )
    human_alert = service.from_detection(human)
    assert human_alert["severity"] == "critical"
    assert human_alert["alert_type"] == "human_detected"


def test_new_tiger_and_repeat_alerts(db, tmp_settings):
    _seed_camera(db, "Camera_01")
    TigerRepository(db).upsert_seen("T001", "2026-08-17T08:00:00")
    first = _seed_detection(
        db,
        camera_id="Camera_01",
        stamp="2026-08-17T08:00:00",
        file_hash="a",
        filename="a.jpg",
        tiger_id="T001",
    )
    second = _seed_detection(
        db,
        camera_id="Camera_01",
        stamp="2026-08-17T09:00:00",
        file_hash="b",
        filename="b.jpg",
        tiger_id="T001",
    )
    third = _seed_detection(
        db,
        camera_id="Camera_01",
        stamp="2026-08-17T10:00:00",
        file_hash="c",
        filename="c.jpg",
        tiger_id="T001",
    )
    service = AlertService(db, tmp_settings)
    produced = service.from_identity(first, "T001", created=True)
    types = {item["alert_type"] for item in produced}
    assert "new_tiger" in types
    service.from_identity(second, "T001")
    service.from_identity(third, "T001")
    repeats = [item for item in service.list_alerts() if item["alert_type"] == "tiger_repeat"]
    assert len(repeats) == 1
    assert repeats[0]["event_key"] == "tiger_repeat:T001:Camera_01"


def test_gnn_prediction_alert_uses_real_prediction(tmp_settings, db):
    _seed_identified_history(db, tiger_id="T001", with_coords=True)
    from backend.api.deps import get_gnn_service

    prediction = get_gnn_service().predict_for_tiger("T001")
    assert prediction["available"] is True
    service = AlertService(db, tmp_settings)
    alert = service.from_prediction(prediction)
    assert alert is not None
    assert alert["alert_type"] == "gnn_prediction"
    assert alert["tiger_id"] == "T001"
    assert "predicted to move toward" in alert["explanation"]
    again = service.from_prediction(prediction)
    assert again is None


def test_alerts_api_read_and_clear(tmp_settings, db):
    _seed_camera(db, "Camera_01")
    TigerRepository(db).upsert_seen("T001", "2026-08-17T14:32:01")
    row = _seed_detection(
        db,
        camera_id="Camera_01",
        stamp="2026-08-17T14:32:01",
        file_hash="t1",
        filename="t1.jpg",
        tiger_id="T001",
    )
    AlertService(db, tmp_settings).from_detection(row, tiger_id="T001")
    client = _client(tmp_settings)
    listed = client.get("/api/alerts").json()
    assert listed["count"] == 1
    alert_id = listed["alerts"][0]["alert_id"]
    summary = client.get("/api/alerts/summary").json()
    assert summary["unread"] == 1
    read = client.post(f"/api/alerts/{alert_id}/read")
    assert read.status_code == 200
    assert read.json()["read"] is True
    cleared = client.post(f"/api/alerts/{alert_id}/clear")
    assert cleared.status_code == 200
    remaining = client.get("/api/alerts").json()
    assert remaining["count"] == 0


def test_pipeline_emits_alert_and_does_not_duplicate(tmp_settings, tmp_path, db):
    CameraRepository(db).create("Camera_01", latitude=21.65, longitude=79.24)
    folder = tmp_path / "Camera_01"
    make_jpeg(folder / "scene.jpg")
    client = _client(tmp_settings)
    job = client.post(
        "/api/import",
        json={"folder_path": str(folder), "camera_id": "Camera_01"},
    ).json()
    import time

    deadline = time.time() + 10
    body = None
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job['job_id']}").json()
        if body["job"]["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert body["job"]["status"] == "completed"
    alerts = client.get("/api/alerts").json()["alerts"]
    tiger_alerts = [item for item in alerts if item["alert_type"] == "tiger_detected"]
    assert len(tiger_alerts) == 1
    classified = Path(tmp_settings.classified_dir)
    tiger_files = list((classified / "tiger").rglob("*.jpg"))
    assert tiger_files
    original = folder / "scene.jpg"
    assert original.is_file()
    assert original.read_bytes()
