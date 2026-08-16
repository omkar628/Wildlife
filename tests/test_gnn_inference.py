"""GNN inference tests using application-native cameras/observations.

Does not load the synthetic 20-forest parquet dataset.
"""

from __future__ import annotations

from backend.api.deps import get_gnn_service, get_pipeline
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
from backend.services.gnn_service import INSUFFICIENT, GNNService
from backend.services.pipeline import PipelineService
from tests.fakes import FakeDetector
from tests.test_api import ApiClient


def _client(tmp_settings) -> ApiClient:
    app = create_app()
    pipeline = PipelineService(get_database(), get_settings(), FakeDetector())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return ApiClient(app)


def _seed_camera(db, camera_id: str, lat=None, lon=None, habitat="dry deciduous", metadata=None):
    return CameraRepository(db).upsert(
        camera_id,
        latitude=lat,
        longitude=lon,
        habitat=habitat,
        metadata=metadata,
    )


def _seed_observation(
    db,
    *,
    camera_id: str,
    tiger_id: str,
    stamp: str,
    file_hash: str,
    filename: str,
):
    images = ImageRepository(db)
    detections = DetectionRepository(db)
    observations = ObservationRepository(db)
    image_id = images.create(
        file_hash,
        f"{filename}",
        filename,
        camera_id,
        stamp,
        "exif",
        64,
        48,
        None,
    )
    detection_id = detections.create(
        image_id, 0, "tiger", 0.91, 1, 1, 20, 20, True, "none", 0, "tiger"
    )
    observations.create(detection_id, tiger_id, 0.88, None, stamp)


def _seed_identified_history(db, tiger_id: str = "T017", with_coords: bool = True):
    cameras = [
        ("C01", 19.10, 73.00),
        ("C02", 19.14, 73.04),
        ("C03", 19.18, 73.08),
    ]
    for camera_id, lat, lon in cameras:
        if with_coords:
            _seed_camera(db, camera_id, lat, lon)
        else:
            _seed_camera(db, camera_id)

    TigerRepository(db).upsert_seen(tiger_id, "2026-01-01T08:00:00")
    path = [
        ("C01", "2026-01-01T08:00:00"),
        ("C02", "2026-01-01T12:00:00"),
        ("C01", "2026-01-02T07:00:00"),
        ("C03", "2026-01-02T16:00:00"),
        ("C02", "2026-01-03T09:00:00"),
    ]
    for index, (camera_id, stamp) in enumerate(path, start=1):
        _seed_observation(
            db,
            camera_id=camera_id,
            tiger_id=tiger_id,
            stamp=stamp,
            file_hash=f"{tiger_id}-{index}",
            filename=f"{tiger_id}_{index}.jpg",
        )


def test_gnn_model_loads(tmp_settings):
    service = GNNService(tmp_settings, get_database())
    status = service.status()
    assert status["loaded"] is True
    assert status["path"].endswith("gnn_model_v3_optimized_best.pt")
    assert status["device"] in {"cpu", "cuda", "cuda:0"}
    assert status["version"] == "v3.1_optimized_multi_forest"
    assert status["reason"] is None


def test_health_reports_gnn_status(tmp_settings):
    client = _client(tmp_settings)
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    gnn = body["gnn"]
    assert gnn["loaded"] is True
    assert "device" in gnn
    assert gnn["path"]
    assert gnn["reason"] is None


def test_insufficient_history_returns_structured_response(tmp_settings, db):
    _seed_camera(db, "C01", 19.1, 73.0)
    _seed_camera(db, "C02", 19.2, 73.1)
    TigerRepository(db).upsert_seen("T017", "2026-01-01T08:00:00")
    _seed_observation(
        db,
        camera_id="C01",
        tiger_id="T017",
        stamp="2026-01-01T08:00:00",
        file_hash="short-1",
        filename="short1.jpg",
    )
    _seed_observation(
        db,
        camera_id="C02",
        tiger_id="T017",
        stamp="2026-01-01T12:00:00",
        file_hash="short-2",
        filename="short2.jpg",
    )

    service = get_gnn_service()
    result = service.predict_for_tiger("T017")
    assert result["available"] is False
    assert result["reason"] == INSUFFICIENT
    assert "5 identified" in result["detail"]

    client = _client(tmp_settings)
    body = client.get("/api/graph/predictions", params={"tiger_id": "T017"}).json()
    assert body["available"] is False
    assert body["reason"] == INSUFFICIENT


def test_missing_coordinates_returns_insufficient(tmp_settings, db):
    _seed_identified_history(db, with_coords=False)
    service = get_gnn_service()
    result = service.predict_for_tiger("T017")
    assert result["available"] is False
    assert result["reason"] == INSUFFICIENT
    assert "latitude" in result["detail"].lower() or "longitude" in result["detail"].lower()


def test_prediction_endpoint_does_not_crash_without_tiger(tmp_settings):
    client = _client(tmp_settings)
    missing = client.get("/api/graph/predictions")
    assert missing.status_code == 200
    body = missing.json()
    assert body["available"] is False
    assert body["reason"] == INSUFFICIENT

    unknown = client.get("/api/graph/predictions", params={"tiger_id": "T999"})
    assert unknown.status_code == 200
    payload = unknown.json()
    assert payload["available"] is False
    assert payload["reason"] == INSUFFICIENT


def test_valid_inference_with_application_native_data(tmp_settings, db):
    _seed_identified_history(db, tiger_id="T017", with_coords=True)
    service = get_gnn_service()
    result = service.predict_for_tiger("T017")

    assert result["available"] is True, result
    assert result["tiger_id"] == "T017"
    assert result["predicted_camera_id"] in {"C01", "C03"}
    assert result["predicted_camera_id"] != "C02"
    assert 0.0 <= float(result["confidence"]) <= 1.0
    assert result["ranked_candidates"]
    assert result["ranked_candidates"][0]["camera_id"] == result["predicted_camera_id"]
    assert [item["camera_id"] for item in result["history"]] == [
        "C01",
        "C02",
        "C01",
        "C03",
        "C02",
    ]
    assert result["feature_degraded"] is True
    assert result["prediction_timestamp"]
    assert result["model"]["version"] == "v3.1_optimized_multi_forest"

    client = _client(tmp_settings)
    body = client.get("/api/graph/predictions", params={"tiger_id": "T017"}).json()
    assert body["available"] is True
    assert body["predicted_camera_id"] == result["predicted_camera_id"]

    graph = client.get("/api/graph").json()
    assert graph["gnn"]["loaded"] is True
    assert graph["gnn"]["predictions"]
    assert graph["gnn"]["predictions"][0]["available"] is True
