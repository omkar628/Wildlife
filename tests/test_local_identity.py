"""Human-confirmed local field-tiger identity. No ATRW gallery, no encoder."""

from __future__ import annotations

from backend.api.deps import get_gnn_service, get_identity_service, get_pipeline
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
from backend.reid.gallery import ENCODER_DISABLED
from backend.reid.identity import LocalIdentityService
from backend.services.gnn_service import INSUFFICIENT
from backend.services.pipeline import PipelineService
from tests.fakes import FakeDetector
from tests.test_api import ApiClient


def _client(tmp_settings) -> ApiClient:
    app = create_app()
    pipeline = PipelineService(get_database(), get_settings(), FakeDetector())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return ApiClient(app)


def _seed_observation(db, *, camera_id: str, stamp: str, file_hash: str, filename: str) -> int:
    CameraRepository(db).upsert(camera_id, latitude=19.1, longitude=73.0)
    image_id = ImageRepository(db).create(
        file_hash, filename, filename, camera_id, stamp, "exif", 64, 48, None
    )
    detection_id = DetectionRepository(db).create(
        image_id, 0, "tiger", 0.91, 1, 1, 20, 20, True, "none", 0, "tiger"
    )
    return ObservationRepository(db).create(
        detection_id, None, None, f"data/crops/{filename}", stamp
    )


def test_create_new_tiger_assigns_t001(tmp_settings, db):
    obs_id = _seed_observation(
        db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="a", filename="a.jpg"
    )
    service = LocalIdentityService(db)
    result = service.assign(obs_id, action="create")
    assert result["tiger_id"] == "T001"
    assert result["created"] is True
    row = ObservationRepository(db).get(obs_id)
    assert row["tiger_id"] == "T001"
    assert row["human_verified"] == 1
    assert TigerRepository(db).get("T001") is not None


def test_same_local_id_across_cameras(tmp_settings, db):
    first = _seed_observation(
        db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="c01", filename="c01.jpg"
    )
    second = _seed_observation(
        db, camera_id="C03", stamp="2026-01-02T09:00:00", file_hash="c03", filename="c03.jpg"
    )
    service = get_identity_service()
    created = service.assign(first, action="create")
    assigned = service.assign(second, action="assign", tiger_id=created["tiger_id"])
    assert created["tiger_id"] == assigned["tiger_id"] == "T001"
    history = ObservationRepository(db).list_for_tiger("T001")
    assert [row["camera_id"] for row in history] == ["C01", "C03"]
    catalog = service.gallery.catalog()
    assert catalog[0]["tiger_id"] == "T001"
    assert catalog[0]["observation_count"] == 2


def test_changing_mistaken_assignment(tmp_settings, db):
    one = _seed_observation(
        db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="m1", filename="m1.jpg"
    )
    two = _seed_observation(
        db, camera_id="C02", stamp="2026-01-01T12:00:00", file_hash="m2", filename="m2.jpg"
    )
    three = _seed_observation(
        db, camera_id="C03", stamp="2026-01-02T08:00:00", file_hash="m3", filename="m3.jpg"
    )
    service = LocalIdentityService(db)
    service.assign(one, action="create")
    service.assign(two, action="create")
    assert ObservationRepository(db).get(two)["tiger_id"] == "T002"
    service.assign(three, action="assign", tiger_id="T002")
    moved = service.assign(three, action="assign", tiger_id="T001")
    assert moved["tiger_id"] == "T001"
    assert ObservationRepository(db).get(three)["tiger_id"] == "T001"
    assert [row["observation_id"] for row in ObservationRepository(db).list_for_tiger("T001")] == [one, three]
    assert [row["observation_id"] for row in ObservationRepository(db).list_for_tiger("T002")] == [two]


def test_rejects_atrw_numeric_ids(tmp_settings, db):
    obs_id = _seed_observation(
        db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="atrw", filename="atrw.jpg"
    )
    service = LocalIdentityService(db)
    try:
        service.assign(obs_id, action="assign", tiger_id="250")
        raise AssertionError("ATRW ID should have been rejected")
    except ValueError as exc:
        assert "ATRW" in str(exc)
    assert ObservationRepository(db).get(obs_id)["tiger_id"] is None


def test_encoder_stays_disabled_without_backend(tmp_settings, db):
    gallery = LocalIdentityService(db).gallery
    assert gallery.encoder_enabled() is False
    result = gallery.identify_crop(tmp_settings.project_root / "missing.jpg")
    assert result.available is False
    assert result.tiger_id is None
    assert result.reason == ENCODER_DISABLED


def test_gnn_unavailable_with_fewer_than_five_observations(tmp_settings, db):
    service = LocalIdentityService(db)
    first_id = None
    for index, camera in enumerate(("C01", "C02", "C03", "C01"), start=1):
        obs_id = _seed_observation(
            db,
            camera_id=camera,
            stamp=f"2026-01-0{index}T08:00:00",
            file_hash=f"g{index}",
            filename=f"g{index}.jpg",
        )
        if first_id is None:
            first_id = service.assign(obs_id, action="create")["tiger_id"]
        else:
            service.assign(obs_id, action="assign", tiger_id=first_id)
    assert first_id == "T001"
    assert ObservationRepository(db).list_for_tiger("T001")
    assert len(ObservationRepository(db).list_for_tiger("T001")) == 4

    prediction = get_gnn_service().predict_for_tiger("T001")
    assert prediction["available"] is False
    assert prediction["reason"] == INSUFFICIENT
    assert "5 identified" in prediction["detail"]


def test_identity_api_create_and_list(tmp_settings, db):
    obs_id = _seed_observation(
        db, camera_id="C07", stamp="2026-01-03T10:00:00", file_hash="api", filename="api.jpg"
    )
    client = _client(tmp_settings)
    queue = client.get("/api/observations/unidentified").json()
    assert queue["pending"] == 1
    assert queue["next_tiger_id"] == "T001"

    created = client.post(
        f"/api/observations/{obs_id}/identity",
        json={"action": "create"},
    )
    assert created.status_code == 200
    assert created.json()["tiger_id"] == "T001"

    blocked = client.post(
        f"/api/observations/{obs_id}/identity",
        json={"action": "assign", "tiger_id": "250"},
    )
    assert blocked.status_code == 400

    tiger = client.get("/api/tigers/T001").json()
    assert tiger["tiger"]["tiger_id"] == "T001"
    assert tiger["history"]
    assert tiger["history"][0]["camera_id"] == "C07"

    health = client.get("/api/health").json()
    assert health["reid"]["uses_atrw_gallery"] is False
    assert health["reid"]["local_identity"]["assigns_atrw_ids"] is False
    assert health["reid"]["local_identity"]["id_namespace"] == "T001+"
