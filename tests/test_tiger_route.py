"""Observed route, occupancy, home range, and GNN-backed predicted station."""

from __future__ import annotations

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
from backend.graph.builder import (
    NOT_ENOUGH_OBSERVATIONS,
    PREDICTION_UNAVAILABLE,
    GraphService,
    infer_zone_type,
)
from backend.main import create_app
from backend.services.pipeline import PipelineService
from tests.fakes import FakeDetector
from tests.test_api import ApiClient
from tests.test_gnn_inference import _seed_identified_history, _seed_observation


def _client(tmp_settings) -> ApiClient:
    app = create_app()
    pipeline = PipelineService(get_database(), get_settings(), FakeDetector())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return ApiClient(app)


def _seed_camera(db, camera_id, lat=None, lon=None, habitat=None, metadata=None):
    return CameraRepository(db).upsert(
        camera_id,
        latitude=lat,
        longitude=lon,
        habitat=habitat,
        metadata=metadata,
    )


def _seed_detection(db, *, camera_id, stamp, file_hash, filename, class_name="tiger", tiger_id=None):
    images = ImageRepository(db)
    detections = DetectionRepository(db)
    image_id = images.create(
        file_hash,
        filename,
        filename,
        camera_id,
        stamp,
        "exif",
        64,
        48,
        None,
    )
    detection_id = detections.create(
        image_id,
        0 if class_name == "tiger" else 1,
        class_name,
        0.91,
        1,
        1,
        20,
        20,
        True,
        "none",
        0 if class_name == "tiger" else 1,
        class_name,
    )
    if class_name == "tiger":
        ObservationRepository(db).create(detection_id, tiger_id, 0.88, None, stamp)
    return detection_id


def test_infer_zone_type_from_recorded_fields_only():
    assert infer_zone_type("core zone", None) == "core"
    assert infer_zone_type("buffer", None) == "buffer"
    assert infer_zone_type("village fringe", None) == "village-adjacent"
    assert infer_zone_type("dry deciduous", '{"zone_type":"buffer"}') == "buffer"
    assert infer_zone_type("dry deciduous", None) is None
    assert infer_zone_type(None, None) is None


def test_observed_route_is_chronological_with_registered_coordinates(db):
    _seed_camera(db, "C01", 21.65, 79.24, habitat="core")
    _seed_camera(db, "C03", 21.70, 79.30, habitat="buffer")
    _seed_camera(db, "C07", 21.68, 79.28, habitat="village-adjacent")
    TigerRepository(db).upsert_seen("T017", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="a", filename="a.jpg", tiger_id="T017")
    _seed_detection(db, camera_id="C03", stamp="2026-01-01T12:00:00", file_hash="b", filename="b.jpg", tiger_id="T017")
    _seed_detection(db, camera_id="C07", stamp="2026-01-02T09:00:00", file_hash="c", filename="c.jpg", tiger_id="T017")

    payload = GraphService(db).build_tiger_route("T017")
    assert [stop["camera_id"] for stop in payload["observed_route"]] == ["C01", "C03", "C07"]
    assert payload["observed_route"][0]["latitude"] == 21.65
    assert payload["observed_route"][0]["longitude"] == 79.24
    assert payload["current_station"]["camera_id"] == "C07"
    assert payload["current_station"]["timestamp"] == "2026-01-02T09:00:00"
    assert payload["last_observed_station"] == "C07"
    assert payload["visited_stations"] == ["C01", "C03", "C07"]
    assert payload["observation_count"] == 3


def test_current_station_is_latest_observation(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02", 21.66, 79.25)
    TigerRepository(db).upsert_seen("T017", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T18:00:00", file_hash="late", filename="late.jpg", tiger_id="T017")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="early", filename="early.jpg", tiger_id="T017")

    payload = GraphService(db).build_tiger_route("T017")
    assert payload["current_station"]["camera_id"] == "C02"
    assert payload["observed_route"][0]["camera_id"] == "C01"
    assert payload["observed_route"][1]["camera_id"] == "C02"


def test_predicted_station_uses_existing_gnn_and_registered_coordinates(tmp_settings, db):
    _seed_identified_history(db, tiger_id="T017", with_coords=True)
    client = _client(tmp_settings)
    body = client.get("/api/graph/tigers/T017/route").json()

    assert body["tiger_id"] == "T017"
    assert [stop["camera_id"] for stop in body["observed_route"]] == [
        "C01",
        "C02",
        "C01",
        "C03",
        "C02",
    ]
    assert body["current_station"]["camera_id"] == "C02"
    assert body["predictions"]["available"] is True, body["predictions"]
    predicted = body["predictions"]["predicted_camera_id"]
    assert predicted in {"C01", "C03"}
    assert predicted != "C02"

    registered = {
        camera["camera_id"]: camera
        for camera in CameraRepository(db).list_all()
    }
    assert predicted in registered
    assert body["predictions"]["latitude"] == registered[predicted]["latitude"]
    assert body["predictions"]["longitude"] == registered[predicted]["longitude"]
    for candidate in body["predictions"]["ranked_candidates"]:
        assert candidate["camera_id"] in registered
        if candidate["latitude"] is not None:
            assert candidate["latitude"] == registered[candidate["camera_id"]]["latitude"]
            assert candidate["longitude"] == registered[candidate["camera_id"]]["longitude"]
        assert candidate["registered"] is True


def test_predicted_coordinates_never_invented(tmp_settings, db):
    _seed_identified_history(db, tiger_id="T017", with_coords=True)
    registered_ids = {row["camera_id"] for row in CameraRepository(db).list_all()}
    client = _client(tmp_settings)
    body = client.get("/api/graph/tigers/T017/route").json()
    for candidate in body["predictions"]["ranked_candidates"]:
        assert candidate["camera_id"] in registered_ids
        assert candidate["registered"] is True


def test_occupancy_from_actual_observations(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02", 21.70, 79.30)
    _seed_camera(db, "C03", 21.68, 79.28)
    TigerRepository(db).upsert_seen("T017", "2026-01-01T08:00:00")
    TigerRepository(db).upsert_seen("T018", "2026-01-01T09:00:00")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="t1", filename="t1.jpg", tiger_id="T017")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T10:00:00", file_hash="t2", filename="t2.jpg", tiger_id="T018")
    _seed_detection(
        db,
        camera_id="C01",
        stamp="2026-01-01T11:00:00",
        file_hash="p1",
        filename="p1.jpg",
        class_name="prey",
    )
    _seed_detection(db, camera_id="C02", stamp="2026-01-02T09:00:00", file_hash="t3", filename="t3.jpg", tiger_id="T017")

    occupancy = GraphService(db).station_occupancy(tiger_id="T017")
    by_id = {item["camera_id"]: item for item in occupancy["stations"]}
    assert by_id["C01"]["tiger_captures"] == 2
    assert by_id["C01"]["unique_tigers"] == 2
    assert by_id["C01"]["all_species_detections"] == 3
    assert by_id["C01"]["selected_tiger_captures"] == 1
    assert by_id["C01"]["latest_tiger_id"] == "T018"
    assert by_id["C02"]["tiger_captures"] == 1
    assert by_id["C02"]["unique_tigers"] == 1
    assert by_id["C02"]["selected_tiger_captures"] == 1
    assert by_id["C03"]["tiger_captures"] == 0
    assert by_id["C03"]["all_species_detections"] == 0
    assert by_id["C03"]["occupancy_level_tiger"] == "none"
    assert occupancy["supported_modes"] == [
        "all_species",
        "tiger",
        "prey",
        "rival",
        "human",
        "selected_tiger",
    ]
    assert occupancy["label"] == "Observed activity / occupancy"


def test_home_range_requires_three_unique_registered_coordinates(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02", 21.70, 79.30)
    _seed_camera(db, "C03", 21.68, 79.28)
    TigerRepository(db).upsert_seen("T017", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="a", filename="a.jpg", tiger_id="T017")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T12:00:00", file_hash="b", filename="b.jpg", tiger_id="T017")

    short = GraphService(db).build_tiger_route("T017")
    assert short["home_range"]["available"] is False
    assert short["home_range"]["reason"] == NOT_ENOUGH_OBSERVATIONS
    assert short["home_range"]["polygon"] == []

    _seed_detection(db, camera_id="C03", stamp="2026-01-02T09:00:00", file_hash="c", filename="c.jpg", tiger_id="T017")
    ready = GraphService(db).build_tiger_route("T017")
    assert ready["home_range"]["available"] is True
    assert ready["home_range"]["label"] == "Estimated home range"
    assert len(ready["home_range"]["polygon"]) >= 3
    registered = {
        (camera["latitude"], camera["longitude"])
        for camera in CameraRepository(db).list_all()
    }
    for point in ready["home_range"]["polygon"]:
        assert (point["latitude"], point["longitude"]) in registered


def test_tiger_with_no_observations(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    TigerRepository(db).upsert_seen("T099", "2026-01-01T08:00:00")
    payload = GraphService(db).build_tiger_route("T099")
    assert payload["observed_route"] == []
    assert payload["current_station"] is None
    assert payload["observation_count"] == 0
    assert payload["home_range"]["available"] is False
    assert payload["home_range"]["reason"] == NOT_ENOUGH_OBSERVATIONS
    assert payload["predictions"]["available"] is False
    assert payload["predictions"]["summary"] == PREDICTION_UNAVAILABLE


def test_camera_with_missing_coordinates_is_listed_but_not_plotted(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02")
    _seed_camera(db, "C03", 21.68, 79.28)
    TigerRepository(db).upsert_seen("T017", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="a", filename="a.jpg", tiger_id="T017")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T12:00:00", file_hash="b", filename="b.jpg", tiger_id="T017")
    _seed_detection(db, camera_id="C03", stamp="2026-01-02T09:00:00", file_hash="c", filename="c.jpg", tiger_id="T017")

    payload = GraphService(db).build_tiger_route("T017")
    by_id = {stop["observation_id"]: stop for stop in payload["observed_route"]}
    missing = next(stop for stop in payload["observed_route"] if stop["camera_id"] == "C02")
    assert missing["latitude"] is None
    assert missing["longitude"] is None
    assert missing["missing_coordinates"] is True
    assert payload["current_station"]["camera_id"] == "C03"
    assert payload["home_range"]["available"] is False
    occupancy = {item["camera_id"]: item for item in payload["occupancy"]["stations"]}
    assert occupancy["C02"]["tiger_captures"] == 1
    assert occupancy["C02"]["missing_coordinates"] is True
    assert by_id


def test_missing_prediction_does_not_break_graph_or_route(tmp_settings, db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02", 21.70, 79.30)
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

    client = _client(tmp_settings)
    graph = client.get("/api/graph")
    assert graph.status_code == 200
    graph_body = graph.json()
    assert "camera_graph" in graph_body
    assert "occupancy" in graph_body
    assert {item["camera_id"] for item in graph_body["occupancy"]["stations"]} == {"C01", "C02"}

    route = client.get("/api/graph/tigers/T017/route")
    assert route.status_code == 200
    body = route.json()
    assert [stop["camera_id"] for stop in body["observed_route"]] == ["C01", "C02"]
    assert body["current_station"]["camera_id"] == "C02"
    assert body["predictions"]["available"] is False
    assert body["predictions"]["summary"] == PREDICTION_UNAVAILABLE
    assert body["home_range"]["available"] is False


def test_unknown_tiger_route_still_returns_occupancy(tmp_settings, db):
    _seed_camera(db, "C01", 21.65, 79.24)
    client = _client(tmp_settings)
    body = client.get("/api/graph/tigers/T999/route").json()
    assert body["tiger_id"] == "T999"
    assert body["observed_route"] == []
    assert body["current_station"] is None
    assert body["predictions"]["available"] is False
    assert body["occupancy"]["stations"][0]["camera_id"] == "C01"
