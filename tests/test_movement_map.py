"""Camera-trap movement map: cameras, folder mapping, edges, occupancy, GNN."""

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
from backend.graph.builder import GraphService, OCCUPANCY_LABEL
from backend.ingestion.scanner import match_folder_to_camera
from backend.main import create_app
from backend.services.gnn_service import INSUFFICIENT
from backend.services.pipeline import PipelineService
from tests.fakes import FakeDetector
from tests.image_helpers import make_jpeg
from tests.test_api import ApiClient
from tests.test_gnn_inference import _seed_identified_history, _seed_observation


def _client(tmp_settings) -> ApiClient:
    app = create_app()
    pipeline = PipelineService(get_database(), get_settings(), FakeDetector())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return ApiClient(app)


def _seed_camera(db, camera_id, lat=None, lon=None, habitat=None, name=None, enabled=True):
    return CameraRepository(db).create(
        camera_id,
        name=name,
        latitude=lat,
        longitude=lon,
        habitat=habitat,
        enabled=enabled,
    )


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
):
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
        True,
        "none",
        class_id,
        class_name,
    )
    observation_id = None
    if class_name == "tiger":
        observation_id = ObservationRepository(db).create(
            detection_id, tiger_id, 0.88, None, stamp
        )
    return detection_id, observation_id


def test_camera_creation_and_unique_id(tmp_settings):
    client = _client(tmp_settings)
    created = client.post(
        "/api/cameras",
        json={
            "camera_id": "Camera_01",
            "name": "Ridge cam",
            "latitude": 21.65,
            "longitude": 79.24,
            "habitat": "dry deciduous",
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["camera_id"] == "Camera_01"
    assert body["name"] == "Ridge cam"
    assert body["latitude"] == 21.65
    assert body["longitude"] == 79.24
    assert body["enabled"] in {1, True}

    duplicate = client.post("/api/cameras", json={"camera_id": "Camera_01"})
    assert duplicate.status_code == 409
    listed = client.get("/api/cameras").json()["cameras"]
    assert len(listed) == 1
    assert listed[0]["observation_count"] == 0


def test_camera_coordinate_persistence(tmp_settings):
    client = _client(tmp_settings)
    client.post(
        "/api/cameras",
        json={"camera_id": "Camera_01", "latitude": 21.6501, "longitude": 79.2402},
    )
    updated = client.put(
        "/api/cameras/Camera_01",
        json={"latitude": 21.71, "longitude": 79.33, "habitat": "moist deciduous"},
    )
    assert updated.status_code == 200
    stored = client.get("/api/cameras/Camera_01").json()
    assert stored["latitude"] == 21.71
    assert stored["longitude"] == 79.33
    assert stored["habitat"] == "moist deciduous"

    disabled = client.patch("/api/cameras/Camera_01", json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] in {0, False}


def test_folder_maps_to_registered_camera_id(tmp_settings, tmp_path, db):
    CameraRepository(db).create("Camera_01", latitude=21.65, longitude=79.24)
    CameraRepository(db).create("Camera_02", latitude=21.66, longitude=79.25)
    root = tmp_path / "CameraTrapData"
    make_jpeg(root / "Camera_01" / "image1.jpg")
    make_jpeg(root / "Camera_02" / "image3.jpg")
    client = _client(tmp_settings)
    preview = client.post("/api/import/preview", json={"folder_path": str(root)}).json()
    by_name = {item["folder_name"]: item for item in preview["camera_folders"]}
    assert by_name["Camera_01"]["suggested_camera_id"] == "Camera_01"
    assert by_name["Camera_01"]["match_status"] == "matched"
    assert by_name["Camera_01"]["unknown_camera_folder"] is False
    assert by_name["Camera_02"]["match_status"] == "matched"
    match = match_folder_to_camera("Camera_01", {"Camera_01", "Camera_02"})
    assert match["camera_id"] == "Camera_01"
    assert match["unknown_camera_folder"] is False


def test_unknown_camera_folder_is_not_silently_created(tmp_settings, tmp_path, db):
    CameraRepository(db).create("Camera_01", latitude=21.65, longitude=79.24)
    root = tmp_path / "CameraTrapData"
    make_jpeg(root / "Camera_99" / "image5.jpg")
    client = _client(tmp_settings)
    preview = client.post("/api/import/preview", json={"folder_path": str(root)}).json()
    folder = preview["camera_folders"][0]
    assert folder["folder_name"] == "Camera_99"
    assert folder["unknown_camera_folder"] is True
    assert folder["match_status"] == "unknown"

    refused = client.post(
        "/api/import",
        json={"folder_path": str(root / "Camera_99"), "camera_id": "Camera_99"},
    )
    assert refused.status_code == 400
    assert "Unknown camera folder" in refused.json()["detail"]
    cameras = {item["camera_id"] for item in client.get("/api/cameras").json()["cameras"]}
    assert "Camera_99" not in cameras

    mapped = client.post(
        "/api/import",
        json={"folder_path": str(root / "Camera_99"), "camera_id": "Camera_01"},
    )
    assert mapped.status_code == 200
    assert mapped.json()["camera_id"] == "Camera_01"


def test_detection_becomes_persisted_observation(tmp_settings, tmp_path, db):
    CameraRepository(db).create("Camera_01", latitude=21.65, longitude=79.24)
    folder = tmp_path / "Camera_01"
    make_jpeg(folder / "scene.jpg")
    client = _client(tmp_settings)
    job = client.post(
        "/api/import",
        json={"folder_path": str(folder), "camera_id": "Camera_01"},
    ).json()
    assert job["camera_id"] == "Camera_01"

    import time

    deadline = time.time() + 10
    body = None
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job['job_id']}").json()
        if body["job"]["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert body is not None
    assert body["job"]["status"] == "completed"

    detections = client.get("/api/detections").json()["detections"]
    assert detections
    tiger = next(item for item in detections if item["class_name"] == "tiger")
    observation = ObservationRepository(db).get_by_detection(tiger["detection_id"])
    assert observation is not None
    image = ImageRepository(db).get(tiger["image_id"])
    assert image["camera_id"] == "Camera_01"
    assert observation["tiger_id"] is None


def test_same_tiger_across_cameras_creates_movement_edges(db):
    _seed_camera(db, "Camera_01", 21.65, 79.24)
    _seed_camera(db, "Camera_02", 21.70, 79.30)
    _seed_camera(db, "Camera_03", 21.68, 79.28)
    TigerRepository(db).upsert_seen("T001", "2026-01-01T08:00:00")
    _seed_detection(
        db,
        camera_id="Camera_01",
        stamp="2026-01-01T08:00:00",
        file_hash="t1",
        filename="t1.jpg",
        tiger_id="T001",
    )
    _seed_detection(
        db,
        camera_id="Camera_02",
        stamp="2026-01-01T12:00:00",
        file_hash="t2",
        filename="t2.jpg",
        tiger_id="T001",
    )
    _seed_detection(
        db,
        camera_id="Camera_03",
        stamp="2026-01-02T09:00:00",
        file_hash="t3",
        filename="t3.jpg",
        tiger_id="T001",
    )

    graph = GraphService(db)
    edges = graph.build_movement_edges(tiger_id="T001")
    pairs = [(edge.source, edge.target, edge.tiger_id) for edge in edges]
    assert ("Camera_01", "Camera_02", "T001") in pairs
    assert ("Camera_02", "Camera_03", "T001") in pairs
    assert all(edge.kind == "observed" for edge in edges)
    first = next(edge for edge in edges if edge.source == "Camera_01")
    assert first.distance_km is not None
    assert first.distance_km > 0
    assert first.observation_ids
    route = [stop["camera_id"] for stop in graph.build_observed_route("T001")]
    assert route == ["Camera_01", "Camera_02", "Camera_03"]


def test_different_animal_class_edges_stay_separate(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02", 21.70, 79.30)
    _seed_camera(db, "C03", 21.68, 79.28)
    _seed_camera(db, "C04", 21.72, 79.26)
    _seed_camera(db, "C05", 21.64, 79.22)
    TigerRepository(db).upsert_seen("T001", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="t1", filename="t1.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T12:00:00", file_hash="t2", filename="t2.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T13:00:00", file_hash="p1", filename="p1.jpg", class_name="prey")
    _seed_detection(db, camera_id="C03", stamp="2026-01-01T16:00:00", file_hash="p2", filename="p2.jpg", class_name="prey")
    _seed_detection(db, camera_id="C04", stamp="2026-01-01T09:00:00", file_hash="r1", filename="r1.jpg", class_name="rival")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T14:00:00", file_hash="r2", filename="r2.jpg", class_name="rival")
    _seed_detection(db, camera_id="C05", stamp="2026-01-01T07:00:00", file_hash="h1", filename="h1.jpg", class_name="human")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T11:00:00", file_hash="h2", filename="h2.jpg", class_name="human")

    edges = GraphService(db).build_movement_edges()
    by_class = {}
    for edge in edges:
        by_class.setdefault(edge.animal_class, []).append((edge.source, edge.target, edge.tiger_id))
    assert ("C01", "C02", "T001") in by_class["tiger"]
    assert ("C02", "C03", None) in by_class["prey"]
    assert ("C04", "C02", None) in by_class["rival"]
    assert ("C05", "C01", None) in by_class["human"]
    prey_only = GraphService(db).build_movement_edges(animal_class="prey")
    assert all(edge.animal_class == "prey" for edge in prey_only)
    assert not any(edge.tiger_id for edge in prey_only)


def test_occupancy_from_actual_class_counts(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02", 21.70, 79.30)
    TigerRepository(db).upsert_seen("T001", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="t1", filename="t1.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T09:00:00", file_hash="p1", filename="p1.jpg", class_name="prey")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T10:00:00", file_hash="h1", filename="h1.jpg", class_name="human")
    occupancy = GraphService(db).station_occupancy()
    assert occupancy["label"] == OCCUPANCY_LABEL
    by_id = {item["camera_id"]: item for item in occupancy["stations"]}
    assert by_id["C01"]["tiger_detections"] == 1
    assert by_id["C01"]["prey_detections"] == 1
    assert by_id["C01"]["human_detections"] == 0
    assert by_id["C02"]["human_detections"] == 1
    assert by_id["C02"]["tiger_detections"] == 0
    filtered = GraphService(db).station_occupancy(time_from="2026-01-01T09:30:00")
    filtered_ids = {item["camera_id"]: item for item in filtered["stations"]}
    assert filtered_ids["C01"]["all_species_detections"] == 0
    assert filtered_ids["C02"]["human_detections"] == 1


def test_gnn_next_camera_is_not_an_observed_edge(tmp_settings, db):
    _seed_identified_history(db, tiger_id="T001", with_coords=True)
    service = get_gnn_service()
    prediction = service.predict_for_tiger("T001")
    assert prediction["available"] is True
    predicted = prediction["predicted_camera_id"]
    assert predicted

    graph = GraphService(db)
    edges = graph.build_movement_edges(tiger_id="T001")
    assert all(edge.kind == "observed" for edge in edges)
    observed_pairs = {(edge.source, edge.target) for edge in edges}
    last = graph.build_observed_route("T001")[-1]["camera_id"]
    assert (last, predicted) not in observed_pairs or last == predicted
    payload = graph.export_payload(tiger_id="T001")
    assert all(edge["kind"] == "observed" for edge in payload["movement_edges"])
    assert "kind" not in {edge.get("kind") for edge in payload["movement_edges"] if edge.get("kind") == "prediction"}


def test_insufficient_history_and_missing_coordinates(tmp_settings, db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02")
    TigerRepository(db).upsert_seen("T001", "2026-01-01T08:00:00")
    _seed_observation(
        db,
        camera_id="C01",
        tiger_id="T001",
        stamp="2026-01-01T08:00:00",
        file_hash="short-1",
        filename="short1.jpg",
    )
    result = get_gnn_service().predict_for_tiger("T001")
    assert result["available"] is False
    assert result["reason"] == INSUFFICIENT

    occupancy = GraphService(db).station_occupancy()
    by_id = {item["camera_id"]: item for item in occupancy["stations"]}
    assert by_id["C02"]["missing_coordinates"] is True
    assert by_id["C02"]["latitude"] is None


def test_graph_reconstruction_from_database(db):
    _seed_camera(db, "Camera_01", 21.65, 79.24)
    _seed_camera(db, "Camera_02", 21.70, 79.30)
    _seed_camera(db, "Camera_03", 21.68, 79.28)
    TigerRepository(db).upsert_seen("T001", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="Camera_01", stamp="2026-01-01T08:00:00", file_hash="a", filename="a.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="Camera_02", stamp="2026-01-01T12:00:00", file_hash="b", filename="b.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="Camera_03", stamp="2026-01-02T09:00:00", file_hash="c", filename="c.jpg", tiger_id="T001")

    first = GraphService(db).export_payload()
    second = GraphService(db).export_payload()
    first_edges = [
        (edge["source"], edge["target"], edge["tiger_id"], edge["kind"])
        for edge in first["movement_edges"]
    ]
    second_edges = [
        (edge["source"], edge["target"], edge["tiger_id"], edge["kind"])
        for edge in second["movement_edges"]
    ]
    assert first_edges == second_edges
    assert ("Camera_01", "Camera_02", "T001", "observed") in first_edges
    assert ("Camera_02", "Camera_03", "T001", "observed") in first_edges
    cameras = {node["camera_id"] for node in first["camera_graph"]["nodes"]}
    assert cameras == {"Camera_01", "Camera_02", "Camera_03"}


def test_out_of_range_coordinates_are_treated_as_missing(db):
    _seed_camera(db, "Camera_01", 21.65, 79.24)
    _seed_camera(db, "Camera_03", 99.0, 77.0)
    occupancy = GraphService(db).station_occupancy()
    by_id = {item["camera_id"]: item for item in occupancy["stations"]}
    assert by_id["Camera_01"]["missing_coordinates"] is False
    assert by_id["Camera_03"]["missing_coordinates"] is True
    assert by_id["Camera_03"]["latitude"] is None
    assert by_id["Camera_03"]["longitude"] is None


def test_wildlife_entities_and_activity_area(db):
    _seed_camera(db, "C01", 21.65, 79.24)
    _seed_camera(db, "C02", 21.70, 79.30)
    _seed_camera(db, "C03", 21.68, 79.28)
    TigerRepository(db).upsert_seen("T001", "2026-01-01T08:00:00")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T08:00:00", file_hash="t1", filename="t1.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="C01", stamp="2026-01-01T09:00:00", file_hash="t2", filename="t2.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T12:00:00", file_hash="t3", filename="t3.jpg", tiger_id="T001")
    _seed_detection(db, camera_id="C02", stamp="2026-01-01T13:00:00", file_hash="p1", filename="p1.jpg", class_name="prey")
    _seed_detection(db, camera_id="C03", stamp="2026-01-01T16:00:00", file_hash="p2", filename="p2.jpg", class_name="prey")

    graph = GraphService(db)
    entities = graph.build_wildlife_entities()
    tiger = next(item for item in entities["tigers"] if item["tiger_id"] == "T001")
    assert tiger["last_camera"] == "C02"
    assert tiger["observation_count"] == 3
    assert tiger["most_frequent_camera"] == "C01"
    assert tiger["most_frequent_count"] == 2
    assert tiger["latitude"] == 21.70
    prey_cameras = {item["camera_id"] for item in entities["prey"]}
    assert prey_cameras == {"C02", "C03"}
    assert all(item.get("tiger_id") is None for item in entities["prey"])

    activity = graph.build_activity_area("T001")
    assert activity["label"] == "Observed Activity Area"
    assert activity["strongest_camera"] == "C01"
    assert activity["cameras"][0]["observation_count"] == 2
    payload = graph.build_tiger_route("T001")
    assert payload["activity_area"]["strongest_camera"] == "C01"
    assert payload["last_observed_station"] == "C02"


def test_no_edge_without_chronological_observations(db):
    _seed_camera(db, "Camera_01", 21.65, 79.24)
    _seed_camera(db, "Camera_02", 21.70, 79.30)
    edges = GraphService(db).build_movement_edges()
    assert edges == []


def test_delete_camera_blocked_when_images_exist(tmp_settings, db):
    client = _client(tmp_settings)
    client.post("/api/cameras", json={"camera_id": "Camera_01", "latitude": 21.65, "longitude": 79.24})
    ImageRepository(db).create("hash-1", "a.jpg", "a.jpg", "Camera_01", None, None, 10, 10, None)
    refused = client.delete("/api/cameras/Camera_01")
    assert refused.status_code == 409
    empty = client.post("/api/cameras", json={"camera_id": "Camera_02"})
    assert empty.status_code == 200
    deleted = client.delete("/api/cameras/Camera_02")
    assert deleted.status_code == 200
    remaining = {item["camera_id"] for item in client.get("/api/cameras").json()["cameras"]}
    assert remaining == {"Camera_01"}
