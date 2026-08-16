from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    ImageRepository,
    ObservationRepository,
    TigerRepository,
)
from backend.graph.builder import GraphService


def test_graph_builds_camera_nodes_and_tiger_edges(db):
    cameras = CameraRepository(db)
    cameras.upsert("C01", latitude=1.0, longitude=2.0)
    cameras.upsert("C02", latitude=1.1, longitude=2.1)
    images = ImageRepository(db)
    img1 = images.create("h1", "a.jpg", "a.jpg", "C01", "2026-01-01T10:00:00", "exif", 10, 10, None)
    img2 = images.create("h2", "b.jpg", "b.jpg", "C02", "2026-01-01T12:00:00", "exif", 10, 10, None)
    detections = DetectionRepository(db)
    d1 = detections.create(img1, 0, "tiger", 0.9, 0, 0, 5, 5, True, "none", 0, "tiger")
    d2 = detections.create(img2, 0, "tiger", 0.9, 0, 0, 5, 5, True, "none", 0, "tiger")
    tigers = TigerRepository(db)
    tigers.upsert_seen("T017", "2026-01-01T10:00:00")
    observations = ObservationRepository(db)
    observations.create(d1, "T017", 0.91, None, "2026-01-01T10:00:00")
    observations.create(d2, "T017", 0.88, None, "2026-01-01T12:00:00")

    graph = GraphService(db)
    camera_graph = graph.build_camera_graph()
    assert {node.camera_id for node in camera_graph.nodes} == {"C01", "C02"}
    assert len(camera_graph.edges) == 1
    assert camera_graph.edges[0].source == "C01"
    assert camera_graph.edges[0].target == "C02"
    assert camera_graph.edges[0].tiger_id == "T017"

    history = graph.get_tiger_history("T017")
    assert [event.camera_id for event in history] == ["C01", "C02"]
    payload = graph.export_payload()
    assert payload["gnn"]["implemented"] is True
    assert payload["observation_graph"]["events"]
