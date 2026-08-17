"""MegaDescriptor local Re-ID: gallery, matching, identity, GNN history."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.config import reload_settings
from backend.database.repositories import (
    CameraRepository,
    DetectionRepository,
    EmbeddingRepository,
    ImageRepository,
    JobRepository,
    ObservationRepository,
    TigerRepository,
)
from backend.graph.builder import GraphService
from backend.reid.identity import LocalIdentityService
from backend.reid.matching import decide_match, is_atrw_numeric_id, is_local_tiger_id
from backend.reid.megadescriptor import (
    MegaDescriptorEncoder,
    l2_normalize,
    select_device,
)
from backend.services.pipeline import PipelineService
from tests.fakes import FakeDetector, FakeEncoder
from tests.image_helpers import make_jpeg


def _seed_observation(db, tmp_path: Path, *, camera_id: str, stamp: str, name: str, lat=19.1, lon=73.0) -> int:
    CameraRepository(db).upsert(camera_id, latitude=lat, longitude=lon)
    crop = make_jpeg(tmp_path / name)
    image_id = ImageRepository(db).create(
        name, name, name, camera_id, stamp, "exif", 64, 48, None
    )
    detection_id = DetectionRepository(db).create(
        image_id, 0, "tiger", 0.91, 1, 1, 20, 20, True, "none", 0, "tiger"
    )
    return ObservationRepository(db).create(detection_id, None, None, str(crop), stamp)


def _vec(*values: float) -> np.ndarray:
    return l2_normalize(np.asarray(values, dtype=np.float32))


def test_select_device_cpu_fallback():
    assert select_device("cpu", cuda_available=True) == "cpu"
    assert select_device("auto", cuda_available=False) == "cpu"
    assert select_device("cuda", cuda_available=False) == "cpu"


def test_select_device_uses_cuda_when_available():
    assert select_device("auto", cuda_available=True) == "cuda"
    assert select_device("cuda:0", cuda_available=True) == "cuda:0"


def test_l2_normalize_unit_length():
    vector = l2_normalize(np.asarray([3.0, 4.0], dtype=np.float32))
    assert vector.shape == (2,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6
    assert abs(vector[0] - 0.6) < 1e-6
    zero = l2_normalize(np.zeros(4, dtype=np.float32))
    assert float(np.linalg.norm(zero)) == 0.0


def test_megadescriptor_loads_when_enabled(tmp_settings, monkeypatch):
    pytest.importorskip("timm")
    monkeypatch.setenv("WI_REID_ENABLED", "true")
    settings = reload_settings()
    encoder = MegaDescriptorEncoder(settings)
    if not encoder.is_available():
        pytest.skip(str(encoder.status().get("reason")))
    assert encoder.status()["loaded"] is True
    assert encoder.status()["uses_atrw_gallery"] is False
    assert encoder.device_name in {"cpu", "cuda", "cuda:0"}
    crop = Path(tmp_settings.crops_dir) / "probe.jpg"
    make_jpeg(crop)
    embedding = encoder.embed_crop(crop)
    assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5
    if __import__("torch").cuda.is_available():
        assert encoder.device_name.startswith("cuda")
    else:
        assert encoder.device_name == "cpu"


def test_same_tiger_accumulates_multiple_embeddings(tmp_settings, db, tmp_path):
    encoder = FakeEncoder()
    service = LocalIdentityService(db, tmp_settings, encoder)
    first = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="a.jpg")
    second = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T12:00:00", name="b.jpg")
    third = _seed_observation(db, tmp_path, camera_id="C03", stamp="2026-01-02T08:00:00", name="c.jpg")
    service.assign(first, action="create")
    service.assign(second, action="assign", tiger_id="T001")
    service.assign(third, action="assign", tiger_id="T001")
    assert EmbeddingRepository(db).count_for_tiger("T001") == 3
    assert [row["observation_id"] for row in EmbeddingRepository(db).list_for_tiger("T001")] == [
        first,
        second,
        third,
    ]


def test_matching_returns_correct_local_id(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "t001a.jpg": _vec(1, 0, 0, 0),
            "t002a.jpg": _vec(0, 1, 0, 0),
            "query.jpg": _vec(0.98, 0.02, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    one = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="t001a.jpg")
    two = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T09:00:00", name="t002a.jpg")
    service.assign(one, action="create")
    service.assign(two, action="create")
    query = _seed_observation(db, tmp_path, camera_id="C03", stamp="2026-01-01T10:00:00", name="query.jpg")
    result = service.identify_new_observation(query)
    assert result["matched"] is True
    assert result["tiger_id"] == "T001"
    assert result["needs_review"] is False
    assert ObservationRepository(db).get(query)["tiger_id"] == "T001"


def test_low_confidence_remains_unidentified(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "t001a.jpg": _vec(1, 0, 0, 0),
            "query.jpg": _vec(0.7, 0.71414, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    known = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="t001a.jpg")
    service.assign(known, action="create")
    query = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T10:00:00", name="query.jpg")
    result = service.identify_new_observation(query)
    assert result["matched"] is False
    assert result["tiger_id"] is None
    assert result["needs_review"] is True
    assert result["suggested_tiger_id"] == "T001"
    assert ObservationRepository(db).get(query)["tiger_id"] is None


def test_empty_gallery_auto_creates_first_local_id(tmp_settings, db, tmp_path):
    service = LocalIdentityService(db, tmp_settings, FakeEncoder())
    observation_id = _seed_observation(
        db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="first.jpg"
    )
    result = service.identify_new_observation(observation_id)
    assert result["tiger_id"] == "T001"
    assert result["decision"] == "create"
    assert result["needs_review"] is False
    assert ObservationRepository(db).get(observation_id)["tiger_id"] == "T001"
    assert EmbeddingRepository(db).count_for_tiger("T001") == 1


def test_empty_gallery_does_not_invent_id_without_encoder(tmp_settings, db, tmp_path):
    service = LocalIdentityService(db, tmp_settings, encoder=None)
    observation_id = _seed_observation(
        db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="first.jpg"
    )
    result = service.identify_new_observation(observation_id)
    assert result["tiger_id"] is None
    assert result["needs_review"] is True
    assert ObservationRepository(db).get(observation_id)["tiger_id"] is None
    assert TigerRepository(db).list_all() == []


def test_atrw_numeric_ids_cannot_enter_production():
    assert is_atrw_numeric_id("250") is True
    assert is_local_tiger_id("250") is False
    gallery = [
        {
            "tiger_id": "250",
            "vector": _vec(1, 0, 0, 0).tobytes(),
        }
    ]
    decision = decide_match(_vec(1, 0, 0, 0), gallery, match_threshold=0.5, review_threshold=0.2)
    assert decision.matched is False
    assert decision.tiger_id is None
    assert decision.candidates == []


def test_human_assignment_updates_identity_and_gallery(tmp_settings, db, tmp_path):
    service = LocalIdentityService(db, tmp_settings, FakeEncoder())
    observation_id = _seed_observation(
        db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="human.jpg"
    )
    created = service.assign(observation_id, action="create")
    assert created["tiger_id"] == "T001"
    row = ObservationRepository(db).get(observation_id)
    assert row["tiger_id"] == "T001"
    assert row["human_verified"] == 1
    assert EmbeddingRepository(db).count_for_tiger("T001") == 1


def test_confirmed_identity_is_visible_to_gnn_history(tmp_settings, db, tmp_path):
    service = LocalIdentityService(db, tmp_settings, FakeEncoder())
    first = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="g1.jpg", lat=21.6, lon=79.2)
    second = _seed_observation(db, tmp_path, camera_id="C03", stamp="2026-01-02T09:00:00", name="g2.jpg", lat=21.7, lon=79.3)
    service.assign(first, action="create")
    service.assign(second, action="assign", tiger_id="T001")
    history = GraphService(db).get_tiger_history("T001")
    assert [event.camera_id for event in history] == ["C01", "C03"]
    assert history[0].tiger_id == "T001"


def test_failed_reid_does_not_break_import(db, tmp_settings, tmp_path):
    folder = tmp_path / "C01"
    make_jpeg(folder / "scene.jpg")
    CameraRepository(db).upsert("C01", latitude=21.65, longitude=79.24)
    encoder = FakeEncoder()
    encoder.fail_paths.add("detection_1.jpg")
    identity = LocalIdentityService(db, tmp_settings, encoder)
    jobs = JobRepository(db)
    job_id = jobs.create(str(folder), "C01", tmp_settings.confidence_auto_accept)
    PipelineService(db, tmp_settings, FakeDetector(), identity=identity)._run_job(job_id, folder, "C01")
    job = jobs.get(job_id)
    assert job["status"] == "completed"
    assert job["processed"] == 1
    observation = ObservationRepository(db).list_all()[0]
    assert observation["tiger_id"] is None
    camera = CameraRepository(db).get("C01")
    assert camera["latitude"] == 21.65
    assert camera["longitude"] == 79.24
    image = ImageRepository(db).list_recent()[0]
    assert image["camera_id"] == "C01"


def test_camera_coordinates_stay_on_the_camera(tmp_settings, db, tmp_path):
    encoder = FakeEncoder({"keep.jpg": _vec(1, 0, 0, 0)})
    service = LocalIdentityService(db, tmp_settings, encoder)
    observation_id = _seed_observation(
        db,
        tmp_path,
        camera_id="C07",
        stamp="2026-01-03T10:00:00",
        name="keep.jpg",
        lat=21.11,
        lon=79.55,
    )
    service.assign(observation_id, action="create")
    joined = ObservationRepository(db).get_joined(observation_id)
    assert joined["camera_id"] == "C07"
    camera = CameraRepository(db).get("C07")
    assert camera["latitude"] == 21.11
    assert camera["longitude"] == 79.55


def test_unconfirmed_tiger_class_does_not_auto_assign(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "t001a.jpg": _vec(1, 0, 0, 0),
            "query.jpg": _vec(0.98, 0.02, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    known = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="t001a.jpg")
    service.assign(known, action="create")
    query = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T10:00:00", name="query.jpg")
    DetectionRepository(db).update_review_decision(
        int(ObservationRepository(db).get(query)["detection_id"]),
        0,
        "tiger",
        "pending",
        accepted=False,
    )
    result = service.identify_new_observation(query)
    assert result["matched"] is False
    assert result["tiger_id"] is None
    assert result["suggested_tiger_id"] == "T001"
    assert ObservationRepository(db).get(query)["tiger_id"] is None


def test_keep_unidentified_does_not_assign(tmp_settings, db, tmp_path):
    service = LocalIdentityService(db, tmp_settings, FakeEncoder())
    observation_id = _seed_observation(
        db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="keep.jpg"
    )
    result = service.assign(observation_id, action="keep")
    assert result["tiger_id"] is None
    assert result["kept_unidentified"] is True
    assert ObservationRepository(db).get(observation_id)["tiger_id"] is None


def test_automatic_same_tiger_matching(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "one.jpg": _vec(1, 0, 0, 0),
            "two.jpg": _vec(0.98, 0.199, 0, 0),
            "three.jpg": _vec(0.97, 0.243, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    first = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="one.jpg")
    second = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T10:00:00", name="two.jpg")
    third = _seed_observation(db, tmp_path, camera_id="C03", stamp="2026-01-01T12:00:00", name="three.jpg")
    assert service.identify_new_observation(first)["tiger_id"] == "T001"
    assert service.identify_new_observation(second)["tiger_id"] == "T001"
    assert service.identify_new_observation(third)["tiger_id"] == "T001"
    assert ObservationRepository(db).get(second)["tiger_id"] == "T001"
    assert ObservationRepository(db).get(third)["tiger_id"] == "T001"
    assert EmbeddingRepository(db).count_for_tiger("T001") == 3
    assert service.unidentified() == []


def test_different_tiger_creates_next_local_id(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "t001.jpg": _vec(1, 0, 0, 0),
            "t002.jpg": _vec(0, 1, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    first = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="t001.jpg")
    second = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T10:00:00", name="t002.jpg")
    assert service.identify_new_observation(first)["tiger_id"] == "T001"
    assert service.identify_new_observation(second)["tiger_id"] == "T002"
    assert {row["tiger_id"] for row in TigerRepository(db).list_all()} == {"T001", "T002"}
    assert service.unidentified() == []


def test_close_top_two_matches_go_to_human_review(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "t001.jpg": _vec(1, 0, 0, 0),
            "t002.jpg": _vec(1, 0.2, 0, 0),
            "query.jpg": _vec(1, 0.15, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    one = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="t001.jpg")
    two = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T09:00:00", name="t002.jpg")
    service.assign(one, action="create")
    service.assign(two, action="create")
    query = _seed_observation(db, tmp_path, camera_id="C03", stamp="2026-01-01T10:00:00", name="query.jpg")
    result = service.identify_new_observation(query)
    assert result["tiger_id"] is None
    assert result["needs_review"] is True
    assert result["decision"] == "review"
    assert ObservationRepository(db).get(query)["tiger_id"] is None
    queue = service.unidentified()
    assert [row["observation_id"] for row in queue] == [query]


def test_gnn_history_only_includes_assigned_local_ids(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "known.jpg": _vec(1, 0, 0, 0),
            "unsure.jpg": _vec(0.7, 0.71414, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    first = _seed_observation(
        db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="known.jpg", lat=21.6, lon=79.2
    )
    unsure = _seed_observation(
        db, tmp_path, camera_id="C02", stamp="2026-01-01T10:00:00", name="unsure.jpg", lat=21.7, lon=79.3
    )
    service.identify_new_observation(first)
    service.identify_new_observation(unsure)
    history = GraphService(db).get_tiger_history("T001")
    assert [event.observation_id for event in history] == [first]
    assert all(event.tiger_id == "T001" for event in history)
    assert ObservationRepository(db).get(unsure)["tiger_id"] is None


def test_uncertain_match_stays_in_review_queue(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "cam02.jpg": _vec(1, 0, 0, 0),
            "cam01.jpg": _vec(0.8, 0.6, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    first = _seed_observation(
        db, tmp_path, camera_id="Camera_02", stamp="2026-01-01T08:00:00", name="cam02.jpg"
    )
    second = _seed_observation(
        db, tmp_path, camera_id="Camera_01", stamp="2026-01-01T10:00:00", name="cam01.jpg"
    )
    assert service.identify_new_observation(first)["tiger_id"] == "T001"
    second_result = service.identify_new_observation(second)
    assert second_result["tiger_id"] is None
    assert second_result["suggested_tiger_id"] == "T001"
    queue = service.unidentified()
    assert [row["observation_id"] for row in queue] == [second]
    assert queue[0]["reid"]["candidates"][0]["similarity"] == pytest.approx(0.8, abs=1e-5)


def test_t001_embedding_persists_for_a_new_identity_service(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "first.jpg": _vec(0, 1, 0, 0),
            "second.jpg": _vec(0.12, 0.9928, 0, 0),
        }
    )
    original = LocalIdentityService(db, tmp_settings, encoder)
    first = _seed_observation(
        db, tmp_path, camera_id="C02", stamp="2026-01-01T08:00:00", name="first.jpg"
    )
    assert original.identify_new_observation(first)["tiger_id"] == "T001"
    second = _seed_observation(
        db, tmp_path, camera_id="C01", stamp="2026-01-01T10:00:00", name="second.jpg"
    )

    restarted = LocalIdentityService(db, tmp_settings, encoder)
    gallery = restarted.gallery.embeddings.list_identified()
    assert len(gallery) == 1
    assert gallery[0]["tiger_id"] == "T001"
    result = restarted.identify_new_observation(second)
    assert result["tiger_id"] == "T001"
    assert result["candidates"][0]["similarity"] > 0.90
    assert ObservationRepository(db).get(second)["tiger_id"] == "T001"


def test_real_megadescriptor_gallery_match(tmp_settings, db, tmp_path, monkeypatch):
    pytest.importorskip("timm")
    monkeypatch.setenv("WI_REID_ENABLED", "true")
    settings = reload_settings()
    encoder = MegaDescriptorEncoder(settings)
    if not encoder.is_available():
        pytest.skip(str(encoder.status().get("reason")))

    first = _seed_observation(
        db, tmp_path, camera_id="Camera_02", stamp="2026-01-01T08:00:00", name="ref.jpg"
    )
    service = LocalIdentityService(db, settings, encoder)
    first_result = service.identify_new_observation(first)
    assert first_result["tiger_id"] == "T001"
    assert EmbeddingRepository(db).count_for_tiger("T001") == 1

    second = _seed_observation(
        db, tmp_path, camera_id="Camera_01", stamp="2026-01-01T10:00:00", name="probe.jpg"
    )
    import shutil

    shutil.copyfile(ObservationRepository(db).get(first)["crop_path"], ObservationRepository(db).get(second)["crop_path"])
    second_result = service.identify_new_observation(second)
    assert second_result["tiger_id"] == "T001"
    similarity = float(second_result["similarity"])
    assert 0.90 <= similarity <= 1.0001
    assert ObservationRepository(db).get(second)["tiger_id"] == "T001"
    assert service.unidentified() == []

    restarted = LocalIdentityService(db, settings, encoder)
    reused = restarted.gallery.embeddings.list_identified()
    assert {row["tiger_id"] for row in reused} == {"T001"}
    assert len(reused) == 2


def test_uncertain_match_stays_in_review(tmp_settings, db, tmp_path):
    encoder = FakeEncoder(
        {
            "t001a.jpg": _vec(1, 0, 0, 0),
            "query.jpg": _vec(0.7, 0.71414, 0, 0),
        }
    )
    service = LocalIdentityService(db, tmp_settings, encoder)
    known = _seed_observation(db, tmp_path, camera_id="C01", stamp="2026-01-01T08:00:00", name="t001a.jpg")
    service.assign(known, action="create")
    query = _seed_observation(db, tmp_path, camera_id="C02", stamp="2026-01-01T10:00:00", name="query.jpg")
    result = service.identify_new_observation(query)
    assert result["matched"] is False
    assert result["needs_review"] is True
    assert result["suggested_tiger_id"] == "T001"
    assert ObservationRepository(db).get(query)["tiger_id"] is None
