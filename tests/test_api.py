from __future__ import annotations

import time

import anyio
import httpx

from backend.api.deps import get_pipeline
from backend.config import get_settings
from backend.database.connection import get_database
from backend.main import create_app
from backend.services.pipeline import PipelineService
from tests.image_helpers import make_jpeg
from tests.fakes import FakeDetector


class ApiClient:
    """Sync wrapper around httpx.ASGITransport (async-only in httpx 0.28+)."""

    def __init__(self, app) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def _go() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)

        return anyio.run(_go)

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PUT", url, **kwargs)


def _client(tmp_settings) -> ApiClient:
    app = create_app()
    pipeline = PipelineService(get_database(), get_settings(), FakeDetector())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return ApiClient(app)


def test_health_endpoint(tmp_settings):
    client = _client(tmp_settings)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["offline"] is True
    assert "detector" in body
    assert body["reid"]["implemented"] is False


def test_dashboard_endpoint(tmp_settings):
    client = _client(tmp_settings)
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["images"]["total"] == 0
    assert body["review"]["pending"] == 0


def test_settings_roundtrip(tmp_settings):
    client = _client(tmp_settings)
    response = client.put("/api/settings", json={"confidence_auto_accept": 0.72})
    assert response.status_code == 200
    assert response.json()["confidence_auto_accept"] == 0.72
    again = client.get("/api/settings")
    assert again.json()["confidence_auto_accept"] == 0.72


def test_import_requires_existing_folder(tmp_settings, tmp_path):
    client = _client(tmp_settings)
    response = client.post(
        "/api/import",
        json={"folder_path": str(tmp_path / "missing"), "camera_id": "C01"},
    )
    assert response.status_code == 400


def test_import_and_job_status(tmp_settings, tmp_path):
    folder = tmp_path / "trap"
    make_jpeg(folder / "one.jpg")
    client = _client(tmp_settings)
    response = client.post(
        "/api/import",
        json={"folder_path": str(folder), "camera_id": "C01", "habitat": "teak"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    deadline = time.time() + 10
    body = None
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["job"]["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert body is not None
    assert body["job"]["status"] == "completed"
    assert body["job"]["processed"] == 1

    detections = client.get("/api/detections").json()["detections"]
    assert detections
    reviews = client.get("/api/reviews").json()
    assert reviews["pending"] >= 1

    review_id = reviews["reviews"][0]["review_id"]
    decided = client.post(f"/api/reviews/{review_id}/decide", json={"human_class": "prey"})
    assert decided.status_code == 200
    assert client.get("/api/reviews").json()["pending"] == 0

    cameras = client.get("/api/cameras").json()["cameras"]
    assert cameras[0]["camera_id"] == "C01"
    graph = client.get("/api/graph").json()
    assert "camera_graph" in graph
    assert graph["gnn"]["implemented"] is False
