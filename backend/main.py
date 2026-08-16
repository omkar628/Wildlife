"""FastAPI application entrypoint.

Start from the project root:

    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import (
    cameras,
    dashboard,
    detections,
    graph,
    health,
    images,
    jobs,
    media,
    observations,
    reviews,
    settings as settings_routes,
    tigers,
)
from backend.config import get_settings
from backend.database.connection import get_database
from backend.logging_setup import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.logs_dir)
    database = get_database(settings.database_path)
    from backend.database.repositories import SettingsRepository

    stored = SettingsRepository(database)
    auto = stored.get("confidence_auto_accept")
    detect_min = stored.get("confidence_detect_min")
    if auto is not None:
        settings.confidence_auto_accept = float(auto)
    if detect_min is not None:
        settings.confidence_detect_min = float(detect_min)

    application = FastAPI(
        title="Wildlife Intelligence",
        description="Offline camera-trap analysis: YOLO detection, review, SQLite, graph-ready events.",
        version="0.1.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router, prefix="/api")
    application.include_router(settings_routes.router, prefix="/api")
    application.include_router(jobs.router, prefix="/api")
    application.include_router(dashboard.router, prefix="/api")
    application.include_router(detections.router, prefix="/api")
    application.include_router(reviews.router, prefix="/api")
    application.include_router(images.router, prefix="/api")
    application.include_router(cameras.router, prefix="/api")
    application.include_router(tigers.router, prefix="/api")
    application.include_router(observations.router, prefix="/api")
    application.include_router(graph.router, prefix="/api")
    application.include_router(media.router, prefix="/api")

    @application.get("/")
    def root() -> dict:
        return {
            "name": "Wildlife Intelligence",
            "docs": "/docs",
            "health": "/api/health",
        }

    return application


app = create_app()
