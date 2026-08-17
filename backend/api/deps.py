"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from backend.config import Settings, get_settings
from backend.database.connection import Database, get_database
from backend.detector.service import DetectorService
from backend.graph.builder import GraphService
from backend.reid.adapter import UnavailableReIDAdapter
from backend.reid.identity import LocalIdentityService
from backend.reid.megadescriptor import MegaDescriptorEncoder
from backend.review.service import ReviewService
from backend.services.gnn_service import GNNService
from backend.services.pipeline import PipelineService


@lru_cache(maxsize=1)
def get_detector() -> DetectorService:
    return DetectorService(get_settings())


@lru_cache(maxsize=1)
def get_reid_encoder() -> MegaDescriptorEncoder:
    return MegaDescriptorEncoder(get_settings())


@lru_cache(maxsize=1)
def get_pipeline() -> PipelineService:
    return PipelineService(
        get_database(),
        get_settings(),
        get_detector(),
        identity=get_identity_service(),
    )


@lru_cache(maxsize=1)
def get_review_service() -> ReviewService:
    return ReviewService(get_database(), get_settings(), identity=get_identity_service())


@lru_cache(maxsize=1)
def get_graph_service() -> GraphService:
    return GraphService(get_database())


@lru_cache(maxsize=1)
def get_reid_adapter() -> UnavailableReIDAdapter:
    return UnavailableReIDAdapter(get_settings())


@lru_cache(maxsize=1)
def get_identity_service() -> LocalIdentityService:
    settings = get_settings()
    encoder = get_reid_encoder() if settings.reid_enabled else None
    return LocalIdentityService(get_database(), settings, encoder)


@lru_cache(maxsize=1)
def get_gnn_service() -> GNNService:
    return GNNService(get_settings(), get_database())


def settings_dep() -> Settings:
    return get_settings()


def db_dep() -> Database:
    return get_database()
