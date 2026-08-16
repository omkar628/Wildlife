"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ImportRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)
    camera_id: str = Field(..., min_length=1, max_length=64)
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    habitat: str | None = None


class ImportPreviewRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)


class BatchImportRequest(BaseModel):
    cameras: list[ImportRequest] = Field(..., min_length=1)


class ReviewDecisionRequest(BaseModel):
    human_class: str = Field(..., min_length=1)


class IdentityAssignRequest(BaseModel):
    action: str = Field(..., min_length=1)
    tiger_id: str | None = None


class SettingsUpdateRequest(BaseModel):
    confidence_auto_accept: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_detect_min: float | None = Field(default=None, ge=0.0, le=1.0)


class CameraUpsertRequest(BaseModel):
    camera_id: str
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    habitat: str | None = None
    metadata: dict[str, Any] | None = None
