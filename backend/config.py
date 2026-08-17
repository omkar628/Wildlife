"""Application configuration.

Paths default to the project root so the existing trained assets
(`best.pt`, `tiger_reid_arcface.pth`, …) are found without moving them.

Environment variables prefixed with ``WI_`` override YAML defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml() -> dict[str, Any]:
    yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
    if not yaml_path.is_file():
        return {}
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _yaml_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        path = default
    else:
        path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def discover_detector_model() -> Path:
    """Prefer models/detector/best.pt, then the original root-level file."""
    candidates = [
        PROJECT_ROOT / "models" / "detector" / "best.pt",
        PROJECT_ROOT / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def discover_reid_weights() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "reid" / "tiger_reid_arcface.pth",
        PROJECT_ROOT / "tiger_reid_arcface.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def discover_reid_index() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "reid" / "tiger_vector_index.faiss",
        PROJECT_ROOT / "tiger_vector_index.faiss",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def discover_reid_metadata() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "reid" / "tiger_metadata.pkl",
        PROJECT_ROOT / "tiger_metadata.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def discover_gnn_weights() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "gnn" / "gnn_model_v3_optimized_best.pt",
        PROJECT_ROOT / "models" / "gnn_model_v3_optimized_best.pt",
        PROJECT_ROOT / "gnn_model_v3_optimized_best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WI_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT

    confidence_auto_accept: float = Field(default=0.60, ge=0.0, le=1.0)
    confidence_detect_min: float = Field(default=0.15, ge=0.0, le=1.0)

    detector_model_path: Path | None = None
    detector_batch_size: int = Field(default=8, ge=1, le=64)
    detector_imgsz: int = Field(default=640, ge=32)
    detector_device: str = "auto"
    detector_yolo_conf: float = Field(default=0.15, ge=0.0, le=1.0)

    reid_weights_path: Path | None = None
    reid_index_path: Path | None = None
    reid_metadata_path: Path | None = None
    reid_enabled: bool = True
    reid_model_id: str = "BVRA/MegaDescriptor-S-224"
    reid_model_path: str | None = None
    reid_device: str = "auto"
    reid_match_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    reid_review_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    reid_min_margin: float = Field(default=0.05, ge=0.0, le=1.0)

    gnn_weights_path: Path | None = None
    gnn_device: str = "auto"

    database_path: Path = Path("database/wildlife.db")
    crops_dir: Path = Path("data/crops")
    classified_dir: Path = Path("data/classified")
    logs_dir: Path = Path("logs")
    reports_dir: Path = Path("reports")

    alert_repeat_threshold: int = Field(default=3, ge=2, le=100)
    alert_unusual_min_detections: int = Field(default=4, ge=2, le=100)

    crop_padding: float = Field(default=0.05, ge=0.0, le=0.5)
    crop_jpeg_quality: int = Field(default=95, ge=50, le=100)

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    class_map: dict[int, str] = Field(
        default_factory=lambda: {0: "tiger", 1: "prey", 2: "rival", 3: "human"}
    )

    def model_post_init(self, __context: Any) -> None:
        yaml_data = _load_yaml()

        if "WI_CONFIDENCE_AUTO_ACCEPT" not in os.environ:
            value = _yaml_get(yaml_data, "confidence", "auto_accept")
            if value is not None:
                self.confidence_auto_accept = float(value)
        if "WI_CONFIDENCE_DETECT_MIN" not in os.environ:
            value = _yaml_get(yaml_data, "confidence", "detect_min")
            if value is not None:
                self.confidence_detect_min = float(value)

        if "WI_DETECTOR_BATCH_SIZE" not in os.environ:
            value = _yaml_get(yaml_data, "detector", "batch_size")
            if value is not None:
                self.detector_batch_size = int(value)
        if "WI_DETECTOR_IMGSZ" not in os.environ:
            value = _yaml_get(yaml_data, "detector", "imgsz")
            if value is not None:
                self.detector_imgsz = int(value)
        if "WI_DETECTOR_DEVICE" not in os.environ:
            value = _yaml_get(yaml_data, "detector", "device")
            if value is not None:
                self.detector_device = str(value)
        if "WI_DETECTOR_YOLO_CONF" not in os.environ:
            value = _yaml_get(yaml_data, "detector", "yolo_conf")
            if value is not None:
                self.detector_yolo_conf = float(value)

        yaml_classes = _yaml_get(yaml_data, "classes")
        if isinstance(yaml_classes, dict):
            self.class_map = {int(k): str(v) for k, v in yaml_classes.items()}

        if "WI_CROP_PADDING" not in os.environ:
            value = _yaml_get(yaml_data, "crops", "padding")
            if value is not None:
                self.crop_padding = float(value)
        if "WI_CROP_JPEG_QUALITY" not in os.environ:
            value = _yaml_get(yaml_data, "crops", "jpeg_quality")
            if value is not None:
                self.crop_jpeg_quality = int(value)

        cors = _yaml_get(yaml_data, "server", "cors_origins")
        if cors and "WI_CORS_ORIGINS" not in os.environ:
            self.cors_origins = list(cors)

        self.database_path = _resolve_path(
            self.database_path,
            PROJECT_ROOT / "database" / "wildlife.db",
        )
        self.crops_dir = _resolve_path(self.crops_dir, PROJECT_ROOT / "data" / "crops")
        self.classified_dir = _resolve_path(
            self.classified_dir, PROJECT_ROOT / "data" / "classified"
        )
        self.logs_dir = _resolve_path(self.logs_dir, PROJECT_ROOT / "logs")
        self.reports_dir = _resolve_path(self.reports_dir, PROJECT_ROOT / "reports")

        if "WI_ALERT_REPEAT_THRESHOLD" not in os.environ:
            value = _yaml_get(yaml_data, "alerts", "repeat_threshold")
            if value is not None:
                self.alert_repeat_threshold = int(value)
        if "WI_ALERT_UNUSUAL_MIN_DETECTIONS" not in os.environ:
            value = _yaml_get(yaml_data, "alerts", "unusual_min_detections")
            if value is not None:
                self.alert_unusual_min_detections = int(value)

        if self.detector_model_path is None or str(self.detector_model_path).strip() == "":
            self.detector_model_path = discover_detector_model()
        else:
            self.detector_model_path = _resolve_path(
                self.detector_model_path, discover_detector_model()
            )

        if self.reid_weights_path is None or str(self.reid_weights_path).strip() == "":
            self.reid_weights_path = discover_reid_weights()
        else:
            self.reid_weights_path = _resolve_path(
                self.reid_weights_path, discover_reid_weights()
            )

        if self.reid_index_path is None or str(self.reid_index_path).strip() == "":
            self.reid_index_path = discover_reid_index()
        else:
            self.reid_index_path = _resolve_path(self.reid_index_path, discover_reid_index())

        if self.reid_metadata_path is None or str(self.reid_metadata_path).strip() == "":
            self.reid_metadata_path = discover_reid_metadata()
        else:
            self.reid_metadata_path = _resolve_path(
                self.reid_metadata_path, discover_reid_metadata()
            )

        if "WI_REID_ENABLED" not in os.environ:
            value = _yaml_get(yaml_data, "reid", "enabled")
            if value is not None:
                self.reid_enabled = bool(value)
        if "WI_REID_MODEL_ID" not in os.environ:
            value = _yaml_get(yaml_data, "reid", "model_id")
            if value is not None:
                self.reid_model_id = str(value)
        if "WI_REID_MODEL_PATH" not in os.environ:
            value = _yaml_get(yaml_data, "reid", "model_path")
            if value:
                self.reid_model_path = str(value)
        if "WI_REID_DEVICE" not in os.environ:
            value = _yaml_get(yaml_data, "reid", "device")
            if value is not None:
                self.reid_device = str(value)
        if "WI_REID_MATCH_THRESHOLD" not in os.environ:
            value = _yaml_get(yaml_data, "reid", "match_threshold")
            if value is not None:
                self.reid_match_threshold = float(value)
        if "WI_REID_REVIEW_THRESHOLD" not in os.environ:
            value = _yaml_get(yaml_data, "reid", "review_threshold")
            if value is not None:
                self.reid_review_threshold = float(value)
        if "WI_REID_MIN_MARGIN" not in os.environ:
            value = _yaml_get(yaml_data, "reid", "min_margin")
            if value is not None:
                self.reid_min_margin = float(value)

        if "WI_GNN_DEVICE" not in os.environ:
            value = _yaml_get(yaml_data, "gnn", "device")
            if value is not None:
                self.gnn_device = str(value)

        if self.gnn_weights_path is None or str(self.gnn_weights_path).strip() == "":
            self.gnn_weights_path = discover_gnn_weights()
        else:
            self.gnn_weights_path = _resolve_path(
                self.gnn_weights_path, discover_gnn_weights()
            )

    def ensure_directories(self) -> None:
        for path in (
            self.database_path.parent,
            self.crops_dir,
            self.classified_dir,
            self.logs_dir,
            self.reports_dir,
            PROJECT_ROOT / "data" / "input",
            PROJECT_ROOT / "data" / "processed",
            PROJECT_ROOT / "data" / "review",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def class_name(self, class_id: int) -> str:
        return self.class_map.get(class_id, f"class_{class_id}")

    def class_id(self, class_name: str) -> int | None:
        lowered = class_name.strip().lower()
        for key, value in self.class_map.items():
            if value.lower() == lowered:
                return key
        if lowered == "other":
            return 99
        return None

    def detector_exists(self) -> bool:
        return bool(self.detector_model_path and self.detector_model_path.is_file())

    def reid_assets_exist(self) -> bool:
        return bool(
            self.reid_weights_path
            and self.reid_weights_path.is_file()
            and self.reid_index_path
            and self.reid_index_path.is_file()
            and self.reid_metadata_path
            and self.reid_metadata_path.is_file()
        )

    def gnn_weights_exist(self) -> bool:
        return bool(self.gnn_weights_path and self.gnn_weights_path.is_file())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
