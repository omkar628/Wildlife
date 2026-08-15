"""YOLO11 detector wrapper around the trained ``best.pt`` weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.detector.parser import ParsedDetection, parse_ultralytics_result

logger = logging.getLogger(__name__)


def resolve_device(preferred: str) -> str:
    if preferred and preferred.lower() not in {"auto", ""}:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


class DetectorService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.detector_device)
        self._model: Any = None

    @property
    def model_path(self) -> Path:
        path = self.settings.detector_model_path
        if path is None:
            raise FileNotFoundError("Detector model path is not configured.")
        return Path(path)

    def available(self) -> bool:
        return self.model_path.is_file()

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.available():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.model_path}. "
                "Place best.pt in the project root or models/detector/best.pt."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        logger.info("Loading YOLO detector from %s on %s", self.model_path, self.device)
        self._model = YOLO(str(self.model_path))
        return self._model

    def predict_paths(self, image_paths: list[Path]) -> list[list[ParsedDetection]]:
        """Run batched inference. One result list per input path."""
        if not image_paths:
            return []
        model = self.load()
        results = model.predict(
            source=[str(path) for path in image_paths],
            conf=self.settings.detector_yolo_conf,
            imgsz=self.settings.detector_imgsz,
            device=self.device,
            verbose=False,
            stream=False,
        )
        parsed_batches: list[list[ParsedDetection]] = []
        for result in results:
            parsed_batches.append(parse_ultralytics_result(result, self.settings.class_map))
        return parsed_batches
