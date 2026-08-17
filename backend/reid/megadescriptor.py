"""MegaDescriptor-S-224 inference for local field-tiger embeddings.

Loads BVRA/MegaDescriptor-S-224 once, stays on CUDA when available,
and never consults the ATRW gallery or shipped ArcFace/FAISS assets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "BVRA/MegaDescriptor-S-224"
INPUT_SIZE = 224
IMAGENET_STYLE_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STYLE_STD = (0.5, 0.5, 0.5)


def select_device(preferred: str | None, cuda_available: bool | None = None) -> str:
    """auto → CUDA if present, otherwise CPU. Never invent a GPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if cuda_available is None:
        cuda_available = bool(torch.cuda.is_available())
    text = (preferred or "auto").strip().lower()
    if text in {"", "auto"}:
        return "cuda" if cuda_available else "cpu"
    if text.startswith("cuda"):
        if not cuda_available:
            logger.warning("CUDA requested for MegaDescriptor but unavailable; using CPU.")
            return "cpu"
        return text
    return "cpu"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array / norm).astype(np.float32)


class MegaDescriptorEncoder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.reid_model_id
        self.device_name = "cpu"
        self._torch: Any = None
        self._model: Any = None
        self._transform: Any = None
        self._load_error: str | None = None
        self._embedding_dim: int | None = None
        try:
            self._load()
        except Exception as exc:
            self._model = None
            self._load_error = f"MegaDescriptor failed to start: {exc}"
            logger.exception(self._load_error)

    def is_available(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "implemented": True,
            "backend": "MegaDescriptor-S-224",
            "model_id": self.model_id,
            "loaded": self._model is not None,
            "device": self.device_name,
            "embedding_dim": self._embedding_dim,
            "input_size": INPUT_SIZE,
            "reason": self._load_error,
            "uses_atrw_gallery": False,
        }

    def embed_crop(self, crop_path: Path) -> np.ndarray:
        vectors = self.embed_crops([crop_path])
        if not vectors:
            raise RuntimeError(f"Failed to embed crop: {crop_path}")
        return vectors[0]

    def embed_crops(self, crop_paths: list[Path]) -> list[np.ndarray]:
        if self._model is None or self._torch is None:
            raise RuntimeError(self._load_error or "MegaDescriptor is not loaded.")
        tensors = []
        for path in crop_paths:
            tensors.append(self._preprocess(Path(path)))
        batch = self._torch.stack(tensors, dim=0).to(self._torch.device(self.device_name))
        self._model.eval()
        with self._torch.inference_mode():
            raw = self._model(batch)
            if isinstance(raw, (tuple, list)):
                raw = raw[0]
            if raw.ndim != 2:
                raw = raw.reshape(raw.shape[0], -1)
            normalized = self._torch.nn.functional.normalize(raw.float(), p=2, dim=1)
        vectors = [l2_normalize(row.detach().cpu().numpy()) for row in normalized]
        if vectors and self._embedding_dim is None:
            self._embedding_dim = int(vectors[0].shape[0])
        return vectors

    def _load(self) -> None:
        if not getattr(self.settings, "reid_enabled", True):
            self._load_error = "MegaDescriptor is disabled (WI_REID_ENABLED=false)."
            return
        try:
            import timm
            import torch
            import torchvision.transforms as transforms
        except ImportError as exc:
            self._load_error = f"MegaDescriptor dependencies are missing: {exc}"
            logger.warning(self._load_error)
            return

        self.device_name = select_device(self.settings.reid_device)
        device = torch.device(self.device_name)
        model_source = self.settings.reid_model_path or f"hf-hub:{self.model_id}"
        try:
            model = timm.create_model(model_source, pretrained=True)
            model.to(device)
            model.eval()
        except Exception as exc:
            self._load_error = f"Failed to load MegaDescriptor ({model_source}): {exc}"
            logger.exception(self._load_error)
            return

        self._torch = torch
        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(list(IMAGENET_STYLE_MEAN), list(IMAGENET_STYLE_STD)),
            ]
        )
        self._load_error = None
        logger.info("Loaded MegaDescriptor %s on %s", model_source, self.device_name)

    def _preprocess(self, crop_path: Path):
        from PIL import Image

        if not crop_path.is_file():
            raise FileNotFoundError(f"Tiger crop not found: {crop_path}")
        with Image.open(crop_path) as image:
            rgb = image.convert("RGB")
            return self._transform(rgb)
