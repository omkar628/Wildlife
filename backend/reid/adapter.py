"""Re-ID adapter.

This file inspects the shipped assets and reports what is known vs missing.
It does **not** construct a guessed ConvNeXt / ArcFace network.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.reid.interface import ReIDResult


MISSING_IMPLEMENTATION = (
    "Tiger Re-ID inference code is not in this repository. "
    "The weights, FAISS index, and metadata are present, but the model class, "
    "image preprocessing, embedding normalization, and match threshold are unknown. "
    "Do not guess them. Provide the original training/inference script to enable Re-ID."
)


def _safe_file_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": None}
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def inspect_reid_assets(settings: Settings) -> dict[str, Any]:
    """Read-only inspection of the shipped Re-ID files."""
    weights = _safe_file_info(settings.reid_weights_path)
    index = _safe_file_info(settings.reid_index_path)
    metadata = _safe_file_info(settings.reid_metadata_path)

    metadata_summary: dict[str, Any] = {}
    meta_path = settings.reid_metadata_path
    if meta_path and meta_path.is_file():
        try:
            with meta_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, list):
                ids = []
                for item in payload:
                    if isinstance(item, dict) and "tiger_id" in item:
                        ids.append(str(item["tiger_id"]))
                metadata_summary = {
                    "type": "list",
                    "n_records": len(payload),
                    "unique_tiger_ids": len(set(ids)),
                    "record_keys": sorted({key for item in payload[:1] if isinstance(item, dict) for key in item}),
                    "id_format_note": (
                        "Gallery IDs are numeric strings from the Amur Tiger Re-ID dataset "
                        "(for example '250'), not T017-style field IDs."
                    ),
                }
            else:
                metadata_summary = {"type": type(payload).__name__}
        except Exception as exc:
            metadata_summary = {"error": f"{type(exc).__name__}: {exc}"}

    index_summary: dict[str, Any] = {}
    index_path = settings.reid_index_path
    if index_path and index_path.is_file():
        try:
            header = index_path.read_bytes()[:16]
            magic = header[:4]
            # FAISS IndexFlatIP files begin with ASCII 'IxFI'.
            if magic == b"IxFI":
                import struct

                dimension = struct.unpack_from("<i", header, 4)[0]
                ntotal = struct.unpack_from("<i", header, 8)[0]
                index_summary = {
                    "faiss_type": "IndexFlatIP",
                    "dimension_from_header": dimension,
                    "ntotal_from_header": ntotal,
                    "metric_note": "Inner product. Cosine search only if vectors were L2-normalized when the index was built.",
                }
            else:
                index_summary = {"magic": magic.decode("latin-1", errors="replace")}
        except Exception as exc:
            index_summary = {"error": f"{type(exc).__name__}: {exc}"}

    observed_from_weights = {
        "checkpoint_format": "PyTorch state_dict (OrderedDict of tensors)",
        "observed_modules": [
            "backbone.features.* (ConvNeXt-like staged blocks with 7x7 depthwise conv and layer_scale)",
            "backbone.classifier.0 (768-d)",
            "bottleneck (Linear 768 -> 512)",
            "bn (BatchNorm1d 512)",
            "arcface.weight shape (107, 512)",
        ],
        "likely_embedding_size": 512,
        "likely_gallery_identities": 107,
        "warning": (
            "These observations come from tensor names/shapes only. "
            "They are not a license to reconstruct inference. "
            "Preprocessing (resize, crop, mean/std), whether embeddings are L2-normalized, "
            "and the identification threshold remain unknown."
        ),
    }

    return {
        "implemented": False,
        "reason": MISSING_IMPLEMENTATION,
        "assets": {
            "weights": weights,
            "faiss_index": {**index, **index_summary},
            "metadata": {**metadata, **metadata_summary},
        },
        "observed_from_weights": observed_from_weights,
        "missing_to_enable_inference": [
            "The Python model class used during training (forward that returns the 512-d embedding).",
            "Exact input size and preprocessing (resize, crop, color order, normalization).",
            "Whether embeddings are L2-normalized before FAISS search.",
            "Identification threshold / top-k policy used to assign a tiger ID.",
            "How gallery IDs should be mapped to field IDs such as T017.",
        ],
    }


class UnavailableReIDAdapter:
    """Placeholder backend that never invents identities."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._status = inspect_reid_assets(settings)

    def is_available(self) -> bool:
        return False

    def status(self) -> dict:
        return self._status

    def identify_crop(self, crop_path: Path) -> ReIDResult:
        return ReIDResult(
            available=False,
            tiger_id=None,
            confidence=None,
            matches=[],
            reason=MISSING_IMPLEMENTATION,
        )
