from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.detector.parser import ParsedDetection
from backend.reid.megadescriptor import l2_normalize


class FakeEncoder:
    """Deterministic L2 embeddings for Re-ID tests. Does not load MegaDescriptor."""

    def __init__(self, vectors: dict[str, np.ndarray] | None = None, default: np.ndarray | None = None) -> None:
        self.vectors = vectors or {}
        self.default = l2_normalize(default if default is not None else np.ones(8, dtype=np.float32))
        self.device_name = "cpu"
        self.calls: list[Path] = []
        self.fail_paths: set[str] = set()

    def is_available(self) -> bool:
        return True

    def status(self) -> dict:
        return {"implemented": True, "loaded": True, "device": "cpu", "backend": "fake"}

    def embed_crop(self, crop_path: Path) -> np.ndarray:
        path = Path(crop_path)
        self.calls.append(path)
        if path.name in self.fail_paths or str(path) in self.fail_paths:
            raise RuntimeError("forced encoder failure")
        if path.name in self.vectors:
            return l2_normalize(self.vectors[path.name])
        if str(path) in self.vectors:
            return l2_normalize(self.vectors[str(path)])
        return np.array(self.default, dtype=np.float32, copy=True)


class FakeDetector:
    """Deterministic detector for tests. Does not load best.pt."""

    def __init__(self, mapping: dict[str, list[ParsedDetection]] | None = None) -> None:
        self.mapping = mapping or {}
        self.calls: list[list[Path]] = []

    def predict_paths(self, image_paths: list[Path]) -> list[list[ParsedDetection]]:
        self.calls.append(list(image_paths))
        results: list[list[ParsedDetection]] = []
        for path in image_paths:
            if path.name in self.mapping:
                results.append(self.mapping[path.name])
            else:
                results.append(
                    [
                        ParsedDetection(0, "tiger", 0.91, 8, 8, 30, 24),
                        ParsedDetection(1, "prey", 0.44, 2, 2, 12, 10),
                    ]
                )
        return results
