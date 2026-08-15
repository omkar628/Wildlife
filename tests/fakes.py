from __future__ import annotations

from pathlib import Path

from backend.detector.parser import ParsedDetection


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
