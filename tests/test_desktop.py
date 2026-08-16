"""Helpers that mirror the Electron folder-picker contract."""

from pathlib import Path

from backend.ingestion.scanner import iter_image_files
from tests.image_helpers import make_jpeg


def suggest_camera_id(folder_path: str) -> str:
    trimmed = folder_path.rstrip("\\/")
    name = Path(trimmed).name
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return name if name and all(ch in allowed for ch in name) else ""


def test_suggest_camera_id_from_windows_path():
    assert suggest_camera_id(r"D:\CameraTrap\C01") == "C01"
    assert suggest_camera_id(r"D:\CameraTrap\C01\\") == "C01"


def test_selected_folder_is_scanned_recursively(tmp_path: Path):
    root = tmp_path / "C01"
    make_jpeg(root / "nested" / "shot.jpg")
    found = iter_image_files(root)
    assert [path.name for path in found] == ["shot.jpg"]
    assert found[0].is_absolute()
