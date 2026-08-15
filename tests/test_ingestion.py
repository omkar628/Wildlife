from pathlib import Path

from PIL import Image

from backend.ingestion.metadata import extract_image_metadata
from backend.ingestion.scanner import iter_image_files
from tests.image_helpers import make_jpeg


def test_scanner_finds_nested_images_and_ignores_other_files(tmp_path: Path):
    make_jpeg(tmp_path / "C01" / "one.jpg")
    make_jpeg(tmp_path / "C01" / "nested" / "two.PNG")
    (tmp_path / "C01" / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "C01" / "video.mp4").write_bytes(b"00")
    found = iter_image_files(tmp_path / "C01")
    names = sorted(path.name.lower() for path in found)
    assert names == ["one.jpg", "two.png"]


def test_scanner_missing_folder(tmp_path: Path):
    try:
        iter_image_files(tmp_path / "missing")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_metadata_size_and_filesystem_fallback(tmp_path: Path):
    path = make_jpeg(tmp_path / "plain.jpg", size=(80, 60))
    meta = extract_image_metadata(path)
    assert meta["width"] == 80
    assert meta["height"] == 60
    assert meta["timestamp_source"] in {"filesystem", "unknown"}
    if meta["timestamp_source"] == "filesystem":
        assert meta["timestamp"]


def test_metadata_prefers_exif(tmp_path: Path):
    path = tmp_path / "exif.jpg"
    image = Image.new("RGB", (32, 32), (8, 8, 8))
    exif = image.getexif()
    exif[306] = "2024:01:15 06:30:00"
    image.save(path, format="JPEG", exif=exif)
    meta = extract_image_metadata(path)
    assert meta["timestamp_source"] == "exif"
    assert meta["timestamp"] is not None
    assert "2024-01-15T06:30:00" in meta["timestamp"]


def test_unreadable_image_raises(tmp_path: Path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    try:
        extract_image_metadata(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unreadable" in str(exc)
