from tests.image_helpers import make_jpeg
from backend.ingestion.hasher import hash_file


def test_same_bytes_same_hash(tmp_path):
    a = make_jpeg(tmp_path / "a.jpg", (10, 20, 30))
    b = make_jpeg(tmp_path / "sub" / "b.jpg", (10, 20, 30))
    assert hash_file(a) == hash_file(b)


def test_different_bytes_different_hash(tmp_path):
    a = make_jpeg(tmp_path / "a.jpg", (10, 20, 30))
    b = make_jpeg(tmp_path / "b.jpg", (200, 10, 10))
    assert hash_file(a) != hash_file(b)


def test_hash_is_sha256_hex(tmp_path):
    path = make_jpeg(tmp_path / "a.jpg")
    digest = hash_file(path)
    assert len(digest) == 64
    int(digest, 16)
