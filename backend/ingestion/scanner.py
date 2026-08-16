"""Recursive camera-folder scanning. Yields paths; never loads pixels."""

from __future__ import annotations

from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_allowed_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def iter_image_files(folder: Path) -> list[Path]:
    """Return image paths under ``folder`` (recursive), sorted for resume stability."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {root}")

    found: list[Path] = []
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
            if not is_allowed_image(path):
                continue
            resolved = path.resolve()
            found.append(resolved)
        except OSError:
            continue
    found.sort(key=lambda item: str(item).lower())
    return found


def suggest_folder_camera_id(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return "camera"
    allowed = []
    for char in cleaned:
        if char.isalnum() or char in "._-":
            allowed.append(char)
        elif char in {" ", "\t"}:
            allowed.append("_")
    suggestion = "".join(allowed).strip("._-")
    return suggestion or "camera"


def discover_camera_folders(folder: Path) -> dict:
    """Group images under ``folder`` into camera folders.

    Immediate subdirectories that contain images become separate cameras.
    Images sitting directly in ``folder`` are one camera named after the folder.
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {root}")
    root = root.resolve()
    images = iter_image_files(root)

    grouped: dict[str, list[Path]] = {}
    for image in images:
        relative = image.relative_to(root)
        key = relative.parts[0] if len(relative.parts) > 1 else ""
        grouped.setdefault(key, []).append(image)

    cameras: list[dict] = []
    for key in sorted(grouped, key=lambda item: item.lower()):
        files = grouped[key]
        if key == "":
            folder_path = root
            folder_name = root.name
        else:
            folder_path = root / key
            folder_name = key
        cameras.append(
            {
                "folder_path": str(folder_path),
                "folder_name": folder_name,
                "suggested_camera_id": suggest_folder_camera_id(folder_name),
                "image_count": len(files),
                "sample_names": [path.name for path in files[:8]],
            }
        )

    return {
        "folder_path": str(root),
        "total_images": len(images),
        "camera_count": len(cameras),
        "camera_folders": cameras,
    }
