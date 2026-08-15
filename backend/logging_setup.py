"""Process-wide logging to console and logs/app.log."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


_CONFIGURED = False


def configure_logging(logs_dir: Path, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _CONFIGURED = True
