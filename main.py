"""Convenience launcher for the local FastAPI backend."""

from __future__ import annotations

import uvicorn

from backend.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
