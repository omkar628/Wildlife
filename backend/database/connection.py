"""SQLite connection helpers. Safe for a local FastAPI + worker thread."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from backend.database.schema import initialize_schema


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialized = False

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self.connect() as connection:
                initialize_schema(connection)
            self._initialized = True

    def execute(
        self,
        sql: str,
        params: Iterable[Any] = (),
        *,
        commit: bool = True,
    ) -> sqlite3.Cursor:
        with self._lock:
            with self.connect() as connection:
                cursor = connection.execute(sql, tuple(params))
                if commit:
                    connection.commit()
                return cursor

    def executemany(
        self,
        sql: str,
        seq_of_params: Iterable[Iterable[Any]],
        *,
        commit: bool = True,
    ) -> sqlite3.Cursor:
        with self._lock:
            with self.connect() as connection:
                cursor = connection.executemany(sql, list(seq_of_params))
                if commit:
                    connection.commit()
                return cursor

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            with self.connect() as connection:
                return connection.execute(sql, tuple(params)).fetchone()

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            with self.connect() as connection:
                return connection.execute(sql, tuple(params)).fetchall()

    def execute_returning_id(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            with self.connect() as connection:
                cursor = connection.execute(sql, tuple(params))
                connection.commit()
                return int(cursor.lastrowid)


_DATABASE: Database | None = None
_DATABASE_LOCK = threading.Lock()


def get_database(path: Path | None = None) -> Database:
    global _DATABASE
    if _DATABASE is not None and path is None:
        return _DATABASE
    if path is None:
        from backend.config import get_settings

        path = get_settings().database_path
    with _DATABASE_LOCK:
        if _DATABASE is None or (path is not None and _DATABASE.path != Path(path)):
            _DATABASE = Database(Path(path))
            _DATABASE.initialize()
        return _DATABASE


def reset_database_singleton() -> None:
    global _DATABASE
    with _DATABASE_LOCK:
        _DATABASE = None
