"""
database.py — SQLite Hardware Abstraction Layer.
Persists all system component states across steps within a session.
"""

from __future__ import annotations
import sqlite3
import json
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import tempfile
DB_PATH = Path(tempfile.gettempdir()) / "quest_openenv.db"

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db() -> None:
    """Initialize all required tables."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS env_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS action_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            step       INTEGER NOT NULL,
            task_id    TEXT NOT NULL,
            action     TEXT NOT NULL,
            reward     REAL NOT NULL,
            success    INTEGER NOT NULL,
            message    TEXT NOT NULL,
            timestamp  REAL NOT NULL DEFAULT (unixepoch('now', 'subsec'))
        );

        CREATE TABLE IF NOT EXISTS session_backup (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()


def set_state(key: str, value: Any) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO env_state (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )
    conn.commit()


def get_state(key: str, default: Any = None) -> Any:
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM env_state WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def set_state_bulk(data: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO env_state (key, value) VALUES (?, ?)",
        [(k, json.dumps(v)) for k, v in data.items()]
    )
    conn.commit()


def get_all_state() -> Dict[str, Any]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM env_state").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def log_action(step: int, task_id: str, action: str, reward: float,
               success: bool, message: str) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO action_log (step, task_id, action, reward, success, message)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (step, task_id, action, reward, int(success), message)
    )
    conn.commit()


def backup_session(data: Dict[str, Any]) -> None:
    """Save session data snapshot (simulates persistent workspace)."""
    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO session_backup (key, value) VALUES (?, ?)",
        [(k, json.dumps(v)) for k, v in data.items()]
    )
    conn.commit()


def restore_session() -> Dict[str, Any]:
    """Restore session data from backup."""
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM session_backup").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def clear_all() -> None:
    """Wipe env_state (does NOT touch session_backup or action_log)."""
    conn = get_conn()
    conn.execute("DELETE FROM env_state")
    conn.commit()


def get_action_history(task_id: str, limit: int = 20) -> list:
    conn = get_conn()
    rows = conn.execute(
        """SELECT step, action, reward, success, message
           FROM action_log WHERE task_id = ?
           ORDER BY id DESC LIMIT ?""",
        (task_id, limit)
    ).fetchall()
    return [dict(r) for r in rows]