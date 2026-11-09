"""
storage/db.py — SQLite persistence for Fixion jobs, metrics, patches, and logs.

Zero third-party dependencies: uses Python standard library sqlite3.
Persists all runs locally to enable job history, live resumption, and auditing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(".fixion_cache/jobs.db")


def _get_db_path(db_path: Path | str | None = None) -> Path:
    p = Path(db_path) if db_path else DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def get_connection(db_path: Path | str | None = None):
    """Context manager yielding a SQLite connection and ensuring it closes on exit."""
    path = _get_db_path(db_path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """Initialize the SQLite database schema if not present."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                repo TEXT NOT NULL,
                issue TEXT NOT NULL,
                status TEXT NOT NULL,
                patch TEXT DEFAULT '',
                root_cause TEXT DEFAULT '',
                fix_explanation TEXT DEFAULT '',
                metrics TEXT DEFAULT '{}',
                logs TEXT DEFAULT '',
                pr_url TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_job(
    job_id: str,
    repo: str,
    issue: str,
    status: str = "running",
    patch: str = "",
    root_cause: str = "",
    fix_explanation: str = "",
    metrics: dict[str, Any] | None = None,
    logs: str = "",
    pr_url: str = "",
    db_path: Path | str | None = None,
) -> None:
    """Insert or replace a job record."""
    init_db(db_path)
    now = datetime.now(UTC).isoformat()
    metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, repo, issue, status, patch, root_cause, fix_explanation, metrics, logs, pr_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                repo=excluded.repo,
                issue=excluded.issue,
                status=excluded.status,
                patch=excluded.patch,
                root_cause=excluded.root_cause,
                fix_explanation=excluded.fix_explanation,
                metrics=excluded.metrics,
                logs=excluded.logs,
                pr_url=excluded.pr_url,
                updated_at=excluded.updated_at
            """,
            (
                job_id,
                repo,
                issue,
                status,
                patch,
                root_cause,
                fix_explanation,
                metrics_json,
                logs,
                pr_url,
                now,
                now,
            ),
        )
        conn.commit()


def update_job(
    job_id: str,
    db_path: Path | str | None = None,
    **kwargs: Any,
) -> None:
    """Update specific fields of an existing job."""
    if not kwargs:
        return
    init_db(db_path)
    now = datetime.now(UTC).isoformat()
    kwargs["updated_at"] = now

    if "metrics" in kwargs and isinstance(kwargs["metrics"], dict):
        kwargs["metrics"] = json.dumps(kwargs["metrics"], ensure_ascii=False)

    set_clauses = [f"{k} = ?" for k in kwargs]
    values = list(kwargs.values()) + [job_id]

    sql = f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?"
    with get_connection(db_path) as conn:
        conn.execute(sql, values)
        conn.commit()


def append_job_log(
    job_id: str,
    log_line: str,
    db_path: Path | str | None = None,
) -> None:
    """Append a log line to a job's log history."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET logs = logs || ? || '\n', updated_at = ? WHERE id = ?",
            (log_line, datetime.now(UTC).isoformat(), job_id),
        )
        conn.commit()


def get_job(job_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Retrieve a single job by its ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("metrics"):
            try:
                d["metrics"] = json.loads(d["metrics"])
            except Exception:
                d["metrics"] = {}
        return d


def list_jobs(limit: int = 50, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """List historical jobs ordered by created_at descending."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT id, repo, issue, status, patch, metrics, pr_url, created_at, updated_at FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        jobs: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if d.get("metrics"):
                try:
                    d["metrics"] = json.loads(d["metrics"])
                except Exception:
                    d["metrics"] = {}
            jobs.append(d)
        return jobs
