"""SQLite persistence layer for optimization jobs, progress, and results."""
import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "optimizer.db")


def get_conn():
    # timeout lets concurrent writers (the ProcessPoolExecutor children all log
    # to this same DB) wait on the lock instead of erroring with "database is
    # locked". WAL mode further improves concurrent read/write throughput.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                settings TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                data TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );
        """)


def create_job(job_id: str, settings: dict):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (id, status, settings, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "pending", json.dumps(settings), now, now),
        )


def update_job_status(job_id: str, status: str):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
            (status, now, job_id),
        )


def get_job(job_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def append_log(job_id: str, message: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO logs (job_id, ts, message) VALUES (?, ?, ?)",
            (job_id, ts, message),
        )


def get_logs(job_id: str, offset: int = 0):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ts, message FROM logs WHERE job_id=? ORDER BY id LIMIT -1 OFFSET ?",
            (job_id, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def save_results(job_id: str, records: list):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("DELETE FROM results WHERE job_id=?", (job_id,))
        conn.execute(
            "INSERT INTO results (job_id, data, saved_at) VALUES (?, ?, ?)",
            (job_id, json.dumps(records), now),
        )


def get_results(job_id: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data FROM results WHERE job_id=?", (job_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else []
