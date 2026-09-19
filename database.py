import sqlite3
import json
import os
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "drugsentinel.db"))


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                drug_name TEXT,
                overall_sentiment TEXT NOT NULL,
                overall_confidence REAL NOT NULL,
                aspects_json TEXT NOT NULL,
                source TEXT DEFAULT 'single',
                batch_id TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON analyses (timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_batch ON analyses (batch_id)")


def save_analysis(review_text: str, result: dict, drug_name: str = None, source: str = "single",
                   batch_id: str = None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO analyses (review_text, drug_name, overall_sentiment, overall_confidence,
                                      aspects_json, source, batch_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                review_text,
                drug_name,
                result["overall_sentiment"],
                result["overall_confidence"],
                json.dumps(result["aspects"]),
                source,
                batch_id,
                datetime.now(timezone.utc).isoformat()
            )
        )


def get_recent_analyses(limit: int = 50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_batch_analyses(batch_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE batch_id = ? ORDER BY id ASC", (batch_id,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_summary_stats():
    """Aggregate stats across all stored analyses — used for the dashboard overview.

    Kept deliberately defensive because older/local database rows may contain
    slightly different aspect sentiment values. A malformed historical row
    should not take down the entire analytics dashboard.
    """
    sentiment_dist = {"Positive": 0, "Neutral": 0, "Negative": 0}
    aspect_dist = {}

    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM analyses").fetchone()["c"]
        sentiment_counts = conn.execute(
            "SELECT overall_sentiment, COUNT(*) as c FROM analyses GROUP BY overall_sentiment"
        ).fetchall()
        rows = conn.execute("SELECT aspects_json FROM analyses").fetchall()

    for row in sentiment_counts:
        label = str(row["overall_sentiment"] or "").strip().title()
        if label in sentiment_dist:
            sentiment_dist[label] = row["c"]

    for row in rows:
        try:
            aspects = json.loads(row["aspects_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(aspects, list):
            continue

        for a in aspects:
            if not isinstance(a, dict):
                continue
            name = str(a.get("aspect") or "").strip()
            sentiment = str(a.get("sentiment") or "").strip().title()
            if not name or sentiment not in {"Positive", "Neutral", "Negative"}:
                continue
            if name not in aspect_dist:
                aspect_dist[name] = {"Positive": 0, "Neutral": 0, "Negative": 0}
            aspect_dist[name][sentiment] += 1

    return {
        "total_analyses": total,
        "overall_sentiment_distribution": sentiment_dist,
        "aspect_sentiment_distribution": aspect_dist
    }


def _row_to_dict(row):
    d = dict(row)
    d["aspects"] = json.loads(d.pop("aspects_json"))
    return d
