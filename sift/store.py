"""DuckDB storage.

Tables
  records        one row per Record (all sources)
  steam_news     Steam news posts, used to find patch dates
  major_patches  the curated patch list from config/games.toml

View
  steam_reviews_windowed  each Steam review joined to the most recent major
                          patch at or before it (its "patch window")
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Iterable

import duckdb

from .schema import NewsItem, Record

DEFAULT_DB = Path("data/sift.duckdb")

DDL = """
CREATE TABLE IF NOT EXISTS records (
    id          VARCHAR PRIMARY KEY,
    source      VARCHAR NOT NULL,
    timestamp   TIMESTAMP NOT NULL,     -- UTC
    text        VARCHAR NOT NULL,
    metadata    JSON,
    url         VARCHAR,
    ingested_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC')
);

CREATE TABLE IF NOT EXISTS steam_news (
    id                 VARCHAR PRIMARY KEY,
    appid              INTEGER NOT NULL,
    timestamp          TIMESTAMP NOT NULL,
    title              VARCHAR,
    url                VARCHAR,
    feed               VARCHAR,
    tags               VARCHAR[],
    is_patch_candidate BOOLEAN
);

CREATE TABLE IF NOT EXISTS major_patches (
    appid      INTEGER NOT NULL,
    patch_date TIMESTAMP NOT NULL,
    label      VARCHAR NOT NULL,
    PRIMARY KEY (appid, patch_date)
);
"""

VIEW = """
CREATE OR REPLACE VIEW steam_reviews_windowed AS
WITH reviews AS (
    SELECT r.*, CAST(r.metadata->>'appid' AS INTEGER) AS appid
    FROM records r
    WHERE r.source = 'steam_review'
)
SELECT
    reviews.*,
    COALESCE(p.label, 'before first tracked patch') AS patch_window,
    p.patch_date,
    date_diff('day', p.patch_date, reviews.timestamp) AS days_since_patch
FROM reviews
ASOF LEFT JOIN major_patches p
    ON reviews.appid = p.appid AND reviews.timestamp >= p.patch_date;
"""


def connect(path: Path | str = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(DDL)
    con.execute(VIEW)
    return con


def upsert_records(con: duckdb.DuckDBPyConnection, records: Iterable[Record]) -> int:
    rows = [r.to_row() for r in records]
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO records (id, source, timestamp, text, metadata, url)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            text = excluded.text,
            metadata = excluded.metadata,
            url = excluded.url
        """,
        rows,
    )
    return len(rows)


def upsert_news(con: duckdb.DuckDBPyConnection, items: Iterable[NewsItem]) -> int:
    rows = [
        (
            n.id,
            n.appid,
            n.timestamp.astimezone(timezone.utc).replace(tzinfo=None),
            n.title,
            n.url,
            n.feed,
            list(n.tags),
            n.is_patch_candidate,
        )
        for n in items
    ]
    if not rows:
        return 0
    con.executemany(
        "INSERT OR REPLACE INTO steam_news VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    return len(rows)


def sync_major_patches(con: duckdb.DuckDBPyConnection, games: list[dict]) -> int:
    """Replace the major_patches table with what's in config."""
    con.execute("DELETE FROM major_patches")
    rows = [
        (g["appid"], p["date"], p["label"])
        for g in games
        for p in g.get("major_patches", [])
    ]
    if rows:
        con.executemany("INSERT INTO major_patches VALUES (?, CAST(? AS TIMESTAMP), ?)", rows)
    return len(rows)


def write_run_summary(summary: dict, runs_dir: Path = Path("data/runs")) -> Path:
    """Every count quoted in the README or case study should trace to one of these."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{summary['kind']}_{summary['started_at'].replace(':', '')}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    return out
