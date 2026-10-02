"""Export the public dataset the hosted app reads.

The hosted app (Streamlit Community Cloud) cannot see the local DuckDB file, so
this writes compact Parquet files to public_data/:

  reviews.parquet        review ID, game, date, text, thumbs up or down, patch window,
                         before/after side, theme (no reviewer identity: never stored)
  topics.parquet         themes per game with labels and descriptions
  patches.parquet        the major patches used for the before/after windows
  examples.json          saved Q&A answers whose citations all check out, shown
                         in the app as free examples

load_public() rebuilds the same tables in memory, so the agent and app code
work the same locally and online.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

PUBLIC = Path("public_data")
CITE = re.compile(r"steam:\d+:\d+")


def export(con, out: Path = PUBLIC, answers: Path = Path("data/runs/answers.jsonl")) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (SELECT id, source, timestamp, text, CAST(metadata AS VARCHAR) AS metadata, url
              FROM records WHERE source = 'steam_review' ORDER BY id)
        TO '{out / "records.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.execute(f"COPY (SELECT * FROM topics) TO '{out / 'topics.parquet'}' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * FROM record_topics) TO '{out / 'record_topics.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY (SELECT * FROM major_patches) TO '{out / 'major_patches.parquet'}' (FORMAT PARQUET)")

    ids = {r[0] for r in con.execute("SELECT id FROM records").fetchall()}
    examples = []
    if answers.exists():
        for line in answers.read_text().splitlines():
            a = json.loads(line)
            cited = list(dict.fromkeys(CITE.findall(a.get("text", ""))))
            if a.get("text") and cited and all(c in ids for c in cited):
                examples.append({"question": a["question"], "text": a["text"], "cited": cited,
                                 "model": a.get("model"), "input_tokens": a.get("input_tokens"),
                                 "output_tokens": a.get("output_tokens")})
    # keep the latest answer per question
    latest = {e["question"]: e for e in examples}
    (out / "examples.json").write_text(json.dumps(list(latest.values()), indent=2))
    sizes = {p.name: p.stat().st_size for p in out.iterdir()}
    return {"files": sizes, "examples": len(latest)}


def load_public(folder: Path = PUBLIC) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"""CREATE TABLE records AS
        SELECT id, source, timestamp, text, CAST(metadata AS JSON) AS metadata, url
        FROM read_parquet('{folder / "records.parquet"}')""")
    for t in ("topics", "record_topics", "major_patches"):
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{folder / (t + '.parquet')}')")
    return con
