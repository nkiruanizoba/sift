"""Command line entry point: python -m sift <command>.

  verify    check each app ID in config resolves to the expected game name
  news      pull Steam developer announcements for each game
  patches   list likely patch posts for one game, to choose major_patches
  reviews   pull reviews around each major patch into DuckDB
  stats     counts by game and patch window, straight from the database
"""

from __future__ import annotations

import argparse
import logging
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

from . import store
from .ingest import steam

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

log = logging.getLogger("sift")


def load_games(path: Path, only: int | None = None) -> list[dict]:
    cfg = tomllib.loads(path.read_text())
    defaults = cfg.get("defaults", {})
    games = [{**defaults, **g} for g in cfg["games"]]
    if only is not None:
        games = [g for g in games if g["appid"] == only]
        if not games:
            sys.exit(f"appid {only} is not in {path}")
    return games


def _plain(name: str) -> str:
    return re.sub(r"[\u2122\u00ae\u00a9]", "", name).strip().casefold()


def cmd_verify(args, games, client, con) -> int:
    bad = 0
    for g in games:
        name = steam.fetch_app_name(client, g["appid"])
        ok = name is not None and _plain(name) == _plain(g["name"])
        bad += not ok
        print(f"{'OK ' if ok else 'MISMATCH'}  {g['appid']:>8}  config={g['name']!r}  steam={name!r}")
    return 1 if bad else 0


def cmd_news(args, games, client, con) -> int:
    for g in games:
        items = steam.fetch_news(client, g["appid"])
        n = store.upsert_news(con, items)
        print(f"{g['name']}: {n} announcements stored")
    return 0


def cmd_patches(args, games, client, con) -> int:
    rows = con.execute(
        """
        SELECT strftime(timestamp, '%Y-%m-%d') AS day, title, url
        FROM steam_news
        WHERE appid = ? AND (is_patch_candidate OR ?)
        ORDER BY timestamp DESC
        """,
        [games[0]["appid"], args.all],
    ).fetchall()
    if not rows:
        print("No news stored for this game yet. Run `python -m sift news` first.")
    for day, title, url in rows:
        print(f"{day}  {title}\n            {url}")
    return 0


def _naive(ts):
    return ts.astimezone(timezone.utc).replace(tzinfo=None)


def cmd_reviews(args, games, client, con) -> int:
    store.sync_major_patches(con, games)
    if args.fresh:
        n = con.execute("SELECT count(*) FROM records WHERE source = 'steam_review'").fetchone()[0]
        con.execute("DELETE FROM records WHERE source = 'steam_review'")
        print(f"Cleared {n} previously stored reviews (--fresh).")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"kind": "steam_reviews", "started_at": started, "games": [], "errors": []}

    try:
        for g in games:
            windows = steam.windows_for_game(g)
            per_window = max(1, g["max_reviews"] // len(windows))
            game_summary = {"appid": g["appid"], "name": g["name"], "windows": []}
            summary["games"].append(game_summary)
            for w in windows:
                slices = w.slices()
                per_slice = max(1, per_window // len(slices))
                total = 0
                print(f"{g['name']} | {w.label}: collecting ({len(slices)} weekly slices, up to {per_slice} each)", flush=True)
                for sl in slices:
                    have = 0
                    if sl.start is not None:
                        have = con.execute(
                            """SELECT count(*) FROM records WHERE source = 'steam_review'
                               AND CAST(metadata->>'appid' AS INTEGER) = ?
                               AND timestamp BETWEEN ? AND ?""",
                            [g["appid"], _naive(sl.start), _naive(sl.end)],
                        ).fetchone()[0]
                    if have >= per_slice:
                        total += have
                        continue
                    batch = []
                    try:
                        for rec in steam.fetch_reviews(client, g["appid"], g["name"], sl, per_slice):
                            rec.metadata["sample_window"] = w.label
                            rec.metadata["sample_side"] = w.side(rec.timestamp)
                            batch.append(rec)
                    except Exception as exc:  # keep going; the rerun picks up missing slices
                        msg = f"{g['name']} | {w.label} | slice from {sl.start}: {exc}"
                        summary["errors"].append(msg)
                        print(f"   problem, skipping this week for now: {exc}", flush=True)
                    store.upsert_records(con, batch)
                    total += len(batch)
                print(f"{g['name']} | {w.label}: {total} reviews", flush=True)
                game_summary["windows"].append(
                    {"label": w.label, "start": w.start, "end": w.end, "reviews": total}
                )
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        summary["db_totals"] = stats_rows(con)
        path = store.write_run_summary(summary)
        print(f"Run summary saved: {path}")
        if summary["errors"]:
            print(f"{len(summary['errors'])} weeks had problems. Run the same command again to fill them in.")
    return 0


def stats_rows(con) -> list[dict]:
    rows = con.execute(
        """
        SELECT metadata->>'game' AS game,
               COALESCE(metadata->>'sample_window', 'recent') AS patch,
               COALESCE(metadata->>'sample_side', '') AS side,
               count(*) AS reviews,
               round(avg(CAST(metadata->>'voted_up' AS BOOLEAN)::INT) * 100, 1) AS pct_positive,
               min(timestamp)::DATE AS first_review, max(timestamp)::DATE AS last_review
        FROM records
        WHERE source = 'steam_review'
        GROUP BY ALL ORDER BY game, first_review
        """
    ).fetchall()
    cols = ["game", "patch", "side", "reviews", "pct_positive", "first_review", "last_review"]
    return [dict(zip(cols, r)) for r in rows]


def cmd_stats(args, games, client, con) -> int:
    rows = stats_rows(con)
    if not rows:
        print("No reviews stored yet.")
        return 0
    for r in rows:
        print(
            f"{r['game']:<15} {r['patch']:<34} {r['side']:<7} {r['reviews']:>5} reviews  "
            f"{r['pct_positive']:>5}% positive  {r['first_review']} to {r['last_review']}"
        )
    total = con.execute("SELECT count(*) FROM records WHERE source = 'steam_review'").fetchone()[0]
    print(f"Total Steam reviews: {total}")
    return 0


COMMANDS = {
    "verify": cmd_verify,
    "news": cmd_news,
    "patches": cmd_patches,
    "reviews": cmd_reviews,
    "stats": cmd_stats,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sift", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--game", type=int, help="limit to one Steam app ID")
    p.add_argument("--config", type=Path, default=Path("config/games.toml"))
    p.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    p.add_argument("--fresh", action="store_true", help="reviews: delete stored reviews first and pull again")
    p.add_argument("--all", action="store_true", help="patches: show every announcement, not just likely patches")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "patches" and args.game is None:
        p.error("patches needs --game <appid>")

    games = load_games(args.config, args.game)
    con = store.connect(args.db)
    try:
        return COMMANDS[args.command](args, games, steam.SteamClient(), con)
    finally:
        con.close()
