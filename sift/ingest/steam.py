"""Steam adapter: player reviews and news posts (for patch dates).

Both endpoints are public and need no API key. Requests are throttled and
retried with backoff. Reviewer identity (Steam ID, profile, username) is
dropped at normalization and never stored.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import requests

from ..schema import NewsItem, Record

log = logging.getLogger(__name__)

REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
USER_AGENT = "sift-portfolio-project/0.1 (public research, throttled)"

PATCH_TITLE = re.compile(
    r"\b(patch|update|hotfix|patch notes|v?\d+\.\d+(\.\d+)?)\b", re.IGNORECASE
)


# --------------------------------------------------------------------------- HTTP


class SteamClient:
    def __init__(self, min_interval: float = 2.0, max_retries: int = 6, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last = 0.0

    def get_json(self, url: str, params: dict[str, Any]) -> dict:
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("request error (%s), attempt %d", exc, attempt + 1)
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in (429, 500, 502, 503, 504):
                    resp.raise_for_status()
                if resp.status_code == 429:
                    # Steam is asking us to slow down: back off for longer each time.
                    pause = min(180, 30 * (attempt + 1))
                    print(f"   Steam asked us to slow down, waiting {pause} seconds...", flush=True)
                    time.sleep(pause)
                    continue
                log.warning("HTTP %s, attempt %d", resp.status_code, attempt + 1)
            time.sleep(min(60, 2 ** (attempt + 1)))
        raise RuntimeError(f"giving up on {url} after {self.max_retries} attempts")


# ------------------------------------------------------------------ normalization


def _utc(ts: int | float) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def _clean(text: str) -> str:
    # Keep the review verbatim apart from control characters and outer whitespace.
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "").strip()


def normalize_review(raw: dict, appid: int, game: str) -> Record | None:
    """Map one Steam review to a Record. Returns None for empty reviews.

    Only non-identifying author fields (playtime) are kept. Steam also returns
    steamid, personaname, profile_url, and avatar; none of these are stored. The Steam ID is
    deliberately dropped, so the stored URL is the game's review page, and
    citations use the review ID.
    """
    text = _clean(raw.get("review", ""))
    rid = raw.get("recommendationid")
    if not text or not rid:
        return None
    author = raw.get("author") or {}
    metadata = {
        "appid": appid,
        "game": game,
        "review_id": str(rid),
        "voted_up": bool(raw.get("voted_up")),
        "language": raw.get("language"),
        "votes_up": raw.get("votes_up", 0),
        "votes_funny": raw.get("votes_funny", 0),
        "weighted_vote_score": float(raw.get("weighted_vote_score") or 0),
        "comment_count": raw.get("comment_count", 0),
        "steam_purchase": raw.get("steam_purchase"),
        "received_for_free": raw.get("received_for_free"),
        "refunded": raw.get("refunded"),
        "primarily_steam_deck": raw.get("primarily_steam_deck"),
        "written_during_early_access": raw.get("written_during_early_access"),
        "timestamp_updated": _utc(raw["timestamp_updated"]).isoformat()
        if raw.get("timestamp_updated")
        else None,
        "playtime_at_review_min": author.get("playtime_at_review"),
        "playtime_forever_min": author.get("playtime_forever"),
    }
    return Record(
        id=f"steam:{appid}:{rid}",
        source="steam_review",
        timestamp=_utc(raw["timestamp_created"]),
        text=text,
        metadata=metadata,
        url=f"https://steamcommunity.com/app/{appid}/reviews/",
    )


def normalize_news(raw: dict, appid: int) -> NewsItem:
    tags = tuple(raw.get("tags") or ())
    title = raw.get("title") or ""
    return NewsItem(
        id=f"steamnews:{appid}:{raw['gid']}",
        appid=appid,
        timestamp=_utc(raw["date"]),
        title=title,
        url=raw.get("url") or "",
        feed=raw.get("feedname") or "",
        tags=tags,
        is_patch_candidate="patchnotes" in tags or bool(PATCH_TITLE.search(title)),
    )


# ------------------------------------------------------------------------ reviews


@dataclass
class Window:
    label: str
    start: datetime | None = None   # inclusive, UTC
    end: datetime | None = None     # inclusive, UTC
    patch: datetime | None = None   # the patch this window surrounds

    def side(self, ts: datetime) -> str | None:
        if self.patch is None:
            return None
        return "before" if ts < self.patch else "after"

    def slices(self, days: int = 1) -> list["Window"]:
        """Split into day-long slices so sampling is spread evenly over time.

        Steam returns reviews newest first, so sampling a whole window with one cap
        would fill up on the last few days and miss the weeks before the patch.
        """
        if self.start is None or self.end is None:
            return [self]
        out, cur, step = [], self.start, timedelta(days=days)
        while cur <= self.end:
            nxt = min(cur + step - timedelta(seconds=1), self.end)
            out.append(Window(self.label, cur, nxt, self.patch))
            cur = nxt + timedelta(seconds=1)
        return out

    def contains(self, ts: datetime) -> bool:
        return (self.start is None or ts >= self.start) and (self.end is None or ts <= self.end)


def windows_for_game(game: dict) -> list[Window]:
    """One window around each major patch, or a single 'recent' window if none are set."""
    patches = game.get("major_patches") or []
    if not patches:
        return [Window("recent")]
    before = timedelta(days=game.get("days_before", 30))
    after = timedelta(days=game.get("days_after", 30))
    out = []
    for p in patches:
        d = datetime.fromisoformat(p["date"]).replace(tzinfo=timezone.utc)
        out.append(Window(p["label"], d - before, d + after + timedelta(days=1) - timedelta(seconds=1), d))
    return out


def fetch_reviews(
    client: SteamClient,
    appid: int,
    game: str,
    window: Window,
    cap: int,
    max_pages: int = 400,
) -> Iterator[Record]:
    """Page through reviews newest first, yielding Records inside the window.

    For bounded windows we pass Steam's start_date/end_date filter. If Steam
    ignores it, the client side check below still keeps only in-window reviews
    and stops once reviews fall before the window start.
    """
    params: dict[str, Any] = {
        "json": 1,
        "filter": "recent",
        "language": "english",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": 100,
        "cursor": "*",
    }
    if window.start and window.end:
        params.update(
            start_date=int(window.start.timestamp()),
            end_date=int(window.end.timestamp()),
            date_range_type="include",
        )

    kept, seen_cursors = 0, set()
    for page in range(max_pages):
        data = client.get_json(REVIEWS_URL.format(appid=appid), params)
        if data.get("success") != 1:
            raise RuntimeError(f"Steam returned success={data.get('success')} for {appid}")
        reviews = data.get("reviews") or []
        if not reviews:
            break

        if page == 0 and window.end and any(_utc(r["timestamp_created"]) > window.end for r in reviews):
            log.warning(
                "[%s] Steam ignored the date filter for window %r; filtering client side",
                game, window.label,
            )

        oldest = None
        for raw in reviews:
            ts = _utc(raw["timestamp_created"])
            oldest = ts if oldest is None or ts < oldest else oldest
            if not window.contains(ts):
                continue
            rec = normalize_review(raw, appid, game)
            if rec is None:
                continue
            yield rec
            kept += 1
            if kept >= cap:
                return

        if window.start and oldest and oldest < window.start:
            break  # sorted newest first, so nothing older can be in the window
        cursor = data.get("cursor")
        if not cursor or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
        params["cursor"] = cursor
    else:
        log.warning("[%s] hit max_pages=%d in window %r", game, max_pages, window.label)


# --------------------------------------------------------------------------- news


def fetch_news(client: SteamClient, appid: int, max_items: int = 1000) -> list[NewsItem]:
    """Developer announcements for an app, newest first, paged back by date."""
    items: dict[str, NewsItem] = {}
    enddate = None
    while len(items) < max_items:
        params: dict[str, Any] = {
            "appid": appid,
            "count": 100,
            "maxlength": 1,
            "feeds": "steam_community_announcements",
            "format": "json",
        }
        if enddate:
            params["enddate"] = enddate
        batch = client.get_json(NEWS_URL, params).get("appnews", {}).get("newsitems", [])
        new = [normalize_news(n, appid) for n in batch if f"steamnews:{appid}:{n['gid']}" not in items]
        if not new:
            break
        for n in new:
            items[n.id] = n
        enddate = int(min(n.timestamp for n in new).timestamp()) - 1
    return sorted(items.values(), key=lambda n: n.timestamp, reverse=True)


def fetch_app_name(client: SteamClient, appid: int) -> str | None:
    data = client.get_json(APPDETAILS_URL, {"appids": appid, "filters": "basic"})
    entry = data.get(str(appid)) or {}
    return (entry.get("data") or {}).get("name") if entry.get("success") else None
