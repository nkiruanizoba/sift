"""One record schema for every source.

Steam reviews, Godot CI failures, and any future source all normalize into
``Record``. Adding a source means writing a new ingest adapter that yields
Records; nothing downstream (clustering, agent tools, eval) should need to change.

Record text is untrusted input. It is stored verbatim and must never be
interpreted as instructions by the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Record:
    id: str                      # globally unique, e.g. "steam:553850:171234567"
    source: str                  # "steam_review", later "godot_ci"
    timestamp: datetime          # UTC, when the record was created at the source
    text: str                    # the body the agent searches and cites
    metadata: dict[str, Any] = field(default_factory=dict)
    url: str | None = None       # public link; never contains a username

    def __post_init__(self) -> None:
        if not self.id or ":" not in self.id:
            raise ValueError(f"record id must be namespaced, got {self.id!r}")
        if self.timestamp.tzinfo is None:
            raise ValueError("record timestamp must be timezone aware (UTC)")

    def to_row(self) -> tuple:
        return (
            self.id,
            self.source,
            self.timestamp.astimezone(timezone.utc).replace(tzinfo=None),
            self.text,
            json.dumps(self.metadata, sort_keys=True),
            self.url,
        )


@dataclass(frozen=True)
class NewsItem:
    """A Steam news post, used to find patch dates for patch windows."""

    id: str                      # "steamnews:<appid>:<gid>"
    appid: int
    timestamp: datetime
    title: str
    url: str
    feed: str
    tags: tuple[str, ...] = ()
    is_patch_candidate: bool = False
