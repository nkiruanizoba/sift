"""The Sift agent: Claude plus a small set of tools over the review database.

Instead of one giant prompt, Claude gets four tools and decides which to call:

  search_reviews   find reviews about something (keyword relevance + filters)
  compare_patch    before vs after numbers for one patch, including topic shifts
  topic_trends     each theme's size and sentiment, before vs after the patches
  get_review       the full text of one review

Every claim in the final answer must cite review IDs like [steam:553850:162345678].
After the answer comes back, Sift checks each cited ID: it must exist and must have
been shown to Claude by a tool during this question. Citations that fail are flagged.

Guardrails: review text is wrapped as untrusted input; the number of tool rounds
and the tokens per question are capped; reviewer identities are never stored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import wrap_untrusted

CITE = re.compile(r"steam:\d+:\d+")  # any review ID in the answer, including several inside one bracket

GAMES = {"helldivers 2": 553850, "cyberpunk 2077": 1091500, "no man's sky": 275850}

SYSTEM = """You are Sift, an insights analyst for game teams. You answer questions about
player feedback using ONLY evidence from the tools. Steam reviews are in the database for
Helldivers 2, Cyberpunk 2077 and No Man's Sky, sampled around specific major patches.

How to work:
- Use the tools before answering. Start with compare_patch or topic_trends for questions
  about change, and search_reviews to find what players actually said.
- Every factual claim must end with one or more citations in square brackets using the
  exact review IDs returned by the tools, e.g. [steam:553850:162345678]. Numbers from
  compare_patch or topic_trends should be cited as [stats].
- Never invent review IDs. Only cite IDs you saw in tool results.
- If the data cannot answer the question (wrong game, a time period or patch that was not
  sampled, information reviews would not contain), say so plainly instead of guessing.
- Keep answers short: a 1 to 2 sentence summary, then 3 to 5 bullet points with citations.
- Do not quote or mention reviewer names. Reviews are anonymous here.
"""

TOOLS = [
    {
        "name": "search_reviews",
        "description": "Find reviews relevant to a query. Returns up to `limit` reviews with ID, game, date, patch window, before/after side, thumbs up or down, theme, and an excerpt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look for, in plain words"},
                "game": {"type": "string", "enum": list(GAMES)},
                "patch": {"type": "string", "description": "patch window label, e.g. 'Account Linking Update'"},
                "side": {"type": "string", "enum": ["before", "after"]},
                "sentiment": {"type": "string", "enum": ["positive", "negative"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15},
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_patch",
        "description": "Before vs after numbers for one patch of one game: review counts, percent positive, and the themes whose share of reviews changed most.",
        "input_schema": {
            "type": "object",
            "properties": {"game": {"type": "string", "enum": list(GAMES)},
                           "patch": {"type": "string"}},
            "required": ["game", "patch"],
        },
    },
    {
        "name": "topic_trends",
        "description": "All themes for a game with review counts, percent positive, and counts before vs after its patches. Also lists the game's sampled patch windows.",
        "input_schema": {
            "type": "object",
            "properties": {"game": {"type": "string", "enum": list(GAMES)}},
            "required": ["game"],
        },
    },
    {
        "name": "get_review",
        "description": "Full text and details of one review by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"review_id": {"type": "string"}},
            "required": ["review_id"],
        },
    },
]


@dataclass
class Answer:
    question: str
    text: str
    cited: list = field(default_factory=list)
    verified: list = field(default_factory=list)
    unverified: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_early: str | None = None


class ReviewIndex:
    """Keyword relevance over all reviews (TF-IDF cosine), built once per session."""

    def __init__(self, con):
        rows = con.execute(
            """SELECT r.id, r.text, r.timestamp,
                      CAST(r.metadata->>'appid' AS INTEGER),
                      r.metadata->>'game', r.metadata->>'sample_window', r.metadata->>'sample_side',
                      CAST(r.metadata->>'voted_up' AS BOOLEAN), t.label
               FROM records r
               LEFT JOIN record_topics rt ON rt.record_id = r.id
               LEFT JOIN topics t ON t.appid = rt.appid AND t.topic_id = rt.topic_id
               WHERE r.source = 'steam_review' ORDER BY r.id"""
        ).fetchall()
        self.rows = rows
        self.by_id = {r[0]: i for i, r in enumerate(rows)}
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2,
                                   ngram_range=(1, 2), max_features=60000)
        self.X = self.vec.fit_transform([r[1] for r in rows])
        # Favor reviews with enough words to explain *why*; one-word reviews like "psn"
        # match keywords perfectly but carry no evidence.
        words = np.array([len(r[1].split()) for r in rows], dtype=float)
        self.length_weight = np.sqrt(np.minimum(words, 40) / 40)

    def search(self, query, game=None, patch=None, side=None, sentiment=None, limit=8):
        q = self.vec.transform([query])
        scores = (self.X @ q.T).toarray().ravel() * self.length_weight
        mask = np.ones(len(self.rows), dtype=bool)
        for i, r in enumerate(self.rows):
            if game and r[3] != GAMES.get(game.lower()):
                mask[i] = False
            elif patch and (r[5] or "").lower() != patch.lower():
                mask[i] = False
            elif side and r[6] != side:
                mask[i] = False
            elif sentiment and r[7] != (sentiment == "positive"):
                mask[i] = False
        scores = np.where(mask & (scores > 0), scores, -1)
        top = [i for i in np.argsort(-scores)[: max(1, min(limit, 15))] if scores[i] > 0]
        return [self.rows[i] for i in top]


def _fmt(r, excerpt=500) -> dict:
    return {
        "review_id": r[0], "game": r[4], "date": str(r[2])[:10], "patch": r[5], "side": r[6],
        "thumbs": "up" if r[7] else "down", "theme": r[8],
        "text": wrap_untrusted(r[1], excerpt),
    }


class Agent:
    def __init__(self, con, client, max_rounds: int = 6, max_input_tokens: int = 80000,
                 index: ReviewIndex | None = None):
        self.con = con
        self.client = client
        self.index = index or ReviewIndex(con)
        self.max_rounds = max_rounds
        self.max_input_tokens = max_input_tokens

    # ---------------------------------------------------------------- tools

    def _tool(self, name: str, args: dict, seen: set) -> str:
        if name == "search_reviews":
            rows = self.index.search(args.get("query", ""), args.get("game"), args.get("patch"),
                                     args.get("side"), args.get("sentiment"), args.get("limit", 8))
            seen.update(r[0] for r in rows)
            return json.dumps([_fmt(r) for r in rows]) if rows else "No matching reviews."
        if name == "get_review":
            i = self.index.by_id.get(args.get("review_id", ""))
            if i is None:
                return "No review with that ID."
            seen.add(self.index.rows[i][0])
            return json.dumps(_fmt(self.index.rows[i], 3000))
        if name == "compare_patch":
            return json.dumps(self._compare(args.get("game", ""), args.get("patch", "")), default=str)
        if name == "topic_trends":
            return json.dumps(self._trends(args.get("game", "")), default=str)
        return f"Unknown tool {name}."

    def _appid(self, game: str):
        return GAMES.get((game or "").lower())

    def _compare(self, game, patch):
        appid = self._appid(game)
        sides = self.con.execute(
            """SELECT metadata->>'sample_side', count(*),
                      round(100.0 * avg(CASE WHEN CAST(metadata->>'voted_up' AS BOOLEAN) THEN 1 ELSE 0 END), 1)
               FROM records WHERE CAST(metadata->>'appid' AS INTEGER) = ?
               AND lower(metadata->>'sample_window') = lower(?) GROUP BY 1""",
            [appid, patch]).fetchall()
        if not sides:
            return {"error": f"No sampled window called '{patch}' for {game}.",
                    "available_patches": self._patches(appid)}
        shifts = self.con.execute(
            """WITH x AS (
                 SELECT t.label, r.metadata->>'sample_side' AS side, count(*) AS n
                 FROM records r JOIN record_topics rt ON rt.record_id = r.id
                 JOIN topics t ON t.appid = rt.appid AND t.topic_id = rt.topic_id
                 WHERE rt.appid = ? AND lower(r.metadata->>'sample_window') = lower(?)
                 GROUP BY 1, 2),
               tot AS (SELECT side, sum(n) AS total FROM x GROUP BY 1)
               SELECT x.label, x.side, round(100.0 * x.n / tot.total, 1)
               FROM x JOIN tot USING (side)""",
            [appid, patch]).fetchall()
        share = {}
        for label, side, pct in shifts:
            share.setdefault(label, {"before": 0.0, "after": 0.0})[side] = float(pct)
        ranked = sorted(share.items(), key=lambda kv: -abs(kv[1]["after"] - kv[1]["before"]))[:6]
        return {
            "game": game, "patch": patch,
            "sides": {s: {"reviews": n, "pct_positive": p} for s, n, p in sides},
            "biggest_theme_shifts_pct_of_reviews": [
                {"theme": k, "before": v["before"], "after": v["after"]} for k, v in ranked],
        }

    def _patches(self, appid):
        return [r[0] for r in self.con.execute(
            "SELECT label FROM major_patches WHERE appid = ? ORDER BY patch_date", [appid]).fetchall()]

    def _trends(self, game):
        appid = self._appid(game)
        rows = self.con.execute(
            """SELECT t.label, count(*) AS n,
                      round(100.0 * avg(CASE WHEN CAST(r.metadata->>'voted_up' AS BOOLEAN) THEN 1 ELSE 0 END), 1),
                      sum(CASE WHEN r.metadata->>'sample_side' = 'before' THEN 1 ELSE 0 END),
                      sum(CASE WHEN r.metadata->>'sample_side' = 'after' THEN 1 ELSE 0 END)
               FROM topics t JOIN record_topics rt ON rt.appid = t.appid AND rt.topic_id = t.topic_id
               JOIN records r ON r.id = rt.record_id
               WHERE t.appid = ? GROUP BY 1 ORDER BY n DESC""", [appid]).fetchall()
        return {"game": game, "patches": self._patches(appid),
                "themes": [{"theme": a, "reviews": b, "pct_positive": c, "before": d, "after": e}
                           for a, b, c, d, e in rows]}

    # ----------------------------------------------------------------- loop

    def ask(self, question: str) -> Answer:
        start_in, start_out = self.client.input_tokens, self.client.output_tokens
        messages = [{"role": "user", "content": question}]
        seen: set = set()
        ans = Answer(question=question, text="")
        for round_no in range(self.max_rounds + 1):
            last = round_no == self.max_rounds
            if last:
                messages.append({"role": "user", "content":
                                 "Tool budget reached. Answer now using only the evidence above."})
            resp = self.client.messages(SYSTEM, messages, tools=None if last else TOOLS)
            content = resp.get("content", [])
            messages.append({"role": "assistant", "content": content})
            calls = [b for b in content if b.get("type") == "tool_use"]
            if not calls:
                ans.text = "".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
                break
            results = []
            for c in calls:
                ans.tool_calls.append({"tool": c["name"], "input": c.get("input", {})})
                results.append({"type": "tool_result", "tool_use_id": c["id"],
                                "content": self._tool(c["name"], c.get("input", {}), seen)})
            messages.append({"role": "user", "content": results})
            if self.client.input_tokens - start_in > self.max_input_tokens:
                ans.stopped_early = "token cap"
                messages.append({"role": "user", "content": "Token budget reached. Answer now."})
                resp = self.client.messages(SYSTEM, messages, tools=None)
                ans.text = "".join(b.get("text", "") for b in resp.get("content", [])
                                   if b.get("type") == "text").strip()
                break

        ans.cited = list(dict.fromkeys(CITE.findall(ans.text)))
        ans.verified = [c for c in ans.cited if c in seen and c in self.index.by_id]
        ans.unverified = [c for c in ans.cited if c not in ans.verified]
        ans.input_tokens = self.client.input_tokens - start_in
        ans.output_tokens = self.client.output_tokens - start_out
        return ans
