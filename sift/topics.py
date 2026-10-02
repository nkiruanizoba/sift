"""Topic clustering for Steam reviews.

For each game: TF-IDF features, then NMF (non-negative matrix factorization) to
find recurring themes. Each review is assigned its strongest theme. Claude then
reads the top terms and a sample of the most typical reviews for each theme and
writes a short label. Review text goes to Claude wrapped as untrusted input.

Results are stored in two tables:
  topics         one row per (game, topic): label, description, top terms, size
  record_topics  one row per review: its topic and how strongly it matches
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .llm import wrap_untrusted

DDL = """
CREATE TABLE IF NOT EXISTS topics (
    appid INTEGER, topic_id INTEGER, label VARCHAR, description VARCHAR,
    top_terms VARCHAR[], size INTEGER, PRIMARY KEY (appid, topic_id)
);
CREATE TABLE IF NOT EXISTS record_topics (
    record_id VARCHAR PRIMARY KEY, appid INTEGER, topic_id INTEGER, weight DOUBLE
);
"""

SHORT = 99  # reviews under MIN_WORDS words, or with no distinctive terms
SHORT_LABEL = "Short or generic reviews"
SHORT_DESC = "Reviews too short or too general to place in a specific theme (for example, 'great game')."
MIN_WORDS = 15

# Words that appear in almost every review and say nothing about *what* players
# discuss. Without these, the themes collapse into "good", "fun", "love".
GENERIC = set("""
game games play played playing player players like just really good great fun love best amazing nice
awesome cool yes pretty it's i've i'm don't doesn't can't won't time hours hour buy bought recommend
recommended worth money price lot lots bit thing things way make makes made want wanted feel feels got
get getting still even also much many well think know say said actually definitely absolutely overall
review reviews rating bad better worst ever little new old fact sure going gonna wanna need people guys
friend friends day days year years
""".split())

# Each game's own name, which otherwise shows up as a "theme".
GAME_NAME_WORDS = {
    553850: "helldivers helldiver hd2 arrowhead",
    1091500: "cyberpunk 2077 cdpr cd projekt red",
    275850: "sky man's mans man hello nms",
}

LABEL_SYSTEM = (
    "You label themes found in Steam player reviews for a game insights tool. The reader "
    "has never played the game, so every label must be concrete and self-explanatory.\n"
    "Rules for the label (3 to 7 words):\n"
    "- Name the specific thing players are reacting to: the feature, policy, update, bug "
    "type, piece of content or design choice. Use the game's own terms where players do.\n"
    "- Do not use vague words such as: business practices, concerns, issues, experience, "
    "changes, quality, aspects, decisions, overall.\n"
    "- Neutral wording; do not judge whether players are right.\n"
    "Bad: 'Sony business practices and data security concerns'. "
    "Good: 'Forced PSN account linking and past data breaches'.\n"
    "Bad: 'Game quality changes with recent updates'. "
    "Good: 'Weapon damage nerfs in recent patches'.\n"
    "The label and description must match what most reviews in the theme say, and their tone "
    "must match the share of reviews that recommend the game.\n"
    "Rules for the description (one sentence, under 30 words): say concretely what players "
    "complain about or praise, so someone new to the game understands it. Ground everything "
    "only in the evidence given."
)


def _fit(texts: list[str], n_topics: int, appid: int = 0, seed: int = 0):
    stop = ENGLISH_STOP_WORDS.union(GENERIC).union(GAME_NAME_WORDS.get(appid, "").split())
    vec = TfidfVectorizer(
        stop_words=sorted(stop), min_df=8, max_df=0.3, ngram_range=(1, 2),
        max_features=20000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    X = vec.fit_transform(texts)
    n_topics = max(2, min(n_topics, X.shape[1] - 1, X.shape[0] - 1))
    nmf = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    return vec, nmf, W


def build_topics(con, appid: int, labeler=None, n_topics: int = 14):
    rows = con.execute(
        """SELECT id, text, CAST(metadata->>'voted_up' AS BOOLEAN) FROM records
           WHERE source = 'steam_review' AND CAST(metadata->>'appid' AS INTEGER) = ? ORDER BY id""",
        [appid],
    ).fetchall()
    if len(rows) < 50:
        return []
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    ups = [bool(r[2]) for r in rows]
    long_idx = [i for i, t in enumerate(texts) if len(t.split()) >= MIN_WORDS]
    if len(long_idx) < 30:
        return []

    vec, nmf, W = _fit([texts[i] for i in long_idx], n_topics, appid)
    terms = np.array(vec.get_feature_names_out())
    strength = W.max(axis=1)
    best = np.where(strength > 0, W.argmax(axis=1), SHORT)

    assigned = np.full(len(texts), SHORT)
    weight = np.zeros(len(texts))
    for j, i in enumerate(long_idx):
        assigned[i] = best[j]
        weight[i] = strength[j]

    con.execute(DDL)
    con.execute("DELETE FROM topics WHERE appid = ?", [appid])
    con.execute("DELETE FROM record_topics WHERE appid = ?", [appid])

    out = []
    for k in range(nmf.n_components):
        members = [j for j in range(len(long_idx)) if best[j] == k]
        if not members:
            continue
        top_terms = [str(t) for t in terms[np.argsort(nmf.components_[k])[::-1][:12]]]
        typical = sorted(members, key=lambda j: -W[j, k])[:12]
        sample = [texts[long_idx[j]] for j in typical]
        label, desc = f"Topic {k + 1}", ""
        if labeler is not None:
            pct_up = round(100 * sum(ups[long_idx[j]] for j in members) / len(members))
            label, desc = labeler(top_terms, sample, pct_up)
        out.append({"topic_id": k, "label": label, "description": desc,
                    "top_terms": top_terms, "size": len(members)})
    n_short = int((assigned == SHORT).sum())
    if n_short:
        out.append({"topic_id": SHORT, "label": SHORT_LABEL, "description": SHORT_DESC,
                    "top_terms": [], "size": n_short})

    con.executemany(
        "INSERT INTO topics VALUES (?, ?, ?, ?, ?, ?)",
        [(appid, t["topic_id"], t["label"], t["description"], t["top_terms"], t["size"]) for t in out],
    )
    con.executemany(
        "INSERT INTO record_topics VALUES (?, ?, ?, ?)",
        [(ids[i], appid, int(assigned[i]), float(weight[i])) for i in range(len(ids))],
    )
    return out


def claude_labeler(client):
    def label(top_terms: list[str], sample: list[str], pct_up: int | None = None) -> tuple[str, str]:
        reviews = "\n".join(wrap_untrusted(s, 400) for s in sample)
        user = (
            f"Most distinctive terms for this theme: {', '.join(top_terms)}\n\n"
            f"Typical reviews in this theme:\n{reviews}\n\n"
            + (f"{pct_up}% of all reviews in this theme recommend the game.\n\n" if pct_up is not None else "")
            + "Describe what MOST of these reviews talk about, not one striking review. "
            "If they are mixed, say so.\n\n"
            'Return {"label": "...", "description": "one sentence on what players say"}.'
        )
        try:
            data = client.complete_json(LABEL_SYSTEM, user, max_tokens=200)
            return str(data["label"]).strip()[:80], str(data.get("description", "")).strip()[:300]
        except Exception:
            return ", ".join(top_terms[:3]), ""
    return label


def topic_report(con, appid: int) -> list[tuple]:
    """Each topic's size and share of reviews before vs after the patches."""
    return con.execute(
        """
        SELECT t.label, t.size,
               round(100.0 * avg(CASE WHEN CAST(r.metadata->>'voted_up' AS BOOLEAN) THEN 1 ELSE 0 END), 1) AS pct_pos,
               sum(CASE WHEN r.metadata->>'sample_side' = 'before' THEN 1 ELSE 0 END) AS n_before,
               sum(CASE WHEN r.metadata->>'sample_side' = 'after' THEN 1 ELSE 0 END) AS n_after
        FROM topics t
        JOIN record_topics rt ON rt.appid = t.appid AND rt.topic_id = t.topic_id
        JOIN records r ON r.id = rt.record_id
        WHERE t.appid = ?
        GROUP BY t.label, t.size ORDER BY t.size DESC
        """,
        [appid],
    ).fetchall()


def run_summary(games_out: dict, client) -> dict:
    return {
        "kind": "topics",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": getattr(client, "model", None),
        "input_tokens": getattr(client, "input_tokens", 0),
        "output_tokens": getattr(client, "output_tokens", 0),
        "games": json.loads(json.dumps(games_out, default=str)),
    }
