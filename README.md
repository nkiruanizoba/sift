# Sift

**Ask questions of messy text, get answers you can check.**

> Status: in development, public launch October 2026.

Sift is an agentic insights engine. You ask a question in plain language, for example "Why did player sentiment drop after the last major patch?" Sift searches and clusters the underlying records and answers with citations that link back to the exact sources, so every claim can be checked.

## What it does

- **Natural-language Q&A with citations:** every claim in an answer points to the records that support it.
- **Automated topic clustering:** groups thousands of records into labeled themes.
- **Trends over time:** shows how topics change across patches or release windows.
- **One platform, many sources:** every dataset maps to a single record schema, so adding a new source only takes a new ingest adapter.

## Datasets (all public)

1. **Player feedback:** public Steam reviews for a few games, across major updates.
2. **Engineering signals:** failed CI runs from the Godot game engine's public GitHub Actions history (coming after the first launch).

## How it's evaluated

Sift is graded against a hand-labeled eval set, not just demoed:

- **Feedback questions:** the share of answer claims that the cited sources actually support; usefulness; and whether it correctly declines questions the data can't answer.
- **CI failures:** classification accuracy against hand labels.

Results will be published here and in the app once the eval runs.

## Guardrails

- Answers must cite sources, and uncited claims are treated as failures.
- Reviews and logs are treated as untrusted input, and instructions inside them are ignored.
- Reviewer usernames are never displayed.
- Each question has a token cap, to control cost.

## Roadmap

- [ ] Steam review ingest and clustering
- [ ] Agent with search, trend, and compare tools
- [ ] 30-question eval and results page
- [ ] Public app
- [ ] CI dataset (Godot) and CI health metrics

## Running it locally

Requires Python 3.9 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Pull Steam news for each game and list likely patch announcements
python -m sift news
python -m sift patches --game 553850

# 2. Mark the major patches in config/games.toml, then pull reviews around them
python -m sift reviews

# 3. Check what landed in the database
python -m sift stats
```

Data is stored locally in `data/sift.duckdb` and is not committed. Each ingest run writes a summary to `data/runs/`, and those summaries are the source for any count quoted in this repo.

## Project layout

```
sift/
  schema.py            one record schema shared by every source
  store.py             DuckDB storage and the patch window view
  ingest/steam.py      Steam reviews and Steam news adapter
  cli.py               command line entry point
config/games.toml      games, caps, and major patch dates
tests/                 offline tests with fixture data
```

## Author

Built by Nkiru Anizoba, Senior Technical Product Manager, AI Strategy. [LinkedIn](https://www.linkedin.com/in/nkiruanizoba/)
