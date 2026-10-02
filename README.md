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

1. **Player feedback:** 26k public Steam reviews of Helldivers 2, Cyberpunk 2077 and No Man's Sky, collected day by day in the weeks just before and just after three major updates per game.
2. **Engineering signals:** failed CI runs from the Godot game engine's public GitHub Actions history (coming after the first launch).

## How it's evaluated

Sift is graded against a hand-labeled eval set, not just demoed:

- **Feedback questions:** the share of answer claims that the cited sources actually support; usefulness; and whether it correctly declines questions the data can't answer.
- **CI failures:** classification accuracy against hand labels.

Results will be published here and in the app once the eval runs.

## Guardrails

- Answers must cite sources, and uncited claims are treated as failures.
- Every cited review ID is checked automatically: it must exist and must have been shown to the model while answering. Anything else is flagged.
- Reviews and logs are written by the public, so they are treated only as evidence. Instructions inside them are never followed.
- Reviewer usernames and profile IDs are dropped at collection and never stored.
- Each question has a cap on tool calls and tokens, and each app visitor gets a few live questions, to control cost.

## Roadmap

- [x] Steam review ingest and topic clustering
- [x] Agent with search, compare, and topic trend tools, plus citation checking
- [ ] 30-question eval and results page
- [ ] Public app
- [ ] CI dataset (Godot) and CI health metrics

## Running it locally

Requires Python 3.9 or later and an Anthropic API key in a `.env` file (`ANTHROPIC_API_KEY=...`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pip install -r requirements.txt

python -m sift news                 # Steam announcements, to choose patch dates
python -m sift reviews              # reviews around each patch in config/games.toml
python -m sift topics               # find and label themes
python -m sift ask "Why did Helldivers 2 players turn negative after the Account Linking Update?"
python -m sift export               # write public_data/ for the app
streamlit run streamlit_app.py      # the web app
```

Raw data stays in `data/` and is not committed. Each run writes a summary to `data/runs/`, and those summaries are the source for any number quoted in this repo.

## Project layout

```
sift/
  schema.py            one record schema shared by every source
  store.py             DuckDB storage
  ingest/steam.py      Steam reviews and Steam news adapter
  topics.py            theme finding (TF-IDF + NMF) and Claude labels
  agent.py             Claude with tools, plus citation checking
  export.py            public dataset for the hosted app
  cli.py               command line entry point
streamlit_app.py       web app: Ask, Patches, Topics, How it works
public_data/           reviews, themes and example answers used by the app
config/games.toml      games, caps, and major patch dates
tests/                 offline tests
```

## Author

Built by Nkiru Anizoba, Senior Technical Product Manager, AI Strategy. [LinkedIn](https://www.linkedin.com/in/nkiruanizoba/)
