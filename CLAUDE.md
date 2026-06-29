# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test file
pytest tests/test_sentiment.py

# Run a single test by name
pytest tests/test_strategy.py::TestSignalGenerator::test_generate_not_implemented
```

All API credentials live in `.env` (copy from `.env.example`). The code reads them via `os.environ[...]` — no key is optional; missing keys raise `KeyError` at instantiation time, not at call time.

## Architecture

The pipeline flows left to right through five independent modules:

```
NewsAPI / Reddit → SentimentScorer → SignalGenerator → BacktestRunner
                                                      → AlpacaTrader
```

Each module is a single class in its respective file. They are deliberately decoupled — no module imports another. The orchestration layer (not yet built) will wire them together.

### `src/ingestion/`
- `newsapi.py` — `NewsAPIClient.fetch_headlines()` hits `/v2/everything`. Free tier: 100 req/day, 30-day historical limit.
- `reddit.py` — `RedditClient.fetch_posts()` wraps PRAW. Intended subreddits: `r/wallstreetbets`, `r/stocks`, `r/investing`.
- Both return `list[dict]` so the schema is uniform before hitting the scorer.

### `src/sentiment/`
- `scorer.py` — `SentimentScorer.score()` runs `ProsusAI/finbert` (~420 MB, downloaded on first use).
- The pipeline is lazy-loaded (`_load()` called on first `score()` invocation) to avoid startup cost in tests and backtest mode.
- Output contract: `pd.DataFrame` with columns `[text, positive, negative, neutral]`.

### `src/strategy/`
- `signals.py` — `SignalGenerator.generate()` receives the sentiment DataFrame and returns `[timestamp, ticker, signal]` where `signal ∈ {buy, sell, hold}`.
- Aggregation logic (time-window, score threshold, per-ticker weighting) is intentionally not yet defined — implement here.

### `src/backtest/`
- `runner.py` — `BacktestRunner.run()` takes a signals DataFrame and replays it against OHLCV CSV data stored under `data/`.
- Returns a metrics dict (PnL, Sharpe ratio, max drawdown).
- Price data format is not yet fixed; establish the schema when implementing.

### `src/trading/`
- `alpaca.py` — `AlpacaTrader` wraps `alpaca-py`'s `TradingClient` with `paper=True` hardcoded.
- `submit_order()` and `get_positions()` are the only two methods needed for the MVP.

### `data/`
Historical price CSVs and news JSON snapshots go here. Not committed to git (only `.gitkeep` is tracked). Name convention TBD when implementing the backtest data pipeline.

## Key constraints
- **FinBERT is CPU-runnable** but slow; GPU is recommended for backtesting large headline sets. The `device` argument can be passed to the HuggingFace `pipeline()` call.
- **Alpaca paper trading only** — `paper=True` is hardcoded in `AlpacaTrader`. Never change this without explicit intent.
- **NewsAPI free tier** caps historical data at 30 days. Any backtest beyond 30 days requires a different price/news data source.
- Tests mock external I/O (API calls, model loading) — do not make real network calls in `tests/`.
