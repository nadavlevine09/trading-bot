# News-Sentiment Trading Bot

A Python 3.11+ paper-trading bot that scores financial news headlines with FinBERT and converts aggregated sentiment into buy/sell signals executed via Alpaca's paper-trading API.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

| Variable | Where to get it |
|---|---|
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) — free tier gives 100 req/day |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — create a "script" app |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | [app.alpaca.markets](https://app.alpaca.markets) — Paper Trading dashboard |

```bash
cp .env.example .env
# edit .env with your keys
```

## Running Tests

```bash
pytest
```

## Project Layout

```
data/               historical price and news snapshots (gitignored except .gitkeep)
src/
  ingestion/        pull headlines from NewsAPI + Reddit
  sentiment/        FinBERT scoring (ProsusAI/finbert via HuggingFace)
  strategy/         aggregate scores → buy/sell/hold signals
  backtest/         replay signals against historical OHLCV data
  trading/          Alpaca paper-trade execution
tests/              unit tests mirroring src/ layout
```

## Notes

- All classes in `src/` have `raise NotImplementedError` stubs — implement them module by module.
- The FinBERT model (`ProsusAI/finbert`) downloads ~420 MB on first use via `transformers`.
- Alpaca is configured for **paper trading** (`paper=True`). Do not switch to live without explicit intent.
- NewsAPI free tier enforces a 30-day historical lookback; plan backtest data collection accordingly.
