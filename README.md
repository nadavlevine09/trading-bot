# News-Sentiment Trading Bot

- **What it does:** reads daily news about eight large-cap stocks, scores each headline with FinBERT, and turns strong positive or negative sentiment into paper trades through Alpaca, with risk limits on every trade.
- **Backtest 1 (Dec 2025 – Mar 2026):** −0.91% over 7 trades, every one a loss.
- **Backtest 2 (Mar – Jun 2026):** +4.28% over 23 trades, 70% win rate. Same code, same settings.
- **What I learned:** one backtest proves almost nothing. The gap came from market conditions, and every feature I added (momentum confirmation, earnings data, portfolio optimization) helped in one kind of market and hurt in the other.
- **Also:** news data quality matters as much as the model. Full write-up in [WRITEUP.md](WRITEUP.md).

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

- Note: the Reddit source (src/ingestion/reddit.py) isn't implemented yet; the bot runs on the news sources.
- The FinBERT model (`ProsusAI/finbert`) downloads ~420 MB on first use via `transformers`.
- Alpaca is configured for **paper trading** (`paper=True`). Do not switch to live without explicit intent.
- NewsAPI free tier enforces a 30-day historical lookback; plan backtest data collection accordingly.
