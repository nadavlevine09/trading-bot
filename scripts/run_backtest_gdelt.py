"""End-to-end backtest: GDELT news + Alpaca prices → sentiment → signals → backtest.

Steps
-----
1. Pull 95 days of daily OHLCV from Alpaca (covers the full 90-day GDELT window).
2. Load GDELT articles for each ticker from data/gdelt/<TICKER>.json.
3. Score all articles with FinBERT and aggregate to daily sentiment scores.
4. Generate buy/sell/hold signals via SignalGenerator.
5. Replay signals through BacktestRunner against the downloaded price data.
6. Print a day-by-day decision log and a full performance summary.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("src").setLevel(logging.INFO)
_log = logging.getLogger("run_backtest_gdelt")
logging.getLogger("run_backtest_gdelt").setLevel(logging.DEBUG)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.backtest.runner import BacktestRunner
from src.portfolio import PortfolioConfig
from src.sentiment.scorer import SentimentScorer, aggregate_daily
from src.strategy.signals import SignalGenerator

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
PRICE_DATA_DIR = Path(__file__).parent.parent / "data" / "prices"
GDELT_DATA_DIR = Path(__file__).parent.parent / "data" / "gdelt"
PORTFOLIO_TOML = Path(__file__).parent.parent / "portfolio.toml"
PRICE_DAYS     = 95      # enough to cover the 90-day GDELT window with warmup
INITIAL_CASH   = 100_000.0


# ── Step 1: Download OHLCV ─────────────────────────────────────────────────────

def fetch_prices(tickers: list[str]) -> dict[str, bool]:
    """Fetch 95 days of daily bars from Alpaca; save to data/prices/<TICKER>.csv."""
    api_key = os.environ["ALPACA_API_KEY"]
    secret  = os.environ["ALPACA_SECRET_KEY"]
    client  = StockHistoricalDataClient(api_key=api_key, secret_key=secret)

    now   = datetime.now(timezone.utc)
    start = now - timedelta(days=PRICE_DAYS)

    PRICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, bool] = {}

    for ticker in tickers:
        print(f"  {ticker}", end=" … ", flush=True)
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=now,
                feed="iex",
            )
            bars = client.get_stock_bars(req)
            df   = bars.df

            if df.empty:
                print("no data")
                fetched[ticker] = False
                continue

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level=0)

            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"

            col_map = {"open": "open", "high": "high", "low": "low",
                       "close": "close", "volume": "volume"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]
            df.sort_index(inplace=True)

            out = PRICE_DATA_DIR / f"{ticker}.csv"
            df.to_csv(out)
            print(f"{len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
            fetched[ticker] = True
        except Exception as exc:
            print(f"ERROR: {exc}")
            fetched[ticker] = False

    return fetched


# ── Step 2: Load GDELT articles ────────────────────────────────────────────────

def load_gdelt_articles(tickers: list[str]) -> list[dict]:
    """Load pre-fetched GDELT JSON files for all tickers."""
    all_articles: list[dict] = []
    for ticker in tickers:
        path = GDELT_DATA_DIR / f"{ticker}.json"
        if not path.exists():
            print(f"  {ticker}: MISSING ({path}) — skipping")
            continue
        articles = json.loads(path.read_text(encoding="utf-8"))
        print(f"  {ticker}: {len(articles):,} articles  "
              f"({articles[0]['published_at'][:10] if articles else '?'} → "
              f"{articles[-1]['published_at'][:10] if articles else '?'})")
        all_articles.extend(articles)
    return all_articles


# ── Step 3+4: Score + signal ───────────────────────────────────────────────────

def score_and_signal(articles: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run FinBERT + aggregate + SignalGenerator. Returns (signals_df, daily_df)."""
    if not articles:
        return (
            pd.DataFrame(columns=["timestamp", "ticker", "signal"]),
            pd.DataFrame(columns=["ticker", "date", "sentiment_score", "n_headlines"]),
        )

    print(f"\n  Scoring {len(articles):,} headlines with FinBERT …")
    scorer  = SentimentScorer(device=-1)
    scored  = scorer.score(articles)

    daily   = aggregate_daily(scored)
    print(f"  Aggregated to {len(daily):,} (ticker, date) rows")

    gen     = SignalGenerator()
    signals = gen.generate(daily)

    rows = [{"timestamp": s.timestamp, "ticker": s.ticker, "signal": s.action}
            for s in signals]
    return pd.DataFrame(rows), daily


# ── Step 5: Run backtest ───────────────────────────────────────────────────────

def run_backtest(signals_df: pd.DataFrame) -> tuple[dict, BacktestRunner]:
    config  = PortfolioConfig.from_toml(PORTFOLIO_TOML)
    runner  = BacktestRunner(
        price_data_path=PRICE_DATA_DIR,
        portfolio_config=config,
        initial_cash=INITIAL_CASH,
    )
    return runner.run(signals_df), runner


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_daily_scores(daily: pd.DataFrame) -> None:
    if daily.empty:
        print("  (no daily scores computed)")
        return
    gen = SignalGenerator()
    print(f"\n  {'Ticker':<6}  {'Date':<12}  {'Score':>7}  {'n_art':>5}  Signal")
    print(f"  {'-'*6}  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*6}")
    for _, row in daily.sort_values(["ticker", "date"]).iterrows():
        ticker = str(row["ticker"])
        n      = int(row["n_headlines"])
        score  = float(row["sentiment_score"])
        action = gen.classify(score, n, ticker=ticker)
        flag   = " ←" if action != "hold" else ""
        print(f"  {ticker:<6}  {str(row['date']):<12}  {score:+.4f}  {n:>5}  {action.upper()}{flag}")
    print(f"\n  (buy>{gen.buy_threshold}, sell<{gen.sell_threshold}, min_headlines={gen.min_headlines})")


def print_signal_table(signals_df: pd.DataFrame) -> None:
    if signals_df.empty:
        print("  (no signals generated)")
        return

    non_hold = signals_df[signals_df["signal"] != "hold"].copy()
    non_hold["date"] = pd.to_datetime(non_hold["timestamp"]).dt.date

    print(f"\n  {'Date':<12} {'Ticker':<6} {'Signal':<6}")
    print(f"  {'-'*12} {'-'*6} {'-'*6}")
    for _, row in non_hold.sort_values("timestamp").iterrows():
        print(f"  {str(row['date']):<12} {row['ticker']:<6} {row['signal'].upper():<6}")

    buys  = (signals_df["signal"] == "buy").sum()
    sells = (signals_df["signal"] == "sell").sum()
    holds = (signals_df["signal"] == "hold").sum()
    print(f"\n  Signals: {len(signals_df)} total — {buys} BUY, {sells} SELL, {holds} HOLD")


def print_decision_log(runner: BacktestRunner) -> None:
    log = runner.decision_log
    if not log:
        print("\n  (no trades executed)")
        return
    print(f"\n  {'Date':<12} {'Ticker':<6} {'Action':<5} {'Price':>9} {'Qty':>8}  Reason")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*9} {'-'*8}  {'-'*20}")
    for entry in log:
        qty_str   = f"{entry['qty']:.2f}" if entry["qty"] else "—"
        price_str = f"${entry['price']:.2f}"
        print(f"  {entry['date']:<12} {entry['ticker']:<6} "
              f"{entry['action'].upper():<5} {price_str:>9} {qty_str:>8}  {entry['reason']}")


def print_portfolio_history(runner: BacktestRunner) -> None:
    hist = runner.portfolio_history
    if not hist:
        return
    print(f"\n  {'Date':<12}  {'Portfolio Value':>16}  {'Daily Return':>13}")
    print(f"  {'-'*12}  {'-'*16}  {'-'*13}")
    prev = None
    for date_str, value in hist:
        ret_str = f"{(value - prev) / prev * 100:+.2f}%" if prev else "—"
        print(f"  {date_str:<12}  ${value:>15,.2f}  {ret_str:>13}")
        prev = value


def print_metrics(metrics: dict) -> None:
    total_ret = metrics["total_return"] * 100
    sharpe    = metrics["sharpe_ratio"]
    max_dd    = metrics["max_drawdown"] * 100
    win_rate  = metrics["win_rate"] * 100
    n_trades  = metrics["n_trades"]
    n_days    = metrics["n_days"]
    final     = metrics["final_value"]
    initial   = metrics["initial_cash"]

    print(f"""
┌─────────────────────────────────────────┐
│          BACKTEST PERFORMANCE           │
├─────────────────────────────────────────┤
│  Initial capital   ${initial:>18,.2f}  │
│  Final value       ${final:>18,.2f}  │
│  Total return      {total_ret:>17.2f}%  │
│  Sharpe ratio      {sharpe:>18.3f}  │
│  Max drawdown      {max_dd:>17.2f}%  │
│  Win rate          {win_rate:>17.2f}%  │
│  Trades executed   {n_trades:>18d}  │
│  Trading days      {n_days:>18d}  │
└─────────────────────────────────────────┘""")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("STEP 1: Downloading OHLCV price data from Alpaca (95 days)")
    print("=" * 60)
    price_ok  = fetch_prices(TICKERS)
    available = [t for t, ok in price_ok.items() if ok]
    missing   = [t for t, ok in price_ok.items() if not ok]
    if missing:
        print(f"\n  Warning: no price data for {missing} — their signals will be skipped")

    print("\n" + "=" * 60)
    print("STEP 2: Loading GDELT articles from data/gdelt/")
    print("=" * 60)
    articles = load_gdelt_articles(available)
    print(f"\n  Total articles across all tickers: {len(articles):,}")

    if not articles:
        print("\nNo GDELT articles found.  Run scripts/fetch_gdelt_history.py first.")
        return

    print("\n" + "=" * 60)
    print("STEP 3+4: Scoring with FinBERT and generating signals")
    print("=" * 60)
    signals_df, daily_df = score_and_signal(articles)

    print("\n── Daily sentiment scores (all ticker-date pairs) ──────")
    print_daily_scores(daily_df)
    print("\n── Non-hold signals ────────────────────────────────────")
    print_signal_table(signals_df)

    print("\n" + "=" * 60)
    print("STEP 5: Running backtest")
    print("=" * 60)
    metrics, runner = run_backtest(signals_df)

    print("\n── Day-by-day portfolio value ──────────────────────────")
    print_portfolio_history(runner)

    print("\n── Trade log ───────────────────────────────────────────")
    print_decision_log(runner)

    print("\n── Performance summary ─────────────────────────────────")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
