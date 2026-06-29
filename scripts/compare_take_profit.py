"""Compare backtest performance with and without the 18% take-profit rule.

Runs both configurations against the same signal set (scored once) so the
only difference between the two runs is the take_profit_pct threshold.

Usage
-----
    # 1. Fetch GDELT for the target window:
    python scripts/fetch_gdelt_history.py \\
        --start 2025-12-17 --end 2026-03-17 --out-dir data/gdelt_prev

    # 2. Run the comparison (fetches Alpaca prices automatically):
    python scripts/compare_take_profit.py \\
        --gdelt-dir data/gdelt_prev \\
        --prices-dir data/prices_prev \\
        --price-end 2026-03-17
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src").setLevel(logging.INFO)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.backtest.runner import BacktestRunner
from src.portfolio import PortfolioConfig
from src.sentiment.scorer import SentimentScorer, aggregate_daily
from src.strategy.signals import SignalGenerator

TICKERS        = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
PORTFOLIO_TOML = Path(__file__).parent.parent / "portfolio.toml"
PRICE_DAYS     = 95
INITIAL_CASH   = 100_000.0


# ── Price fetch ────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], prices_dir: Path, price_end: date) -> list[str]:
    """Fetch PRICE_DAYS of daily bars up to price_end from Alpaca IEX."""
    api_key = os.environ["ALPACA_API_KEY"]
    secret  = os.environ["ALPACA_SECRET_KEY"]
    client  = StockHistoricalDataClient(api_key=api_key, secret_key=secret)

    end_dt   = datetime(price_end.year, price_end.month, price_end.day,
                        23, 59, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=PRICE_DAYS)

    prices_dir.mkdir(parents=True, exist_ok=True)
    available: list[str] = []

    for ticker in tickers:
        print(f"  {ticker}", end=" … ", flush=True)
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                feed="iex",
            )
            bars = client.get_stock_bars(req)
            df   = bars.df
            if df.empty:
                print("no data")
                continue
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level=0)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]
            df.sort_index(inplace=True)
            df.to_csv(prices_dir / f"{ticker}.csv")
            print(f"{len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
            available.append(ticker)
        except Exception as exc:
            print(f"ERROR: {exc}")

    return available


# ── GDELT load ─────────────────────────────────────────────────────────────────

def load_gdelt(tickers: list[str], gdelt_dir: Path) -> list[dict]:
    articles: list[dict] = []
    for ticker in tickers:
        path = gdelt_dir / f"{ticker}.json"
        if not path.exists():
            print(f"  {ticker}: MISSING — skipping")
            continue
        batch = json.loads(path.read_text(encoding="utf-8"))
        print(f"  {ticker}: {len(batch):,} articles  "
              f"({batch[0]['published_at'][:10] if batch else '?'} → "
              f"{batch[-1]['published_at'][:10] if batch else '?'})")
        articles.extend(batch)
    return articles


# ── Score + signal (run once, shared across both backtests) ───────────────────

def score_and_signal(articles: list[dict]) -> pd.DataFrame:
    """Return signals_df; also prints daily scores."""
    print(f"\n  Scoring {len(articles):,} headlines with FinBERT …")
    scorer  = SentimentScorer(device=-1)
    scored  = scorer.score(articles)
    daily   = aggregate_daily(scored)
    print(f"  Aggregated to {len(daily):,} (ticker, date) rows")

    gen     = SignalGenerator()
    signals = gen.generate(daily)
    rows    = [{"timestamp": s.timestamp, "ticker": s.ticker, "signal": s.action}
               for s in signals]
    signals_df = pd.DataFrame(rows)

    non_hold = (signals_df["signal"] != "hold").sum()
    buys     = (signals_df["signal"] == "buy").sum()
    sells    = (signals_df["signal"] == "sell").sum()
    print(f"  {len(signals_df)} signals — {buys} BUY, {sells} SELL, "
          f"{len(signals_df) - non_hold} HOLD")
    return signals_df


# ── Single backtest run ────────────────────────────────────────────────────────

def run_one(
    signals_df: pd.DataFrame,
    prices_dir: Path,
    config: PortfolioConfig,
    label: str,
) -> tuple[dict, BacktestRunner]:
    runner  = BacktestRunner(
        price_data_path=prices_dir,
        portfolio_config=config,
        initial_cash=INITIAL_CASH,
    )
    metrics = runner.run(signals_df)
    return metrics, runner


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_trade_log(runner: BacktestRunner, label: str) -> None:
    log = runner.decision_log
    print(f"\n  ── {label} trade log {'─' * max(0, 48 - len(label))}")
    if not log:
        print("    (no trades executed)")
        return
    print(f"  {'Date':<12} {'Ticker':<6} {'Action':<5} {'Price':>9} {'Qty':>7}  Reason")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*9} {'-'*7}  {'-'*30}")
    for e in log:
        qty_str = f"{e['qty']:.0f}"
        print(f"  {e['date']:<12} {e['ticker']:<6} {e['action'].upper():<5} "
              f"${e['price']:>8.2f} {qty_str:>7}  {e['reason']}")


def print_comparison(
    window_label: str,
    m_base: dict,
    m_tp: dict,
) -> None:
    def fmt_ret(v): return f"{v*100:+.2f}%"
    def fmt_pct(v): return f"{v*100:.2f}%"
    def fmt_f(v):   return f"{v:.3f}"
    def fmt_i(v):   return str(v)

    rows = [
        ("Total return",      fmt_ret(m_base["total_return"]),   fmt_ret(m_tp["total_return"])),
        ("Sharpe ratio",      fmt_f(m_base["sharpe_ratio"]),     fmt_f(m_tp["sharpe_ratio"])),
        ("Max drawdown",      fmt_pct(m_base["max_drawdown"]),   fmt_pct(m_tp["max_drawdown"])),
        ("Win rate",          fmt_pct(m_base["win_rate"]),       fmt_pct(m_tp["win_rate"])),
        ("Trades executed",   fmt_i(m_base["n_trades"]),         fmt_i(m_tp["n_trades"])),
        ("Trading days",      fmt_i(m_base["n_days"]),           fmt_i(m_tp["n_days"])),
        ("Final value",       f"${m_base['final_value']:,.2f}",  f"${m_tp['final_value']:,.2f}"),
    ]

    col0 = max(len(r[0]) for r in rows) + 2
    col1 = max(max(len(r[1]) for r in rows), len("No take-profit")) + 2
    col2 = max(max(len(r[2]) for r in rows), len("+18% take-profit")) + 2

    sep  = f"  ├{'─'*(col0+2)}┼{'─'*(col1+2)}┼{'─'*(col2+2)}┤"
    top  = f"  ┌{'─'*(col0+2)}┬{'─'*(col1+2)}┬{'─'*(col2+2)}┐"
    bot  = f"  └{'─'*(col0+2)}┴{'─'*(col1+2)}┴{'─'*(col2+2)}┘"

    def row(label, a, b):
        return f"  │ {label:<{col0}} │ {a:>{col1}} │ {b:>{col2}} │"

    print(f"\n  Window: {window_label}")
    print(top)
    print(row("Metric", "No take-profit", "+18% take-profit"))
    print(sep)
    for label, a, b in rows:
        print(row(label, a, b))
    print(bot)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare backtest with/without 18%% take-profit on a given GDELT window."
    )
    parser.add_argument("--gdelt-dir",   required=True, help="Directory with GDELT JSON files")
    parser.add_argument("--prices-dir",  required=True, help="Directory for price CSVs (created if absent)")
    parser.add_argument(
        "--price-end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Latest date for price bars (default: yesterday). "
             "Set this to the GDELT window end date.",
    )
    parser.add_argument(
        "--skip-price-fetch",
        action="store_true",
        help="Skip Alpaca fetch; use existing CSVs in --prices-dir",
    )
    args = parser.parse_args()

    gdelt_dir  = Path(args.gdelt_dir)
    prices_dir = Path(args.prices_dir)
    price_end  = (date.fromisoformat(args.price_end)
                  if args.price_end else date.today() - timedelta(days=1))

    # ── Step 1: Prices ──────────────────────────────────────────────────────────
    if args.skip_price_fetch:
        print("Skipping price fetch (--skip-price-fetch set)")
        available = [t for t in TICKERS if (prices_dir / f"{t}.csv").exists()]
        print(f"  Found price CSVs for: {' '.join(available)}")
    else:
        print("=" * 60)
        print(f"STEP 1: Fetching Alpaca prices  (ending {price_end})")
        print("=" * 60)
        available = fetch_prices(TICKERS, prices_dir, price_end)

    if not available:
        print("ERROR: no price data available", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Load GDELT ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading GDELT articles")
    print("=" * 60)
    articles = load_gdelt(available, gdelt_dir)
    print(f"\n  Total: {len(articles):,} articles")

    if not articles:
        print("ERROR: no GDELT articles found", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Score + signals (once) ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: FinBERT scoring + signal generation")
    print("=" * 60)
    signals_df = score_and_signal(articles)

    if signals_df.empty:
        print("No signals generated — nothing to backtest.")
        return

    # ── Step 4: Two backtests ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Running backtests")
    print("=" * 60)

    base_config = PortfolioConfig.from_toml(PORTFOLIO_TOML)
    cfg_no_tp   = replace(base_config, take_profit_pct=1.0)   # effectively disabled
    cfg_with_tp = base_config                                   # uses toml value (0.18)

    print("\n  [A] Trailing stops only (no take-profit) …", flush=True)
    m_base, runner_base = run_one(signals_df, prices_dir, cfg_no_tp,   "No take-profit")

    print("  [B] Trailing stops + 18% take-profit …",     flush=True)
    m_tp,   runner_tp   = run_one(signals_df, prices_dir, cfg_with_tp, "+18% take-profit")

    # ── Step 5: Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    window_label = f"{price_end - timedelta(days=PRICE_DAYS)} → {price_end}"
    print_comparison(window_label, m_base, m_tp)

    print_trade_log(runner_base, "No take-profit")
    print_trade_log(runner_tp,   "+18% take-profit")


if __name__ == "__main__":
    main()
