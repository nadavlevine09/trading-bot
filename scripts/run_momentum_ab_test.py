"""A/B test: momentum confirmation ON vs OFF, both 90-day backtest windows.

Uses pre-fetched data only — no network calls:
  data/gdelt_prev/  + data/prices_prev/  → window 1: 2025-12-17 → 2026-03-17
  data/gdelt/       + data/prices/       → window 2: 2026-03-20 → 2026-06-22

FinBERT scoring happens once per window; two backtests (on/off) share that run.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src").setLevel(logging.INFO)

from src.backtest.runner import BacktestRunner, _load_ticker_csv
from src.portfolio import PortfolioConfig
from src.sentiment.scorer import SentimentScorer, aggregate_daily
from src.strategy.signals import SignalGenerator

ROOT          = Path(__file__).parent.parent
PORTFOLIO_TOML = ROOT / "portfolio.toml"
INITIAL_CASH  = 100_000.0
TICKERS       = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]


WINDOWS = [
    {
        "label":      "Window 1: 2025-12-17 → 2026-03-17",
        "gdelt_dir":  ROOT / "data" / "gdelt_prev",
        "price_dir":  ROOT / "data" / "prices_prev",
    },
    {
        "label":      "Window 2: 2026-03-20 → 2026-06-22",
        "gdelt_dir":  ROOT / "data" / "gdelt",
        "price_dir":  ROOT / "data" / "prices",
    },
]


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_gdelt(gdelt_dir: Path) -> list[dict]:
    articles = []
    for ticker in TICKERS:
        path = gdelt_dir / f"{ticker}.json"
        if not path.exists():
            print(f"    {ticker}: missing {path.name} — skipped")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"    {ticker}: {len(data):,} articles")
        articles.extend(data)
    return articles


def load_prices(price_dir: Path) -> dict[str, pd.DataFrame]:
    ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        df = _load_ticker_csv(price_dir, ticker)
        if df is not None:
            ohlcv[ticker] = df
    return ohlcv


# ── Score once, generate signals twice ────────────────────────────────────────

def score_articles(articles: list[dict]) -> pd.DataFrame:
    print(f"\n  Scoring {len(articles):,} headlines with FinBERT …")
    scorer = SentimentScorer(device=-1)
    scored = scorer.score(articles)
    daily  = aggregate_daily(scored)
    print(f"  Aggregated to {len(daily):,} (ticker, date) rows")
    return daily


def make_signals_df(signals) -> pd.DataFrame:
    rows = [{"timestamp": s.timestamp, "ticker": s.ticker, "signal": s.action}
            for s in signals]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "ticker", "signal"]
    )


def run_backtest(signals_df: pd.DataFrame, price_dir: Path) -> tuple[dict, BacktestRunner]:
    config = PortfolioConfig.from_toml(PORTFOLIO_TOML)
    runner = BacktestRunner(
        price_data_path=price_dir,
        portfolio_config=config,
        initial_cash=INITIAL_CASH,
    )
    return runner.run(signals_df), runner


# ── Reporting ─────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _fmt(v: float) -> str:
    return f"${v:,.2f}"


def print_comparison(
    label: str,
    off_m: dict,
    on_m: dict,
    off_r: BacktestRunner,
    on_r: BacktestRunner,
    n_off_buy: int,
    n_off_sell: int,
    n_on_buy: int,
    n_on_sell: int,
) -> None:
    def delta(key: str, pct: bool = False) -> str:
        d = on_m[key] - off_m[key]
        if pct:
            return f"{d * 100:+.2f}pp"
        return f"{d:+.3f}"

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  {'Metric':<22}  {'WITHOUT':>10}  {'WITH':>10}  {'Δ':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")

    rows = [
        ("Total return",    _pct(off_m["total_return"]),  _pct(on_m["total_return"]),   delta("total_return", True)),
        ("Sharpe ratio",    f"{off_m['sharpe_ratio']:.3f}", f"{on_m['sharpe_ratio']:.3f}", delta("sharpe_ratio")),
        ("Max drawdown",    _pct(off_m["max_drawdown"]),  _pct(on_m["max_drawdown"]),   delta("max_drawdown", True)),
        ("Win rate",        _pct(off_m["win_rate"]),       _pct(on_m["win_rate"]),        delta("win_rate", True)),
        ("Trades executed", str(off_m["n_trades"]),        str(on_m["n_trades"]),         str(on_m["n_trades"] - off_m["n_trades"])),
        ("Trading days",    str(off_m["n_days"]),          str(on_m["n_days"]),           "—"),
        ("Final value",     _fmt(off_m["final_value"]),    _fmt(on_m["final_value"]),     "—"),
    ]
    for name, v_off, v_on, d in rows:
        print(f"  {name:<22}  {v_off:>10}  {v_on:>10}  {d:>10}")

    print(f"\n  Signal mix (BUY/SELL actionable):")
    print(f"    WITHOUT momentum: {n_off_buy} BUY, {n_off_sell} SELL")
    print(f"    WITH    momentum: {n_on_buy} BUY, {n_on_sell} SELL")
    filtered = (n_off_buy + n_off_sell) - (n_on_buy + n_on_sell)
    print(f"    Filtered out by momentum check: {filtered} signals → HOLD")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gen_off = SignalGenerator(momentum_confirm=False)
    gen_on  = SignalGenerator(momentum_confirm=True)

    for win in WINDOWS:
        print(f"\n{'#' * 62}")
        print(f"  {win['label']}")
        print(f"{'#' * 62}")

        # Load articles
        print(f"\n  Loading GDELT articles from {win['gdelt_dir'].name}/")
        articles = load_gdelt(win["gdelt_dir"])
        if not articles:
            print("  No articles — skipping window.")
            continue

        # Load prices (used both for backtest runner AND momentum check)
        print(f"\n  Loading price CSVs from {win['price_dir'].name}/")
        ohlcv = load_prices(win["price_dir"])
        print(f"  Loaded price data for: {list(ohlcv.keys())}")

        # Score once
        daily = score_articles(articles)

        # Generate signals — two variants from the same daily scores
        signals_off = gen_off.generate(daily)
        signals_on  = gen_on.generate(daily, price_history=ohlcv)

        df_off = make_signals_df(signals_off)
        df_on  = make_signals_df(signals_on)

        n_off_buy  = int((df_off["signal"] == "buy").sum())
        n_off_sell = int((df_off["signal"] == "sell").sum())
        n_on_buy   = int((df_on["signal"] == "buy").sum())
        n_on_sell  = int((df_on["signal"] == "sell").sum())

        # Run two backtests
        print("\n  Running backtest WITHOUT momentum confirmation …")
        metrics_off, runner_off = run_backtest(df_off, win["price_dir"])

        print("  Running backtest WITH momentum confirmation …")
        metrics_on, runner_on = run_backtest(df_on, win["price_dir"])

        print_comparison(
            win["label"],
            metrics_off, metrics_on,
            runner_off, runner_on,
            n_off_buy, n_off_sell,
            n_on_buy, n_on_sell,
        )

    print(f"\n{'=' * 62}")
    print("  Done.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
