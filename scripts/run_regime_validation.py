"""Regime-gated backtest validation on a fresh (untouched) data window.

Window
------
  GDELT articles : 2025-09-19 → 2025-12-17  (90 days; no overlap with W1/W2)
  Prices fetched : 2025-07-01 → 2025-12-31  (extra warmup for Markowitz + momentum)

Three backtest configurations compared
---------------------------------------
  (a) baseline      – no momentum, no smoothing (α=1.0), no Markowitz
  (b) best_config   – momentum=True, α=0.5, Markowitz, applied uniformly
  (c) regime_gated  – same features, but gated by per-day regime:
                       trending  → momentum=True  + Markowitz
                       choppy    → momentum=False + fixed sizing
                       α=0.5 in both regimes (as recommended from prior finding)

Outputs
-------
  logs/regime_validation.log        — detailed run log
  logs/regime_validation_summary.md — concise result table + honest verdict
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
LOGS_DIR     = ROOT / "logs"
LOG_FILE     = LOGS_DIR / "regime_validation.log"
SUMMARY_FILE = LOGS_DIR / "regime_validation_summary.md"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("src").setLevel(logging.INFO)
_log = logging.getLogger("regime_validation")
logging.getLogger("regime_validation").setLevel(logging.DEBUG)


def log(msg: str = "") -> None:
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# ── Imports ───────────────────────────────────────────────────────────────────
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.backtest.runner import BacktestRunner, _load_ticker_csv
from src.config.companies import COMPANIES
from src.ingestion.gdelt import GDELTClient, GDELTError
from src.portfolio import MarkowitzOptimizer, PortfolioConfig
from src.sentiment.scorer import SentimentScorer, aggregate_daily, smooth_daily_scores
from src.strategy.regime import RegimeDetector
from src.strategy.signals import SignalGenerator

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
PORTFOLIO_TOML = ROOT / "portfolio.toml"
INITIAL_CASH   = 100_000.0

# Window W3: the fresh, untouched window
W3_GDELT_START = date(2025, 9, 19)
W3_GDELT_END   = date(2025, 12, 17)
W3_PRICE_START = date(2025, 7, 1)    # extra warmup for Markowitz (60-day) + regime (80-day)
W3_PRICE_END   = date(2025, 12, 31)

# Data directories (separate from W1/W2 to avoid contaminating prior runs)
W3_GDELT_DIR   = ROOT / "data" / "gdelt_w3"
W3_PRICE_DIR   = ROOT / "data" / "prices_w3"

# Best-config parameters (from prior A/B results)
BEST_ALPHA            = 0.5
BEST_MOMENTUM         = True
BEST_MARKOWITZ        = True
MARKOWITZ_LOOKBACK    = 60
MARKOWITZ_MAX_WEIGHT  = 0.25

# Regime detector parameters (configurable)
REGIME_VOL_WINDOW        = 20
REGIME_VOL_LOOKBACK      = 60
REGIME_RATIO_THRESHOLD   = 1.2
REGIME_SMOOTH_ALPHA      = 0.5

# GDELT fetch config — DOC API sub-chunks internally to 7-day windows.
# Pass the full window as one "chunk" per ticker; dedup happens inside the client.
CHUNK_DAYS_DEFAULT = 90   # one call covering the full window; client sub-chunks
CHUNK_DAYS: dict[str, int] = {}   # no per-ticker override needed with DOC API
MAX_ROWS_CHUNK = 5_000


# ── Step 1: Fetch GDELT ───────────────────────────────────────────────────────

def fetch_gdelt_w3(force: bool = False) -> bool:
    """Fetch GDELT articles for W3 into data/gdelt_w3/. Skip if already present."""
    W3_GDELT_DIR.mkdir(parents=True, exist_ok=True)

    all_present = all((W3_GDELT_DIR / f"{t}.json").exists() for t in TICKERS)
    if all_present and not force:
        log("  [cache] All W3 GDELT files already present — skipping fetch")
        return True

    log(f"  Fetching GDELT {W3_GDELT_START} → {W3_GDELT_END}")
    import time as _time; _time.sleep(60)   # let any prior rate-limit block expire

    try:
        client = GDELTClient()
    except Exception as exc:
        log(f"  ERROR: GDELTClient init failed — {exc}")
        return False

    ok = True
    for ticker in TICKERS:
        info        = COMPANIES.get(ticker)
        entity_name = info.legal_entity_name if info else ticker
        chunk_days  = CHUNK_DAYS.get(ticker, CHUNK_DAYS_DEFAULT)

        chunks = _date_chunks(W3_GDELT_START, W3_GDELT_END, chunk_days)
        log(f"  {ticker} ({entity_name}): {len(chunks)}×{chunk_days}d chunks")

        seen_urls: set[str] = set()
        articles:  list[dict] = []
        capped = False

        for (cs, ce) in chunks:
            try:
                batch = client.fetch_for_ticker(
                    ticker=ticker,
                    company_name=entity_name,
                    from_date=cs,
                    to_date=ce,
                    max_rows=MAX_ROWS_CHUNK,
                )
            except GDELTError as exc:
                log(f"    [{cs}→{ce}] ERROR: {exc}")
                continue

            fresh = []
            for a in batch:
                url = a.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                fresh.append(a)
            if len(batch) >= MAX_ROWS_CHUNK:
                capped = True
            articles.extend(fresh)
            log(f"    [{cs} → {ce}]  {len(batch):>4} raw → {len(fresh):>4} new")

        articles.sort(key=lambda a: a.get("published_at", ""))
        out = W3_GDELT_DIR / f"{ticker}.json"
        out.write_text(json.dumps(articles, indent=2, ensure_ascii=False), encoding="utf-8")

        dates = {a["published_at"][:10] for a in articles if a.get("published_at")}
        log(f"    saved {len(articles):,} articles, {len(dates)} active days"
            + ("  ⚠ chunk cap hit" if capped else ""))

    return ok


def _date_chunks(start: date, end: date, size: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=size - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


# ── Step 2: Fetch prices ──────────────────────────────────────────────────────

def fetch_prices_w3(force: bool = False) -> dict[str, bool]:
    """Fetch historical OHLCV for the W3 warmup+backtest window."""
    W3_PRICE_DIR.mkdir(parents=True, exist_ok=True)

    all_present = all((W3_PRICE_DIR / f"{t}.csv").exists() for t in TICKERS)
    if all_present and not force:
        log("  [cache] All W3 price files already present — skipping fetch")
        return {t: True for t in TICKERS}

    api_key = os.environ["ALPACA_API_KEY"]
    secret  = os.environ["ALPACA_SECRET_KEY"]
    client  = StockHistoricalDataClient(api_key=api_key, secret_key=secret)

    start_ts = datetime(W3_PRICE_START.year, W3_PRICE_START.month, W3_PRICE_START.day,
                        tzinfo=timezone.utc)
    end_ts   = datetime(W3_PRICE_END.year, W3_PRICE_END.month, W3_PRICE_END.day,
                        23, 59, 59, tzinfo=timezone.utc)

    results: dict[str, bool] = {}
    for ticker in TICKERS:
        print(f"  {ticker} … ", end="", flush=True)
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_ts,
                end=end_ts,
                feed="iex",
            )
            bars = client.get_stock_bars(req)
            df   = bars.df

            if df.empty:
                log("no data")
                results[ticker] = False
                continue

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level=0)

            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]
            df.sort_index(inplace=True)

            out = W3_PRICE_DIR / f"{ticker}.csv"
            df.to_csv(out)
            log(f"{len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
            results[ticker] = True
        except Exception as exc:
            log(f"ERROR: {exc}")
            results[ticker] = False

    return results


# ── Step 3: Load + score GDELT articles ───────────────────────────────────────

def load_articles() -> list[dict]:
    articles: list[dict] = []
    for ticker in TICKERS:
        path = W3_GDELT_DIR / f"{ticker}.json"
        if not path.exists():
            log(f"  {ticker}: MISSING — skipping")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        log(f"  {ticker}: {len(data):,} articles")
        articles.extend(data)
    return articles


def sample_articles(articles: list[dict], max_per_ticker_per_day: int = 100) -> list[dict]:
    """Cap articles per (ticker, date) to keep FinBERT scoring tractable."""
    import random
    random.seed(42)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for art in articles:
        key = (art.get("ticker", ""), (art.get("published_at") or "")[:10])
        groups[key].append(art)
    result = []
    for group in groups.values():
        if len(group) <= max_per_ticker_per_day:
            result.extend(group)
        else:
            result.extend(random.sample(group, max_per_ticker_per_day))
    log(f"  Sampled {len(result):,} articles (cap={max_per_ticker_per_day}/ticker/day, was {len(articles):,})")
    return result


_SCORE_CACHE_OVERRIDE: Path | None = None

def score_articles(articles: list[dict]) -> pd.DataFrame:
    cache_path = _SCORE_CACHE_OVERRIDE or (ROOT / "data" / "cache" / "w3_scored.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        log(f"  [cache] Loading scored data from {cache_path.name}")
        return pd.read_pickle(cache_path)

    log(f"  Scoring {len(articles):,} headlines with FinBERT …")
    scorer = SentimentScorer(device=-1)
    scored = scorer.score(articles)
    scored.to_pickle(cache_path)
    log(f"  Cached to {cache_path.name}")
    return scored


# ── Step 4: Signal generation helpers ─────────────────────────────────────────

def make_signals_df(signals: list) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame(columns=["timestamp", "ticker", "signal"])
    return pd.DataFrame(
        [{"timestamp": s.timestamp, "ticker": s.ticker, "signal": s.action}
         for s in signals]
    )


def generate_baseline_signals(daily: pd.DataFrame) -> pd.DataFrame:
    """No momentum, no smoothing, no earnings — pure threshold signals."""
    gen = SignalGenerator(momentum_confirm=False, earnings_confirm=False)
    return make_signals_df(gen.generate(daily))


def generate_best_config_signals(
    daily: pd.DataFrame,
    price_history: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Uniformly apply the best config: alpha=0.5 smoothing + momentum=True.
    (Markowitz is applied at the runner level, not signal generation.)
    """
    daily_smooth = smooth_daily_scores(daily, alpha=BEST_ALPHA)
    gen = SignalGenerator(momentum_confirm=BEST_MOMENTUM, earnings_confirm=False)
    return make_signals_df(gen.generate(daily_smooth, price_history=price_history))


def generate_regime_gated_signals(
    daily: pd.DataFrame,
    price_history: dict[str, pd.DataFrame],
    regime_map: dict[pd.Timestamp, str],
) -> pd.DataFrame:
    """
    Per-day regime-gated signals:
      trending → momentum_confirm=True
      choppy   → momentum_confirm=False
    Both use alpha=0.5 smoothing (smoothing applied uniformly, regime only gates momentum).
    """
    daily_smooth = smooth_daily_scores(daily, alpha=BEST_ALPHA)

    all_signals = []
    for signal_date, day_df in daily_smooth.groupby("date"):
        dt_norm  = pd.Timestamp(signal_date).normalize()
        regime   = regime_map.get(dt_norm, "trending")
        use_mom  = (regime == "trending")

        gen  = SignalGenerator(momentum_confirm=use_mom, earnings_confirm=False)
        sigs = gen.generate(day_df, price_history=price_history)
        all_signals.extend(sigs)

        for s in sigs:
            if s.action != "hold":
                _log.debug(
                    "regime_gated %s %s: regime=%s momentum=%s → %s",
                    signal_date, s.ticker, regime, use_mom, s.action,
                )

    return make_signals_df(all_signals)


# ── Step 5: Regime-aware Markowitz ────────────────────────────────────────────

class RegimeAwareOptimizer:
    """
    Wrapper around MarkowitzOptimizer that returns empty weights on choppy days.
    On choppy days the BacktestRunner falls back to fixed-percentage sizing.
    """
    def __init__(
        self,
        optimizer: MarkowitzOptimizer,
        regime_map: dict[pd.Timestamp, str],
    ) -> None:
        self._optimizer  = optimizer
        self._regime_map = regime_map

    def compute_weights(
        self,
        price_history: dict[str, pd.DataFrame],
        tickers: list[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        regime = self._regime_map.get(as_of.normalize(), "trending")
        if regime == "choppy":
            _log.debug("RegimeAwareOptimizer: choppy on %s — returning empty weights", as_of.date())
            return {}
        return self._optimizer.compute_weights(price_history, tickers, as_of)


# ── Step 6: Run backtests ─────────────────────────────────────────────────────

def run_bt(
    sig_df: pd.DataFrame,
    optimizer=None,
) -> tuple[dict, BacktestRunner]:
    cfg    = PortfolioConfig.from_toml(PORTFOLIO_TOML)
    runner = BacktestRunner(
        price_data_path=W3_PRICE_DIR,
        portfolio_config=cfg,
        initial_cash=INITIAL_CASH,
        optimizer=optimizer,
    )
    return runner.run(sig_df), runner


def count_signals(sig_df: pd.DataFrame) -> tuple[int, int, int]:
    buy  = int((sig_df["signal"] == "buy").sum())
    sell = int((sig_df["signal"] == "sell").sum())
    hold = int((sig_df["signal"] == "hold").sum())
    return buy, sell, hold


# ── Step 7: Reporting ─────────────────────────────────────────────────────────

def fmt_metrics(m: dict) -> dict[str, str]:
    return {
        "total_return": f"{m['total_return'] * 100:+.2f}%",
        "sharpe_ratio": f"{m['sharpe_ratio']:+.3f}",
        "max_drawdown": f"{m['max_drawdown'] * 100:.2f}%",
        "win_rate":     f"{m['win_rate'] * 100:.1f}%",
        "n_trades":     str(m["n_trades"]),
        "n_days":       str(m["n_days"]),
    }


def metrics_table(
    labels: list[str],
    metrics: list[dict],
    signal_counts: list[tuple[int, int, int]],
) -> str:
    col_w = 18
    rows = [
        ("Total return",    [f"{m['total_return']*100:+.2f}%" for m in metrics]),
        ("Sharpe ratio",    [f"{m['sharpe_ratio']:+.3f}"      for m in metrics]),
        ("Max drawdown",    [f"{m['max_drawdown']*100:.2f}%"  for m in metrics]),
        ("Win rate",        [f"{m['win_rate']*100:.1f}%"      for m in metrics]),
        ("Trades executed", [str(m["n_trades"])               for m in metrics]),
        ("Trading days",    [str(m["n_days"])                 for m in metrics]),
    ]

    header = f"  {'Metric':<26}" + "".join(f"  {lbl:>{col_w}}" for lbl in labels)
    sep    = f"  {'-'*26}" + "".join(f"  {'-'*col_w}" for _ in labels)
    lines  = [header, sep]
    for name, vals in rows:
        lines.append(f"  {name:<26}" + "".join(f"  {v:>{col_w}}" for v in vals))

    lines.append("")
    lines.append("  Signal counts (BUY / SELL / HOLD):")
    for lbl, (b, s, h) in zip(labels, signal_counts):
        lines.append(f"    {lbl}: {b} BUY  {s} SELL  {h} HOLD")

    return "\n".join(lines)


def write_summary(
    regime_map: dict[pd.Timestamp, str],
    labels: list[str],
    metrics: list[dict],
    signal_counts: list[tuple[int, int, int]],
) -> None:
    n_trending = sum(1 for v in regime_map.values() if v == "trending")
    n_choppy   = sum(1 for v in regime_map.values() if v == "choppy")
    total      = len(regime_map)

    table = metrics_table(labels, metrics, signal_counts)

    # Determine verdict
    m_base  = metrics[0]
    m_best  = metrics[1]
    m_gated = metrics[2]

    gated_beats_base  = m_gated["total_return"] > m_base["total_return"]
    gated_beats_best  = m_gated["total_return"] > m_best["total_return"]
    gated_sharpe_base = m_gated["sharpe_ratio"] > m_base["sharpe_ratio"]
    gated_sharpe_best = m_gated["sharpe_ratio"] > m_best["sharpe_ratio"]

    if gated_beats_base and gated_beats_best:
        verdict = (
            "**The regime-gated version OUTPERFORMS both the naive baseline AND the "
            "uniformly-applied best config on this fresh, untouched window.** "
            "The regime gate is adding genuine value beyond what was seen on prior windows."
        )
    elif gated_beats_base and not gated_beats_best:
        verdict = (
            "The regime-gated version outperforms the naive baseline but NOT the "
            "uniformly-applied best config on this window. "
            "The gate partially helps but does not improve on applying the best config uniformly."
        )
    elif not gated_beats_base and gated_beats_best:
        verdict = (
            "The regime-gated version beats the uniformly-applied best config but NOT the "
            "naive baseline on this window. This is an unusual result — the gate is useful "
            "relative to the best config but the market may have been in a regime where "
            "fewer trades was net-positive (similar to W1 EWM smoothing pattern)."
        )
    else:
        verdict = (
            "**The regime-gated version does NOT outperform either the naive baseline or the "
            "uniformly-applied best config on this fresh window.** "
            "This is an honest null result: the regime gate may have overfit patterns from W1/W2, "
            "or the W3 window may have required different parameters."
        )

    # Sharpe addendum
    sharpe_note = ""
    if gated_sharpe_base and gated_sharpe_best:
        sharpe_note = "  Risk-adjusted (Sharpe): regime-gated also wins on both comparisons.\n"
    elif gated_sharpe_base != gated_beats_base or gated_sharpe_best != gated_beats_best:
        sharpe_note = (
            f"  Note: Sharpe comparison diverges from return comparison — "
            f"gated Sharpe ({m_gated['sharpe_ratio']:+.3f}) vs baseline "
            f"({m_base['sharpe_ratio']:+.3f}) and best config "
            f"({m_best['sharpe_ratio']:+.3f}).\n"
        )

    regime_detail = "\n".join(
        f"  {dt.date()}  →  {r}"
        for dt, r in sorted(regime_map.items())
    )

    content = f"""# Regime Validation: W3 (2025-09-19 → 2025-12-17)

*Run completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC*

## Window

Fresh, untouched data window with zero overlap to W1 (2025-12-17→2026-03-17)
or W2 (2026-03-20→2026-06-22).  Neither the regime parameters nor the feature
configs were tuned against this data.

## Regime Classification

Detector parameters:
  vol_window={REGIME_VOL_WINDOW}, vol_lookback={REGIME_VOL_LOOKBACK},
  ratio_threshold={REGIME_RATIO_THRESHOLD}, smooth_alpha={REGIME_SMOOTH_ALPHA}

  Total days classified : {total}
  Trending              : {n_trending} ({100*n_trending/total:.0f}%)
  Choppy                : {n_choppy} ({100*n_choppy/total:.0f}%)

Per-day labels:

{regime_detail}

## Configuration Summary

  (a) baseline    : momentum=False, earnings=False, α=1.0, no Markowitz
  (b) best_config : momentum=True,  earnings=False, α=0.5, Markowitz (lookback=60, max_w=0.25)
  (c) regime_gated: momentum=regime, earnings=False, α=0.5, Markowitz only on trending days

## Results

{table}

## Verdict

{verdict}

{sharpe_note}
### Regime breakdown note

In this window, {100*n_trending/total:.0f}% of days were classified as trending and
{100*n_choppy/total:.0f}% as choppy.  In a window dominated by trending days, the
regime gate will behave similarly to the best config (few days where it differs).
In a window with mixed or predominantly choppy days, the gate's value should be
more apparent.

### Configuration lineage

The best_config parameters (momentum=True, α=0.5, Markowitz) were chosen from
A/B results on W1 (2025-12-17→2026-03-17) and W2 (2026-03-20→2026-06-22).
The regime detector was not tuned against either window — parameters
(20-day vol, 60-day baseline, 1.2× threshold) are conventional values.
"""

    SUMMARY_FILE.write_text(content, encoding="utf-8")
    log(f"\nSummary written to {SUMMARY_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdelt-dir", default=None,
                        help="Override W3_GDELT_DIR (path to pre-fetched JSON files)")
    parser.add_argument("--score-cache", default=None,
                        help="Override pickle cache path for scored articles")
    args = parser.parse_args()

    global W3_GDELT_DIR
    if args.gdelt_dir:
        W3_GDELT_DIR = Path(args.gdelt_dir)
        log(f"  [override] GDELT dir: {W3_GDELT_DIR}")

    global _SCORE_CACHE_OVERRIDE
    _SCORE_CACHE_OVERRIDE = Path(args.score_cache) if args.score_cache else None

    log("=" * 65)
    log("REGIME VALIDATION — W3: 2025-09-19 → 2025-12-17")
    log("=" * 65)

    # ── GDELT fetch ────────────────────────────────────────────────────────────
    log("\nSTEP 1: Fetching GDELT articles for W3")
    log("-" * 65)
    fetch_gdelt_w3()

    # ── Price fetch ────────────────────────────────────────────────────────────
    log("\nSTEP 2: Fetching historical prices for W3")
    log("-" * 65)
    price_ok = fetch_prices_w3()
    available = [t for t, ok in price_ok.items() if ok]
    missing   = [t for t, ok in price_ok.items() if not ok]
    if missing:
        log(f"  Warning: no price data for {missing}")

    # ── Load + score ───────────────────────────────────────────────────────────
    log("\nSTEP 3: Loading and scoring GDELT articles")
    log("-" * 65)
    articles = load_articles()
    log(f"  Total: {len(articles):,} articles across all tickers")
    if not articles:
        log("ERROR: no articles — run fetch_gdelt_history.py or check GDELTClient")
        return

    articles = sample_articles(articles, max_per_ticker_per_day=100)
    scored = score_articles(articles)
    daily  = aggregate_daily(scored)
    log(f"  Aggregated to {len(daily):,} (ticker, date) rows")

    # ── Load price history (for momentum + regime detection) ───────────────────
    log("\nSTEP 4: Loading price history")
    log("-" * 65)
    price_history: dict[str, pd.DataFrame] = {}
    for ticker in available:
        df = _load_ticker_csv(W3_PRICE_DIR, ticker)
        if df is not None:
            price_history[ticker] = df
            log(f"  {ticker}: {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    # ── Regime detection ───────────────────────────────────────────────────────
    log("\nSTEP 5: Classifying regimes")
    log("-" * 65)
    detector = RegimeDetector(
        vol_window=REGIME_VOL_WINDOW,
        vol_lookback=REGIME_VOL_LOOKBACK,
        vol_ratio_threshold=REGIME_RATIO_THRESHOLD,
        smooth_alpha=REGIME_SMOOTH_ALPHA,
    )

    # Classify the dates spanned by the backtest (GDELT window only)
    backtest_dates = sorted({
        pd.Timestamp(d).normalize()
        for d in daily["date"].unique()
    })
    regime_map = detector.classify_dates(price_history, backtest_dates)
    detector.log_summary(regime_map)

    n_trending = sum(1 for v in regime_map.values() if v == "trending")
    n_choppy   = sum(1 for v in regime_map.values() if v == "choppy")
    log(f"  {len(regime_map)} dates: {n_trending} trending, {n_choppy} choppy")

    # ── Signal generation ──────────────────────────────────────────────────────
    log("\nSTEP 6: Generating signals (3 configurations)")
    log("-" * 65)

    log("  (a) baseline …")
    sig_baseline = generate_baseline_signals(daily)
    log(f"      {count_signals(sig_baseline)}")

    log("  (b) best_config (momentum + α=0.5 smoothing) …")
    sig_best = generate_best_config_signals(daily, price_history)
    log(f"      {count_signals(sig_best)}")

    log("  (c) regime_gated …")
    sig_gated = generate_regime_gated_signals(daily, price_history, regime_map)
    log(f"      {count_signals(sig_gated)}")

    # ── Backtests ──────────────────────────────────────────────────────────────
    log("\nSTEP 7: Running backtests")
    log("-" * 65)

    mkz_optimizer = MarkowitzOptimizer(
        lookback=MARKOWITZ_LOOKBACK,
        max_weight=MARKOWITZ_MAX_WEIGHT,
    )
    regime_optimizer = RegimeAwareOptimizer(mkz_optimizer, regime_map)

    log("  (a) baseline …")
    m_baseline, _ = run_bt(sig_baseline, optimizer=None)
    log(f"      return={m_baseline['total_return']*100:+.2f}%  "
        f"sharpe={m_baseline['sharpe_ratio']:+.3f}  "
        f"dd={m_baseline['max_drawdown']*100:.2f}%  "
        f"trades={m_baseline['n_trades']}")

    log("  (b) best_config (Markowitz) …")
    m_best, _ = run_bt(sig_best, optimizer=mkz_optimizer)
    log(f"      return={m_best['total_return']*100:+.2f}%  "
        f"sharpe={m_best['sharpe_ratio']:+.3f}  "
        f"dd={m_best['max_drawdown']*100:.2f}%  "
        f"trades={m_best['n_trades']}")

    log("  (c) regime_gated …")
    m_gated, _ = run_bt(sig_gated, optimizer=regime_optimizer)
    log(f"      return={m_gated['total_return']*100:+.2f}%  "
        f"sharpe={m_gated['sharpe_ratio']:+.3f}  "
        f"dd={m_gated['max_drawdown']*100:.2f}%  "
        f"trades={m_gated['n_trades']}")

    # ── Print table ────────────────────────────────────────────────────────────
    log("\nRESULTS")
    log("=" * 65)
    labels  = ["(a) baseline", "(b) best_config", "(c) regime_gated"]
    metrics = [m_baseline, m_best, m_gated]
    scounts = [count_signals(sig_baseline), count_signals(sig_best), count_signals(sig_gated)]
    log(metrics_table(labels, metrics, scounts))

    # ── Write summary ──────────────────────────────────────────────────────────
    log("\nSTEP 8: Writing summary")
    write_summary(regime_map, labels, metrics, scounts)
    log(f"\nDone.  Summary at: {SUMMARY_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\nFATAL ERROR:")
        log(traceback.format_exc())
        sys.exit(1)
