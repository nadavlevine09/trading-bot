#!/usr/bin/env python3
"""
Dry-run integration script — fetch, score, aggregate, signal. No trades placed.

Usage (from repo root):
    source .venv/bin/activate
    python scripts/dry_run.py

Purpose:
    Sanity-check the full pipeline end-to-end before connecting to Alpaca.
    Nothing is written to disk and no orders are submitted.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make `src/` importable when running from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.newsapi import NewsAPIClient, NewsAPIError, RateLimitError
from src.sentiment.scorer import SentimentScorer, aggregate_daily
from src.strategy.signals import SignalGenerator

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS: dict[str, str] = {
    "AAPL": "Apple",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
}

BUY_THRESHOLD  =  0.3
SELL_THRESHOLD = -0.3
MIN_HEADLINES  =  8
DAYS_BACK      =  7

TOP_HEADLINES_PER_TICKER = 3   # shown in the sample section at the end

# ── Formatting helpers ─────────────────────────────────────────────────────────

W = 64  # output width


def rule(char: str = "─") -> None:
    print(char * W)


def header(title: str) -> None:
    print()
    rule("━")
    print(f"  {title}")
    rule("━")


def step(n: int, total: int, label: str) -> None:
    print(f"\n[{n}/{total}] {label}")


def score_bar(score: float, width: int = 12) -> str:
    """Render a compact ASCII bar: negative left of centre, positive right."""
    half = width // 2
    filled = round(abs(score) * half)
    filled = min(filled, half)
    if score >= 0:
        return " " * half + "+" * filled + " " * (half - filled)
    else:
        return " " * (half - filled) + "-" * filled + " " * half


def signal_label(action: str) -> str:
    return {"buy": "BUY ", "sell": "SELL", "hold": "HOLD"}.get(action, action.upper())


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def fetch_headlines(client: NewsAPIClient) -> list[dict]:
    all_headlines: list[dict] = []
    for ticker, company in TICKERS.items():
        try:
            articles = client.fetch_for_ticker(ticker, company, days_back=DAYS_BACK)
            status = f"{len(articles):3d} articles"
        except RateLimitError as exc:
            articles = []
            status = f"RATE LIMIT — {exc}"
        except NewsAPIError as exc:
            articles = []
            status = f"ERROR — {exc}"
        print(f"  {ticker:5s} ({company:6s}): {status}")
        all_headlines.extend(articles)
    return all_headlines


def score_headlines(headlines: list[dict]):
    print("  Loading FinBERT (ProsusAI/finbert) …")
    print("  First run downloads ~420 MB — subsequent runs use the local cache.")
    t0 = time.monotonic()
    scorer = SentimentScorer()
    scored = scorer.score(headlines)
    elapsed = time.monotonic() - t0
    print(f"  {len(scored)} headlines scored in {elapsed:.1f}s")
    return scored


def print_signal_table(signals, daily_df) -> None:
    rule()
    col = f"  {'Ticker':<7} {'Date':<12} {'Score':>7}  {'Bar':^14}  {'N':>4}  Signal  Reason"
    print(col)
    rule()

    # generate() iterates daily_df in order and emits one signal per row,
    # so zip is the right alignment — no lookup needed.
    for sig, (_, row) in zip(signals, daily_df.iterrows()):
        score    = float(row["sentiment_score"])
        n        = int(row["n_headlines"])
        date_str = str(row["date"])

        bar  = f"[{score_bar(score)}]"
        lbl  = signal_label(sig.action)

        if n < MIN_HEADLINES:
            reason = f"thin data ({n} headlines < min {MIN_HEADLINES})"
        elif sig.action == "buy":
            reason = f"score {score:+.3f} > {BUY_THRESHOLD}"
        elif sig.action == "sell":
            reason = f"score {score:+.3f} < {SELL_THRESHOLD}"
        else:
            reason = f"score {score:+.3f} in [{SELL_THRESHOLD}, {BUY_THRESHOLD}]"

        print(
            f"  {sig.ticker:<7} {date_str:<12} {score:>+7.3f}  {bar:<14}  "
            f"{n:>4}  {lbl}    {reason}"
        )

    rule()


def print_sample_headlines(scored_df, n: int = TOP_HEADLINES_PER_TICKER) -> None:
    print(f"\n  Top {n} headlines per ticker (sorted by |sentiment_score|):\n")
    for ticker in TICKERS:
        subset = (
            scored_df[scored_df["ticker"] == ticker]
            .copy()
            .assign(_abs=lambda df: df["sentiment_score"].abs())
            .sort_values("_abs", ascending=False)
            .head(n)
        )
        if subset.empty:
            print(f"  {ticker}: (no headlines)")
            continue
        print(f"  {ticker}:")
        for _, row in subset.iterrows():
            s     = row["sentiment_score"]
            src   = (row["source"] or "?")[:14]
            text  = (row["text"] or "")[:72]
            arrow = "+" if s > 0 else ("-" if s < 0 else " ")
            print(f"    {arrow}{abs(s):.2f}  [{src:<14}]  {text}")
        print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    header(
        f"DRY RUN — News Sentiment Pipeline\n"
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"tickers: {', '.join(TICKERS)}\n"
        f"  thresholds: buy > {BUY_THRESHOLD}  /  sell < {SELL_THRESHOLD}"
        f"  /  min {MIN_HEADLINES} headlines"
    )

    # 1. Fetch -----------------------------------------------------------------
    step(1, 4, f"Fetching headlines via NewsAPI (last {DAYS_BACK} days) …")
    client = NewsAPIClient()
    headlines = fetch_headlines(client)

    if not headlines:
        print("\n  No headlines fetched. Check NEWSAPI_KEY in .env and network access.")
        sys.exit(1)

    print(f"\n  Total: {len(headlines)} unique, relevant headlines")

    # 2. Score -----------------------------------------------------------------
    step(2, 4, "Scoring sentiment with FinBERT …")
    try:
        scored_df = score_headlines(headlines)
    except Exception as exc:
        print(f"\n  ERROR loading or running FinBERT: {exc}")
        print("  Check that torch and transformers are installed correctly.")
        sys.exit(1)

    if scored_df.empty:
        print("\n  Scoring returned no rows. Exiting.")
        sys.exit(1)

    # 3. Aggregate -------------------------------------------------------------
    step(3, 4, "Aggregating to daily scores per ticker …")
    daily_df = aggregate_daily(scored_df)
    print(f"  {len(daily_df)} (ticker × date) rows")

    # 4. Generate signals ------------------------------------------------------
    step(4, 4, "Generating signals …")
    gen = SignalGenerator(
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
        min_headlines=MIN_HEADLINES,
    )
    signals = gen.generate(daily_df, as_of=datetime.now(timezone.utc).replace(tzinfo=None))
    buy_n  = sum(1 for s in signals if s.action == "buy")
    sell_n = sum(1 for s in signals if s.action == "sell")
    hold_n = sum(1 for s in signals if s.action == "hold")
    print(f"  {len(signals)} signals: {buy_n} buy / {sell_n} sell / {hold_n} hold")

    # Print results ------------------------------------------------------------
    header("SIGNAL SUMMARY")
    print_signal_table(signals, daily_df)
    header("SAMPLE HEADLINES")
    print_sample_headlines(scored_df)

    rule("━")
    print("  No trades executed — dry run complete.")
    rule("━")
    print()


if __name__ == "__main__":
    main()
