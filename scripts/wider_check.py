#!/usr/bin/env python3
"""
Wider dry-run: 8 tickers × 7 days with marginal / thin-data / contamination flags.
No trades are placed.

Usage (from repo root):
    source .venv/bin/activate
    python scripts/wider_check.py
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.ingestion.newsapi import NewsAPIClient, NewsAPIError, RateLimitError, SUBSIDIARY_MAP
from src.sentiment.scorer import SentimentScorer, aggregate_daily
from src.strategy.signals import SignalGenerator

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS: dict[str, str] = {
    "AAPL":  "Apple",
    "NVDA":  "Nvidia",
    "TSLA":  "Tesla",
    "MSFT":  "Microsoft",
    "AMZN":  "Amazon",
    "GOOGL": "Google",
    "META":  "Meta",
    "AMD":   "AMD",
}

BUY_THRESHOLD  =  0.3
SELL_THRESHOLD = -0.3
MIN_HEADLINES  =  8     # hard filter: fewer than this → forced hold (matches signals.py default)
THIN_THRESHOLD =  8     # soft warning: signal fires but data is sparse
MARGINAL_BAND  =  0.05  # flag if |score − threshold| < this
CONTAM_SCORE   =  0.7   # headline |score| must reach this to be checked
DAYS_BACK      =  7

# ── Formatting ─────────────────────────────────────────────────────────────────

W = 80


def rule(c: str = "─") -> None:
    print(c * W)


def header(title: str) -> None:
    print()
    rule("━")
    print(f"  {title}")
    rule("━")


def step(n: int, total: int, label: str) -> None:
    print(f"\n[{n}/{total}] {label}")


def score_bar(score: float, width: int = 10) -> str:
    half = width // 2
    filled = min(round(abs(score) * half), half)
    if score >= 0:
        return " " * half + "+" * filled + " " * (half - filled)
    return " " * (half - filled) + "-" * filled + " " * half


def signal_label(action: str) -> str:
    return {"buy": "BUY ", "sell": "SELL", "hold": "HOLD"}.get(action, action.upper())


def _flag_str(is_marginal: bool, is_thin: bool, is_contam: bool) -> str:
    parts = []
    if is_marginal:
        parts.append("MRGN")
    if is_thin:
        parts.append("THIN")
    if is_contam:
        parts.append("CONT")
    return ("[" + "][".join(parts) + "]") if parts else ""


# ── Contamination detection ────────────────────────────────────────────────────

def _title_mentions_ticker(headline: dict, ticker: str, company: str) -> bool:
    """
    True if the headline title contains the ticker symbol, company name, or any
    known subsidiary/brand alias.  Subsidiary terms (e.g. "Xbox" for MSFT) are
    treated as legitimate company mentions so that brand-level articles are not
    falsely flagged as contamination.
    """
    title = headline.get("title") or ""
    subs  = SUBSIDIARY_MAP.get(ticker, [])
    terms = {ticker, company, company.split()[0]} | set(subs)
    return any(
        re.search(r"\b" + re.escape(t) + r"\b", title, re.IGNORECASE)
        for t in terms
    )


def _is_marginal(score: float) -> bool:
    """True when score is within MARGINAL_BAND of either threshold (buy or sell)."""
    return (
        abs(score - BUY_THRESHOLD)  < MARGINAL_BAND or
        abs(score - SELL_THRESHOLD) < MARGINAL_BAND
    )


def find_contamination_suspects(
    scored_df: pd.DataFrame,
    url_index:  dict[str, dict],
) -> list[dict]:
    """
    Return one entry per strong headline (|score| ≥ CONTAM_SCORE) where the
    ticker / company name does NOT appear as a whole word in the article title.

    The intuition: the relevance filter in fetch_for_ticker() only requires the
    term to appear anywhere (title OR description), so a headline like
    "LeCun slams Musk's xAI" that mentions 'Tesla CEO' in its description
    passes the filter but is really about xAI, not Tesla the stock.
    """
    suspects: list[dict] = []
    seen: set[tuple] = set()

    strong = scored_df[scored_df["sentiment_score"].abs() >= CONTAM_SCORE]
    for _, row in strong.iterrows():
        headline = url_index.get(row.get("url", ""))
        if not headline:
            continue

        ticker  = str(row["ticker"])
        company = TICKERS.get(ticker, "")

        if _title_mentions_ticker(headline, ticker, company):
            continue  # headline is cleanly about this ticker

        date_str = str(pd.to_datetime(row["published_at"], utc=True).date())
        key      = (ticker, date_str, headline.get("title", ""))
        if key in seen:
            continue
        seen.add(key)

        suspects.append({
            "ticker":  ticker,
            "date":    date_str,
            "title":   headline.get("title", ""),
            "score":   float(row["sentiment_score"]),
            "source":  str(row.get("source", "")),
        })

    return suspects


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    header(
        f"WIDER CHECK — News Sentiment Pipeline\n"
        f"  {now.strftime('%Y-%m-%d %H:%M')}  |  "
        f"{len(TICKERS)} tickers: {', '.join(TICKERS)}\n"
        f"  thresholds: buy > {BUY_THRESHOLD}  /  sell < {SELL_THRESHOLD}  "
        f"|  marginal band: ±{MARGINAL_BAND}  |  thin-data: n < {THIN_THRESHOLD}"
    )

    TOTAL_STEPS = 5

    # 1. Fetch -----------------------------------------------------------------
    step(1, TOTAL_STEPS, f"Fetching headlines via NewsAPI (last {DAYS_BACK} days) …")
    client = NewsAPIClient()
    all_headlines: list[dict] = []

    for ticker, company in TICKERS.items():
        try:
            articles = client.fetch_for_ticker(ticker, company, days_back=DAYS_BACK)
        except RateLimitError as exc:
            print(f"  {ticker:5s}: RATE LIMIT — {exc}")
            continue
        except NewsAPIError as exc:
            print(f"  {ticker:5s}: ERROR — {exc}")
            continue
        print(f"  {ticker:5s} ({company:9s}): {len(articles):3d} articles")
        all_headlines.extend(articles)

    if not all_headlines:
        print("\n  No headlines fetched. Check NEWSAPI_KEY in .env and network access.")
        sys.exit(1)

    print(f"\n  Total: {len(all_headlines)} unique, relevant headlines")

    # Index by URL for O(1) lookup during contamination analysis.
    url_index: dict[str, dict] = {h["url"]: h for h in all_headlines if h.get("url")}

    # 2. Score -----------------------------------------------------------------
    step(2, TOTAL_STEPS, "Scoring sentiment with FinBERT …")
    t0 = time.monotonic()
    try:
        scored_df = SentimentScorer().score(all_headlines)
    except Exception as exc:
        print(f"\n  ERROR loading or running FinBERT: {exc}")
        sys.exit(1)
    print(f"  {len(scored_df)} rows scored in {time.monotonic() - t0:.1f}s")

    # 3. Aggregate -------------------------------------------------------------
    step(3, TOTAL_STEPS, "Aggregating to daily scores per ticker …")
    daily_df = aggregate_daily(scored_df)
    print(f"  {len(daily_df)} (ticker × date) rows across {daily_df['ticker'].nunique()} tickers")

    # 4. Generate signals ------------------------------------------------------
    step(4, TOTAL_STEPS, "Generating signals …")
    gen = SignalGenerator(
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
        min_headlines=MIN_HEADLINES,
    )
    signals = gen.generate(daily_df, as_of=now)
    buy_n  = sum(1 for s in signals if s.action == "buy")
    sell_n = sum(1 for s in signals if s.action == "sell")
    hold_n = sum(1 for s in signals if s.action == "hold")
    print(f"  {len(signals)} signals: {buy_n} buy / {sell_n} sell / {hold_n} hold")

    # 5. Analyse flags ---------------------------------------------------------
    step(5, TOTAL_STEPS, "Analysing flags …")
    contam_suspects = find_contamination_suspects(scored_df, url_index)
    contam_keys = {(s["ticker"], s["date"]) for s in contam_suspects}
    print(
        f"  {len(contam_suspects)} contamination suspect(s) across "
        f"{len(contam_keys)} ticker-day(s)"
    )

    # ── Signal table ──────────────────────────────────────────────────────────

    header("SIGNAL TABLE")

    # Column header
    rule()
    print(
        f"  {'Ticker':<6} {'Date':<12} {'Score':>7}  {'Bar':^12}  "
        f"{'N':>4}  {'Signal':<6}  Flags"
    )
    rule()

    marginal_rows: list[dict] = []
    thin_rows:     list[dict] = []
    prev_ticker = None

    for sig, (_, row) in zip(signals, daily_df.iterrows()):
        score    = float(row["sentiment_score"])
        n        = int(row["n_headlines"])
        date_str = str(row["date"])
        ticker   = sig.ticker

        is_marginal = _is_marginal(score)
        is_thin     = (n < THIN_THRESHOLD)
        is_contam   = (ticker, date_str) in contam_keys
        flags       = _flag_str(is_marginal, is_thin, is_contam)

        if is_marginal:
            marginal_rows.append(
                {"ticker": ticker, "date": date_str, "score": score, "action": sig.action}
            )
        if is_thin:
            thin_rows.append(
                {"ticker": ticker, "date": date_str, "n": n, "action": sig.action}
            )

        # Separator between ticker groups for readability.
        if ticker != prev_ticker and prev_ticker is not None:
            print(f"  {'·' * (W - 4)}")
        prev_ticker = ticker

        bar = f"[{score_bar(score)}]"
        lbl = signal_label(sig.action)
        print(
            f"  {ticker:<6} {date_str:<12} {score:>+7.3f}  {bar:<14}  "
            f"{n:>4}  {lbl}    {flags}"
        )

    rule()

    # Legend (only if flags appeared)
    any_flags = marginal_rows or thin_rows or contam_suspects
    if any_flags:
        print(
            f"  Flags: "
            f"[MRGN] score within ±{MARGINAL_BAND} of threshold  "
            f"[THIN] n < {THIN_THRESHOLD} headlines  "
            f"[CONT] title doesn't mention ticker"
        )

    # ── Marginal calls detail ─────────────────────────────────────────────────

    header(f"MARGINAL CALLS  ({len(marginal_rows)} rows)")
    if not marginal_rows:
        print(f"  None — no signal is within ±{MARGINAL_BAND} of a threshold.")
    else:
        print(
            f"  These rows would flip signal with a ±{MARGINAL_BAND} move in "
            f"the daily average score.\n"
        )
        for r in marginal_rows:
            dist_buy  = abs(r["score"] - BUY_THRESHOLD)
            dist_sell = abs(r["score"] - SELL_THRESHOLD)
            closer = (
                f"buy ({dist_buy:.3f} away)"  if dist_buy < dist_sell
                else f"sell ({dist_sell:.3f} away)"
            )
            print(
                f"  {r['ticker']:<6} {r['date']}  score {r['score']:>+.3f}  "
                f"→ {r['action'].upper():<4}   closer to {closer} threshold"
            )

    # ── Thin-data detail ──────────────────────────────────────────────────────

    header(f"THIN-DATA ROWS  ({len(thin_rows)} rows with n < {THIN_THRESHOLD})")
    if not thin_rows:
        print(f"  None — all rows have ≥ {THIN_THRESHOLD} headlines.")
    else:
        print(
            f"  Signals in these rows are noisier; consider raising MIN_HEADLINES\n"
            f"  or treating the action as 'hold' for live execution.\n"
        )
        for r in thin_rows:
            flag = "  ← signal fires despite thin data" if r["action"] != "hold" else ""
            print(
                f"  {r['ticker']:<6} {r['date']}  n={r['n']:2d}  → {r['action'].upper()}{flag}"
            )

    # ── Contamination detail ──────────────────────────────────────────────────

    header(f"CONTAMINATION SUSPECTS  ({len(contam_suspects)} headlines flagged)")
    if not contam_suspects:
        print(
            f"  None — every strong-scoring headline (|score| ≥ {CONTAM_SCORE}) "
            f"mentions the ticker in its title."
        )
    else:
        print(
            f"  Criteria: |headline score| ≥ {CONTAM_SCORE} AND ticker/company\n"
            f"  NOT found as a whole word in the headline title.\n"
            f"  These headlines likely drove the daily score but describe a different\n"
            f"  entity. Review before trusting the signal.\n"
        )
        # Group by (ticker, date)
        by_td: dict[tuple, list] = {}
        for s in contam_suspects:
            by_td.setdefault((s["ticker"], s["date"]), []).append(s)

        for (ticker, date), items in sorted(by_td.items()):
            score_sum = sum(i["score"] for i in items)
            print(f"  {ticker} / {date}  ({len(items)} headline(s), total Δscore {score_sum:+.2f}):")
            for item in sorted(items, key=lambda x: abs(x["score"]), reverse=True):
                arrow  = "+" if item["score"] > 0 else "-"
                source = item["source"][:14]
                title  = item["title"][:64]
                print(f"    {arrow}{abs(item['score']):.2f}  [{source:<14}]  {title}")
            print()

    # ── Summary ───────────────────────────────────────────────────────────────

    contam_ticker_days = len(contam_keys)

    header("SUMMARY")
    print(f"  Tickers checked     : {', '.join(TICKERS)}")
    print(f"  Headlines fetched   : {len(all_headlines)}")
    print(f"  Ticker-day rows     : {len(daily_df)}")
    print()
    print(f"  Total signals       : {len(signals)}")
    print(f"  Buy / Sell / Hold   : {buy_n} / {sell_n} / {hold_n}")
    print()
    print(f"  [MRGN] Marginal     : {len(marginal_rows)} row(s)  "
          f"(score within ±{MARGINAL_BAND} of buy or sell threshold)")
    print(f"  [THIN] Thin-data    : {len(thin_rows)} row(s)  "
          f"(n_headlines < {THIN_THRESHOLD})")
    print(f"  [CONT] Contaminated : {len(contam_suspects)} headline(s)  "
          f"across {contam_ticker_days} ticker-day(s)")
    print()
    rule("━")
    print("  No trades executed — wider check complete.")
    rule("━")
    print()


if __name__ == "__main__":
    main()
