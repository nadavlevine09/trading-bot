"""Pull 3 months of GDELT GKG history for tracked tickers.

Fetches in time-windowed chunks so ORDER BY DATE DESC + LIMIT does not bias
results toward the most recent dates within each chunk.  Chunk size is
configurable per ticker — high-volume tickers (AMZN ~500 articles/day) use
weekly windows to stay under MAX_ROWS_CHUNK; others use 30-day windows.

Usage
-----
    # Full run — all 8 tickers
    python scripts/fetch_gdelt_history.py

    # Targeted re-fetch (overwrites only those files)
    python scripts/fetch_gdelt_history.py AMZN AMD

Output
------
data/gdelt/<TICKER>.json  — JSON array of normalised article dicts, sorted by
published_at ascending.  Overwrites any previous run for the given ticker(s).

Summary (printed per ticker)
----------------------------
  total        total articles saved (after cross-chunk URL deduplication)
  date range   earliest and latest published_at in the saved file
  active days  number of distinct calendar days with ≥ 1 article
               (the key signal: 90/90 means perfect coverage across the window)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.companies import COMPANIES
from src.ingestion.gdelt import GDELTClient, GDELTError

# ── Config ─────────────────────────────────────────────────────────────────────

ALL_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
OUT_DIR     = Path(__file__).parent.parent / "data" / "gdelt"

# Per-chunk row cap.  Kept at 5 000 — well above the per-chunk expected volume
# for monthly windows once chunk sizes are correctly tuned per ticker.
MAX_ROWS_CHUNK = 5_000

# Chunk size (days) per ticker.  Tickers whose daily article volume would
# saturate MAX_ROWS_CHUNK in a 30-day window get smaller windows instead.
#   AMZN: ~500 articles/day → 30-day window ≈ 15 000 rows (3× the cap)
#          7-day window  ≈  3 500 rows (safely under cap)
CHUNK_DAYS_DEFAULT = 30
CHUNK_DAYS: dict[str, int] = {
    "AMZN": 7,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _date_chunks(start: date, end: date, size: int) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive windows of at most *size* days."""
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=size - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _day_of(iso_ts: str) -> str:
    """Return the YYYY-MM-DD portion of an ISO 8601 timestamp."""
    return iso_ts[:10] if iso_ts else ""


def _summarise(articles: list[dict]) -> dict:
    """Compute summary stats over a list of normalised article dicts."""
    if not articles:
        return {"total": 0, "date_range": ("—", "—"), "active_days": 0}
    timestamps = sorted(a["published_at"] for a in articles if a.get("published_at"))
    days = {_day_of(ts) for ts in timestamps if ts}
    return {
        "total":       len(articles),
        "date_range":  (timestamps[0], timestamps[-1]),
        "active_days": len(days),
    }


def fetch_ticker(
    client: GDELTClient,
    ticker: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Fetch and deduplicate all chunks for one ticker. Returns sorted articles."""
    info        = COMPANIES.get(ticker)
    entity_name = info.legal_entity_name if info else ticker
    chunk_days  = CHUNK_DAYS.get(ticker, CHUNK_DAYS_DEFAULT)
    chunks      = _date_chunks(start_date, end_date, chunk_days)

    print(f"{'─' * 60}")
    print(f"{ticker}  ({entity_name})  —  {len(chunks)} × {chunk_days}-day chunks")

    seen_urls: set[str] = set()
    all_articles: list[dict] = []
    capped = False

    for chunk_start, chunk_end in chunks:
        label = f"  [{chunk_start} → {chunk_end}]"
        try:
            batch = client.fetch_for_ticker(
                ticker=ticker,
                company_name=entity_name,
                from_date=chunk_start,
                to_date=chunk_end,
                max_rows=MAX_ROWS_CHUNK,
            )
        except GDELTError as exc:
            print(f"{label}  ERROR: {exc}")
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

        all_articles.extend(fresh)
        print(f"{label}  {len(batch):>4} raw  →  {len(fresh):>4} new unique")

    all_articles.sort(key=lambda a: a.get("published_at", ""))

    stats = _summarise(all_articles)
    cap_warn = "  ⚠ chunk cap hit — reduce chunk size further" if capped else ""

    print()
    print(f"  total        : {stats['total']:,}")
    dr = stats["date_range"]
    print(f"  date range   : {dr[0][:10]}  →  {dr[1][:10]}")
    print(f"  active days  : {stats['active_days']}")
    if cap_warn:
        print(cap_warn)

    return all_articles


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch GDELT GKG history for one or more tickers."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        default=ALL_TICKERS,
        metavar="TICKER",
        help=f"Ticker(s) to fetch (default: all {len(ALL_TICKERS)}). "
             f"Example: AMZN AMD",
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="Window start date (default: today - 90 days)",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Window end date inclusive (default: yesterday)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help=f"Output directory (default: {OUT_DIR})",
    )
    args    = parser.parse_args()
    tickers = [t.upper() for t in args.tickers]

    unknown = [t for t in tickers if t not in ALL_TICKERS]
    if unknown:
        print(f"ERROR: unknown ticker(s): {unknown}", file=sys.stderr)
        sys.exit(1)

    today      = date.today()
    end_date   = date.fromisoformat(args.end)   if args.end   else today - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else today - timedelta(days=90)
    out_dir    = Path(args.out_dir) if args.out_dir else OUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    print("GDELT history fetch")
    print(f"  Tickers : {' '.join(tickers)}")
    print(f"  Window  : {start_date} → {end_date}  ({(end_date - start_date).days + 1} days)")
    print(f"  Max rows: {MAX_ROWS_CHUNK:,} per chunk")
    print(f"  Output  : {out_dir}/")
    print()

    try:
        client = GDELTClient()
    except Exception as exc:
        print(f"ERROR: could not initialise GDELTClient — {exc}")
        sys.exit(1)

    results: dict[str, dict] = {}

    for ticker in tickers:
        all_articles = fetch_ticker(client, ticker, start_date, end_date)

        out_path = out_dir / f"{ticker}.json"
        out_path.write_text(
            json.dumps(all_articles, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  saved        : {out_path}")
        print()

        results[ticker] = _summarise(all_articles)

    # ── Summary table ───────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"{'TICKER':<6}  {'TOTAL':>6}  {'ACTIVE DAYS':>11}  {'EARLIEST':>10}  {'LATEST':>10}")
    print(f"{'──────':<6}  {'──────':>6}  {'───────────':>11}  {'──────────':>10}  {'──────────':>10}")
    for ticker in tickers:
        s = results.get(ticker)
        if s is None:
            print(f"{ticker:<6}  {'ERROR':>6}")
            continue
        dr       = s["date_range"]
        earliest = dr[0][:10] if dr[0] != "—" else "—"
        latest   = dr[1][:10] if dr[1] != "—" else "—"
        print(
            f"{ticker:<6}  {s['total']:>6,}  {s['active_days']:>11,}"
            f"  {earliest:>10}  {latest:>10}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
