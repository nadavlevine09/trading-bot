"""Build FinBERT parquet cache — score at most MAX_PER_TICKER_DATE articles
per (ticker, date) cell to keep runtime under ~15 minutes per window.

Full scoring of 35 k articles at 10 art/s would take ~56 min. Capping at
50 articles per ticker-date cell reduces the corpus to ~3-4 k articles (≈5 min)
while keeping all date coverage intact for the backtest.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.sentiment.scorer import SentimentScorer, aggregate_daily

TICKERS            = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
MAX_PER_TICKER_DATE = 15   # cap per (ticker, date) → keeps total ≈ 10 k articles/window (~20 min)
CACHE_DIR          = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    {"gdelt_dir": ROOT / "data" / "gdelt_prev", "cache": CACHE_DIR / "prev_scored.parquet"},
    {"gdelt_dir": ROOT / "data" / "gdelt",      "cache": CACHE_DIR / "current_scored.parquet"},
]


def load_gdelt(gdelt_dir: Path) -> list[dict]:
    articles: list[dict] = []
    for ticker in TICKERS:
        p = gdelt_dir / f"{ticker}.json"
        if p.exists():
            articles.extend(json.loads(p.read_text()))
    return articles


def subsample(articles: list[dict], cap: int) -> list[dict]:
    """Keep at most *cap* articles per (ticker, date) cell."""
    from collections import defaultdict
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for a in articles:
        date_str = (a.get("published_at") or "")[:10]
        key = (a.get("ticker", ""), date_str)
        if len(buckets[key]) < cap:
            buckets[key].append(a)
    return [a for bucket in buckets.values() for a in bucket]


scorer = SentimentScorer(device=-1)

for win in WINDOWS:
    if win["cache"].exists():
        print(f"  {win['cache'].name}: cached ✓")
        daily = aggregate_daily(pd.read_parquet(win["cache"]))
        print(f"  → {len(daily)} (ticker,date) rows")
        continue

    all_articles = load_gdelt(win["gdelt_dir"])
    sampled      = subsample(all_articles, MAX_PER_TICKER_DATE)
    print(
        f"  {win['gdelt_dir'].name}: {len(all_articles):,} total → "
        f"{len(sampled):,} sampled (cap={MAX_PER_TICKER_DATE}/ticker/day). Scoring …",
        flush=True,
    )
    scored = scorer.score(sampled)
    scored.to_parquet(win["cache"], index=False)
    daily  = aggregate_daily(scored)
    print(f"  → {len(scored):,} scored, {len(daily)} (ticker,date) rows — saved {win['cache'].name}")

print("Cache ready.")
