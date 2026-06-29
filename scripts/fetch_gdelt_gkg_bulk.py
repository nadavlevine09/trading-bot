#!/usr/bin/env python3
"""
Fetch GDELT W3 data (2025-09-19 to 2025-12-17) via GKG v2.1 bulk files.

Downloads all 15-minute GKG zips for the date range (~8,500 files / ~22 GB),
filters locally for 8 companies, saves to data/gdelt_w3/{TICKER}.json,
and deletes each zip after processing (raw data never persists on disk).

Logs to logs/v3_fetch.log and stdout.
"""
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_FILE = ROOT / "logs" / "v3_fetch.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gkg_bulk_fetch")

from src.ingestion.gdelt import GKGBulkClient

# Date range for W3 window
W3_START = date(2025, 9, 19)
W3_END   = date(2025, 12, 17)
OUT_DIR  = ROOT / "data" / "gdelt_w3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keywords for each ticker — prefix-matched against GKG orgs field (lowercase)
# Kept conservative to avoid false positives
TICKER_KEYWORDS: dict[str, list[str]] = {
    "AAPL":  ["apple"],
    "NVDA":  ["nvidia"],
    "TSLA":  ["tesla"],
    "MSFT":  ["microsoft"],
    "AMZN":  ["amazon"],
    "GOOGL": ["google", "alphabet"],
    "META":  ["meta platforms", "facebook"],   # bare "meta" too noisy in GKG orgs
    "AMD":   ["advanced micro devices", "amd"],
}

# Company names for the article schema
COMPANY_NAMES: dict[str, str] = {
    "AAPL":  "Apple Inc",
    "NVDA":  "NVIDIA Corp",
    "TSLA":  "Tesla Inc",
    "MSFT":  "Microsoft Corp",
    "AMZN":  "Amazon.com",
    "GOOGL": "Alphabet Inc",
    "META":  "Meta Platforms Inc",
    "AMD":   "Advanced Micro Devices",
}


def patch_company_name(articles: list[dict], ticker: str) -> list[dict]:
    """Replace the placeholder company field with the proper legal name."""
    name = COMPANY_NAMES.get(ticker, ticker)
    for a in articles:
        a["company"] = name
    return articles


def main() -> None:
    log.info("=" * 70)
    log.info("GKG bulk fetch — W3 window: %s to %s", W3_START, W3_END)
    log.info("Output dir: %s", OUT_DIR)
    log.info("=" * 70)

    t0 = time.time()
    client = GKGBulkClient(workers=4, timeout=60)

    results = client.fetch_all_tickers(
        ticker_keywords=TICKER_KEYWORDS,
        from_date=W3_START,
        to_date=W3_END,
        max_per_ticker=200_000,
    )

    log.info("Download+parse complete in %.1f minutes", (time.time() - t0) / 60)

    log.info("=== Final article counts ===")
    for ticker, articles in results.items():
        articles = patch_company_name(articles, ticker)
        out_path = OUT_DIR / f"{ticker}.json"
        out_path.write_text(json.dumps(articles, indent=2))
        # Date coverage
        dates = sorted({a["published_at"][:10] for a in articles if a.get("published_at")})
        log.info(
            "  %s: %d articles, %d active days (%s → %s)",
            ticker, len(articles), len(dates),
            dates[0] if dates else "N/A",
            dates[-1] if dates else "N/A",
        )

    log.info("All JSON files written to %s", OUT_DIR)
    log.info("Total elapsed: %.1f minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
