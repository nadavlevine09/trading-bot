#!/usr/bin/env python3
"""Fetch GDELT W3 data for remaining tickers (MSFT, GOOGL, META, AMZN, AMD).

AAPL, NVDA, TSLA already fetched. Logs to logs/v3_fetch.log.
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
log = logging.getLogger("fetch_remaining")

from src.ingestion.gdelt import GDELTClient

TICKERS = {
    "MSFT":  "Microsoft Corp",
    "GOOGL": "Alphabet Inc",
    "META":  "Meta Platforms",
    "AMZN":  "Amazon.com",
    "AMD":   "Advanced Micro Devices",
}

W3_START = date(2025, 9, 19)
W3_END   = date(2025, 12, 17)
OUT_DIR  = ROOT / "data" / "gdelt_w3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    log.info("=== Starting W3 GDELT fetch for remaining 5 tickers ===")
    log.info("Date range: %s -> %s", W3_START, W3_END)
    log.info("Output dir: %s", OUT_DIR)
    log.info("Flat 15s delay between every API call")

    # 30-second startup pause so IP rate limit fully resets
    log.info("Waiting 30s before first request...")
    time.sleep(30)

    client = GDELTClient(request_delay=15.0)

    for ticker, company in TICKERS.items():
        out_path = OUT_DIR / f"{ticker}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if len(existing) > 0:
                log.info("%s: already has %d articles, skipping", ticker, len(existing))
                continue

        log.info("Fetching %s (%s)...", ticker, company)
        try:
            articles = client.fetch_for_ticker(
                ticker=ticker,
                company_name=company,
                from_date=W3_START,
                to_date=W3_END,
                max_rows=5000,
            )
        except Exception as exc:
            log.error("%s: fetch failed: %s", ticker, exc)
            articles = []

        out_path.write_text(json.dumps(articles, indent=2))
        log.info("%s: saved %d articles to %s", ticker, len(articles), out_path)

        # Extra pause between tickers
        if ticker != list(TICKERS.keys())[-1]:
            log.info("Pausing 30s before next ticker...")
            time.sleep(30)

    log.info("=== Fetch complete ===")
    log.info("Summary:")
    for ticker in list(TICKERS.keys()) + ["AAPL", "NVDA", "TSLA"]:
        p = OUT_DIR / f"{ticker}.json"
        if p.exists():
            n = len(json.loads(p.read_text()))
            log.info("  %s: %d articles", ticker, n)
        else:
            log.info("  %s: MISSING", ticker)

if __name__ == "__main__":
    main()
