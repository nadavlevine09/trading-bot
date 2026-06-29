"""Fetch congressional trading disclosures from Quiver Quant and save to
data/congress/<TICKER>.json.

Requires QUIVERQUANT_KEY in .env — free tier: https://www.quiverquant.com/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.congress import QuiverQuantClient

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
OUT_DIR = Path(__file__).parent.parent / "data" / "congress"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = QuiverQuantClient()

    for ticker in TICKERS:
        out = OUT_DIR / f"{ticker}.json"
        if out.exists():
            print(f"  {ticker}: cached — skip")
            continue
        print(f"  {ticker}: fetching …", end=" ", flush=True)
        try:
            rows = client.fetch_trades(ticker)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"{len(rows)} disclosed filings")
        except Exception as exc:
            print(f"ERROR: {exc}")
        time.sleep(1)

    print("Done.")


if __name__ == "__main__":
    main()
