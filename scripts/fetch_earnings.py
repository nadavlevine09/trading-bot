"""Fetch quarterly earnings from Alpha Vantage and save to data/earnings/<TICKER>.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.alpha_vantage import AlphaVantageClient

TICKERS   = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
OUT_DIR   = Path(__file__).parent.parent / "data" / "earnings"

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = AlphaVantageClient()

    for ticker in TICKERS:
        out = OUT_DIR / f"{ticker}.json"
        if out.exists():
            print(f"  {ticker}: cached — skip")
            continue
        print(f"  {ticker}: fetching …", end=" ", flush=True)
        try:
            rows = client.fetch_earnings(ticker)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"{len(rows)} quarters")
        except Exception as exc:
            print(f"ERROR: {exc}")
        import time; time.sleep(13)  # stay under 5 calls/min free-tier cap

    print("Done.")

if __name__ == "__main__":
    main()
