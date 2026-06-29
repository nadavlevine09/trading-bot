#!/usr/bin/env python3
"""
Fetch W3 GDELT data (2025-09-19 → 2025-12-17) via GDELT DOC 2.0 API.

Rate-limit rules (strict):
  - Flat 20 s delay between EVERY API call, no exceptions
  - On 429: wait 5 minutes (300 s) before retrying, then resume 20 s pacing
  - 60 s extra pause when switching to a new ticker
  - 30-day chunks → 3 calls per ticker, 24 total (+ retries)

Logs every call with timestamps to logs/v3_doc_api_retry.log.
Output: data/gdelt_w3_v2/{TICKER}.json  (same schema as existing pipeline)
"""
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_FILE = ROOT / "logs" / "v3_doc_api_retry.log"
OUT_DIR  = ROOT / "data" / "gdelt_w3_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

# ── Timing constants ───────────────────────────────────────────────────────────
CALL_DELAY    = 20.0   # flat seconds between every API call
RETRY_WAIT    = 300.0  # 5-minute wait on 429
TICKER_PAUSE  = 60.0   # extra seconds between tickers
MAX_RETRIES   = 6      # per chunk

# ── DOC API constants ──────────────────────────────────────────────────────────
DOC_API_URL  = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS  = 250
CHUNK_DAYS   = 30
TIMEOUT      = 30

# ── Tickers and company phrases ────────────────────────────────────────────────
W3_START = date(2025, 9, 19)
W3_END   = date(2025, 12, 17)

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
COMPANY_NAMES = {
    "AAPL":  "Apple Inc",
    "NVDA":  "NVIDIA Corp",
    "TSLA":  "Tesla Inc",
    "MSFT":  "Microsoft Corp",
    "AMZN":  "Amazon.com",
    "GOOGL": "Alphabet Inc",
    "META":  "Meta Platforms",
    "AMD":   "Advanced Micro Devices",
}


# ── Logging ────────────────────────────────────────────────────────────────────
_log_file = open(LOG_FILE, "a", buffering=1)

def log(msg: str = "") -> None:
    line = msg
    print(line, flush=True)
    _log_file.write(line + "\n")

def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ── DOC API helpers ────────────────────────────────────────────────────────────

def date_chunks(start: date, end: date, size: int) -> list[tuple[date, date]]:
    chunks, cur = [], start
    while cur <= end:
        chunks.append((cur, min(cur + timedelta(days=size - 1), end)))
        cur = chunks[-1][1] + timedelta(days=1)
    return chunks


def seendate_to_iso(s: str) -> str:
    m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?$", s.strip())
    if m:
        y, mo, d, h, mi, sec = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{sec}Z"
    m2 = re.match(r"^(\d{4})(\d{2})(\d{2})$", s.strip())
    if m2:
        y, mo, d = m2.groups()
        return f"{y}-{mo}-{d}T00:00:00Z"
    return ""


def title_from_url(url: str) -> str:
    if not url:
        return ""
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"\.[a-z]{2,5}$", "", slug)
    slug = re.sub(r"[-_]+", " ", slug).strip()
    return slug.title() if slug else url


def normalise(item: dict, ticker: str, company: str) -> dict:
    url      = (item.get("url") or "").strip()
    title    = (item.get("title") or "").strip() or title_from_url(url)
    domain   = (item.get("domain") or "").strip()
    seendate = (item.get("seendate") or "").strip()
    return {
        "ticker":          ticker,
        "company":         company,
        "title":           title,
        "description":     "",
        "content_snippet": "",
        "source":          domain or "unknown",
        "published_at":    seendate_to_iso(seendate),
        "url":             url,
    }


def fetch_chunk(
    session: requests.Session,
    ticker: str,
    company: str,
    from_date: date,
    to_date: date,
) -> list[dict]:
    """
    Make one DOC API call. On 429 waits RETRY_WAIT then retries.
    The 20 s inter-call delay is applied by the CALLER, not here.
    """
    params = {
        "query":         f'"{company}"',
        "mode":          "ArtList",
        "format":        "json",
        "maxrecords":    MAX_RECORDS,
        "sort":          "DateAsc",
        "startdatetime": from_date.strftime("%Y%m%d") + "000000",
        "enddatetime":   to_date.strftime("%Y%m%d") + "235959",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        log(f"  [{ts()}] CALL #{attempt}  {ticker} [{from_date} → {to_date}]")
        t0 = time.time()
        try:
            resp = session.get(DOC_API_URL, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            elapsed = time.time() - t0
            log(f"  [{ts()}] NETWORK ERROR in {elapsed:.1f}s: {exc}")
            if attempt < MAX_RETRIES:
                log(f"  [{ts()}] Waiting {CALL_DELAY}s then retrying…")
                time.sleep(CALL_DELAY)
            continue

        elapsed = time.time() - t0
        log(f"  [{ts()}] RESPONSE  HTTP {resp.status_code} in {elapsed:.1f}s")

        if resp.status_code == 429:
            log(f"  [{ts()}] RATE-LIMITED (429). Waiting {RETRY_WAIT/60:.0f} min…")
            time.sleep(RETRY_WAIT)
            log(f"  [{ts()}] Resuming after rate-limit wait — attempt {attempt+1}/{MAX_RETRIES}")
            continue

        if resp.status_code != 200:
            log(f"  [{ts()}] BAD STATUS {resp.status_code}: {resp.text[:200]}")
            if attempt < MAX_RETRIES:
                log(f"  [{ts()}] Waiting {CALL_DELAY}s then retrying…")
                time.sleep(CALL_DELAY)
            continue

        try:
            data = resp.json()
        except ValueError as exc:
            log(f"  [{ts()}] JSON PARSE ERROR: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(CALL_DELAY)
            continue

        raw = data.get("articles") or []
        if not isinstance(raw, list):
            log(f"  [{ts()}] Unexpected articles type: {type(raw)}")
            return []

        articles = [normalise(a, ticker, company) for a in raw if isinstance(a, dict)]
        log(f"  [{ts()}] OK — {len(articles)} articles")
        return articles

    log(f"  [{ts()}] CHUNK FAILED after {MAX_RETRIES} attempts — skipping")
    return []


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log("=" * 70)
    log(f"[{ts()}] W3 GDELT DOC 2.0 API fetch  ({W3_START} → {W3_END})")
    log(f"  CALL_DELAY={CALL_DELAY}s  RETRY_WAIT={RETRY_WAIT}s  TICKER_PAUSE={TICKER_PAUSE}s")
    log(f"  CHUNK_DAYS={CHUNK_DAYS}  MAX_RETRIES={MAX_RETRIES}  MAX_RECORDS={MAX_RECORDS}")
    log(f"  Output dir: {OUT_DIR}")
    log("=" * 70)

    session = requests.Session()
    session.headers["User-Agent"] = "trading-bot-research/1.0 (academic sentiment backtest)"

    t_global = time.time()

    for ticker_idx, ticker in enumerate(TICKERS):
        company = COMPANY_NAMES[ticker]

        # 60 s inter-ticker pause (except before the first ticker)
        if ticker_idx > 0:
            log(f"\n[{ts()}] ── Inter-ticker pause {TICKER_PAUSE}s before {ticker} ──")
            time.sleep(TICKER_PAUSE)

        log(f"\n[{ts()}] ════ {ticker} ({company}) ════")

        out_path = OUT_DIR / f"{ticker}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing:
                log(f"  [{ts()}] Already fetched ({len(existing)} articles) — skipping")
                continue

        chunks    = date_chunks(W3_START, W3_END, CHUNK_DAYS)
        seen_urls: set[str] = set()
        articles:  list[dict] = []

        for chunk_idx, (cs, ce) in enumerate(chunks):
            # 20 s delay before every call (except the very first call of the whole run)
            if ticker_idx > 0 or chunk_idx > 0:
                log(f"  [{ts()}] Waiting {CALL_DELAY}s before next call…")
                time.sleep(CALL_DELAY)

            batch = fetch_chunk(session, ticker, company, cs, ce)

            fresh = []
            for a in batch:
                url = a.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                fresh.append(a)

            articles.extend(fresh)
            log(f"  [{ts()}] Chunk {chunk_idx+1}/{len(chunks)}: "
                f"{len(batch)} raw → {len(fresh)} new  (total={len(articles)})")

        articles.sort(key=lambda a: a.get("published_at", ""))
        out_path.write_text(json.dumps(articles, indent=2, ensure_ascii=False))

        dates = sorted({a["published_at"][:10] for a in articles if a.get("published_at")})
        log(f"  [{ts()}] SAVED {len(articles)} articles, {len(dates)} active days"
            + (f"  ({dates[0]} → {dates[-1]})" if dates else "  (no dates)"))

    elapsed = (time.time() - t_global) / 60
    log(f"\n[{ts()}] ═══ Fetch complete — {elapsed:.1f} min elapsed ═══")

    log("\nFinal counts:")
    total_ok = 0
    for ticker in TICKERS:
        p = OUT_DIR / f"{ticker}.json"
        if p.exists():
            arts = json.loads(p.read_text())
            dates = sorted({a["published_at"][:10] for a in arts if a.get("published_at")})
            log(f"  {ticker:5s}: {len(arts):>4} articles  {len(dates):>3} days"
                + (f"  {dates[0]} → {dates[-1]}" if dates else ""))
            total_ok += 1
        else:
            log(f"  {ticker:5s}: MISSING")

    log(f"\n{total_ok}/{len(TICKERS)} tickers fetched successfully.")
    _log_file.close()


if __name__ == "__main__":
    main()
