"""GDELT ingestion — GKG v2.1 bulk file downloader.

Downloads the 15-minute GKG zip files directly from data.gdeltproject.org,
filters locally for the target companies using the organizations NER field,
and outputs the same normalized JSON schema used by the rest of the pipeline.

Each zip is downloaded, parsed, filtered, and deleted immediately — only
the filtered article records are kept in memory and written to the output
JSON files at the end.

GKG v2.1 tab-delimited column layout (0-based):
  0  GKGRECORDID
  1  DATE  (YYYYMMDDHHMMSS)
  2  V2SOURCECOLLECTIONIDENTIFIER
  3  V2SOURCECOMMONNAME  (outlet name)
  4  V2DOCUMENTIDENTIFIER  (article URL)
  5  V2.1COUNTS
  6  V2.1ENHANCEDCOUNTS
  7  V2.1THEMES
  8  V2.1ENHANCEDTHEMES
  9  V2.1LOCATIONS
 10  V2.1ENHANCEDLOCATIONS
 11  V2.1PERSONS
 12  V2.1ENHANCEDPERSONS
 13  V2.1ORGANIZATIONS  (semicolon-delimited lowercase org names)
 14  V2.1ENHANCEDORGANIZATIONS
 15  V2.1TONE
 ... (12 more fields)
"""
from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

_log = logging.getLogger(__name__)

_BASE_URL    = "http://data.gdeltproject.org/gdeltv2"
_INTERVAL    = timedelta(minutes=15)
_WORKERS     = 4          # parallel download threads
_TIMEOUT     = 60         # HTTP timeout per file
_RETRY_WAIT  = 5.0        # seconds before retry on transient error
_MAX_RETRIES = 3

# GKG column indices
_C_DATE   = 1
_C_SOURCE = 3
_C_URL    = 4
_C_ORGS   = 13


class GDELTError(Exception):
    """Any GDELT ingestion failure."""


class GDELTAuthError(GDELTError):
    """Kept for interface compatibility."""


class GKGBulkClient:
    """
    Fetch company news from GDELT GKG v2.1 bulk 15-minute files.

    Downloads all GKG zip files covering the requested date range,
    filters rows where the organizations NER field mentions one of the
    target companies, and returns normalized article dicts.
    """

    def __init__(self, workers: int = _WORKERS, timeout: int = _TIMEOUT) -> None:
        self._workers = workers
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "trading-bot-research/1.0 (academic sentiment backtest)"
        )
        _log.info(
            "GKGBulkClient initialised — %d workers, %ds timeout", workers, timeout
        )

    def fetch_all_tickers(
        self,
        ticker_keywords: dict[str, list[str]],
        from_date: date,
        to_date: date,
        max_per_ticker: int = 10_000,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Download all GKG files covering [from_date, to_date] and filter for
        each ticker's keywords.

        Args:
            ticker_keywords: {ticker: [keyword, ...]} — keywords are matched
                against each semicolon-delimited org name (case-insensitive
                prefix match).
            from_date / to_date: Inclusive date range.
            max_per_ticker: Cap on articles per ticker (deduped by URL).

        Returns:
            {ticker: [article_dict, ...]} sorted by published_at ascending.
        """
        urls = _gen_file_urls(from_date, to_date)
        total = len(urls)
        _log.info(
            "GKG bulk fetch: %d files, %s -> %s, %d tickers",
            total, from_date, to_date, len(ticker_keywords),
        )

        results: dict[str, dict[str, dict]] = {t: {} for t in ticker_keywords}
        done = 0
        skipped = 0

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {
                pool.submit(self._process_file, url, ticker_keywords): url
                for url in urls
            }
            for fut in as_completed(futures):
                url = futures[fut]
                done += 1
                try:
                    hits = fut.result()
                except Exception as exc:
                    skipped += 1
                    _log.debug("skip %s: %s", url.split("/")[-1], exc)
                    continue
                for ticker, articles in hits.items():
                    seen = results[ticker]
                    for art in articles:
                        u = art.get("url", "")
                        if u and u not in seen:
                            if len(seen) < max_per_ticker:
                                seen[u] = art
                if done % 200 == 0 or done == total:
                    counts = {t: len(v) for t, v in results.items()}
                    _log.info(
                        "Progress: %d/%d files processed (%d skipped). "
                        "Article counts: %s",
                        done, total, skipped, counts,
                    )

        out: dict[str, list[dict[str, Any]]] = {}
        for ticker, seen in results.items():
            arts = sorted(seen.values(), key=lambda a: a.get("published_at", ""))
            out[ticker] = arts
            _log.info("%s: %d articles after dedup", ticker, len(arts))
        return out

    def _process_file(
        self,
        url: str,
        ticker_keywords: dict[str, list[str]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Download one GKG zip, parse, filter, return matches. Raises on failure."""
        content = self._download(url)
        try:
            z = zipfile.ZipFile(io.BytesIO(content))
            fname = z.namelist()[0]
            with z.open(fname) as fh:
                lines = fh.readlines()
        except Exception as exc:
            raise GDELTError(f"Bad zip {url}: {exc}") from exc

        hits: dict[str, list[dict]] = {t: [] for t in ticker_keywords}
        for raw in lines:
            try:
                row = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            except Exception:
                continue
            if len(row) <= _C_ORGS:
                continue
            orgs_field = row[_C_ORGS]
            if not orgs_field:
                continue
            matched = _match_orgs(orgs_field, ticker_keywords)
            if not matched:
                continue
            art_url    = row[_C_URL].strip() if len(row) > _C_URL else ""
            art_source = row[_C_SOURCE].strip() if len(row) > _C_SOURCE else ""
            art_date   = row[_C_DATE].strip() if len(row) > _C_DATE else ""
            pub_at     = _gkg_date_to_iso(art_date)
            title      = _title_from_url(art_url)
            for ticker in matched:
                hits[ticker].append({
                    "ticker":          ticker,
                    "company":         ticker_keywords[ticker][0] if ticker_keywords[ticker] else ticker,
                    "title":           title,
                    "description":     "",
                    "content_snippet": "",
                    "source":          art_source or "unknown",
                    "published_at":    pub_at,
                    "url":             art_url,
                })
        return hits

    def _download(self, url: str) -> bytes:
        """Download url with retries. Raises GDELTError on persistent failure."""
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(url, timeout=self._timeout)
                if resp.status_code == 404:
                    raise GDELTError(f"404 {url}")
                if resp.status_code != 200:
                    raise GDELTError(f"HTTP {resp.status_code} {url}")
                return resp.content
            except GDELTError:
                raise
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_WAIT)
                    continue
                raise GDELTError(f"Download failed after {_MAX_RETRIES} tries: {url}") from exc
        raise GDELTError(f"unreachable {url}")


# ── Module-level helpers ───────────────────────────────────────────────────────

def _gen_file_urls(from_date: date, to_date: date) -> list[str]:
    """Generate all expected 15-min GKG file URLs for the inclusive date range."""
    urls: list[str] = []
    dt = datetime(from_date.year, from_date.month, from_date.day, 0, 0, 0)
    end = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
    while dt <= end:
        fname = dt.strftime("%Y%m%d%H%M%S") + ".gkg.csv.zip"
        urls.append(f"{_BASE_URL}/{fname}")
        dt += _INTERVAL
    return urls


def _match_orgs(
    orgs_field: str,
    ticker_keywords: dict[str, list[str]],
) -> list[str]:
    """Return list of tickers whose keywords prefix-match any org in the field."""
    orgs = [o.strip().lower() for o in orgs_field.split(";") if o.strip()]
    matched: list[str] = []
    for ticker, keywords in ticker_keywords.items():
        for kw in keywords:
            kw_lo = kw.lower()
            for org in orgs:
                if org == kw_lo or org.startswith(kw_lo + " ") or org.startswith(kw_lo + ","):
                    matched.append(ticker)
                    break
            else:
                continue
            break
    return matched


def _gkg_date_to_iso(dt_str: str) -> str:
    """Convert GKG DATE field 'YYYYMMDDHHMMSS' to ISO 8601."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$", dt_str.strip())
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}Z"
    return ""


def _title_from_url(url: str) -> str:
    """Derive a human-readable title from a URL slug."""
    if not url:
        return ""
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"\.[a-z]{2,5}$", "", slug)        # strip extension
    slug = re.sub(r"[-_]+", " ", slug).strip()
    return slug.title() if slug else url


# ── Backward-compat shim (DOC API client kept for interface compatibility) ─────

class GDELTClient:
    """Legacy DOC 2.0 API client — kept for interface compatibility.

    New code should use GKGBulkClient.fetch_all_tickers() instead.
    """

    def __init__(self, project=None, request_delay=15.0, timeout=30):
        self._delay = request_delay
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "trading-bot-research/1.0"
        _log.warning(
            "GDELTClient (DOC API) is deprecated; prefer GKGBulkClient for bulk fetches"
        )

    def fetch_for_ticker(self, ticker, company_name, from_date, to_date, max_rows=1_000):
        """Thin shim: delegates to GKGBulkClient for a single ticker."""
        bulk = GKGBulkClient(workers=1, timeout=60)
        # Build minimal keywords from company name
        keywords = [company_name.lower().split()[0]]
        results = bulk.fetch_all_tickers(
            {ticker: keywords}, from_date, to_date, max_per_ticker=max_rows
        )
        return results.get(ticker, [])
