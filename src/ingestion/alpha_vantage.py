"""Alpha Vantage client — earnings surprise data for confirmation signals."""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

_log = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"
# Free tier: 25 calls/day, 5 calls/min → sleep 12.5 s between calls to stay safe
_MIN_SLEEP_SECS = 12.5


class AlphaVantageClient:
    """Thin wrapper around the Alpha Vantage REST API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["ALPHA_VANTAGE_KEY"]
        self._session = requests.Session()

    def fetch_earnings(self, ticker: str) -> list[dict]:
        """
        Fetch quarterly earnings history for *ticker*.

        Returns a list of dicts:
          ticker, reportedDate, fiscalDateEnding,
          reportedEPS, estimatedEPS, surprisePercentage
        """
        resp = self._session.get(
            _BASE_URL,
            params={
                "function": "EARNINGS",
                "symbol":   ticker,
                "apikey":   self._api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if "Note" in data or "Information" in data:
            raise RateLimitError(data.get("Note") or data.get("Information"))

        rows = []
        for q in data.get("quarterlyEarnings", []):
            rows.append({
                "ticker":            ticker,
                "reportedDate":      q.get("reportedDate", ""),
                "fiscalDateEnding":  q.get("fiscalDateEnding", ""),
                "reportedEPS":       _to_float(q.get("reportedEPS")),
                "estimatedEPS":      _to_float(q.get("estimatedEPS")),
                "surprisePercentage": _to_float(q.get("surprisePercentage")),
            })
        return rows

    def fetch_all_earnings(
        self,
        tickers: list[str],
        sleep_secs: float = _MIN_SLEEP_SECS,
    ) -> dict[str, list[dict]]:
        """Fetch earnings for all *tickers*, rate-limited."""
        result: dict[str, list[dict]] = {}
        for i, ticker in enumerate(tickers):
            _log.info("Fetching earnings: %s (%d/%d)", ticker, i + 1, len(tickers))
            try:
                result[ticker] = self.fetch_earnings(ticker)
                _log.info("  %s: %d quarters", ticker, len(result[ticker]))
            except RateLimitError as exc:
                _log.error("Rate limited — stopping: %s", exc)
                break
            except Exception as exc:
                _log.warning("  %s failed: %s", ticker, exc)
                result[ticker] = []
            if i < len(tickers) - 1:
                time.sleep(sleep_secs)
        return result


class EarningsCache:
    """
    Fast look-up: "was there a recent earnings report for this ticker
    near a given signal date, and if so what was the surprise?"

    Build once from the raw dict returned by ``fetch_all_earnings``
    and reuse across many ``recent_surprise()`` calls.
    """

    def __init__(self, earnings: dict[str, list[dict]]) -> None:
        # {ticker → [(report_date, surprise_pct), ...], newest-first}
        self._by_ticker: dict[str, list[tuple[date, float]]] = {}
        for ticker, rows in earnings.items():
            pairs: list[tuple[date, float]] = []
            for r in rows:
                rd  = _parse_date(r.get("reportedDate", ""))
                sp  = r.get("surprisePercentage")
                if rd is not None and sp is not None:
                    pairs.append((rd, float(sp)))
            self._by_ticker[ticker] = sorted(pairs, key=lambda x: x[0], reverse=True)

    def recent_surprise(
        self,
        ticker: str,
        signal_date: date | datetime,
        lookback_days: int = 7,
    ) -> float | None:
        """
        Return the surprise percentage of the most recent earnings report
        if it falls within ``[signal_date − lookback_days, signal_date]``.
        Returns ``None`` when no earnings were reported in that window.
        """
        if isinstance(signal_date, datetime):
            signal_date = signal_date.date()
        cutoff = signal_date - timedelta(days=lookback_days)
        for report_date, surprise_pct in self._by_ticker.get(ticker, []):
            if cutoff <= report_date <= signal_date:
                return surprise_pct
        return None

    def __repr__(self) -> str:
        counts = {t: len(v) for t, v in self._by_ticker.items()}
        return f"EarningsCache({counts})"


class RateLimitError(Exception):
    pass


# ── Private helpers ────────────────────────────────────────────────────────────

def _to_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_date(val: str) -> date | None:
    try:
        return date.fromisoformat(val)
    except (ValueError, TypeError):
        return None
