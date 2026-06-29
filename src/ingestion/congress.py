"""Congressional trading disclosures — confirmation signal source.

Data comes from STOCK Act periodic transaction reports (PTRs), via Quiver
Quant's cleaned/free-tier API rather than scraping raw House/Senate PDFs.

Two timestamps matter and they are NOT the same day:
  - transaction_date: when the member actually bought/sold
  - filing_date / disclosure_date: when the public could first see it

The STOCK Act allows up to 45 days between the two. A signal built on
transaction_date is unbacktestable in real life — you would not have known
about the trade yet. Every lookup in this module is keyed on filing_date,
because that's the earliest moment the information is actually tradeable.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

_log = logging.getLogger(__name__)

_BASE_URL = "https://api.quiverquant.com/beta/live/congresstrading"
_MIN_SLEEP_SECS = 1.0  # Quiver free tier is generous relative to Alpha Vantage


class QuiverQuantClient:
    """Thin wrapper around Quiver Quant's congressional trading endpoint."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["QUIVERQUANT_KEY"]
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

    def fetch_trades(self, ticker: str) -> list[dict]:
        """
        Fetch congressional trading history for *ticker*.

        Returns a list of dicts with (at minimum):
          ticker, Representative, Transaction, TransactionDate, ReportDate,
          Amount, Party, House
        """
        resp = self._session.get(
            f"{_BASE_URL}/{ticker}",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "error" in data:
            raise RateLimitError(data["error"])

        rows = []
        for r in data:
            rows.append({
                "ticker":          ticker,
                "representative":  r.get("Representative", ""),
                "transaction":     r.get("Transaction", ""),       # "Purchase" | "Sale"
                "transaction_date": r.get("TransactionDate", ""),
                "report_date":     r.get("ReportDate", ""),         # disclosure date
                "amount":          r.get("Amount", ""),
                "party":           r.get("Party", ""),
                "chamber":         r.get("House", ""),
            })
        return rows

    def fetch_all_trades(
        self,
        tickers: list[str],
        sleep_secs: float = _MIN_SLEEP_SECS,
    ) -> dict[str, list[dict]]:
        """Fetch congressional trades for all *tickers*, rate-limited."""
        result: dict[str, list[dict]] = {}
        for i, ticker in enumerate(tickers):
            _log.info("Fetching congress trades: %s (%d/%d)", ticker, i + 1, len(tickers))
            try:
                result[ticker] = self.fetch_trades(ticker)
                _log.info("  %s: %d filings", ticker, len(result[ticker]))
            except RateLimitError as exc:
                _log.error("Rate limited — stopping: %s", exc)
                break
            except Exception as exc:
                _log.warning("  %s failed: %s", ticker, exc)
                result[ticker] = []
            if i < len(tickers) - 1:
                time.sleep(sleep_secs)
        return result


class CongressTradeCache:
    """
    Fast look-up: "was there a recent congressional disclosure for this
    ticker near a given signal date, and if so, which direction?"

    Mirrors ``EarningsCache``'s shape so it slots into the same
    ``SignalGenerator`` confirmation-layer pattern.

    Critically keyed on ``report_date`` (disclosure date), not
    ``transaction_date`` — see module docstring. A trade is only visible to
    the cache as of the day it became public.
    """

    def __init__(self, trades: dict[str, list[dict]]) -> None:
        # {ticker → [(report_date, direction), ...], newest-first}
        # direction: +1.0 (purchase) / -1.0 (sale)
        self._by_ticker: dict[str, list[tuple[date, float]]] = {}
        for ticker, rows in trades.items():
            pairs: list[tuple[date, float]] = []
            for r in rows:
                rd = _parse_date(r.get("report_date", ""))
                direction = _direction(r.get("transaction", ""))
                if rd is not None and direction is not None:
                    pairs.append((rd, direction))
            self._by_ticker[ticker] = sorted(pairs, key=lambda x: x[0], reverse=True)

    def recent_trade_direction(
        self,
        ticker: str,
        signal_date: date | datetime,
        lookback_days: int = 14,
    ) -> float | None:
        """
        Return the net direction (-1.0 to +1.0) of congressional trades
        disclosed within ``[signal_date - lookback_days, signal_date]``.

        Multiple filings in the window are averaged (a buy and a sell in
        the same window cancel toward zero, which then fails to confirm
        either action — this is intentional; mixed congressional sentiment
        is not a confirming signal).

        Returns ``None`` when no disclosures fall in that window.
        """
        if isinstance(signal_date, datetime):
            signal_date = signal_date.date()
        cutoff = signal_date - timedelta(days=lookback_days)

        directions = [
            d for report_date, d in self._by_ticker.get(ticker, [])
            if cutoff <= report_date <= signal_date
        ]
        if not directions:
            return None
        return sum(directions) / len(directions)

    def __repr__(self) -> str:
        counts = {t: len(v) for t, v in self._by_ticker.items()}
        return f"CongressTradeCache({counts})"


class RateLimitError(Exception):
    pass


# ── Private helpers ────────────────────────────────────────────────────────────

def _direction(transaction: str) -> float | None:
    t = transaction.strip().lower()
    if "purchase" in t or t == "buy":
        return 1.0
    if "sale" in t or t == "sell":
        return -1.0
    return None  # e.g. "Exchange" — ambiguous, excluded


def _parse_date(val: str) -> date | None:
    try:
        return date.fromisoformat(val)
    except (ValueError, TypeError):
        return None
