"""Convert aggregated daily sentiment scores into buy/sell/hold signals."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.portfolio.models import Signal

if TYPE_CHECKING:
    from src.ingestion.alpha_vantage import EarningsCache
    from src.ingestion.congress import CongressTradeCache

_log = logging.getLogger(__name__)

_BUY_DEFAULT   =  0.3
_SELL_DEFAULT  = -0.3
_MIN_HEADLINES =  8
_MA_WINDOW     = 10
_EARNINGS_LOOKBACK = 7    # days to look back for a recent earnings report
_EARNINGS_THRESHOLD = 0.0 # min absolute surprise % to count as a signal; 0 = any beats/miss
_CONGRESS_LOOKBACK = 14   # days to look back for a recent congressional disclosure
_CONGRESS_THRESHOLD = 0.0 # min absolute net direction to count as confirming; 0 = any net lean


class SignalGenerator:
    """
    Threshold-based signal generator with optional confirmation layers.

    Maps a daily aggregated sentiment score (from ``aggregate_daily()``) to one
    of: ``buy`` | ``sell`` | ``hold``.

    Confirmation layers (all opt-in, applied in order):
      1. Thin-data guard: fewer than ``min_headlines`` → forced hold.
      2. Momentum filter (``momentum_confirm=True``): BUY requires close >
         ``ma_window``-day MA; SELL requires close < MA.  Insufficient history
         also falls back to hold.
      3. Earnings surprise filter (``earnings_confirm=True``): only active near
         earnings dates.  If an earnings report falls within the last
         ``earnings_lookback`` days, a BUY requires a positive surprise and a
         SELL requires a negative surprise; neutral/contradictory surprises
         downgrade to hold.  When no recent earnings exist the signal passes
         through unchanged (the filter is irrelevant that day).
      4. Congressional trading filter (``congress_confirm=True``): only active
         when a STOCK Act disclosure for the ticker was filed within the last
         ``congress_lookback`` days. Keyed on disclosure (filing) date, not
         transaction date, since that's the earliest point the trade is
         actually public. A BUY requires net congressional buying and a SELL
         requires net congressional selling over that window; disagreement
         downgrades to hold. No recent disclosure → signal passes through
         unchanged.

    Boundary values (score == threshold) always yield ``hold``.
    """

    def __init__(
        self,
        buy_threshold: float = _BUY_DEFAULT,
        sell_threshold: float = _SELL_DEFAULT,
        min_headlines: int = _MIN_HEADLINES,
        momentum_confirm: bool = False,
        ma_window: int = _MA_WINDOW,
        earnings_confirm: bool = False,
        earnings_lookback: int = _EARNINGS_LOOKBACK,
        earnings_threshold: float = _EARNINGS_THRESHOLD,
        congress_confirm: bool = False,
        congress_lookback: int = _CONGRESS_LOOKBACK,
        congress_threshold: float = _CONGRESS_THRESHOLD,
    ) -> None:
        if buy_threshold <= sell_threshold:
            raise ValueError(
                f"buy_threshold ({buy_threshold}) must be strictly greater than "
                f"sell_threshold ({sell_threshold})"
            )
        if min_headlines < 0:
            raise ValueError(f"min_headlines must be >= 0, got {min_headlines}")
        if ma_window < 1:
            raise ValueError(f"ma_window must be >= 1, got {ma_window}")
        if earnings_lookback < 0:
            raise ValueError(f"earnings_lookback must be >= 0, got {earnings_lookback}")
        if congress_lookback < 0:
            raise ValueError(f"congress_lookback must be >= 0, got {congress_lookback}")

        self.buy_threshold        = buy_threshold
        self.sell_threshold       = sell_threshold
        self.min_headlines        = min_headlines
        self.momentum_confirm     = momentum_confirm
        self.ma_window            = ma_window
        self.earnings_confirm     = earnings_confirm
        self.earnings_lookback    = earnings_lookback
        self.earnings_threshold   = earnings_threshold
        self.congress_confirm     = congress_confirm
        self.congress_lookback    = congress_lookback
        self.congress_threshold   = congress_threshold

    # ── Core classification ────────────────────────────────────────────────────

    def classify(
        self,
        sentiment_score: float,
        n_headlines: int,
        ticker: str = "",
    ) -> str:
        """
        Return ``'buy'``, ``'sell'``, or ``'hold'`` for a single data point.

        Args:
            sentiment_score: Aggregated daily score in [−1, 1].
            n_headlines:     Number of articles scored that day.
            ticker:          Optional label used for debug logging only.
        """
        if n_headlines < self.min_headlines:
            _log.debug(
                "%s: hold (thin data — %d headlines, min %d)",
                ticker or "?", n_headlines, self.min_headlines,
            )
            return "hold"
        if sentiment_score > self.buy_threshold:
            return "buy"
        if sentiment_score < self.sell_threshold:
            return "sell"
        return "hold"

    # ── Batch generation ───────────────────────────────────────────────────────

    def generate(
        self,
        daily_scores: pd.DataFrame,
        as_of: datetime | None = None,
        price_history: dict[str, pd.DataFrame] | None = None,
        earnings_cache: "EarningsCache | None" = None,
        congress_cache: "CongressTradeCache | None" = None,
    ) -> list[Signal]:
        """
        Generate one Signal per row of a daily aggregated scores DataFrame.

        Every row produces a signal — including those that yield ``hold`` — so
        the caller (and the audit log) has full visibility into every decision.

        Args:
            daily_scores:   DataFrame with columns
                            ``[ticker, date, sentiment_score, n_headlines]``
                            as returned by ``sentiment.scorer.aggregate_daily()``.
            as_of:          Timestamp to stamp on every Signal.
                            • ``None`` (default) → midnight on the row's own date;
                              suitable for backtesting.
                            • Pass ``datetime.utcnow()`` or a market-open time for
                              live signal generation.
            price_history:  Per-ticker OHLCV DataFrames (DatetimeIndex, ``close``
                            column required).  Consulted when
                            ``momentum_confirm=True``; pass ``None`` to skip.
            earnings_cache: ``EarningsCache`` built from Alpha Vantage data.
                            Consulted when ``earnings_confirm=True``; pass
                            ``None`` to skip even when the flag is on.
            congress_cache: ``CongressTradeCache`` built from Quiver Quant
                            disclosure data. Consulted when
                            ``congress_confirm=True``; pass ``None`` to skip
                            even when the flag is on.

        Returns:
            List of ``Signal`` objects ready to pass to
            ``PortfolioManager.process_signal()``.
        """
        if daily_scores.empty:
            return []

        signals: list[Signal] = []
        for _, row in daily_scores.iterrows():
            ticker = str(row["ticker"])
            score  = float(row["sentiment_score"])
            n      = int(row.get("n_headlines", 0))
            action = self.classify(score, n, ticker=ticker)
            ts     = as_of if as_of is not None else _date_to_datetime(row["date"])

            # Layer 2: momentum confirmation
            if (
                self.momentum_confirm
                and price_history is not None
                and action in ("buy", "sell")
                and not self._momentum_agrees(ticker, row["date"], action, price_history)
            ):
                _log.debug("%s: momentum disagrees with %s — downgraded to hold", ticker, action)
                action = "hold"

            # Layer 3: earnings surprise confirmation (only active near earnings dates)
            if (
                self.earnings_confirm
                and earnings_cache is not None
                and action in ("buy", "sell")
            ):
                action = self._apply_earnings_filter(
                    ticker, row["date"], action, earnings_cache
                )

            # Layer 4: congressional trading confirmation (only active near disclosures)
            if (
                self.congress_confirm
                and congress_cache is not None
                and action in ("buy", "sell")
            ):
                action = self._apply_congress_filter(
                    ticker, row["date"], action, congress_cache
                )

            signals.append(Signal(timestamp=ts, ticker=ticker, action=action))
            _log.debug("%s: score=%.3f n=%d → %s", ticker, score, n, action)

        return signals

    # ── Earnings surprise confirmation ────────────────────────────────────────

    def _apply_earnings_filter(
        self,
        ticker: str,
        signal_date: Any,
        action: str,
        earnings_cache: "EarningsCache",
    ) -> str:
        """
        Apply earnings surprise gate for BUY/SELL signals.

        Only active when there is a recent earnings report (within
        ``earnings_lookback`` days).  When no earnings exist in that window,
        the signal passes through unchanged (filter is neutral).

        If an earnings report IS found:
          • BUY  + positive surprise (≥ threshold)  → passes
          • SELL + negative surprise (≤ -threshold) → passes
          • Any other combination                   → downgraded to hold
        """
        sig_date = _to_date(signal_date)
        surprise = earnings_cache.recent_surprise(
            ticker, sig_date, lookback_days=self.earnings_lookback
        )

        if surprise is None:
            # No recent earnings → filter not applicable; let signal through
            return action

        threshold = abs(self.earnings_threshold)
        if action == "buy" and surprise >= threshold:
            return action
        if action == "sell" and surprise <= -threshold:
            return action

        _log.debug(
            "%s: earnings surprise %.2f%% disagrees with %s — downgraded to hold",
            ticker, surprise, action,
        )
        return "hold"

    # ── Congressional trading confirmation ────────────────────────────────────

    def _apply_congress_filter(
        self,
        ticker: str,
        signal_date: Any,
        action: str,
        congress_cache: "CongressTradeCache",
    ) -> str:
        """
        Apply congressional-disclosure gate for BUY/SELL signals.

        Only active when a STOCK Act disclosure for the ticker was filed
        within ``congress_lookback`` days. When no recent disclosure exists,
        the signal passes through unchanged (filter is neutral).

        If a disclosure IS found (net direction over the window):
          • BUY  + net direction ≥ threshold   → passes
          • SELL + net direction ≤ -threshold  → passes
          • Any other combination               → downgraded to hold
        """
        sig_date = _to_date(signal_date)
        direction = congress_cache.recent_trade_direction(
            ticker, sig_date, lookback_days=self.congress_lookback
        )

        if direction is None:
            # No recent disclosure → filter not applicable; let signal through
            return action

        threshold = abs(self.congress_threshold)
        if action == "buy" and direction >= threshold:
            return action
        if action == "sell" and direction <= -threshold:
            return action

        _log.debug(
            "%s: congressional net direction %.2f disagrees with %s — downgraded to hold",
            ticker, direction, action,
        )
        return "hold"

    # ── Momentum confirmation ──────────────────────────────────────────────────

    def _momentum_agrees(
        self,
        ticker: str,
        signal_date: Any,
        intended_action: str,
        price_history: dict[str, pd.DataFrame],
    ) -> bool:
        """
        Return True if short-term price momentum supports ``intended_action``.

        BUY:  close > ``ma_window``-day moving average of close.
        SELL: close < ``ma_window``-day moving average of close.
        Fewer than ``ma_window`` bars available → return False (conservative hold).
        """
        df = price_history.get(ticker)
        if df is None or df.empty:
            _log.debug("%s: no price history — momentum fallback to hold", ticker)
            return False

        sig_ts = pd.Timestamp(signal_date).normalize()
        hist   = df[df.index <= sig_ts]

        if len(hist) < self.ma_window:
            _log.debug(
                "%s: only %d bar(s) up to %s (need %d) — momentum fallback to hold",
                ticker, len(hist), sig_ts.date(), self.ma_window,
            )
            return False

        ma    = float(hist["close"].tail(self.ma_window).mean())
        close = float(hist["close"].iloc[-1])

        if intended_action == "buy":
            return close > ma
        if intended_action == "sell":
            return close < ma
        return True


# ── Private helpers ────────────────────────────────────────────────────────────

def _to_date(val: Any) -> date:
    """Convert various date-like values to ``datetime.date``."""
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    ts = pd.Timestamp(val)
    if pd.isna(ts):
        return datetime.utcnow().date()
    return ts.date()


def _date_to_datetime(val: Any) -> datetime:
    """
    Convert a row's ``date`` value to a timezone-naive datetime at midnight.

    Handles: ``datetime.date``, ``datetime.datetime``, ``pd.Timestamp``, ``NaT``.
    Falls back to ``datetime.utcnow()`` on unparseable input.
    """
    if isinstance(val, datetime):          # datetime is a subclass of date
        return val.replace(tzinfo=None)
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
    ts = pd.Timestamp(val)
    if pd.isna(ts):
        return datetime.utcnow()
    return ts.to_pydatetime().replace(tzinfo=None)
