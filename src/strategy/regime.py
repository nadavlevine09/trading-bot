"""Regime detection: classify each trading day as 'trending' or 'choppy/volatile'.

Algorithm
---------
1. Compute 20-day rolling realized volatility for each ticker using log returns.
2. Average across tickers to get a single market-wide vol series.
3. Compute the 60-day rolling mean of that series (the longer-term baseline).
4. For each date: if current_20d_vol > baseline × vol_ratio_threshold → 'choppy';
   otherwise → 'trending'.
5. Apply a 3-day EWM smoother to avoid single-day flip-flopping at the boundary.

All parameters are configurable at construction time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# ── Defaults (all configurable) ───────────────────────────────────────────────

_VOL_WINDOW     = 20    # bars: short-term realized vol window
_VOL_LOOKBACK   = 60    # bars: how far back to average for baseline
_RATIO_THRESHOLD = 1.2  # current_vol / baseline_vol above which = choppy
_SMOOTH_ALPHA   = 0.5   # EWM alpha applied to the vol-ratio series before thresholding
_MIN_BARS       = 21    # minimum bars required before classifying (else 'trending')


class RegimeDetector:
    """
    Classify each trading date as ``'trending'`` or ``'choppy'``.

    Parameters
    ----------
    vol_window : int
        Rolling window (bars) for short-term realized volatility.
    vol_lookback : int
        Rolling window (bars) for the longer-term vol baseline.
    vol_ratio_threshold : float
        If smoothed(current_vol / baseline_vol) > this → 'choppy'.
    smooth_alpha : float
        EWM alpha applied to the vol-ratio series to reduce flip-flopping.
    min_bars : int
        Minimum bars required before making a classification.
        Dates with fewer bars of history default to 'trending'.
    """

    def __init__(
        self,
        vol_window: int = _VOL_WINDOW,
        vol_lookback: int = _VOL_LOOKBACK,
        vol_ratio_threshold: float = _RATIO_THRESHOLD,
        smooth_alpha: float = _SMOOTH_ALPHA,
        min_bars: int = _MIN_BARS,
    ) -> None:
        if vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {vol_window}")
        if vol_lookback <= vol_window:
            raise ValueError(
                f"vol_lookback ({vol_lookback}) must exceed vol_window ({vol_window})"
            )
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError(f"smooth_alpha must be in (0, 1], got {smooth_alpha}")

        self.vol_window          = vol_window
        self.vol_lookback        = vol_lookback
        self.vol_ratio_threshold = vol_ratio_threshold
        self.smooth_alpha        = smooth_alpha
        self.min_bars            = min_bars

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify_dates(
        self,
        price_history: dict[str, pd.DataFrame],
        dates: list[pd.Timestamp] | None = None,
    ) -> dict[pd.Timestamp, str]:
        """
        Classify each date in *dates* as ``'trending'`` or ``'choppy'``.

        Parameters
        ----------
        price_history : dict[ticker → OHLCV DataFrame]
            Full price history available.  Each DataFrame must have a
            DatetimeIndex and a ``close`` column.
        dates : list of pd.Timestamp, optional
            Dates to classify.  If None, classifies every date present in
            any ticker's price history.

        Returns
        -------
        dict mapping each date (normalized to midnight) to a regime label.
        Also logs each date's regime at DEBUG level.
        """
        if not price_history:
            _log.warning("RegimeDetector: no price history provided; defaulting all to 'trending'")
            return {d: "trending" for d in (dates or [])}

        # Build combined daily log-return series (mean across tickers per day)
        log_returns_per_ticker: list[pd.Series] = []
        for ticker, df in price_history.items():
            if "close" not in df.columns or df.empty:
                continue
            close = df["close"].sort_index()
            log_ret = np.log(close / close.shift(1)).dropna()
            log_returns_per_ticker.append(log_ret.rename(ticker))

        if not log_returns_per_ticker:
            _log.warning("RegimeDetector: no usable price data; defaulting to 'trending'")
            return {d: "trending" for d in (dates or [])}

        combined = pd.concat(log_returns_per_ticker, axis=1).mean(axis=1).sort_index()

        # Rolling realized vol (std of log-returns over vol_window days)
        rolling_vol = combined.rolling(window=self.vol_window, min_periods=self.vol_window).std()

        # Longer-term baseline: rolling mean of short-term vol
        baseline_vol = rolling_vol.rolling(window=self.vol_lookback, min_periods=self.vol_window).mean()

        # Vol ratio
        vol_ratio = rolling_vol / baseline_vol.replace(0, np.nan)

        # Smooth the ratio to reduce flip-flopping
        vol_ratio_smooth = vol_ratio.ewm(alpha=self.smooth_alpha, adjust=False).mean()

        # Classify
        target_dates = dates if dates is not None else sorted(combined.index.normalize().unique())

        regime_map: dict[pd.Timestamp, str] = {}
        for dt in target_dates:
            dt_norm = pd.Timestamp(dt).normalize()
            # Number of bars available up to and including this date
            n_bars = int((combined.index.normalize() <= dt_norm).sum())

            if n_bars < self.min_bars:
                regime = "trending"
                _log.debug(
                    "Regime %s: trending (insufficient history: %d < %d bars)",
                    dt_norm.date(), n_bars, self.min_bars,
                )
            elif dt_norm not in vol_ratio_smooth.index:
                # Date is outside the price series — find nearest earlier date
                prior = vol_ratio_smooth[vol_ratio_smooth.index.normalize() <= dt_norm]
                ratio = float(prior.iloc[-1]) if not prior.empty else np.nan
                if np.isnan(ratio):
                    regime = "trending"
                else:
                    regime = "choppy" if ratio > self.vol_ratio_threshold else "trending"
            else:
                ratio = float(vol_ratio_smooth.get(dt_norm, np.nan))
                if np.isnan(ratio):
                    regime = "trending"
                else:
                    regime = "choppy" if ratio > self.vol_ratio_threshold else "trending"

            regime_map[dt_norm] = regime
            _log.debug("Regime %s: %s (vol_ratio=%.3f)", dt_norm.date(), regime,
                       ratio if 'ratio' in dir() and not np.isnan(ratio) else float("nan"))

        return regime_map

    def log_summary(self, regime_map: dict[pd.Timestamp, str]) -> None:
        """Print a human-readable table of regime labels to the logger (INFO)."""
        if not regime_map:
            _log.info("RegimeDetector: no dates to summarise")
            return

        n_trending = sum(1 for v in regime_map.values() if v == "trending")
        n_choppy   = sum(1 for v in regime_map.values() if v == "choppy")
        total      = len(regime_map)

        _log.info(
            "Regime summary: %d days total — %d trending (%.0f%%), %d choppy (%.0f%%)",
            total,
            n_trending, 100 * n_trending / total if total else 0,
            n_choppy,   100 * n_choppy   / total if total else 0,
        )

        for dt in sorted(regime_map):
            _log.info("  %s  →  %s", dt.date(), regime_map[dt])
