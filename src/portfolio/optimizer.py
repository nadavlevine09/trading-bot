"""Markowitz mean-variance portfolio optimizer."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_log = logging.getLogger(__name__)


class MarkowitzOptimizer:
    """
    Minimum-variance portfolio optimizer (long-only, fully-invested).

    Uses ``lookback`` trading days of closing-price history to estimate the
    covariance matrix, then solves:

        min  w^T Σ w
        s.t. Σw = 1,  min_weight ≤ w_i ≤ max_weight  ∀i

    When fewer than ``min_bars`` bars are available for any ticker the
    optimizer falls back to equal weights (no optimisation performed).
    """

    def __init__(
        self,
        lookback: int = 60,
        min_weight: float = 0.0,
        max_weight: float = 0.25,
        min_bars: int = 20,
    ) -> None:
        if max_weight <= 0 or max_weight > 1.0:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if min_weight < 0 or min_weight >= max_weight:
            raise ValueError(
                f"min_weight must be in [0, max_weight), got {min_weight}"
            )
        self.lookback   = lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_bars   = min_bars

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_weights(
        self,
        price_history: dict[str, pd.DataFrame],
        tickers: list[str],
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """
        Compute minimum-variance portfolio weights for *tickers*.

        Args:
            price_history: Per-ticker DataFrames with a DatetimeIndex and
                           at least a ``close`` column.
            tickers:       Ordered list of tickers to include.
            as_of:         Upper bound for the history window.  Pass the
                           current bar's timestamp to avoid look-ahead;
                           defaults to the latest date available.

        Returns:
            ``{ticker: weight}`` dict.  Tickers with insufficient history
            receive equal weight (the fallback is applied portfolio-wide).
        """
        if not tickers:
            return {}

        # Build aligned return matrix (rows = dates, cols = tickers)
        returns = self._build_returns(price_history, tickers, as_of)

        if returns is None:
            _log.debug("Markowitz: insufficient data — equal weights")
            eq = 1.0 / len(tickers)
            return {t: eq for t in tickers}

        cov = returns.cov().values
        n   = len(tickers)

        # Solve min-variance
        w0          = np.full(n, 1.0 / n)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds      = [(self.min_weight, self.max_weight)] * n

        try:
            result = minimize(
                fun=lambda w: float(w @ cov @ w),
                x0=w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if not result.success:
                _log.debug("Markowitz solver did not converge: %s — equal weights", result.message)
                weights = w0
            else:
                weights = np.clip(result.x, 0.0, 1.0)
                # Renormalise after clipping
                total = weights.sum()
                weights = weights / total if total > 0 else w0
        except Exception as exc:
            _log.warning("Markowitz optimisation failed: %s — equal weights", exc)
            weights = w0

        return dict(zip(tickers, weights.tolist()))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_returns(
        self,
        price_history: dict[str, pd.DataFrame],
        tickers: list[str],
        as_of: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """
        Build a (dates × tickers) DataFrame of daily log-returns.
        Returns None when any ticker has fewer than ``min_bars`` observations.
        """
        series: dict[str, pd.Series] = {}
        for ticker in tickers:
            df = price_history.get(ticker)
            if df is None or df.empty:
                return None
            closes = df["close"]
            if as_of is not None:
                closes = closes[closes.index <= as_of]
            closes = closes.tail(self.lookback + 1)
            if len(closes) < self.min_bars:
                return None
            rets = np.log(closes / closes.shift(1)).dropna()
            series[ticker] = rets

        if not series:
            return None

        combined = pd.DataFrame(series).dropna()
        if len(combined) < self.min_bars:
            return None
        return combined
