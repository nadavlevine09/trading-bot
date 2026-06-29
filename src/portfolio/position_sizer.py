"""Volatility-adjusted position sizing with hard caps."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PortfolioConfig


class PositionSizer:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config

    def calculate_qty(
        self,
        ticker: str,
        price: float,
        portfolio_value: float,
        cash: float,
        price_history: pd.DataFrame | None,
        target_weight: float | None = None,
    ) -> float:
        """
        Return whole-share quantity for a new position.

        Sizing uses three constraints; the binding (smallest) wins:
          1. Volatility target: size so daily dollar vol of position ≈
             volatility_target × portfolio_value.
             Replaced by ``target_weight × portfolio_value`` when provided
             (Markowitz mode — uses optimizer weight as the hard cap).
          2. Max allocation cap: no position > max_position_pct × portfolio_value.
             Skipped in Markowitz mode (target_weight acts as the cap).
          3. Deployable cash: cash minus the configured cash buffer.

        Falls back to cap (2) when price history is too short.
        Returns 0.0 if price ≤ 0 or deployable cash is exhausted.
        """
        if price <= 0:
            return 0.0

        deployable = cash - self.config.cash_buffer_pct * portfolio_value
        if deployable <= 0:
            return 0.0

        if target_weight is not None:
            # Markowitz mode: honour optimizer weight as both target and cap
            cap_qty  = (target_weight * portfolio_value) / price
            cash_qty = deployable / price
            qty = min(cap_qty, cash_qty)
        else:
            vol_qty  = self._vol_adjusted_qty(price, portfolio_value, price_history)
            cap_qty  = (self.config.max_position_pct * portfolio_value) / price
            cash_qty = deployable / price
            qty = min(vol_qty, cap_qty, cash_qty)

        return max(0.0, float(int(qty)))  # floor to whole shares

    def _vol_adjusted_qty(
        self,
        price: float,
        portfolio_value: float,
        price_history: pd.DataFrame | None,
    ) -> float:
        min_bars = self.config.volatility_window + 1
        if price_history is None or len(price_history) < min_bars:
            return (self.config.max_position_pct * portfolio_value) / price

        closes = price_history["close"].tail(min_bars)
        daily_vol = closes.pct_change().dropna().std()

        if daily_vol == 0 or np.isnan(daily_vol):
            return (self.config.max_position_pct * portfolio_value) / price

        # qty such that: price × qty × daily_vol == volatility_target × portfolio_value
        return (self.config.volatility_target * portfolio_value) / (price * daily_vol)
