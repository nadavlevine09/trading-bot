"""Shared data models for the portfolio module."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class Signal:
    timestamp: datetime
    ticker: str
    action: str  # "buy" | "sell" | "hold"


@dataclass
class Position:
    ticker: str
    qty: float
    avg_cost: float
    sector: str
    entry_date: datetime
    peak_price: float  # high-watermark used by trailing-stop logic

    def market_value(self, current_price: float) -> float:
        return self.qty * current_price

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost

    def update_peak(self, current_price: float) -> None:
        if current_price > self.peak_price:
            self.peak_price = current_price


@dataclass
class MarketData:
    """Snapshot of market state passed into PortfolioManager on each evaluation."""
    current_prices: dict[str, float]
    # ticker → DataFrame with at minimum a 'close' column and DatetimeIndex
    price_history: dict[str, pd.DataFrame]
    avg_daily_volume: dict[str, float]
    # Optional Markowitz target weights (ticker → fraction of portfolio).
    # When non-empty, PositionSizer uses these instead of max_position_pct.
    target_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class TradeDecision:
    action: str   # "buy" | "sell" | "skip" | "queued"
    ticker: str
    qty: float
    price: float  # mid-market signal price; slippage applied separately at execution
    reason: str
    timestamp: datetime
    portfolio_snapshot: dict[str, Any] = field(default_factory=dict)
