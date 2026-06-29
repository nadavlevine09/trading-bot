"""Portfolio-level configuration. All thresholds live here — nothing hardcoded elsewhere."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PortfolioConfig:
    # ── Position sizing ────────────────────────────────────────────────────────
    max_position_pct: float = 0.10      # max fraction of portfolio per position
    cash_buffer_pct: float = 0.05       # minimum uninvested cash fraction
    max_positions: int = 20             # max concurrent open positions
    volatility_window: int = 20         # rolling window (days) for daily return std
    volatility_target: float = 0.02     # target daily return vol per position

    # ── Concentration risk ─────────────────────────────────────────────────────
    max_sector_pct: float = 0.30        # max portfolio fraction in any one sector
    sector_map: dict[str, str] = field(default_factory=dict)  # ticker → sector

    # ── Drawdown protection ────────────────────────────────────────────────────
    circuit_breaker_pct: float = 0.10   # halt buys if portfolio drops this far from peak
    trailing_stop_pct: float = 0.08     # per-position trailing stop from position peak price
    take_profit_pct: float = 0.18       # sell when unrealized gain exceeds this (vs avg_cost)

    # ── Rebalancing ────────────────────────────────────────────────────────────
    at_capacity_action: str = "skip"    # "skip" | "queue" | "sell_weakest"
    signal_queue_maxsize: int = 50      # max pending signals held in queue
    rebalance_interval_days: int = 7    # minimum days between periodic rebalance runs

    # ── Liquidity / transaction costs ──────────────────────────────────────────
    min_avg_daily_volume: int = 500_000  # ADV gate; tickers below this are skipped
    slippage_bps: float = 10.0           # simulated round-trip slippage in basis points

    @classmethod
    def from_toml(cls, path: Path) -> PortfolioConfig:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data.get("portfolio", {}))
