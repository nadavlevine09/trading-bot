"""At-capacity routing and periodic rebalance logic."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from .config import PortfolioConfig
from .models import Position, Signal


class Rebalancer:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config
        self._queue: deque[Signal] = deque(maxlen=config.signal_queue_maxsize)
        self._last_rebalance: datetime | None = None

    # ── Signal queue ───────────────────────────────────────────────────────────

    def enqueue(self, signal: Signal) -> None:
        self._queue.append(signal)

    def drain(self) -> list[Signal]:
        signals = list(self._queue)
        self._queue.clear()
        return signals

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    # ── Weakest-position selection ─────────────────────────────────────────────

    def weakest_position(
        self,
        positions: dict[str, Position],
        current_prices: dict[str, float],
    ) -> str | None:
        """Return ticker of the position with the lowest unrealized PnL %."""
        if not positions:
            return None
        return min(
            positions,
            key=lambda t: positions[t].unrealized_pnl_pct(
                current_prices.get(t, positions[t].avg_cost)
            ),
        )

    # ── Periodic rebalance scheduling ─────────────────────────────────────────

    def is_due(self, now: datetime) -> bool:
        if self._last_rebalance is None:
            return True
        return (now - self._last_rebalance) >= timedelta(
            days=self.config.rebalance_interval_days
        )

    def mark_done(self, now: datetime) -> None:
        self._last_rebalance = now

    # ── Drift detection ────────────────────────────────────────────────────────

    def overweight_positions(
        self,
        positions: dict[str, Position],
        current_prices: dict[str, float],
        portfolio_value: float,
    ) -> list[tuple[str, float, float, str]]:
        """
        Return (ticker, price, excess_qty, reason) for positions whose weight
        has drifted above max_position_pct × 1.5 (a 50% drift buffer).

        Trim quantity is chosen to bring the position back to max_position_pct.
        """
        drift_limit = self.config.max_position_pct * 1.5
        candidates: list[tuple[str, float, float, str]] = []
        for ticker, pos in positions.items():
            price = current_prices.get(ticker, pos.avg_cost)
            weight = pos.market_value(price) / portfolio_value if portfolio_value else 0
            if weight > drift_limit:
                target_qty = int((self.config.max_position_pct * portfolio_value) / price)
                excess_qty = pos.qty - target_qty
                if excess_qty > 0:
                    candidates.append((
                        ticker,
                        price,
                        float(excess_qty),
                        f"rebalance trim: {ticker} weight {weight:.1%} "
                        f"> drift limit {drift_limit:.1%}",
                    ))
        return candidates
