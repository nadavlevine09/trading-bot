"""Risk checks: circuit breaker, concentration, capacity, liquidity, trailing stops."""
from __future__ import annotations

from .config import PortfolioConfig
from .models import Position


class RiskChecker:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config

    # ── Circuit breaker ────────────────────────────────────────────────────────

    def is_circuit_breaker_triggered(
        self, portfolio_value: float, peak_portfolio_value: float
    ) -> tuple[bool, str]:
        if peak_portfolio_value <= 0:
            return False, ""
        drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
        if drawdown >= self.config.circuit_breaker_pct:
            return True, (
                f"circuit breaker: portfolio down {drawdown:.1%} from peak "
                f"(threshold {self.config.circuit_breaker_pct:.1%})"
            )
        return False, ""

    # ── Sector concentration ───────────────────────────────────────────────────

    def check_sector_concentration(
        self,
        ticker: str,
        new_position_value: float,
        portfolio_value: float,
        positions: dict[str, Position],
        current_prices: dict[str, float],
    ) -> tuple[bool, str]:
        """Returns (allowed, reason). allowed=True means the trade passes."""
        sector = self.config.sector_map.get(ticker, "unknown")
        existing_sector_value = sum(
            pos.market_value(current_prices.get(pos.ticker, pos.avg_cost))
            for pos in positions.values()
            if self.config.sector_map.get(pos.ticker, "unknown") == sector
        )
        post_trade_pct = (existing_sector_value + new_position_value) / portfolio_value
        if post_trade_pct > self.config.max_sector_pct:
            return False, (
                f"sector concentration: {ticker} ({sector}) would bring "
                f"sector to {post_trade_pct:.1%} (max {self.config.max_sector_pct:.1%})"
            )
        return True, ""

    # ── Capacity ───────────────────────────────────────────────────────────────

    def check_capacity(self, n_positions: int) -> tuple[bool, str]:
        if n_positions >= self.config.max_positions:
            return False, (
                f"at capacity: {n_positions}/{self.config.max_positions} positions open"
            )
        return True, ""

    # ── Liquidity ──────────────────────────────────────────────────────────────

    def check_liquidity(self, ticker: str, avg_daily_volume: float) -> tuple[bool, str]:
        if avg_daily_volume < self.config.min_avg_daily_volume:
            return False, (
                f"liquidity: {ticker} ADV {avg_daily_volume:,.0f} "
                f"< minimum {self.config.min_avg_daily_volume:,}"
            )
        return True, ""

    # ── Take-profit ────────────────────────────────────────────────────────────

    def positions_hitting_take_profit(
        self,
        positions: dict[str, Position],
        current_prices: dict[str, float],
    ) -> list[tuple[str, float, str]]:
        """
        Return (ticker, price, reason) for every position whose unrealized gain
        vs avg_cost meets or exceeds take_profit_pct.  Does NOT update peak prices
        — that responsibility stays with positions_hitting_trailing_stop.
        """
        hits: list[tuple[str, float, str]] = []
        for ticker, pos in positions.items():
            price = current_prices.get(ticker)
            if price is None or pos.avg_cost <= 0:
                continue
            gain = (price - pos.avg_cost) / pos.avg_cost
            if gain >= self.config.take_profit_pct:
                hits.append((
                    ticker,
                    price,
                    f"take profit: {ticker} at {price:.2f} "
                    f"(+{gain:.1%} vs cost {pos.avg_cost:.2f}, "
                    f"threshold {self.config.take_profit_pct:.1%})",
                ))
        return hits

    # ── Trailing stops ─────────────────────────────────────────────────────────

    def positions_hitting_trailing_stop(
        self,
        positions: dict[str, Position],
        current_prices: dict[str, float],
    ) -> list[tuple[str, float, str]]:
        """
        Update each position's peak price, then return (ticker, price, reason)
        for every position whose current price has fallen below its trailing stop.
        """
        hits: list[tuple[str, float, str]] = []
        for ticker, pos in positions.items():
            price = current_prices.get(ticker)
            if price is None:
                continue
            pos.update_peak(price)
            stop_price = pos.peak_price * (1 - self.config.trailing_stop_pct)
            if price <= stop_price:
                hits.append((
                    ticker,
                    price,
                    f"trailing stop: {ticker} at {price:.2f} ≤ stop {stop_price:.2f} "
                    f"({self.config.trailing_stop_pct:.1%} below peak {pos.peak_price:.2f})",
                ))
        return hits
