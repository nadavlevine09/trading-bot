"""
PortfolioManager — the single interface used by both backtest and live trading.

Callers follow this contract:
  1. Call process_signal() / check_trailing_stops() / periodic_rebalance()
     to get TradeDecision objects.  These calls are READ-ONLY with one
     exception: queuing a signal is a side-effect of process_signal() when
     at_capacity_action == "queue" or "sell_weakest".
  2. Call apply_decision() with the returned decision (and an optional
     actual fill_price from a live broker) to commit the trade to state.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import PortfolioConfig
from .logger import DecisionLogger
from .models import MarketData, Position, Signal, TradeDecision
from .position_sizer import PositionSizer
from .rebalancer import Rebalancer
from .risk import RiskChecker


class PortfolioManager:
    def __init__(self, config: PortfolioConfig, initial_cash: float) -> None:
        self.config = config
        self._cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._peak_portfolio_value = initial_cash

        self._sizer = PositionSizer(config)
        self._risk = RiskChecker(config)
        self._rebalancer = Rebalancer(config)
        self._decision_logger = DecisionLogger()

    # ──────────────────────────────────────────────────────────────────────────
    # Primary interface
    # ──────────────────────────────────────────────────────────────────────────

    def process_signal(self, signal: Signal, market_data: MarketData) -> TradeDecision:
        """
        Evaluate a signal against all risk constraints and return a TradeDecision.
        Does NOT mutate positions or cash; call apply_decision() afterwards.
        May update the signal queue as a side-effect (queue / sell_weakest paths).
        """
        prices = market_data.current_prices
        price = prices.get(signal.ticker, 0.0)
        pv = self._portfolio_value(prices)
        self._update_peak(pv)
        snapshot = self.snapshot(prices)
        now = signal.timestamp

        if signal.action == "hold":
            return self._mk("skip", signal.ticker, 0.0, price, "signal=hold", now, snapshot)

        if signal.action == "sell":
            return self._eval_sell(signal, price, now, snapshot)

        return self._eval_buy(signal, market_data, price, pv, now, snapshot)

    def check_trailing_stops(
        self,
        current_prices: dict[str, float],
        now: datetime | None = None,
    ) -> list[TradeDecision]:
        """
        Return sell decisions for every position that has breached its take-profit
        level or trailing stop.  Take-profit is evaluated first and takes priority:
        if both conditions apply to the same position, only the take-profit decision
        is emitted (since it represents the better outcome and subsumes the stop).
        """
        now = now or datetime.utcnow()
        decisions: list[TradeDecision] = []

        # Take-profit: checked first, wins when both conditions could apply.
        tp_tickers: set[str] = set()
        for ticker, price, reason in self._risk.positions_hitting_take_profit(
            self._positions, current_prices
        ):
            tp_tickers.add(ticker)
            snapshot = self.snapshot(current_prices)
            decisions.append(
                self._mk("sell", ticker, self._positions[ticker].qty, price, reason, now, snapshot)
            )

        # Trailing stops: also updates peak high-watermarks for all positions.
        # Skip tickers already handled by take-profit above.
        for ticker, price, reason in self._risk.positions_hitting_trailing_stop(
            self._positions, current_prices
        ):
            if ticker in tp_tickers:
                continue
            snapshot = self.snapshot(current_prices)
            decisions.append(
                self._mk("sell", ticker, self._positions[ticker].qty, price, reason, now, snapshot)
            )

        return decisions

    def periodic_rebalance(
        self,
        market_data: MarketData,
        now: datetime | None = None,
    ) -> list[TradeDecision]:
        """
        Periodic check independent of new signals. Runs only when
        rebalance_interval_days have elapsed since the last run.

        Steps:
          1. Trim positions that have drifted above max_position_pct × 1.5.
          2. Process queued signals now that capacity may have opened up.

        Returns TradeDecisions. Caller must apply each one.
        """
        now = now or datetime.utcnow()
        if not self._rebalancer.is_due(now):
            return []
        self._rebalancer.mark_done(now)

        prices = market_data.current_prices
        pv = self._portfolio_value(prices)
        decisions: list[TradeDecision] = []

        # Step 1: drift trims
        for ticker, price, qty, reason in self._rebalancer.overweight_positions(
            self._positions, prices, pv
        ):
            decisions.append(
                self._mk("sell", ticker, qty, price, reason, now, self.snapshot(prices))
            )

        # Step 2: flush signal queue
        for queued in self._rebalancer.drain():
            d = self.process_signal(queued, market_data)
            if d.action not in ("skip", "queued"):
                decisions.append(d)

        return decisions

    def apply_decision(
        self,
        decision: TradeDecision,
        fill_price: float | None = None,
    ) -> None:
        """
        Commit a buy or sell decision to portfolio state.

        fill_price:
          • Live trading — pass the actual broker fill price.
          • Backtesting — omit; configured slippage is applied to decision.price.

        Skipped / queued decisions are no-ops.
        """
        if decision.action not in ("buy", "sell"):
            self._decision_logger.log(decision)
            return

        fp = (
            fill_price
            if fill_price is not None
            else self._with_slippage(decision.price, decision.action)
        )

        if decision.action == "buy":
            self._open_position(decision, fp)
        else:
            self._close_position(decision, fp)

        self._decision_logger.log(decision)

    # ──────────────────────────────────────────────────────────────────────────
    # State queries
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def snapshot(self, current_prices: dict[str, float]) -> dict[str, Any]:
        pv = self._portfolio_value(current_prices)
        pos_rows = []
        sector_exposure: dict[str, float] = {}
        for ticker, pos in self._positions.items():
            price = current_prices.get(ticker, pos.avg_cost)
            value = pos.market_value(price)
            weight = value / pv if pv else 0.0
            sector = self.config.sector_map.get(ticker, "unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight
            pos_rows.append({
                "ticker": ticker,
                "qty": pos.qty,
                "avg_cost": pos.avg_cost,
                "current_price": price,
                "value": value,
                "weight": weight,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct(price),
                "sector": sector,
            })
        peak = self._peak_portfolio_value
        return {
            "portfolio_value": pv,
            "cash": self._cash,
            "n_positions": len(self._positions),
            "exposure_pct": 1.0 - (self._cash / pv) if pv else 0.0,
            "drawdown_from_peak": (peak - pv) / peak if peak > 0 else 0.0,
            "sector_exposure": sector_exposure,
            "positions": pos_rows,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _eval_buy(
        self,
        signal: Signal,
        market_data: MarketData,
        price: float,
        pv: float,
        now: datetime,
        snapshot: dict,
    ) -> TradeDecision:
        ticker = signal.ticker
        prices = market_data.current_prices

        if ticker in self._positions:
            return self._mk("skip", ticker, 0.0, price, "already holding", now, snapshot)

        adv = market_data.avg_daily_volume.get(ticker, 0.0)
        ok, reason = self._risk.check_liquidity(ticker, adv)
        if not ok:
            return self._mk("skip", ticker, 0.0, price, reason, now, snapshot)

        triggered, reason = self._risk.is_circuit_breaker_triggered(
            pv, self._peak_portfolio_value
        )
        if triggered:
            return self._mk("skip", ticker, 0.0, price, reason, now, snapshot)

        ok, reason = self._risk.check_capacity(len(self._positions))
        if not ok:
            return self._handle_at_capacity(signal, price, reason, now, snapshot, prices)

        history       = market_data.price_history.get(ticker)
        target_weight = market_data.target_weights.get(ticker) if market_data.target_weights else None
        qty = self._sizer.calculate_qty(ticker, price, pv, self._cash, history, target_weight)
        if qty <= 0:
            return self._mk("skip", ticker, 0.0, price, "insufficient cash after buffer", now, snapshot)

        ok, reason = self._risk.check_sector_concentration(
            ticker, qty * price, pv, self._positions, prices
        )
        if not ok:
            return self._mk("skip", ticker, 0.0, price, reason, now, snapshot)

        return self._mk(
            "buy", ticker, qty, price,
            f"vol-adjusted qty={qty:.0f} @ {price:.2f}",
            now, snapshot,
        )

    def _eval_sell(
        self,
        signal: Signal,
        price: float,
        now: datetime,
        snapshot: dict,
    ) -> TradeDecision:
        pos = self._positions.get(signal.ticker)
        if pos is None:
            return self._mk("skip", signal.ticker, 0.0, price, "no position to sell", now, snapshot)
        return self._mk(
            "sell", signal.ticker, pos.qty, price,
            f"sell signal @ {price:.2f}",
            now, snapshot,
        )

    def _handle_at_capacity(
        self,
        signal: Signal,
        price: float,
        capacity_reason: str,
        now: datetime,
        snapshot: dict,
        current_prices: dict[str, float],
    ) -> TradeDecision:
        action = self.config.at_capacity_action

        if action == "queue":
            self._rebalancer.enqueue(signal)
            return self._mk(
                "queued", signal.ticker, 0.0, price,
                f"{capacity_reason} → signal queued (depth={self._rebalancer.queue_depth})",
                now, snapshot,
            )

        if action == "sell_weakest":
            weakest = self._rebalancer.weakest_position(self._positions, current_prices)
            if weakest:
                pos = self._positions[weakest]
                wp = current_prices.get(weakest, pos.avg_cost)
                # Queue the incoming buy so it fires on the next periodic_rebalance
                # after the caller applies the sell and capacity reopens.
                self._rebalancer.enqueue(signal)
                return self._mk(
                    "sell", weakest, pos.qty, wp,
                    f"sell weakest ({weakest}, pnl={pos.unrealized_pnl_pct(wp):.1%}) "
                    f"to make room; {signal.ticker} queued",
                    now, self.snapshot(current_prices),
                )

        # Default: "skip"
        return self._mk("skip", signal.ticker, 0.0, price, capacity_reason, now, snapshot)

    def _open_position(self, decision: TradeDecision, fill_price: float) -> None:
        ticker = decision.ticker
        cost = fill_price * decision.qty
        self._cash -= cost
        if ticker in self._positions:
            pos = self._positions[ticker]
            total_qty = pos.qty + decision.qty
            pos.avg_cost = (
                (pos.avg_cost * pos.qty + fill_price * decision.qty) / total_qty
            )
            pos.qty = total_qty
            pos.update_peak(fill_price)
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                qty=decision.qty,
                avg_cost=fill_price,
                sector=self.config.sector_map.get(ticker, "unknown"),
                entry_date=decision.timestamp,
                peak_price=fill_price,
            )

    def _close_position(self, decision: TradeDecision, fill_price: float) -> None:
        pos = self._positions.get(decision.ticker)
        if pos is None:
            return
        self._cash += fill_price * decision.qty
        remaining = pos.qty - decision.qty
        if remaining <= 0:
            del self._positions[decision.ticker]
        else:
            pos.qty = remaining

    def _with_slippage(self, price: float, action: str) -> float:
        factor = self.config.slippage_bps / 10_000
        return price * (1 + factor) if action == "buy" else price * (1 - factor)

    def _portfolio_value(self, current_prices: dict[str, float]) -> float:
        invested = sum(
            pos.market_value(current_prices.get(t, pos.avg_cost))
            for t, pos in self._positions.items()
        )
        return self._cash + invested

    def _update_peak(self, portfolio_value: float) -> None:
        if portfolio_value > self._peak_portfolio_value:
            self._peak_portfolio_value = portfolio_value

    @staticmethod
    def _mk(
        action: str,
        ticker: str,
        qty: float,
        price: float,
        reason: str,
        timestamp: datetime,
        snapshot: dict,
    ) -> TradeDecision:
        return TradeDecision(
            action=action,
            ticker=ticker,
            qty=qty,
            price=price,
            reason=reason,
            timestamp=timestamp,
            portfolio_snapshot=snapshot,
        )
