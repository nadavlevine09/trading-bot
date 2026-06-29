"""Alpaca paper-trading execution."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.models import Order, Position
from alpaca.trading.requests import MarketOrderRequest

from src.portfolio import MarketData, PortfolioConfig, PortfolioManager, Signal

_log = logging.getLogger(__name__)

_TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
}
_FILL_POLL_INTERVAL = 0.5   # seconds between status checks
_FILL_TIMEOUT = 30.0        # seconds before giving up on a fill


class AlpacaTrader:
    def __init__(self, portfolio_config: PortfolioConfig | None = None) -> None:
        self.client = TradingClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
        self.portfolio = PortfolioManager(
            config=portfolio_config or PortfolioConfig(),
            initial_cash=self._fetch_cash(),
        )

    def evaluate_and_trade(self, signal: Signal, market_data: MarketData) -> None:
        """
        Run the signal through PortfolioManager, then execute approved decisions.

        Workflow:
          1. Check trailing stops for all open positions.
          2. Evaluate the incoming signal.
          3. For each resulting 'buy' or 'sell' decision, submit a market order
             to Alpaca and call portfolio.apply_decision(d, fill_price=actual_fill).
          Skipped / queued decisions are passed to apply_decision for logging only.
        """
        now = datetime.utcnow()

        # Trailing-stop sells first so they free cash before any new buy is attempted.
        decisions = self.portfolio.check_trailing_stops(market_data.current_prices, now=now)
        decisions.append(self.portfolio.process_signal(signal, market_data))

        for decision in decisions:
            if decision.action not in ("buy", "sell"):
                self.portfolio.apply_decision(decision)
                continue
            try:
                fill_price = self.submit_order(decision.ticker, decision.qty, decision.action)
                self.portfolio.apply_decision(decision, fill_price=fill_price)
            except Exception as exc:
                _log.error(
                    "Order failed — %s %s qty=%.0f: %s",
                    decision.action, decision.ticker, decision.qty, exc,
                )

    def submit_order(self, ticker: str, qty: float, side: str) -> float:
        """
        Submit a market DAY order and block until filled.
        Returns the actual average fill price.
        Raises TimeoutError if the order is not filled within _FILL_TIMEOUT seconds.
        Raises RuntimeError if the order is rejected, cancelled, or expired.
        """
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order: Order = self.client.submit_order(req)
        _log.info("Submitted %s %s qty=%.0f order_id=%s", side, ticker, qty, order.id)

        deadline = time.monotonic() + _FILL_TIMEOUT
        while order.status not in _TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Order {order.id} ({side} {ticker} qty={qty}) "
                    f"not filled within {_FILL_TIMEOUT}s (status={order.status})"
                )
            time.sleep(_FILL_POLL_INTERVAL)
            order = self.client.get_order_by_id(order.id)

        if order.status != OrderStatus.FILLED:
            raise RuntimeError(
                f"Order {order.id} ({side} {ticker} qty={qty}) "
                f"ended in non-fill status: {order.status}"
            )

        fill_price = float(order.filled_avg_price)
        _log.info("Filled %s %s qty=%.0f @ %.4f", side, ticker, qty, fill_price)
        return fill_price

    def get_positions(self) -> list[Position]:
        """Return all open positions from the Alpaca paper account."""
        return self.client.get_all_positions()

    def _fetch_cash(self) -> float:
        """Return available cash balance from the Alpaca paper account."""
        return float(self.client.get_account().cash)
