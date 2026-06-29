"""Backtest engine: replay signals against historical price data.

CSV Schema  (<price_data_path>/<TICKER>.csv)
--------------------------------------------
Columns (header required): date,open,high,low,close,volume
  date   – YYYY-MM-DD, one row per trading day, ascending
  open / high / low / close – float, USD
  volume – int, shares traded

One CSV file per ticker.  Fills execute at each bar's closing price
adjusted by PortfolioConfig.slippage_bps (default 10 bps one-way).
Signals timestamped on a day that has no matching price file entry are
skipped with a debug-level log message.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio import MarketData, MarkowitzOptimizer, PortfolioConfig, PortfolioManager, Signal

_log = logging.getLogger(__name__)

_METRIC_KEYS = frozenset({
    "total_return", "sharpe_ratio", "max_drawdown",
    "win_rate", "n_trades", "n_days", "final_value", "initial_cash",
})


class BacktestRunner:
    def __init__(
        self,
        price_data_path: Path,
        portfolio_config: PortfolioConfig | None = None,
        initial_cash: float = 100_000.0,
        optimizer: MarkowitzOptimizer | None = None,
    ) -> None:
        self.price_data_path = price_data_path
        self._initial_cash   = initial_cash
        self._config         = portfolio_config or PortfolioConfig()
        self._optimizer      = optimizer
        # Exposed as an attribute so callers can inspect post-run state.
        self.portfolio = PortfolioManager(
            config=self._config,
            initial_cash=initial_cash,
        )
        # Populated by run(); cleared at the start of each call.
        self.decision_log: list[dict] = []       # one entry per executed trade
        self.portfolio_history: list[tuple[str, float]] = []  # (date_str, value)

    def run(self, signals: pd.DataFrame) -> dict:
        """
        Replay signals against OHLCV data in price_data_path.

        Parameters
        ----------
        signals:
            DataFrame with columns ``[timestamp, ticker, signal]`` where
            ``signal ∈ {buy, sell, hold}``.  An ``action`` column is
            accepted as an alias for ``signal``.  Timestamps are matched
            to price bars by calendar date; time-of-day is ignored.

        Returns
        -------
        dict with keys: total_return, sharpe_ratio, max_drawdown,
        win_rate, n_trades, n_days, final_value, initial_cash.

        Metrics
        -------
        total_return  – (final_value − initial_cash) / initial_cash
        sharpe_ratio  – annualised (√252 × mean/std of daily returns);
                        risk-free rate = 0; 0.0 when fewer than 2 bars
        max_drawdown  – peak-to-trough magnitude (positive, e.g. 0.15
                        means the portfolio fell 15% from its peak)
        win_rate      – fraction of closed positions with positive PnL;
                        0.0 when no positions were closed
        n_trades      – total executed fills (buys + sells combined)
        """
        # Reset portfolio and logs so run() is safe to call more than once.
        self.portfolio = PortfolioManager(
            config=self._config,
            initial_cash=self._initial_cash,
        )
        self.decision_log      = []
        self.portfolio_history = []
        # Cache last computed weights to avoid redundant optimisations per bar.
        self._last_weights: dict[str, float] = {}

        signals = _normalise_signals(signals)
        if signals.empty:
            return _zero_metrics(self._initial_cash)

        tickers: list[str] = signals["ticker"].unique().tolist()

        # ── Load OHLCV data ────────────────────────────────────────────────────
        ohlcv: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = _load_ticker_csv(self.price_data_path, ticker)
            if df is not None:
                ohlcv[ticker] = df
            else:
                _log.warning(
                    "No price data for %s in %s — signals skipped.",
                    ticker, self.price_data_path,
                )

        if not ohlcv:
            _log.warning("No OHLCV data loaded; returning zero metrics.")
            return _zero_metrics(self._initial_cash)

        # ── Trading calendar: union of all bar timestamps ──────────────────────
        all_timestamps: list[pd.Timestamp] = sorted(
            set().union(*[set(df.index) for df in ohlcv.values()])
        )

        # ── Index signals by normalised date (midnight) ────────────────────────
        sig_by_date: dict[pd.Timestamp, list[Signal]] = {}
        for _, row in signals.iterrows():
            day = pd.Timestamp(row["timestamp"]).normalize()
            sig_by_date.setdefault(day, []).append(
                Signal(
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    ticker=str(row["ticker"]),
                    action=str(row["signal"]),
                )
            )

        # ── Simulation loop ────────────────────────────────────────────────────
        adv_window = self._config.volatility_window
        portfolio_values: list[float] = []
        closed_trades: list[tuple[float, float]] = []   # (avg_cost, exit_fill)
        n_executed = 0

        for bar_ts in all_timestamps:
            # Current close prices for tickers with a bar today.
            current_prices: dict[str, float] = {
                ticker: float(df.at[bar_ts, "close"])
                for ticker, df in ohlcv.items()
                if bar_ts in df.index
            }
            if not current_prices:
                continue

            # Price history and ADV up to and including today.
            price_history: dict[str, pd.DataFrame] = {}
            avg_daily_volume: dict[str, float] = {}
            for ticker, df in ohlcv.items():
                hist = df[df.index <= bar_ts]
                if not hist.empty:
                    price_history[ticker] = hist
                    avg_daily_volume[ticker] = float(
                        hist["volume"].tail(adv_window).mean()
                    )

            # Recompute Markowitz weights weekly (every 5 bars) to balance
            # accuracy vs. computational cost.
            if self._optimizer is not None:
                if len(portfolio_values) % 5 == 0:
                    self._last_weights = self._optimizer.compute_weights(
                        price_history=price_history,
                        tickers=list(ohlcv.keys()),
                        as_of=bar_ts,
                    )

            market_data = MarketData(
                current_prices=current_prices,
                price_history=price_history,
                avg_daily_volume=avg_daily_volume,
                target_weights=self._last_weights,
            )
            bar_dt: datetime = bar_ts.to_pydatetime()

            # 1. Trailing stops
            for decision in self.portfolio.check_trailing_stops(
                current_prices, now=bar_dt
            ):
                n_executed, closed_trades = self._apply(
                    decision, n_executed, closed_trades
                )

            # 2. Signals for this bar
            for signal in sig_by_date.get(bar_ts, []):
                if signal.ticker not in current_prices:
                    _log.debug(
                        "No price for %s on %s — signal skipped.",
                        signal.ticker, bar_ts.date(),
                    )
                    continue
                decision = self.portfolio.process_signal(signal, market_data)
                n_executed, closed_trades = self._apply(
                    decision, n_executed, closed_trades
                )

            # 3. Periodic rebalance
            for decision in self.portfolio.periodic_rebalance(
                market_data, now=bar_dt
            ):
                n_executed, closed_trades = self._apply(
                    decision, n_executed, closed_trades
                )

            # End-of-day portfolio value (marks open positions to close).
            day_value = self.portfolio._portfolio_value(current_prices)
            portfolio_values.append(day_value)
            self.portfolio_history.append((bar_ts.date().isoformat(), day_value))

        return _compute_metrics(
            portfolio_values=portfolio_values,
            initial_cash=self._initial_cash,
            closed_trades=closed_trades,
            n_executed=n_executed,
        )

    def _apply(
        self,
        decision,
        n_executed: int,
        closed_trades: list[tuple[float, float]],
    ) -> tuple[int, list[tuple[float, float]]]:
        """
        Apply a decision and capture trade metadata for win-rate calculation.

        Must be called BEFORE apply_decision() so the position still exists
        when we read avg_cost for sell decisions.
        """
        if decision.action == "sell":
            pos = self.portfolio.positions.get(decision.ticker)
            if pos is not None:
                slip = self._config.slippage_bps / 10_000
                exit_fill = decision.price * (1.0 - slip)
                closed_trades.append((pos.avg_cost, exit_fill))

        if decision.action in ("buy", "sell"):
            n_executed += 1
            self.decision_log.append({
                "date":   decision.timestamp.date().isoformat(),
                "ticker": decision.ticker,
                "action": decision.action,
                "price":  decision.price,
                "qty":    decision.qty,
                "reason": decision.reason,
            })

        self.portfolio.apply_decision(decision)
        return n_executed, closed_trades


# ── Module-level helpers ───────────────────────────────────────────────────────

def _load_ticker_csv(directory: Path, ticker: str) -> pd.DataFrame | None:
    """
    Look for <ticker>.csv (case-insensitive) inside directory.
    Returns a DataFrame with a DatetimeIndex sorted ascending, or None.
    """
    for name in (ticker, ticker.upper(), ticker.lower()):
        path = directory / f"{name}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["date"], index_col="date")
                df.sort_index(inplace=True)
                for col in ("open", "high", "low", "close", "volume"):
                    if col not in df.columns:
                        raise ValueError(
                            f"{path.name}: missing required column '{col}'"
                        )
                return df
            except Exception as exc:
                _log.error("Failed to load %s: %s", path, exc)
                return None
    return None


def _normalise_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept 'action' as an alias for 'signal'.
    Validates required columns; raises ValueError if any are absent.
    """
    if df.empty:
        return df
    if "signal" not in df.columns and "action" in df.columns:
        df = df.rename(columns={"action": "signal"})
    required = {"timestamp", "ticker", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"signals DataFrame is missing required column(s): {sorted(missing)}"
        )
    return df.reset_index(drop=True)


def _compute_metrics(
    portfolio_values: list[float],
    initial_cash: float,
    closed_trades: list[tuple[float, float]],
    n_executed: int,
) -> dict:
    if not portfolio_values:
        return _zero_metrics(initial_cash)

    final_value = portfolio_values[-1]
    total_return = (final_value - initial_cash) / initial_cash if initial_cash else 0.0

    # Sharpe ratio (annualised, rf = 0)
    pv = pd.Series(portfolio_values, dtype=float)
    daily_returns = pv.pct_change().dropna()
    sharpe = 0.0
    if len(daily_returns) >= 2:
        std = float(daily_returns.std())
        if std > 0.0 and not math.isnan(std):
            sharpe = float(daily_returns.mean() / std * math.sqrt(252))

    # Max drawdown (peak-to-trough, returned as a positive magnitude)
    arr = pv.to_numpy()
    peak = np.maximum.accumulate(arr)
    drawdowns = np.where(peak > 0, (arr - peak) / peak, 0.0)
    max_drawdown = float(-drawdowns.min()) if len(drawdowns) else 0.0

    # Win rate across closed (sell-side) fills
    win_rate = 0.0
    if closed_trades:
        wins = sum(1 for avg_cost, exit_fill in closed_trades if exit_fill > avg_cost)
        win_rate = wins / len(closed_trades)

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_trades": n_executed,
        "n_days": len(portfolio_values),
        "final_value": final_value,
        "initial_cash": initial_cash,
    }


def _zero_metrics(initial_cash: float) -> dict:
    return {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "n_trades": 0,
        "n_days": 0,
        "final_value": initial_cash,
        "initial_cash": initial_cash,
    }
