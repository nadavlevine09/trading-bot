"""Tests for src/portfolio/. All external I/O is avoided."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.portfolio.config import PortfolioConfig
from src.portfolio.manager import PortfolioManager
from src.portfolio.models import MarketData, Signal
from src.portfolio.optimizer import MarkowitzOptimizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NOW = datetime(2025, 1, 10, 9, 30)
TICKER = "AAPL"
PRICE = 100.0
ADV = 10_000_000


def _cfg(**kwargs) -> PortfolioConfig:
    defaults = dict(
        max_position_pct=0.10,
        cash_buffer_pct=0.05,
        max_positions=5,
        volatility_window=5,
        volatility_target=0.02,
        max_sector_pct=0.30,
        sector_map={TICKER: "technology"},
        circuit_breaker_pct=0.10,
        trailing_stop_pct=0.08,
        at_capacity_action="skip",
        signal_queue_maxsize=10,
        rebalance_interval_days=7,
        min_avg_daily_volume=500_000,
        slippage_bps=10.0,
    )
    defaults.update(kwargs)
    return PortfolioConfig(**defaults)


def _pm(cfg: PortfolioConfig | None = None, cash: float = 100_000.0) -> PortfolioManager:
    return PortfolioManager(cfg or _cfg(), initial_cash=cash)


def _price_history(n: int = 30, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0, vol, n)
    closes = 100 * np.cumprod(1 + returns)
    dates = pd.date_range("2024-12-01", periods=n, freq="B")
    return pd.DataFrame({"close": closes}, index=dates)


def _buy_signal(ticker: str = TICKER) -> Signal:
    return Signal(timestamp=NOW, ticker=ticker, action="buy")


def _sell_signal(ticker: str = TICKER) -> Signal:
    return Signal(timestamp=NOW, ticker=ticker, action="sell")


def _mkt(
    price: float = PRICE,
    adv: float = ADV,
    history: pd.DataFrame | None = None,
    ticker: str = TICKER,
) -> MarketData:
    return MarketData(
        current_prices={ticker: price},
        price_history={ticker: _price_history() if history is None else history},
        avg_daily_volume={ticker: adv},
    )


def _buy_and_apply(pm: PortfolioManager, **mkt_kwargs) -> None:
    mkt = _mkt(**mkt_kwargs)
    d = pm.process_signal(_buy_signal(), mkt)
    assert d.action == "buy", f"Expected buy, got: {d.action} — {d.reason}"
    pm.apply_decision(d)


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizing:
    def test_qty_never_exceeds_max_position_pct(self):
        pm = _pm()
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "buy"
        max_qty = (pm.config.max_position_pct * pm.cash) / PRICE
        assert d.qty <= max_qty

    def test_low_volatility_produces_larger_qty_than_high_volatility(self):
        # Test the sizer directly: use a small volatility_target so vol-adjusted
        # qty is the binding constraint (below both the position cap and cash limit).
        from src.portfolio.position_sizer import PositionSizer
        cfg = _cfg(max_position_pct=0.50, cash_buffer_pct=0.0, volatility_target=0.001)
        sizer = PositionSizer(cfg)
        low_qty = sizer.calculate_qty(TICKER, PRICE, 100_000, 100_000, _price_history(vol=0.005))
        high_qty = sizer.calculate_qty(TICKER, PRICE, 100_000, 100_000, _price_history(vol=0.05))
        assert low_qty > high_qty

    def test_insufficient_cash_returns_skip(self):
        # buffer = 99 % of $1k = $990; only $10 deployable, price $100 → floor qty = 0
        pm = PortfolioManager(_cfg(cash_buffer_pct=0.99), initial_cash=1_000)
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "skip"
        assert "cash" in d.reason

    def test_qty_floored_to_whole_shares(self):
        pm = _pm()
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "buy"
        assert d.qty == int(d.qty)

    def test_short_history_falls_back_to_cap(self):
        short_hist = _price_history(n=3)  # less than volatility_window + 1
        pm = _pm()
        d = pm.process_signal(_buy_signal(), _mkt(history=short_hist))
        max_qty = (pm.config.max_position_pct * pm.cash) / PRICE
        assert d.action == "buy"
        assert d.qty <= max_qty


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def _pm_with_drawdown(self, drawdown_pct: float) -> PortfolioManager:
        pm = _pm()
        # Manually set a peak above current portfolio value
        pm._peak_portfolio_value = pm.cash / (1 - drawdown_pct)
        return pm

    def test_triggered_halts_buy(self):
        pm = self._pm_with_drawdown(0.15)  # 15 % drawdown > 10 % threshold
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "skip"
        assert "circuit breaker" in d.reason

    def test_not_triggered_below_threshold(self):
        pm = self._pm_with_drawdown(0.05)  # 5 % drawdown < 10 % threshold
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "buy"

    def test_does_not_affect_sell_signals(self):
        pm = self._pm_with_drawdown(0.20)
        # Give it a position to sell
        pm._positions["AAPL"] = _make_position()
        d = pm.process_signal(_sell_signal(), _mkt())
        assert d.action == "sell"


# ─────────────────────────────────────────────────────────────────────────────
# Trailing stops
# ─────────────────────────────────────────────────────────────────────────────

def _make_position(
    ticker: str = TICKER,
    qty: float = 100,
    avg_cost: float = PRICE,
    peak_price: float = PRICE,
) -> "Position":
    from src.portfolio.models import Position
    return Position(
        ticker=ticker,
        qty=qty,
        avg_cost=avg_cost,
        sector="technology",
        entry_date=NOW,
        peak_price=peak_price,
    )


class TestTrailingStop:
    def test_stop_triggered_when_price_falls_below_threshold(self):
        pm = _pm()
        pm._positions[TICKER] = _make_position(peak_price=100.0)
        # Price drops 10 % from peak (threshold is 8 %)
        stop_price = 89.0
        decisions = pm.check_trailing_stops({TICKER: stop_price}, now=NOW)
        assert len(decisions) == 1
        assert decisions[0].action == "sell"
        assert "trailing stop" in decisions[0].reason

    def test_stop_not_triggered_when_price_above_stop(self):
        pm = _pm()
        pm._positions[TICKER] = _make_position(peak_price=100.0)
        # Price drops only 5 % (below 8 % threshold)
        decisions = pm.check_trailing_stops({TICKER: 95.0}, now=NOW)
        assert decisions == []

    def test_peak_updates_as_price_rises(self):
        pm = _pm()
        pos = _make_position(peak_price=100.0)
        pm._positions[TICKER] = pos
        pm.check_trailing_stops({TICKER: 120.0}, now=NOW)
        assert pos.peak_price == 120.0

    def test_stop_uses_updated_peak(self):
        pm = _pm()
        pos = _make_position(peak_price=100.0)
        pm._positions[TICKER] = pos
        pm.check_trailing_stops({TICKER: 120.0}, now=NOW)  # peak → 120
        # 8 % stop from 120 = 110.4; price of 112 should be safe
        decisions = pm.check_trailing_stops({TICKER: 112.0}, now=NOW)
        assert decisions == []
        # Price of 109 (below 110.4) should trigger
        decisions = pm.check_trailing_stops({TICKER: 109.0}, now=NOW)
        assert len(decisions) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Take-profit
# ─────────────────────────────────────────────────────────────────────────────

class TestTakeProfit:
    def test_below_threshold_does_not_trigger(self):
        # 17 % gain is below the 18 % threshold → no decision.
        pm = _pm(_cfg(take_profit_pct=0.18))
        pm._positions[TICKER] = _make_position(avg_cost=100.0, peak_price=100.0)
        decisions = pm.check_trailing_stops({TICKER: 117.0}, now=NOW)
        assert decisions == []

    def test_at_threshold_triggers_sell(self):
        # Exactly at 18 % gain → fires.
        pm = _pm(_cfg(take_profit_pct=0.18))
        pm._positions[TICKER] = _make_position(avg_cost=100.0, peak_price=100.0)
        decisions = pm.check_trailing_stops({TICKER: 118.0}, now=NOW)
        assert len(decisions) == 1
        assert decisions[0].action == "sell"
        assert "take profit" in decisions[0].reason

    def test_above_threshold_triggers_sell(self):
        # 30 % gain well above threshold.
        pm = _pm(_cfg(take_profit_pct=0.18))
        pm._positions[TICKER] = _make_position(avg_cost=100.0, peak_price=100.0)
        decisions = pm.check_trailing_stops({TICKER: 130.0}, now=NOW)
        assert len(decisions) == 1
        assert decisions[0].action == "sell"
        assert "take profit" in decisions[0].reason

    def test_take_profit_priority_over_trailing_stop(self):
        # Both conditions fire simultaneously:
        #   - avg_cost=100, price=118 → +18% gain ≥ take_profit_pct (0.18)
        #   - peak_price=150, stop=150*0.92=138, price 118 ≤ 138 → trailing stop also fires
        # Only one decision should be emitted, carrying the take-profit reason.
        pm = _pm(_cfg(take_profit_pct=0.18, trailing_stop_pct=0.08))
        pm._positions[TICKER] = _make_position(avg_cost=100.0, peak_price=150.0)
        decisions = pm.check_trailing_stops({TICKER: 118.0}, now=NOW)
        assert len(decisions) == 1
        assert "take profit" in decisions[0].reason
        assert "trailing stop" not in decisions[0].reason

    def test_peak_still_updated_when_take_profit_fires(self):
        # Trailing-stop logic runs for all positions (to maintain peak watermarks)
        # even when take-profit has already claimed a ticker.  Verify peak is updated.
        pm = _pm(_cfg(take_profit_pct=0.18, trailing_stop_pct=0.08))
        pos = _make_position(avg_cost=100.0, peak_price=100.0)
        pm._positions[TICKER] = pos
        pm.check_trailing_stops({TICKER: 125.0}, now=NOW)  # 25 % gain → take profit
        assert pos.peak_price == 125.0

    def test_negative_pnl_does_not_trigger_take_profit(self):
        # Position is underwater — take-profit must never fire.
        pm = _pm(_cfg(take_profit_pct=0.18))
        pm._positions[TICKER] = _make_position(avg_cost=100.0, peak_price=100.0)
        decisions = pm.check_trailing_stops({TICKER: 80.0}, now=NOW)
        # Only trailing-stop may fire (80 < 92), not take-profit.
        take_profit_decisions = [d for d in decisions if "take profit" in d.reason]
        assert take_profit_decisions == []


# ─────────────────────────────────────────────────────────────────────────────
# Sector concentration
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorConcentration:
    def test_blocks_buy_that_would_exceed_sector_cap(self):
        cfg = _cfg(
            max_sector_pct=0.30,
            sector_map={"AAPL": "technology", "MSFT": "technology"},
            max_positions=10,
        )
        pm = PortfolioManager(cfg, initial_cash=100_000)
        # Pre-load AAPL at 27 % of portfolio → adding MSFT at ~10 % would push to 37 %
        pm._positions["AAPL"] = _make_position(ticker="AAPL", qty=270, avg_cost=100)
        mkt = MarketData(
            current_prices={"AAPL": 100.0, "MSFT": 100.0},
            price_history={"AAPL": _price_history(), "MSFT": _price_history()},
            avg_daily_volume={"AAPL": ADV, "MSFT": ADV},
        )
        d = pm.process_signal(Signal(NOW, "MSFT", "buy"), mkt)
        assert d.action == "skip"
        assert "sector" in d.reason

    def test_allows_buy_within_sector_cap(self):
        cfg = _cfg(
            max_sector_pct=0.30,
            sector_map={"AAPL": "technology", "MSFT": "technology"},
            max_positions=10,
        )
        pm = PortfolioManager(cfg, initial_cash=100_000)
        # AAPL at 15 % → MSFT at ~10 % keeps sector at 25 % ≤ 30 %
        pm._positions["AAPL"] = _make_position(ticker="AAPL", qty=150, avg_cost=100)
        mkt = MarketData(
            current_prices={"AAPL": 100.0, "MSFT": 100.0},
            price_history={"AAPL": _price_history(), "MSFT": _price_history()},
            avg_daily_volume={"AAPL": ADV, "MSFT": ADV},
        )
        d = pm.process_signal(Signal(NOW, "MSFT", "buy"), mkt)
        assert d.action == "buy"


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity filter
# ─────────────────────────────────────────────────────────────────────────────

class TestLiquidity:
    def test_low_volume_ticker_is_skipped(self):
        pm = _pm()
        d = pm.process_signal(_buy_signal(), _mkt(adv=100_000))  # below 500k min
        assert d.action == "skip"
        assert "liquidity" in d.reason

    def test_adequate_volume_passes(self):
        pm = _pm()
        d = pm.process_signal(_buy_signal(), _mkt(adv=1_000_000))
        assert d.action == "buy"


# ─────────────────────────────────────────────────────────────────────────────
# At-capacity routing
# ─────────────────────────────────────────────────────────────────────────────

def _fill_to_capacity(pm: PortfolioManager, n: int | None = None) -> None:
    cap = n or pm.config.max_positions
    for i in range(cap):
        ticker = f"T{i:02d}"
        pm._positions[ticker] = _make_position(ticker=ticker)


class TestAtCapacity:
    def test_skip_action(self):
        pm = _pm(_cfg(at_capacity_action="skip"))
        _fill_to_capacity(pm)
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "skip"
        assert "capacity" in d.reason

    def test_queue_action_stores_signal(self):
        pm = _pm(_cfg(at_capacity_action="queue"))
        _fill_to_capacity(pm)
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "queued"
        assert pm._rebalancer.queue_depth == 1

    def test_sell_weakest_returns_sell_and_queues_buy(self):
        pm = _pm(_cfg(at_capacity_action="sell_weakest", max_positions=2))
        # Add two positions with different PnL
        pm._positions["T0"] = _make_position(ticker="T0", avg_cost=80.0, peak_price=100.0)
        pm._positions["T1"] = _make_position(ticker="T1", avg_cost=110.0, peak_price=110.0)
        mkt = MarketData(
            current_prices={"T0": 100.0, "T1": 100.0, TICKER: PRICE},
            price_history={TICKER: _price_history()},
            avg_daily_volume={TICKER: ADV},
        )
        d = pm.process_signal(_buy_signal(), mkt)
        # Should sell the weakest (T1, which is underwater)
        assert d.action == "sell"
        assert d.ticker == "T1"
        # Incoming AAPL buy should be queued
        assert pm._rebalancer.queue_depth == 1

    def test_queued_signal_executes_on_periodic_rebalance(self):
        pm = _pm(_cfg(at_capacity_action="queue", max_positions=1))
        pm._positions["T0"] = _make_position(ticker="T0")
        # Queue a buy for AAPL
        pm.process_signal(_buy_signal(), _mkt())
        assert pm._rebalancer.queue_depth == 1
        # Free capacity
        del pm._positions["T0"]
        pm._cash = 100_000.0
        # Force rebalance to be due
        pm._rebalancer._last_rebalance = None
        decisions = pm.periodic_rebalance(_mkt(), now=NOW)
        assert any(d.action == "buy" and d.ticker == TICKER for d in decisions)


# ─────────────────────────────────────────────────────────────────────────────
# apply_decision state transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyDecision:
    def test_buy_reduces_cash_and_creates_position(self):
        pm = _pm()
        initial_cash = pm.cash
        _buy_and_apply(pm)
        assert pm.cash < initial_cash
        assert TICKER in pm.positions

    def test_sell_removes_position_and_restores_cash(self):
        pm = _pm()
        _buy_and_apply(pm)
        cash_after_buy = pm.cash
        sell_d = pm.process_signal(_sell_signal(), _mkt())
        pm.apply_decision(sell_d)
        assert TICKER not in pm.positions
        assert pm.cash > cash_after_buy

    def test_partial_sell_reduces_qty(self):
        from src.portfolio.models import TradeDecision
        pm = _pm()
        _buy_and_apply(pm)
        full_qty = pm.positions[TICKER].qty
        partial = TradeDecision(
            action="sell", ticker=TICKER, qty=full_qty / 2,
            price=PRICE, reason="partial sell test",
            timestamp=NOW,
        )
        pm.apply_decision(partial, fill_price=PRICE)
        assert pm.positions[TICKER].qty == full_qty / 2

    def test_slippage_applied_on_buy_when_no_fill_price(self):
        pm = _pm()
        d = pm.process_signal(_buy_signal(), _mkt())
        d.action = "buy"
        pre_cash = pm.cash
        pm.apply_decision(d)  # no fill_price → applies slippage
        expected_cost = d.price * (1 + pm.config.slippage_bps / 10_000) * d.qty
        assert abs((pre_cash - pm.cash) - expected_cost) < 0.01

    def test_actual_fill_price_used_when_provided(self):
        pm = _pm()
        d = pm.process_signal(_buy_signal(), _mkt())
        fill = 105.0
        pre_cash = pm.cash
        pm.apply_decision(d, fill_price=fill)
        assert abs((pre_cash - pm.cash) - fill * d.qty) < 0.01

    def test_skip_is_noop(self):
        pm = _pm()
        _fill_to_capacity(pm)
        cash_before = pm.cash
        d = pm.process_signal(_buy_signal(), _mkt())
        pm.apply_decision(d)
        assert pm.cash == cash_before


# ─────────────────────────────────────────────────────────────────────────────
# Periodic rebalance
# ─────────────────────────────────────────────────────────────────────────────

class TestPeriodicRebalance:
    def test_not_due_returns_empty(self):
        pm = _pm()
        pm._rebalancer.mark_done(NOW)
        decisions = pm.periodic_rebalance(_mkt(), now=NOW + timedelta(days=1))
        assert decisions == []

    def test_trims_overweight_position(self):
        pm = _pm(_cfg(max_position_pct=0.10))
        # Position worth 25 % of 100k portfolio (2.5× drift limit of 15 %)
        pm._positions[TICKER] = _make_position(qty=250, avg_cost=100.0, peak_price=100.0)
        pm._rebalancer._last_rebalance = None
        decisions = pm.periodic_rebalance(_mkt(), now=NOW)
        sell_decisions = [d for d in decisions if d.action == "sell" and d.ticker == TICKER]
        assert len(sell_decisions) == 1
        assert sell_decisions[0].qty > 0

    def test_does_not_trim_within_drift_buffer(self):
        pm = _pm(_cfg(max_position_pct=0.10))
        # Position at 12 % — within drift buffer (10 % × 1.5 = 15 %)
        pm._positions[TICKER] = _make_position(qty=120, avg_cost=100.0, peak_price=100.0)
        pm._rebalancer._last_rebalance = None
        decisions = pm.periodic_rebalance(_mkt(), now=NOW)
        sell_decisions = [d for d in decisions if d.ticker == TICKER]
        assert sell_decisions == []


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_snapshot_keys_present(self):
        pm = _pm()
        snap = pm.snapshot({TICKER: PRICE})
        for key in ("portfolio_value", "cash", "n_positions", "exposure_pct",
                    "drawdown_from_peak", "sector_exposure", "positions"):
            assert key in snap

    def test_snapshot_drawdown_reflects_peak(self):
        pm = _pm(cash=100_000)
        pm._peak_portfolio_value = 110_000
        snap = pm.snapshot({})
        assert abs(snap["drawdown_from_peak"] - (10_000 / 110_000)) < 1e-6

    def test_snapshot_sector_exposure_sums_correctly(self):
        pm = _pm(_cfg(sector_map={"AAPL": "technology", "MSFT": "technology"}))
        pm._positions["AAPL"] = _make_position(ticker="AAPL", qty=100, avg_cost=100)
        pm._positions["MSFT"] = _make_position(ticker="MSFT", qty=100, avg_cost=100)
        pm._cash = 80_000.0  # reflect that $20k was spent on the positions
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        snap = pm.snapshot(prices)
        # 20k invested in tech out of (80k cash + 10k AAPL + 10k MSFT = 100k)
        assert abs(snap["sector_exposure"]["technology"] - 0.20) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Misc edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_sell_signal_with_no_position_is_skipped(self):
        pm = _pm()
        d = pm.process_signal(_sell_signal(), _mkt())
        assert d.action == "skip"
        assert "no position" in d.reason

    def test_hold_signal_always_skipped(self):
        pm = _pm()
        sig = Signal(NOW, TICKER, "hold")
        d = pm.process_signal(sig, _mkt())
        assert d.action == "skip"

    def test_already_holding_buy_is_skipped(self):
        pm = _pm()
        _buy_and_apply(pm)
        d = pm.process_signal(_buy_signal(), _mkt())
        assert d.action == "skip"
        assert "already holding" in d.reason


# ─────────────────────────────────────────────────────────────────────────────
# MarkowitzOptimizer
# ─────────────────────────────────────────────────────────────────────────────

def _mkz_history(
    tickers: list[str],
    n_bars: int = 40,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build synthetic OHLCV DataFrames for multiple tickers (Markowitz tests)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="B")
    result = {}
    for t in tickers:
        closes = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n_bars))
        result[t] = pd.DataFrame({"close": closes, "volume": 1_000_000}, index=idx)
    return result


class TestMarkowitzOptimizer:
    TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]

    def test_weights_sum_to_one(self):
        opt = MarkowitzOptimizer()
        hist = _mkz_history(self.TICKERS)
        weights = opt.compute_weights(hist, self.TICKERS)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_tickers_present_in_output(self):
        opt = MarkowitzOptimizer()
        hist = _mkz_history(self.TICKERS)
        weights = opt.compute_weights(hist, self.TICKERS)
        assert set(weights.keys()) == set(self.TICKERS)

    def test_weights_non_negative(self):
        opt = MarkowitzOptimizer(min_weight=0.0)
        hist = _mkz_history(self.TICKERS)
        weights = opt.compute_weights(hist, self.TICKERS)
        assert all(w >= -1e-9 for w in weights.values())

    def test_weights_respect_max_weight(self):
        opt = MarkowitzOptimizer(max_weight=0.40)
        hist = _mkz_history(self.TICKERS)
        weights = opt.compute_weights(hist, self.TICKERS)
        assert all(w <= 0.40 + 1e-6 for w in weights.values())

    def test_insufficient_history_returns_equal_weights(self):
        # Only 5 bars — below min_bars=20 default → equal weights
        opt = MarkowitzOptimizer(min_bars=20)
        hist = _mkz_history(self.TICKERS, n_bars=5)
        weights = opt.compute_weights(hist, self.TICKERS)
        eq = 1.0 / len(self.TICKERS)
        for w in weights.values():
            assert w == pytest.approx(eq, abs=1e-6)

    def test_empty_tickers_returns_empty(self):
        opt = MarkowitzOptimizer()
        weights = opt.compute_weights({}, [])
        assert weights == {}

    def test_single_ticker_gets_weight_one(self):
        opt = MarkowitzOptimizer()
        hist = _mkz_history(["AAPL"])
        weights = opt.compute_weights(hist, ["AAPL"])
        assert weights["AAPL"] == pytest.approx(1.0, abs=1e-6)

    def test_invalid_max_weight_raises(self):
        with pytest.raises(ValueError, match="max_weight"):
            MarkowitzOptimizer(max_weight=0.0)
        with pytest.raises(ValueError, match="max_weight"):
            MarkowitzOptimizer(max_weight=1.5)

    def test_invalid_min_weight_raises(self):
        with pytest.raises(ValueError, match="min_weight"):
            MarkowitzOptimizer(min_weight=-0.1)

    def test_target_weights_flow_into_position_sizer(self):
        # When MarketData carries target_weights, a buy should use those weights
        cfg = _cfg(max_position_pct=0.10)
        pm = PortfolioManager(cfg, initial_cash=100_000.0)
        # Target weight = 0.20 (double the config cap)
        mkt = MarketData(
            current_prices={"AAPL": PRICE},
            price_history={},
            avg_daily_volume={"AAPL": ADV},
            target_weights={"AAPL": 0.20},
        )
        d = pm.process_signal(Signal(NOW, "AAPL", "buy"), mkt)
        # Qty should reflect 20 % of $100k / $100 = 200 shares
        # (minus cash buffer: deployable = 95k, cap = 20k → 200 shares)
        assert d.action == "buy"
        assert d.qty == pytest.approx(200.0, abs=5.0)  # 0.20×100k/100 = 200

    def test_empty_target_weights_uses_config_cap(self):
        cfg = _cfg(max_position_pct=0.10)
        pm = PortfolioManager(cfg, initial_cash=100_000.0)
        mkt = MarketData(
            current_prices={"AAPL": PRICE},
            price_history={},
            avg_daily_volume={"AAPL": ADV},
            target_weights={},  # empty → use config cap
        )
        d = pm.process_signal(Signal(NOW, "AAPL", "buy"), mkt)
        assert d.action == "buy"
        # Cap = 10 % × 100k / $100 = 100 shares (minus cash buffer cap: 95k cap)
        assert d.qty == pytest.approx(95.0, abs=5.0)
