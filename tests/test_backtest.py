"""Tests for src/backtest/runner.py.

All tests use synthetic OHLCV data written to a tmp_path fixture directory.
PortfolioConfig is tuned to eliminate side-effects (no trailing stop, no
circuit breaker, no liquidity gate, no slippage) so metric values are
directly predictable from the test's price series.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.runner import (
    BacktestRunner,
    _compute_metrics,
    _load_ticker_csv,
    _normalise_signals,
    _zero_metrics,
)
from src.portfolio import PortfolioConfig

# ── Shared constants ───────────────────────────────────────────────────────────

EXPECTED_METRIC_KEYS = {
    "total_return", "sharpe_ratio", "max_drawdown",
    "win_rate", "n_trades", "n_days", "final_value", "initial_cash",
}

# 5 consecutive business days starting 2024-01-02.
_DATES = pd.date_range("2024-01-02", periods=5, freq="B")

INITIAL_CASH = 10_000.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_csv(
    directory: Path,
    ticker: str,
    prices: list[float],
    volume: int = 1_000_000,
) -> None:
    """Write a OHLCV CSV file for one ticker using the global _DATES range."""
    rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": p,
            "high": round(p * 1.01, 4),
            "low": round(p * 0.99, 4),
            "close": p,
            "volume": volume,
        }
        for d, p in zip(_DATES, prices)
    ]
    pd.DataFrame(rows).to_csv(directory / f"{ticker}.csv", index=False)


def _signals(*rows) -> pd.DataFrame:
    """Build a signals DataFrame from (timestamp, ticker, signal) tuples."""
    return pd.DataFrame(rows, columns=["timestamp", "ticker", "signal"])


def _cfg(**overrides) -> PortfolioConfig:
    """
    Test-friendly PortfolioConfig: no liquidity gate, no slippage,
    effectively disabled trailing stop and circuit breaker.
    max_position_pct=0.5 so a single position can absorb half the portfolio.
    """
    defaults = dict(
        min_avg_daily_volume=0,
        slippage_bps=0.0,
        volatility_window=2,
        max_position_pct=0.50,
        cash_buffer_pct=0.0,
        max_positions=20,
        max_sector_pct=1.0,
        rebalance_interval_days=999,
        circuit_breaker_pct=1.0,
        trailing_stop_pct=1.0,
        take_profit_pct=1.0,
    )
    defaults.update(overrides)
    return PortfolioConfig(**defaults)


def _runner(tmp_path: Path, **cfg_overrides) -> BacktestRunner:
    return BacktestRunner(
        price_data_path=tmp_path,
        portfolio_config=_cfg(**cfg_overrides),
        initial_cash=INITIAL_CASH,
    )


# ── BacktestRunner.run() — integration ────────────────────────────────────────

class TestRunReturnsMetrics:
    def test_returns_all_expected_keys(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100, 100, 100, 100, 100])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert set(result.keys()) == EXPECTED_METRIC_KEYS

    def test_initial_cash_echoed_in_result(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        result = _runner(tmp_path).run(_signals((_DATES[0], "AAPL", "hold")))
        assert result["initial_cash"] == INITIAL_CASH

    def test_n_days_equals_bars_in_price_file(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["n_days"] == 5


class TestEmptyAndMissingData:
    def test_empty_signals_returns_zero_metrics(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        result = _runner(tmp_path).run(pd.DataFrame())
        assert result == _zero_metrics(INITIAL_CASH)

    def test_missing_price_file_returns_zero_metrics(self, tmp_path):
        # No CSV written — directory is empty.
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result == _zero_metrics(INITIAL_CASH)

    def test_missing_columns_raises(self, tmp_path):
        # Write a broken CSV (no 'volume' column).
        bad = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in _DATES],
            "close": [100.0] * 5,
        })
        bad.to_csv(tmp_path / "AAPL.csv", index=False)
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        # No price data loads → zero metrics (error logged, not raised).
        result = _runner(tmp_path).run(sigs)
        assert result == _zero_metrics(INITIAL_CASH)

    def test_signal_without_matching_price_date_is_skipped(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        # Signal dated far in the future — no matching bar.
        future = pd.Timestamp("2030-01-02")
        sigs = _signals((future, "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["n_trades"] == 0


class TestTotalReturn:
    def test_hold_signals_produce_zero_return(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = _signals(
            (_DATES[0], "AAPL", "hold"),
            (_DATES[1], "AAPL", "hold"),
        )
        result = _runner(tmp_path).run(sigs)
        assert result["total_return"] == pytest.approx(0.0)

    def test_buy_and_hold_rising_price_gives_positive_return(self, tmp_path):
        # Buy on day 0 at 100; price rises to 110 by day 4.
        # Position qty = floor(max_position_pct * cash / price)
        #             = floor(0.5 * 10_000 / 100) = 50 shares.
        # End-of-run portfolio: 5_000 cash + 50 * 110 = 10_500.
        # Expected total_return = 500 / 10_000 = 5%.
        _write_csv(tmp_path, "AAPL", [100, 102, 104, 106, 110])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["total_return"] == pytest.approx(0.05)
        assert result["final_value"] == pytest.approx(10_500.0)

    def test_buy_and_hold_falling_price_gives_negative_return(self, tmp_path):
        # Buy at 100; price falls to 90 by day 4 → loss of 5%.
        _write_csv(tmp_path, "AAPL", [100, 98, 96, 94, 90])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["total_return"] == pytest.approx(-0.05)

    def test_run_is_idempotent(self, tmp_path):
        """Calling run() twice must produce identical results."""
        _write_csv(tmp_path, "AAPL", [100, 105, 110, 115, 120])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        runner = _runner(tmp_path)
        r1 = runner.run(sigs)
        r2 = runner.run(sigs)
        assert r1 == r2


class TestNTrades:
    def test_buy_signal_counts_as_one_trade(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["n_trades"] == 1

    def test_buy_then_sell_counts_as_two_trades(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100, 100, 120, 120, 120])
        sigs = _signals(
            (_DATES[0], "AAPL", "buy"),
            (_DATES[2], "AAPL", "sell"),
        )
        result = _runner(tmp_path).run(sigs)
        assert result["n_trades"] == 2

    def test_hold_signal_does_not_count_as_trade(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = _signals(
            (_DATES[0], "AAPL", "hold"),
            (_DATES[1], "AAPL", "hold"),
        )
        result = _runner(tmp_path).run(sigs)
        assert result["n_trades"] == 0

    def test_sell_without_position_is_skipped(self, tmp_path):
        # Sell signal with no open position → skip, n_trades stays 0.
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = _signals((_DATES[0], "AAPL", "sell"))
        result = _runner(tmp_path).run(sigs)
        assert result["n_trades"] == 0


class TestWinRate:
    def test_profitable_round_trip_win_rate_is_one(self, tmp_path):
        # Buy at 100 (day 0), sell at 120 (day 2).
        _write_csv(tmp_path, "AAPL", [100, 110, 120, 120, 120])
        sigs = _signals(
            (_DATES[0], "AAPL", "buy"),
            (_DATES[2], "AAPL", "sell"),
        )
        result = _runner(tmp_path).run(sigs)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_losing_round_trip_win_rate_is_zero(self, tmp_path):
        # Buy at 100, sell at 80.
        _write_csv(tmp_path, "AAPL", [100, 90, 80, 80, 80])
        sigs = _signals(
            (_DATES[0], "AAPL", "buy"),
            (_DATES[2], "AAPL", "sell"),
        )
        result = _runner(tmp_path).run(sigs)
        assert result["win_rate"] == pytest.approx(0.0)

    def test_no_closed_trades_win_rate_is_zero(self, tmp_path):
        # Buy but never sell — open position at end, no closed trades.
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["win_rate"] == pytest.approx(0.0)

    def test_mixed_wins_and_losses(self, tmp_path):
        # Two tickers: AAPL profitable, MSFT unprofitable.
        _write_csv(tmp_path, "AAPL", [100, 100, 120, 120, 120])
        _write_csv(tmp_path, "MSFT", [200, 200, 180, 180, 180])
        sigs = _signals(
            (_DATES[0], "AAPL", "buy"),
            (_DATES[0], "MSFT", "buy"),
            (_DATES[2], "AAPL", "sell"),
            (_DATES[2], "MSFT", "sell"),
        )
        result = _runner(tmp_path).run(sigs)
        assert result["win_rate"] == pytest.approx(0.5)


class TestMaxDrawdown:
    def test_monotone_rising_portfolio_has_zero_drawdown(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100, 105, 110, 115, 120])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_price_drop_produces_positive_drawdown(self, tmp_path):
        # Buy at 100; price rises to 120 then falls to 90.
        # Peak portfolio ≈ 5_000 + 50*120 = 11_000.
        # Trough ≈ 5_000 + 50*90 = 9_500.
        # Drawdown ≈ (11_000 - 9_500) / 11_000 ≈ 0.1364.
        _write_csv(tmp_path, "AAPL", [100, 110, 120, 100, 90])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["max_drawdown"] > 0.0
        assert result["max_drawdown"] < 1.0

    def test_no_trades_no_drawdown(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100, 90, 80, 70, 60])
        sigs = _signals((_DATES[0], "AAPL", "hold"))
        result = _runner(tmp_path).run(sigs)
        # Cash-only portfolio never changes → no drawdown.
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


class TestSharpeRatio:
    def test_constant_portfolio_value_gives_zero_sharpe(self, tmp_path):
        # Cash only; no trades → portfolio value is flat → std=0 → sharpe=0.
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = _signals((_DATES[0], "AAPL", "hold"))
        result = _runner(tmp_path).run(sigs)
        assert result["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)

    def test_sharpe_sign_positive_for_rising_portfolio(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100, 110, 121, 133, 146])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["sharpe_ratio"] > 0.0

    def test_sharpe_sign_negative_for_falling_portfolio(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100, 90, 81, 73, 66])
        sigs = _signals((_DATES[0], "AAPL", "buy"))
        result = _runner(tmp_path).run(sigs)
        assert result["sharpe_ratio"] < 0.0


class TestSignalColumnAlias:
    def test_action_column_accepted_as_alias(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        sigs = pd.DataFrame(
            [(_DATES[0], "AAPL", "buy")],
            columns=["timestamp", "ticker", "action"],
        )
        result = _runner(tmp_path).run(sigs)
        assert result["n_trades"] == 1

    def test_missing_required_column_raises(self, tmp_path):
        bad = pd.DataFrame([("AAPL", "buy")], columns=["ticker", "signal"])
        with pytest.raises(ValueError, match="timestamp"):
            _normalise_signals(bad)


class TestMultiTicker:
    def test_two_tickers_each_get_their_own_signals(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100, 105, 110, 115, 120])
        _write_csv(tmp_path, "MSFT", [200, 200, 200, 200, 200])
        sigs = _signals(
            (_DATES[0], "AAPL", "buy"),
            (_DATES[0], "MSFT", "buy"),
        )
        result = _runner(tmp_path).run(sigs)
        # Both buys execute → 2 trades.
        assert result["n_trades"] == 2
        # AAPL rises → positive overall return.
        assert result["total_return"] > 0.0


# ── Unit tests for module-level helpers ───────────────────────────────────────

class TestLoadTickerCsv:
    def test_loads_valid_csv(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        df = _load_ticker_csv(tmp_path, "AAPL")
        assert df is not None
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 5

    def test_index_is_datetime(self, tmp_path):
        _write_csv(tmp_path, "AAPL", [100] * 5)
        df = _load_ticker_csv(tmp_path, "AAPL")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_is_sorted_ascending(self, tmp_path):
        # Write rows in reverse order; loader should sort them.
        rows = [
            {"date": d.strftime("%Y-%m-%d"), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for d in reversed(_DATES)
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "AAPL.csv", index=False)
        df = _load_ticker_csv(tmp_path, "AAPL")
        assert df.index.is_monotonic_increasing

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _load_ticker_csv(tmp_path, "AAPL") is None

    def test_case_insensitive_filename(self, tmp_path):
        _write_csv(tmp_path, "aapl", [100] * 5)   # lowercase on disk
        df = _load_ticker_csv(tmp_path, "AAPL")   # lookup with uppercase
        assert df is not None


class TestNormaliseSignals:
    def test_renames_action_to_signal(self):
        df = pd.DataFrame([("2024-01-02", "AAPL", "buy")],
                          columns=["timestamp", "ticker", "action"])
        out = _normalise_signals(df)
        assert "signal" in out.columns
        assert "action" not in out.columns

    def test_raises_on_missing_timestamp(self):
        df = pd.DataFrame([("AAPL", "buy")], columns=["ticker", "signal"])
        with pytest.raises(ValueError, match="timestamp"):
            _normalise_signals(df)

    def test_raises_on_missing_ticker(self):
        df = pd.DataFrame([("2024-01-02", "buy")], columns=["timestamp", "signal"])
        with pytest.raises(ValueError, match="ticker"):
            _normalise_signals(df)

    def test_empty_dataframe_returned_unchanged(self):
        out = _normalise_signals(pd.DataFrame())
        assert out.empty


class TestComputeMetrics:
    def test_total_return_calculation(self):
        m = _compute_metrics([10_000, 10_500], 10_000, [], 0)
        assert m["total_return"] == pytest.approx(0.05)

    def test_zero_metrics_on_empty_values(self):
        m = _compute_metrics([], 10_000, [], 0)
        assert m == _zero_metrics(10_000)

    def test_max_drawdown_peak_trough(self):
        # Values: 100 → 120 → 90.  Peak=120, trough=90 → dd=(120-90)/120=25%.
        m = _compute_metrics([100, 120, 90], 100, [], 0)
        assert m["max_drawdown"] == pytest.approx(0.25)

    def test_max_drawdown_monotone_never_negative(self):
        m = _compute_metrics([100, 110, 120, 130], 100, [], 0)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_win_rate_all_winners(self):
        trades = [(100.0, 120.0), (200.0, 210.0)]  # both profitable
        m = _compute_metrics([10_000, 10_000], 10_000, trades, 2)
        assert m["win_rate"] == pytest.approx(1.0)

    def test_win_rate_no_trades(self):
        m = _compute_metrics([10_000, 10_000], 10_000, [], 0)
        assert m["win_rate"] == pytest.approx(0.0)

    def test_sharpe_positive_for_rising_series(self):
        values = [1000 * (1.01 ** i) for i in range(50)]
        m = _compute_metrics(values, 1000, [], 0)
        assert m["sharpe_ratio"] > 0.0

    def test_sharpe_zero_for_single_value(self):
        m = _compute_metrics([10_000], 10_000, [], 0)
        assert m["sharpe_ratio"] == pytest.approx(0.0)

    def test_n_days_reflects_portfolio_values_length(self):
        m = _compute_metrics([10_000] * 7, 10_000, [], 0)
        assert m["n_days"] == 7
