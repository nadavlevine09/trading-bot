"""Tests for src/strategy/signals.py."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from src.ingestion.alpha_vantage import EarningsCache
from src.portfolio.models import Signal
from src.strategy.signals import SignalGenerator, _date_to_datetime


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

AS_OF = datetime(2025, 1, 10, 9, 30)  # simulated market-open timestamp


def _gen(**kwargs) -> SignalGenerator:
    defaults = dict(buy_threshold=0.3, sell_threshold=-0.3, min_headlines=3)
    defaults.update(kwargs)
    return SignalGenerator(**defaults)


def _df(*rows: dict) -> pd.DataFrame:
    """Build a daily_scores DataFrame matching aggregate_daily() output."""
    base = {
        "ticker": "AAPL",
        "date": date(2025, 1, 10),
        "sentiment_score": 0.0,
        "n_headlines": 5,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
# Construction & validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_thresholds(self):
        gen = SignalGenerator()
        assert gen.buy_threshold  == pytest.approx(0.3)
        assert gen.sell_threshold == pytest.approx(-0.3)
        assert gen.min_headlines  == 8

    def test_custom_thresholds_stored(self):
        gen = _gen(buy_threshold=0.5, sell_threshold=-0.2, min_headlines=5)
        assert gen.buy_threshold  == pytest.approx(0.5)
        assert gen.sell_threshold == pytest.approx(-0.2)
        assert gen.min_headlines  == 5

    def test_buy_threshold_must_exceed_sell_threshold(self):
        with pytest.raises(ValueError, match="buy_threshold"):
            SignalGenerator(buy_threshold=0.1, sell_threshold=0.2)

    def test_equal_thresholds_raise(self):
        with pytest.raises(ValueError):
            SignalGenerator(buy_threshold=0.3, sell_threshold=0.3)

    def test_negative_min_headlines_raises(self):
        with pytest.raises(ValueError, match="min_headlines"):
            SignalGenerator(min_headlines=-1)

    def test_zero_min_headlines_is_valid(self):
        gen = SignalGenerator(min_headlines=0)
        assert gen.min_headlines == 0


# ─────────────────────────────────────────────────────────────────────────────
# classify()
# ─────────────────────────────────────────────────────────────────────────────

class TestClassify:
    def test_buy_above_threshold(self):
        assert _gen().classify(0.5, 5) == "buy"

    def test_sell_below_threshold(self):
        assert _gen().classify(-0.5, 5) == "sell"

    def test_hold_in_dead_zone(self):
        assert _gen().classify(0.1, 5) == "hold"

    def test_hold_at_exactly_buy_threshold(self):
        # Boundary is strict: score == threshold → hold.
        assert _gen().classify(0.3, 5) == "hold"

    def test_hold_at_exactly_sell_threshold(self):
        assert _gen().classify(-0.3, 5) == "hold"

    def test_hold_overrides_strong_buy_when_headlines_below_min(self):
        assert _gen(min_headlines=3).classify(0.9, 2) == "hold"

    def test_hold_on_zero_headlines(self):
        assert _gen().classify(0.9, 0) == "hold"

    def test_buy_fires_at_exactly_min_headlines(self):
        # n == min_headlines passes the thin-data guard.
        assert _gen(min_headlines=3).classify(0.5, 3) == "buy"

    def test_sell_fires_at_exactly_min_headlines(self):
        assert _gen(min_headlines=3).classify(-0.5, 3) == "sell"

    def test_zero_min_headlines_acts_on_any_article_count(self):
        gen = _gen(min_headlines=0)
        assert gen.classify(0.5, 0) == "buy"
        assert gen.classify(-0.5, 0) == "sell"

    def test_asymmetric_thresholds(self):
        gen = _gen(buy_threshold=0.5, sell_threshold=-0.1)
        assert gen.classify(0.4, 5) == "hold"   # between thresholds
        assert gen.classify(-0.2, 5) == "sell"
        assert gen.classify(0.6, 5) == "buy"

    def test_extreme_positive_score(self):
        assert _gen().classify(1.0, 10) == "buy"

    def test_extreme_negative_score(self):
        assert _gen().classify(-1.0, 10) == "sell"


# ─────────────────────────────────────────────────────────────────────────────
# generate()
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_returns_list_of_signal_objects(self):
        result = _gen().generate(_df({"sentiment_score": 0.5}), as_of=AS_OF)
        assert isinstance(result, list)
        assert all(isinstance(s, Signal) for s in result)

    def test_one_signal_per_row(self):
        df = _df(
            {"ticker": "AAPL", "sentiment_score": 0.5},
            {"ticker": "MSFT", "sentiment_score": -0.4},
        )
        assert len(_gen().generate(df, as_of=AS_OF)) == 2

    def test_empty_dataframe_returns_empty_list(self):
        empty = pd.DataFrame(columns=["ticker", "date", "sentiment_score", "n_headlines"])
        assert _gen().generate(empty) == []

    # ── Signal fields ──────────────────────────────────────────────────────────

    def test_buy_action(self):
        result = _gen().generate(_df({"sentiment_score": 0.5}), as_of=AS_OF)
        assert result[0].action == "buy"

    def test_sell_action(self):
        result = _gen().generate(_df({"sentiment_score": -0.5}), as_of=AS_OF)
        assert result[0].action == "sell"

    def test_hold_action_thin_data(self):
        result = _gen(min_headlines=5).generate(
            _df({"sentiment_score": 0.9, "n_headlines": 2}), as_of=AS_OF
        )
        assert result[0].action == "hold"

    def test_hold_signals_included_not_dropped(self):
        # hold must be emitted (not silently filtered) for full audit visibility.
        result = _gen().generate(_df({"sentiment_score": 0.0}), as_of=AS_OF)
        assert len(result) == 1
        assert result[0].action == "hold"

    def test_ticker_matches_row(self):
        assert _gen().generate(_df({"ticker": "NVDA"}), as_of=AS_OF)[0].ticker == "NVDA"

    def test_timestamp_uses_as_of(self):
        result = _gen().generate(_df({}), as_of=AS_OF)
        assert result[0].timestamp == AS_OF

    def test_timestamp_defaults_to_row_date_midnight(self):
        result = _gen().generate(_df({"date": date(2025, 3, 15)}))
        assert result[0].timestamp == datetime(2025, 3, 15, 0, 0, 0)

    # ── Multi-ticker day ───────────────────────────────────────────────────────

    def test_mixed_signals_across_tickers(self):
        df = _df(
            {"ticker": "AAPL", "sentiment_score":  0.6, "n_headlines": 5},
            {"ticker": "MSFT", "sentiment_score": -0.5, "n_headlines": 4},
            {"ticker": "TSLA", "sentiment_score":  0.1, "n_headlines": 3},
            {"ticker": "NVDA", "sentiment_score":  0.8, "n_headlines": 1},  # thin
        )
        actions = {s.ticker: s.action for s in _gen().generate(df, as_of=AS_OF)}
        assert actions["AAPL"] == "buy"
        assert actions["MSFT"] == "sell"
        assert actions["TSLA"] == "hold"
        assert actions["NVDA"] == "hold"   # thin-data guard overrides strong score

    def test_all_tickers_represented_in_output(self):
        df = _df(
            {"ticker": "AAPL"}, {"ticker": "MSFT"}, {"ticker": "TSLA"}
        )
        tickers = {s.ticker for s in _gen().generate(df, as_of=AS_OF)}
        assert tickers == {"AAPL", "MSFT", "TSLA"}

    def test_signals_compatible_with_portfolio_manager(self):
        # Verify returned objects satisfy the Signal contract consumed by
        # PortfolioManager.process_signal().
        result = _gen().generate(_df({"sentiment_score": 0.5}), as_of=AS_OF)
        sig = result[0]
        assert hasattr(sig, "timestamp")
        assert hasattr(sig, "ticker")
        assert hasattr(sig, "action")
        assert sig.action in ("buy", "sell", "hold")


# ─────────────────────────────────────────────────────────────────────────────
# Momentum confirmation
# ─────────────────────────────────────────────────────────────────────────────

# Signal date used by the default _df() helper is 2025-01-10.
# We build price DataFrames that end on that date so the momentum check sees
# the full history.

def _price_df(closes: list[float], end: str = "2025-01-10") -> pd.DataFrame:
    """Build a single-ticker OHLCV-like DataFrame with a DatetimeIndex."""
    idx = pd.date_range(end=end, periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=idx)


def _above_ma_history() -> dict[str, pd.DataFrame]:
    """AAPL: 11 ascending bars → last close (110) > 10-day MA (105.5)."""
    return {"AAPL": _price_df(list(range(100, 111)))}


def _below_ma_history() -> dict[str, pd.DataFrame]:
    """AAPL: 11 descending bars → last close (100) < 10-day MA (104.5)."""
    return {"AAPL": _price_df(list(range(110, 99, -1)))}


def _thin_history() -> dict[str, pd.DataFrame]:
    """AAPL: only 5 bars (< default ma_window=10) → insufficient."""
    return {"AAPL": _price_df([100.0, 101.0, 102.0, 103.0, 104.0])}


class TestMomentumConfirmation:
    """Tests for the momentum_confirm flag on SignalGenerator."""

    # ── Flag is off by default ─────────────────────────────────────────────────

    def test_momentum_off_by_default(self):
        gen = SignalGenerator()
        assert gen.momentum_confirm is False

    def test_momentum_off_buy_passes_without_price_data(self):
        gen = SignalGenerator(momentum_confirm=False, min_headlines=3)
        result = gen.generate(_df({"sentiment_score": 0.5}), as_of=AS_OF)
        assert result[0].action == "buy"

    def test_momentum_off_ignores_price_history(self):
        # Even if close < MA, no downgrade when flag is off.
        gen = SignalGenerator(momentum_confirm=False, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5}),
            as_of=AS_OF,
            price_history=_below_ma_history(),
        )
        assert result[0].action == "buy"

    # ── Sentiment and momentum agree → signal passes ───────────────────────────

    def test_buy_confirmed_when_close_above_ma(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_above_ma_history(),
        )
        assert result[0].action == "buy"

    def test_sell_confirmed_when_close_below_ma(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": -0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_below_ma_history(),
        )
        assert result[0].action == "sell"

    # ── Sentiment and momentum disagree → downgraded to hold ──────────────────

    def test_buy_downgraded_when_close_below_ma(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_below_ma_history(),  # close < MA → disagrees with BUY
        )
        assert result[0].action == "hold"

    def test_sell_downgraded_when_close_above_ma(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": -0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_above_ma_history(),  # close > MA → disagrees with SELL
        )
        assert result[0].action == "hold"

    # ── Insufficient price history → graceful fallback to hold ─────────────────

    def test_insufficient_history_buy_falls_back_to_hold(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_thin_history(),
        )
        assert result[0].action == "hold"

    def test_insufficient_history_sell_falls_back_to_hold(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": -0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_thin_history(),
        )
        assert result[0].action == "hold"

    def test_missing_ticker_in_price_history_falls_back_to_hold(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history={},  # AAPL not present
        )
        assert result[0].action == "hold"

    # ── price_history=None skips the filter ───────────────────────────────────

    def test_momentum_on_but_no_price_history_skips_check(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=None,
        )
        # No price data → filter is skipped → sentiment signal passes unchanged.
        assert result[0].action == "buy"

    # ── Hold signals are never affected by momentum filter ────────────────────

    def test_hold_signal_unaffected_by_momentum(self):
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.1, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_above_ma_history(),
        )
        assert result[0].action == "hold"

    # ── ma_window validation ───────────────────────────────────────────────────

    def test_ma_window_zero_raises(self):
        with pytest.raises(ValueError, match="ma_window"):
            SignalGenerator(ma_window=0)

    def test_custom_ma_window_respected(self):
        # Use ma_window=3: only 3 bars needed.
        gen = SignalGenerator(momentum_confirm=True, min_headlines=3, ma_window=3)
        # thin_history has 5 bars — sufficient for window=3.
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_thin_history(),  # ascending: last(104) > MA(mean of [102,103,104]=103)
        )
        assert result[0].action == "buy"


# ─────────────────────────────────────────────────────────────────────────────
# Earnings surprise confirmation
# ─────────────────────────────────────────────────────────────────────────────

# Signal date from the default _df() helper is 2025-01-10.

def _earnings_cache(
    surprise_pct: float,
    report_date: date = date(2025, 1, 8),  # 2 days before signal date
) -> EarningsCache:
    raw = {"AAPL": [{"reportedDate": report_date.isoformat(), "surprisePercentage": surprise_pct}]}
    return EarningsCache(raw)


class TestEarningsConfirmation:
    """Tests for the earnings_confirm flag on SignalGenerator."""

    # ── Flag is off by default ─────────────────────────────────────────────────

    def test_earnings_off_by_default(self):
        assert SignalGenerator().earnings_confirm is False

    def test_earnings_off_ignores_cache(self):
        gen = SignalGenerator(earnings_confirm=False, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=-20.0),  # would disagree
        )
        assert result[0].action == "buy"

    # ── No recent earnings → signal passes through unchanged ──────────────────

    def test_no_recent_earnings_buy_passes(self):
        # Report date is 30 days before signal — outside lookback window
        cache = _earnings_cache(surprise_pct=-30.0, report_date=date(2024, 12, 1))
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=cache,
        )
        assert result[0].action == "buy"

    def test_no_recent_earnings_sell_passes(self):
        cache = _earnings_cache(surprise_pct=30.0, report_date=date(2024, 12, 1))
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": -0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=cache,
        )
        assert result[0].action == "sell"

    # ── Sentiment and earnings agree → signal passes ───────────────────────────

    def test_buy_confirmed_by_positive_earnings_surprise(self):
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=5.0),
        )
        assert result[0].action == "buy"

    def test_sell_confirmed_by_negative_earnings_surprise(self):
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": -0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=-5.0),
        )
        assert result[0].action == "sell"

    # ── Sentiment and earnings disagree → downgraded to hold ──────────────────

    def test_buy_downgraded_by_negative_earnings_surprise(self):
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=-10.0),
        )
        assert result[0].action == "hold"

    def test_sell_downgraded_by_positive_earnings_surprise(self):
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": -0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=10.0),
        )
        assert result[0].action == "hold"

    # ── earnings_cache=None skips the filter ──────────────────────────────────

    def test_earnings_on_but_no_cache_skips_filter(self):
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=None,
        )
        assert result[0].action == "buy"

    # ── Custom threshold ───────────────────────────────────────────────────────

    def test_earnings_threshold_respected(self):
        # Require at least 5 % surprise to confirm a BUY
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3, earnings_threshold=5.0)
        # 3 % surprise — below threshold → downgraded
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=3.0),
        )
        assert result[0].action == "hold"

    def test_earnings_threshold_exactly_met_passes(self):
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3, earnings_threshold=5.0)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=5.0),
        )
        assert result[0].action == "buy"

    # ── Hold signals never touched ────────────────────────────────────────────

    def test_hold_never_affected_by_earnings(self):
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.1, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=_earnings_cache(surprise_pct=-50.0),
        )
        assert result[0].action == "hold"

    # ── Earnings lookback boundary ─────────────────────────────────────────────

    def test_earnings_exactly_at_lookback_boundary_applies(self):
        # Default lookback = 7 days.  Report on day D-7 should apply.
        report_date = date(2025, 1, 10) - timedelta(days=7)
        cache = _earnings_cache(surprise_pct=-20.0, report_date=report_date)
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=cache,
        )
        assert result[0].action == "hold"

    def test_earnings_just_outside_lookback_is_neutral(self):
        # Report 8 days before signal → outside default 7-day window
        report_date = date(2025, 1, 10) - timedelta(days=8)
        cache = _earnings_cache(surprise_pct=-20.0, report_date=report_date)
        gen = SignalGenerator(earnings_confirm=True, min_headlines=3)
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            earnings_cache=cache,
        )
        assert result[0].action == "buy"

    # ── Combined: momentum + earnings ─────────────────────────────────────────

    def test_both_filters_must_agree_for_buy_to_pass(self):
        # Momentum agrees (close > MA), earnings agrees (positive surprise) → buy
        gen = SignalGenerator(
            momentum_confirm=True, earnings_confirm=True, min_headlines=3
        )
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_above_ma_history(),
            earnings_cache=_earnings_cache(surprise_pct=5.0),
        )
        assert result[0].action == "buy"

    def test_earnings_disagree_overrides_momentum_agreement(self):
        # Momentum agrees, earnings disagrees → hold
        gen = SignalGenerator(
            momentum_confirm=True, earnings_confirm=True, min_headlines=3
        )
        result = gen.generate(
            _df({"sentiment_score": 0.5, "n_headlines": 5}),
            as_of=AS_OF,
            price_history=_above_ma_history(),
            earnings_cache=_earnings_cache(surprise_pct=-10.0),
        )
        assert result[0].action == "hold"


# ─────────────────────────────────────────────────────────────────────────────
# _date_to_datetime()
# ─────────────────────────────────────────────────────────────────────────────

class TestDateToDatetime:
    def test_date_object_returns_midnight(self):
        assert _date_to_datetime(date(2025, 1, 10)) == datetime(2025, 1, 10, 0, 0)

    def test_datetime_returned_without_tzinfo(self):
        dt = datetime(2025, 1, 10, 15, 30)
        assert _date_to_datetime(dt) == dt

    def test_pandas_timestamp(self):
        assert _date_to_datetime(pd.Timestamp("2025-01-10")) == datetime(2025, 1, 10, 0, 0)

    def test_nat_returns_datetime_without_crashing(self):
        result = _date_to_datetime(pd.NaT)
        assert isinstance(result, datetime)
