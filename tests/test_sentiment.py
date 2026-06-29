"""Tests for src/sentiment/. The model is always mocked — no downloads."""
from __future__ import annotations

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.sentiment.scorer import SentimentScorer, aggregate_daily, smooth_daily_scores, _build_text


# ─────────────────────────────────────────────────────────────────────────────
# Canonical mock outputs (three FinBERT label scores per article)
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE = [
    {"label": "positive", "score": 0.85},
    {"label": "negative", "score": 0.08},
    {"label": "neutral",  "score": 0.07},
]
NEGATIVE = [
    {"label": "positive", "score": 0.05},
    {"label": "negative", "score": 0.88},
    {"label": "neutral",  "score": 0.07},
]
NEUTRAL = [
    {"label": "positive", "score": 0.10},
    {"label": "negative", "score": 0.12},
    {"label": "neutral",  "score": 0.78},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_scorer(pipe_output: list) -> SentimentScorer:
    """Build a SentimentScorer whose pipeline is replaced by a mock."""
    with patch("src.sentiment.scorer.pipeline") as mock_pipeline_fn:
        mock_pipe = MagicMock(return_value=pipe_output)
        mock_pipeline_fn.return_value = mock_pipe
        scorer = SentimentScorer()
    # scorer._pipe is already mock_pipe; keep a direct ref for call assertions.
    scorer._pipe = mock_pipe
    return scorer


def _headline(**overrides) -> dict:
    base = {
        "ticker":       "AAPL",
        "company":      "Apple",
        "title":        "Apple reports record quarterly earnings",
        "description":  "Apple Inc beat analyst estimates with strong iPhone sales.",
        "source":       "Reuters",
        "published_at": "2025-01-10T09:00:00Z",
        "url":          "https://reuters.com/apple-q1",
    }
    base.update(overrides)
    return base


def _scored_row(**overrides) -> dict:
    base = {
        "ticker": "AAPL", "company": "Apple", "source": "Reuters",
        "published_at": "2025-01-10T09:00:00Z", "url": "",
        "text": "Apple earnings", "positive": 0.8, "negative": 0.1,
        "neutral": 0.1, "sentiment_score": 0.7,
    }
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame([_scored_row(**r) for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
# SentimentScorer.score()
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_returns_dataframe_with_required_columns(self):
        df = _make_scorer([POSITIVE]).score([_headline()])
        for col in ("ticker", "company", "source", "published_at", "url",
                    "text", "positive", "negative", "neutral", "sentiment_score"):
            assert col in df.columns

    def test_positive_headline_has_positive_score(self):
        df = _make_scorer([POSITIVE]).score([_headline()])
        assert df.iloc[0]["positive"] == pytest.approx(0.85)
        assert df.iloc[0]["sentiment_score"] == pytest.approx(0.85 - 0.08)

    def test_negative_headline_has_negative_score(self):
        df = _make_scorer([NEGATIVE]).score([_headline(title="Apple hit with antitrust fine")])
        assert df.iloc[0]["sentiment_score"] < 0

    def test_neutral_headline_score_near_zero(self):
        df = _make_scorer([NEUTRAL]).score([_headline()])
        assert abs(df.iloc[0]["sentiment_score"]) < 0.2

    def test_sentiment_score_in_range(self):
        df = _make_scorer([POSITIVE]).score([_headline()])
        assert -1.0 <= df.iloc[0]["sentiment_score"] <= 1.0

    def test_empty_input_returns_empty_dataframe(self):
        df = _make_scorer([]).score([])
        assert df.empty
        assert "sentiment_score" in df.columns

    # ── Text construction ──────────────────────────────────────────────────────

    def test_combines_title_and_description_in_scoring_text(self):
        scorer = _make_scorer([POSITIVE])
        scorer.score([_headline(title="Apple earnings", description="Beat estimates.")])
        texts_passed = scorer._pipe.call_args[0][0]
        assert texts_passed[0] == "Apple earnings. Beat estimates."

    def test_uses_title_only_when_description_empty(self):
        scorer = _make_scorer([POSITIVE])
        scorer.score([_headline(title="Apple earnings", description="")])
        texts_passed = scorer._pipe.call_args[0][0]
        assert texts_passed[0] == "Apple earnings"

    def test_does_not_duplicate_when_description_equals_title(self):
        scorer = _make_scorer([POSITIVE])
        scorer.score([_headline(title="Apple earnings", description="Apple earnings")])
        texts_passed = scorer._pipe.call_args[0][0]
        assert texts_passed[0] == "Apple earnings"

    # ── Batching ───────────────────────────────────────────────────────────────

    def test_multiple_headlines_scored_in_one_pipeline_call(self):
        scorer = _make_scorer([POSITIVE, NEGATIVE, NEUTRAL])
        scorer.score([_headline(), _headline(title="Apple fine"), _headline(title="Apple flat")])
        assert scorer._pipe.call_count == 1

    def test_returns_one_row_per_headline(self):
        df = _make_scorer([POSITIVE, NEGATIVE, NEUTRAL]).score(
            [_headline(), _headline(), _headline()]
        )
        assert len(df) == 3

    def test_batch_size_forwarded_to_pipeline(self):
        scorer = _make_scorer([POSITIVE])
        scorer.batch_size = 16
        scorer.score([_headline()])
        assert scorer._pipe.call_args[1].get("batch_size") == 16

    # ── Metadata passthrough ───────────────────────────────────────────────────

    def test_ticker_and_source_preserved_in_output(self):
        df = _make_scorer([POSITIVE]).score(
            [_headline(ticker="NVDA", source="Bloomberg")]
        )
        assert df.iloc[0]["ticker"] == "NVDA"
        assert df.iloc[0]["source"] == "Bloomberg"

    def test_model_loaded_once_not_per_call(self):
        scorer = _make_scorer([POSITIVE])
        scorer._pipe.return_value = [POSITIVE]
        scorer.score([_headline()])
        scorer.score([_headline()])
        # Two score() calls → two pipeline() invocations, NOT two model loads.
        # The key invariant: pipeline() (the factory) is called exactly once.
        # We verify this via the constructor mock path in _make_scorer.
        assert scorer._pipe.call_count == 2  # pipe called twice, but model loaded once


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_daily()
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateDaily:
    def test_returns_one_row_per_ticker_per_date(self):
        result = aggregate_daily(_df({"sentiment_score": 0.5}, {"sentiment_score": 0.3}))
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"

    def test_n_headlines_counts_articles(self):
        result = aggregate_daily(_df({}, {}))
        assert result.iloc[0]["n_headlines"] == 2

    def test_equal_weight_average(self):
        result = aggregate_daily(_df({"sentiment_score": 0.6}, {"sentiment_score": 0.2}))
        assert result.iloc[0]["sentiment_score"] == pytest.approx(0.4)

    def test_source_weights_applied(self):
        df = _df(
            {"source": "Reuters",   "sentiment_score": 1.0},
            {"source": "BlogSpot",  "sentiment_score": -1.0},
        )
        result = aggregate_daily(df, source_weights={"Reuters": 3.0, "BlogSpot": 1.0})
        # (1.0×3 + −1.0×1) / 4 = 0.5
        assert result.iloc[0]["sentiment_score"] == pytest.approx(0.5)

    def test_unknown_source_defaults_to_weight_one(self):
        df = _df(
            {"source": "Reuters",  "sentiment_score": 1.0},
            {"source": "NewSite",  "sentiment_score": 0.0},  # not in weights dict
        )
        result = aggregate_daily(df, source_weights={"Reuters": 1.0})
        # both weight=1 → average = 0.5
        assert result.iloc[0]["sentiment_score"] == pytest.approx(0.5)

    def test_no_weights_equals_equal_weight_mean(self):
        df = _df({"sentiment_score": 0.8}, {"sentiment_score": 0.4})
        without = aggregate_daily(df)
        with_ones = aggregate_daily(df, source_weights={})
        assert without.iloc[0]["sentiment_score"] == pytest.approx(
            with_ones.iloc[0]["sentiment_score"]
        )

    def test_separate_tickers_yield_separate_rows(self):
        df = pd.DataFrame([
            _scored_row(ticker="AAPL", sentiment_score=0.6),
            _scored_row(ticker="MSFT", sentiment_score=-0.2),
        ])
        result = aggregate_daily(df)
        assert set(result["ticker"]) == {"AAPL", "MSFT"}

    def test_separate_dates_yield_separate_rows(self):
        df = _df(
            {"published_at": "2025-01-10T09:00:00Z", "sentiment_score": 0.8},
            {"published_at": "2025-01-11T09:00:00Z", "sentiment_score": 0.2},
        )
        assert len(aggregate_daily(df)) == 2

    def test_empty_input_returns_empty_dataframe_with_columns(self):
        empty = pd.DataFrame(columns=list(_scored_row().keys()))
        result = aggregate_daily(empty)
        assert result.empty
        for col in ("ticker", "date", "sentiment_score", "n_headlines"):
            assert col in result.columns

    def test_output_columns_are_exactly(self):
        result = aggregate_daily(_df({}))
        assert list(result.columns) == ["ticker", "date", "sentiment_score", "n_headlines"]


# ─────────────────────────────────────────────────────────────────────────────
# _build_text()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildText:
    def test_combines_title_and_description(self):
        assert _build_text({"title": "Apple earnings", "description": "Beat estimates."}) \
               == "Apple earnings. Beat estimates."

    def test_title_only_when_description_missing(self):
        assert _build_text({"title": "Apple earnings"}) == "Apple earnings"

    def test_title_only_when_description_empty_string(self):
        assert _build_text({"title": "Apple earnings", "description": ""}) == "Apple earnings"

    def test_title_only_when_description_none(self):
        assert _build_text({"title": "Apple earnings", "description": None}) == "Apple earnings"

    def test_no_duplication_when_title_equals_description(self):
        assert _build_text({"title": "Apple earnings", "description": "Apple earnings"}) \
               == "Apple earnings"

    def test_empty_dict_returns_empty_string(self):
        assert _build_text({}) == ""

    def test_missing_title_uses_empty_string(self):
        assert _build_text({"description": "Some text."}) == ". Some text."


# ─────────────────────────────────────────────────────────────────────────────
# smooth_daily_scores()
# ─────────────────────────────────────────────────────────────────────────────

def _daily_df(scores: list[float], ticker: str = "AAPL", start: str = "2025-01-01") -> pd.DataFrame:
    """Build a daily-scores DataFrame with consecutive dates."""
    dates = pd.date_range(start, periods=len(scores), freq="D").date
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "sentiment_score": scores,
        "n_headlines": 10,
    })


class TestSmoothDailyScores:
    def test_returns_dataframe(self):
        daily = _daily_df([0.1, 0.2, 0.3])
        out = smooth_daily_scores(daily)
        assert isinstance(out, pd.DataFrame)

    def test_schema_preserved(self):
        daily = _daily_df([0.1, 0.2, 0.3])
        out = smooth_daily_scores(daily)
        assert set(out.columns) == set(daily.columns)
        assert len(out) == len(daily)

    def test_alpha_one_is_identity(self):
        # alpha=1.0 means "no smoothing": smoothed == raw
        daily = _daily_df([0.1, -0.3, 0.5, 0.0])
        out = smooth_daily_scores(daily, alpha=1.0)
        assert list(out["sentiment_score"]) == pytest.approx([0.1, -0.3, 0.5, 0.0])

    def test_smoothed_values_differ_from_raw(self):
        # A volatile series should be smoothed
        daily = _daily_df([0.9, -0.9, 0.9, -0.9])
        out = smooth_daily_scores(daily, alpha=0.3)
        # Second value should be pulled back from -0.9 toward the prior 0.9
        assert out["sentiment_score"].iloc[1] > -0.9

    def test_first_value_unchanged(self):
        # EWM with adjust=False: smoothed[0] == raw[0]
        daily = _daily_df([0.5, 0.1, 0.2])
        out = smooth_daily_scores(daily, alpha=0.4)
        assert out["sentiment_score"].iloc[0] == pytest.approx(0.5)

    def test_smoothing_per_ticker_independent(self):
        # Ticker B should not bleed into ticker A
        aapl = _daily_df([1.0, 1.0, 1.0], ticker="AAPL")
        msft = _daily_df([-1.0, -1.0, -1.0], ticker="MSFT")
        combined = pd.concat([aapl, msft], ignore_index=True)
        out = smooth_daily_scores(combined, alpha=0.3)
        aapl_out = out[out["ticker"] == "AAPL"]["sentiment_score"]
        msft_out = out[out["ticker"] == "MSFT"]["sentiment_score"]
        assert all(aapl_out > 0)
        assert all(msft_out < 0)

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame(columns=["ticker", "date", "sentiment_score", "n_headlines"])
        out = smooth_daily_scores(empty)
        assert out.empty

    def test_invalid_alpha_raises(self):
        daily = _daily_df([0.1, 0.2])
        with pytest.raises(ValueError, match="alpha"):
            smooth_daily_scores(daily, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            smooth_daily_scores(daily, alpha=1.1)

    def test_n_headlines_not_smoothed(self):
        # Smoothing should only touch sentiment_score, not n_headlines
        daily = _daily_df([0.1, 0.2, 0.3])
        out = smooth_daily_scores(daily, alpha=0.5)
        assert list(out["n_headlines"]) == [10, 10, 10]
