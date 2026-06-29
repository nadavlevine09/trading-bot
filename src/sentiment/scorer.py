"""FinBERT-based sentiment scoring for financial headlines."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from transformers import Pipeline, pipeline

_log = logging.getLogger(__name__)

_FINBERT = "ProsusAI/finbert"


class SentimentScorer:
    def __init__(
        self,
        model_name: str = _FINBERT,
        device: int = -1,
        batch_size: int = 32,
    ) -> None:
        """
        Load FinBERT once at construction time.

        Args:
            model_name:  HuggingFace model identifier.
            device:      -1 = CPU; 0, 1, ... = CUDA device index.
                         Use 0 for GPU-accelerated backtests.
            batch_size:  Headlines per forward pass. Increase for GPU.
        """
        _log.info("Loading %s (device=%d) …", model_name, device)
        self._pipe: Pipeline = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,       # return all three label probabilities
            device=device,
            truncation=True,
            max_length=512,
        )
        self.batch_size = batch_size
        _log.info("Model ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, headlines: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Score a list of headline dicts as produced by NewsAPIClient.

        Scoring text is ``title + ". " + description`` (when description is
        present and non-duplicate); title only otherwise.

        Returns:
            DataFrame with columns:
              ticker, company, source, published_at, url,
              text, positive, negative, neutral, sentiment_score

            ``sentiment_score`` = positive_prob − negative_prob ∈ [−1, 1].
            Strongly positive headlines score near 1; negative near −1.
        """
        if not headlines:
            return _empty_frame()

        texts = [_build_text(h) for h in headlines]

        # Single batched call — pipeline handles chunking internally.
        raw: list[list[dict]] = self._pipe(texts, batch_size=self.batch_size)

        rows = []
        for headline, text, label_list in zip(headlines, texts, raw):
            probs = {item["label"].lower(): float(item["score"]) for item in label_list}
            rows.append({
                "ticker":          headline.get("ticker", ""),
                "company":         headline.get("company", ""),
                "source":          headline.get("source", ""),
                "published_at":    headline.get("published_at", ""),
                "url":             headline.get("url", ""),
                "text":            text,
                "positive":        probs.get("positive", 0.0),
                "negative":        probs.get("negative", 0.0),
                "neutral":         probs.get("neutral", 0.0),
                "sentiment_score": probs.get("positive", 0.0) - probs.get("negative", 0.0),
            })

        return pd.DataFrame(rows)


# ── Module-level aggregation ───────────────────────────────────────────────────

def aggregate_daily(
    scored: pd.DataFrame,
    source_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Reduce per-headline scores to one row per (ticker, date).

    Args:
        scored:         DataFrame from ``SentimentScorer.score()``.
        source_weights: Optional mapping of source name → weight multiplier.
                        Sources absent from the dict default to weight 1.0.
                        Pass a populated dict to enable source-credibility
                        weighting (e.g. Reuters > random blog).
                        Pass ``None`` for a simple equal-weight mean.

    Returns:
        DataFrame with columns: ticker, date, sentiment_score, n_headlines.
        ``sentiment_score`` is a weighted average in [−1, 1].
    """
    if scored.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "sentiment_score", "n_headlines"]
        )

    df = scored.copy()

    # Parse ISO-8601 timestamps; coerce unparseable values to NaT.
    df["date"] = (
        pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        .dt.date
    )

    # Apply source weights; default to 1.0 for unrecognised sources.
    sw = source_weights or {}
    df["_weight"] = df["source"].map(lambda s: sw.get(s, 1.0))

    df["_weighted_score"] = df["sentiment_score"] * df["_weight"]

    agg = (
        df.groupby(["ticker", "date"], dropna=False)
        .agg(
            _wsum=("_weighted_score", "sum"),
            _wtotal=("_weight",        "sum"),
            n_headlines=("sentiment_score", "count"),
        )
        .reset_index()
    )

    # Avoid division by zero (all-zero weights would be a misconfiguration).
    agg["sentiment_score"] = agg["_wsum"] / agg["_wtotal"].clip(lower=1e-10)

    return agg[["ticker", "date", "sentiment_score", "n_headlines"]]


# ── Bayesian / exponential-decay sentiment smoothing ─────────────────────────

def smooth_daily_scores(
    daily: pd.DataFrame,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """
    Apply exponential-decay (EWM) smoothing to daily sentiment scores.

    Replaces the flat per-day average with a weighted rolling estimate where
    prior days carry forward with exponentially decaying weight:

        smoothed_t = alpha × raw_t + (1 − alpha) × smoothed_{t-1}

    Args:
        daily:  DataFrame as returned by ``aggregate_daily()``.
                Columns: ticker, date, sentiment_score, n_headlines.
        alpha:  Smoothing factor in (0, 1].
                • alpha = 1.0 → no smoothing (identical to raw daily).
                • alpha = 0.3 → roughly a 6-day half-life.

    Returns:
        New DataFrame with the same schema; ``sentiment_score`` values are
        replaced with the smoothed estimates.  All other columns are unchanged.
    """
    if daily.empty:
        return daily.copy()

    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    result = daily.copy()
    result["_date_ts"] = pd.to_datetime(result["date"])

    for ticker in result["ticker"].unique():
        mask       = result["ticker"] == ticker
        ticker_idx = result[mask].sort_values("_date_ts").index
        raw_scores = result.loc[ticker_idx, "sentiment_score"]
        smoothed   = raw_scores.ewm(alpha=alpha, adjust=False).mean()
        result.loc[ticker_idx, "sentiment_score"] = smoothed.values

    return result.drop(columns=["_date_ts"])


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_text(headline: dict[str, Any]) -> str:
    """Combine title and description into one scoring string."""
    title = (headline.get("title") or "").strip()
    desc  = (headline.get("description") or "").strip()
    if desc and desc != title:
        return f"{title}. {desc}"
    return title


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "ticker", "company", "source", "published_at", "url",
        "text", "positive", "negative", "neutral", "sentiment_score",
    ])
