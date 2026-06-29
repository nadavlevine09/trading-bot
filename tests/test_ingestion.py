"""Unit tests for src/ingestion. No real HTTP calls are made."""
from __future__ import annotations

import pytest
import requests
from unittest.mock import MagicMock, patch

from src.ingestion.newsapi import (
    NewsAPIClient,
    NewsAPIError,
    RateLimitError,
    _deduplicate,
    _drop_removed,
    _is_relevant,
    _normalise,
    _title_key,
)
from src.ingestion.reddit import RedditClient


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_article(**overrides) -> dict:
    base = {
        "source": {"id": "reuters", "name": "Reuters"},
        "author": "Jane Doe",
        "title": "Apple reports record quarterly earnings",
        "description": "Apple Inc beat analyst estimates with strong iPhone sales.",
        "url": "https://reuters.com/apple-earnings",
        "publishedAt": "2025-01-10T09:00:00Z",
        "content": "Apple reported record earnings this quarter... [+1234 chars]",
    }
    base.update(overrides)
    return base


def _api_response(articles: list[dict], total: int | None = None) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.ok = True
    mock.json.return_value = {
        "status": "ok",
        "totalResults": total if total is not None else len(articles),
        "articles": articles,
    }
    return mock


def _client(monkeypatch) -> NewsAPIClient:
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")
    return NewsAPIClient()


# ─────────────────────────────────────────────────────────────────────────────
# fetch_headlines — HTTP layer
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchHeadlines:
    def test_returns_articles_on_success(self, monkeypatch):
        client = _client(monkeypatch)
        articles = [_make_article(), _make_article(title="Apple hires new CFO")]
        with patch.object(client._session, "get", return_value=_api_response(articles)):
            result = client.fetch_headlines("Apple")
        assert len(result) == 2

    def test_raises_rate_limit_error_on_429(self, monkeypatch):
        client = _client(monkeypatch)
        mock_resp = MagicMock(status_code=429, ok=False, headers={"Retry-After": "3600"})
        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(RateLimitError, match="rate limit"):
                client.fetch_headlines("Apple")

    def test_raises_newsapi_error_on_401(self, monkeypatch):
        client = _client(monkeypatch)
        mock_resp = MagicMock(status_code=401, ok=False)
        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(NewsAPIError, match="Invalid or missing"):
                client.fetch_headlines("Apple")

    def test_raises_newsapi_error_on_500(self, monkeypatch):
        client = _client(monkeypatch)
        mock_resp = MagicMock(status_code=500, ok=False, text="Internal Server Error")
        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(NewsAPIError, match="500"):
                client.fetch_headlines("Apple")

    def test_raises_newsapi_error_on_api_level_error(self, monkeypatch):
        client = _client(monkeypatch)
        mock_resp = MagicMock(status_code=200, ok=True)
        mock_resp.json.return_value = {
            "status": "error",
            "code": "maximumResultsReached",
            "message": "You have requested too many results.",
        }
        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(NewsAPIError, match="maximumResultsReached"):
                client.fetch_headlines("Apple")

    def test_raises_newsapi_error_on_network_failure(self, monkeypatch):
        client = _client(monkeypatch)
        with patch.object(
            client._session, "get",
            side_effect=requests.exceptions.ConnectionError("no route"),
        ):
            with pytest.raises(NewsAPIError, match="Network error"):
                client.fetch_headlines("Apple")

    def test_page_size_capped_at_100(self, monkeypatch):
        client = _client(monkeypatch)
        captured = {}
        def fake_get(url, params, timeout):
            captured["pageSize"] = params["pageSize"]
            return _api_response([])
        with patch.object(client._session, "get", side_effect=fake_get):
            client.fetch_headlines("Apple", page_size=999)
        assert captured["pageSize"] == 100

    def test_from_and_to_dates_passed_as_params(self, monkeypatch):
        from datetime import date
        client = _client(monkeypatch)
        captured = {}
        def fake_get(url, params, timeout):
            captured.update(params)
            return _api_response([])
        with patch.object(client._session, "get", side_effect=fake_get):
            client.fetch_headlines("Apple", from_date=date(2025, 1, 1), to_date=date(2025, 1, 7))
        assert captured["from"] == "2025-01-01"
        assert captured["to"] == "2025-01-07"


# ─────────────────────────────────────────────────────────────────────────────
# fetch_for_ticker — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchForTicker:
    def test_returns_normalised_dicts(self, monkeypatch):
        client = _client(monkeypatch)
        with patch.object(client._session, "get", return_value=_api_response([_make_article()])):
            results = client.fetch_for_ticker("AAPL", "Apple")
        assert len(results) == 1
        r = results[0]
        for key in ("ticker", "company", "title", "description", "content_snippet",
                    "source", "published_at", "url"):
            assert key in r

    def test_ticker_and_company_injected(self, monkeypatch):
        client = _client(monkeypatch)
        with patch.object(client._session, "get", return_value=_api_response([_make_article()])):
            results = client.fetch_for_ticker("AAPL", "Apple")
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["company"] == "Apple"

    def test_query_includes_both_ticker_and_company(self, monkeypatch):
        client = _client(monkeypatch)
        captured = {}
        def fake_get(url, params, timeout):
            captured["q"] = params["q"]
            return _api_response([])
        with patch.object(client._session, "get", side_effect=fake_get):
            client.fetch_for_ticker("AAPL", "Apple")
        assert "Apple" in captured["q"]
        assert "AAPL" in captured["q"]

    def test_removed_articles_filtered_out(self, monkeypatch):
        client = _client(monkeypatch)
        articles = [
            _make_article(title="[Removed]", url="https://removed.com"),
            _make_article(title="Apple stock rises"),
        ]
        with patch.object(client._session, "get", return_value=_api_response(articles)):
            results = client.fetch_for_ticker("AAPL", "Apple")
        assert len(results) == 1
        assert results[0]["title"] == "Apple stock rises"

    def test_near_duplicate_titles_deduplicated(self, monkeypatch):
        client = _client(monkeypatch)
        articles = [
            _make_article(title="Apple reports record quarterly earnings"),
            _make_article(
                title="Apple reports record quarterly earnings beat",  # ~90% similar
                source={"id": "bloomberg", "name": "Bloomberg"},
            ),
            _make_article(title="Tesla unveils new Model Y variant"),  # unrelated
        ]
        with patch.object(client._session, "get", return_value=_api_response(articles)):
            results = client.fetch_for_ticker("AAPL", "Apple")
        assert len(results) == 2  # duplicate dropped, unrelated kept


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (unit-tested in isolation)
# ─────────────────────────────────────────────────────────────────────────────

class TestDropRemoved:
    def test_removes_placeholder_articles(self):
        articles = [
            {"title": "[Removed]", "url": "https://removed.com"},
            {"title": "Apple earnings beat", "url": "https://reuters.com/a"},
        ]
        result = _drop_removed(articles)
        assert len(result) == 1
        assert result[0]["title"] == "Apple earnings beat"

    def test_keeps_all_valid_articles(self):
        articles = [_make_article(), _make_article(title="Apple hires new CFO")]
        assert len(_drop_removed(articles)) == 2


class TestIsRelevant:
    def test_company_name_in_title_is_relevant(self):
        a = _make_article(title="Apple reports record quarterly earnings", description="")
        assert _is_relevant(a, "AAPL", "Apple") is True

    def test_ticker_in_title_is_relevant(self):
        a = _make_article(title="AAPL surges 5% after earnings beat", description="")
        assert _is_relevant(a, "AAPL", "Apple") is True

    def test_company_name_in_description_only_is_relevant(self):
        a = _make_article(title="Markets rally on tech optimism", description="Apple leads gains.")
        assert _is_relevant(a, "AAPL", "Apple") is True

    def test_term_only_in_content_is_not_relevant(self):
        # Simulates the MMA recap that mentioned Apple in passing in body text only.
        a = _make_article(
            title="UFC Vegas 119: Manel Kape knocks out rival in stunning fashion",
            description="An exciting night of MMA action at the UFC Vegas 119 event.",
            content="Fighters used Apple AirPods to listen to music before the fight.",
        )
        assert _is_relevant(a, "AAPL", "Apple") is False

    def test_common_noun_homonym_is_known_false_positive(self):
        # A case-insensitive word-boundary filter cannot distinguish "Apple Inc"
        # from the common noun "apple". Eliminating homonym false positives
        # requires named-entity recognition, which is out of scope here.
        # This test documents the known limitation: "apple pie" is flagged as
        # relevant. The sentiment layer is responsible for discarding it.
        a = _make_article(
            title="Best apple pie recipes for the holidays",
            description="Our food editors share their favourite apple dessert ideas.",
        )
        assert _is_relevant(a, "AAPL", "Apple") is True  # known false positive

    def test_word_boundary_prevents_substring_match(self):
        # "Appleton" contains "apple" but should not count as Apple Inc coverage.
        a = _make_article(
            title="Appleton Farms reports bumper harvest season",
            description="The Appleton family farm posted strong results this year.",
        )
        assert _is_relevant(a, "AAPL", "Apple") is False

    def test_case_insensitive_match(self):
        a = _make_article(title="APPLE STOCK HITS ALL-TIME HIGH", description="")
        assert _is_relevant(a, "AAPL", "Apple") is True

    def test_multiword_company_name_matched_as_phrase(self):
        a = _make_article(
            title="Goldman Sachs raises Apple price target to $250",
            description="",
        )
        assert _is_relevant(a, "GS", "Goldman Sachs") is True

    def test_multiword_company_name_absent_is_not_relevant(self):
        a = _make_article(
            title="Goldman raises price target",  # "Goldman" alone, not "Goldman Sachs"
            description="Analysts at Goldman raised the price target.",
        )
        # Neither "GS" nor "Goldman Sachs" (full phrase) appears → not relevant.
        assert _is_relevant(a, "GS", "Goldman Sachs") is False

    def test_subsidiary_in_title_is_relevant_with_extra_terms(self):
        # "Xbox Game Pass" is tagged as ORG by spaCy; _org_matches catches "Xbox".
        a = _make_article(title="Xbox Game Pass hits 50 million subscribers", description="")
        # Without extra_terms, neither NER (no Microsoft ORG) nor string fallback
        # (MSFT/Microsoft absent) finds a match.
        assert _is_relevant(a, "MSFT", "Microsoft") is False
        # With the subsidiary list, NER finds "Xbox Game Pass" ORG → matches "Xbox".
        assert _is_relevant(a, "MSFT", "Microsoft", extra_terms=["Xbox", "Azure"]) is True

    def test_subsidiary_in_description_only_is_not_relevant_with_extra_terms(self):
        # Subsidiary brand appearing ONLY in the description (not the title) must be
        # rejected by the string fallback — description-only "Xbox" pulls in EA /
        # PlayStation articles that mention Xbox only as a passing comparison.
        # Note: spaCy may mislabel "Xbox Game Pass" as PERSON here, so NER also fails.
        a = _make_article(
            title="Gaming subscription revenues surge across platforms",
            description="Xbox Game Pass leads with strong Q2 numbers.",
        )
        assert _is_relevant(a, "MSFT", "Microsoft", extra_terms=["Xbox", "Azure"]) is False

    def test_unrelated_subsidiary_term_does_not_match_other_ticker(self):
        # "Azure" is not in AAPL's extra_terms and "Apple" is not in "Azure cloud costs".
        a = _make_article(title="Azure cloud costs rise amid AI demand", description="")
        assert _is_relevant(a, "AAPL", "Apple") is False

    # ── NER-specific behaviour ─────────────────────────────────────────────────

    def test_ner_matches_long_form_company_name(self):
        # spaCy extracts "Apple Inc." as ORG; _org_matches("Apple Inc.", "Apple") is True
        # because "Apple" appears as a whole word within "Apple Inc."
        a = _make_article(title="Apple Inc. to expand manufacturing in India", description="")
        assert _is_relevant(a, "AAPL", "Apple") is True

    def test_ner_partial_entity_does_not_match_full_company_name(self):
        # spaCy extracts "Goldman" as ORG; _org_matches("Goldman", "Goldman Sachs")
        # returns False because "Goldman Sachs" is NOT a whole word within "Goldman".
        a = _make_article(
            title="Goldman upgrades Microsoft to Buy rating",
            description="Analysts at Goldman raised the price target.",
        )
        assert _is_relevant(a, "GS", "Goldman Sachs") is False

    def test_ner_generic_subsidiary_word_does_not_produce_false_positive(self):
        # "Prime Minister" is in the text.  "Prime" is intentionally excluded from
        # AMZN's SUBSIDIARY_MAP (too generic); production extra_terms are ["Amazon","AWS"].
        # NER: only "UK" (GPE) extracted — no Amazon ORG.
        # String fallback: "AMZN", "Amazon", "AWS" absent → False.
        a = _make_article(
            title="UK PM announces new economic policy",
            description="The Prime Minister outlined plans at a press conference.",
        )
        assert _is_relevant(a, "AMZN", "Amazon", extra_terms=["Amazon", "AWS"]) is False

    def test_ner_and_fallback_both_fail_for_truly_off_topic_article(self):
        # Neither NER (no relevant ORG) nor string fallback ("Tesla"/"TSLA" absent)
        # should accept a football headline.
        a = _make_article(
            title="Champions League final: Real Madrid vs Arsenal",
            description="A thrilling final is expected at the Bernabéu stadium.",
        )
        assert _is_relevant(a, "TSLA", "Tesla") is False


class TestFetchForTickerRelevanceFilter:
    def test_off_topic_article_filtered_out(self, monkeypatch):
        """MMA recap mentioning Apple only in content should be dropped."""
        client = _client(monkeypatch)
        articles = [
            _make_article(
                title="UFC Vegas 119: Manel Kape knocks out rival in stunning fashion",
                description="An exciting night of MMA action at the UFC Vegas 119 event.",
                content="Fighters used Apple AirPods to listen to music before the fight.",
            ),
            _make_article(title="Apple stock hits record high on strong earnings"),
        ]
        with patch.object(client._session, "get", return_value=_api_response(articles)):
            results = client.fetch_for_ticker("AAPL", "Apple")
        assert len(results) == 1
        assert results[0]["title"] == "Apple stock hits record high on strong earnings"

    def test_all_relevant_articles_pass_through(self, monkeypatch):
        client = _client(monkeypatch)
        articles = [
            _make_article(title="Apple unveils new MacBook Pro lineup"),
            _make_article(title="AAPL options volume spikes ahead of earnings"),
        ]
        with patch.object(client._session, "get", return_value=_api_response(articles)):
            results = client.fetch_for_ticker("AAPL", "Apple")
        assert len(results) == 2

    def test_relevance_filter_applied_before_dedup(self, monkeypatch):
        """An off-topic duplicate should not prevent a real article from appearing."""
        client = _client(monkeypatch)
        articles = [
            # Two near-identical off-topic articles — both should be dropped, not
            # one kept because the dedup ran before the relevance check.
            _make_article(
                title="UFC Vegas 119: Manel Kape knocks out rival",
                description="Great fight night at the UFC event.",
            ),
            _make_article(
                title="UFC Vegas 119: Manel Kape knocks out rival in stunning fashion",
                description="Exciting MMA action at UFC Vegas.",
            ),
            _make_article(title="Apple earnings beat Wall Street estimates"),
        ]
        with patch.object(client._session, "get", return_value=_api_response(articles)):
            results = client.fetch_for_ticker("AAPL", "Apple")
        assert len(results) == 1
        assert "Apple" in results[0]["title"]


class TestDeduplicate:
    def test_identical_titles_collapsed_to_one(self):
        articles = [_make_article(), _make_article()]
        assert len(_deduplicate(articles)) == 1

    def test_dissimilar_titles_both_kept(self):
        articles = [
            _make_article(title="Apple earnings beat expectations"),
            _make_article(title="Tesla recalls 50000 vehicles over software glitch"),
        ]
        assert len(_deduplicate(articles)) == 2

    def test_near_duplicate_dropped(self):
        articles = [
            _make_article(title="Apple Inc reports record Q1 earnings"),
            _make_article(title="Apple Inc reports record Q1 earnings results"),
        ]
        assert len(_deduplicate(articles)) == 1

    def test_empty_titles_skipped(self):
        articles = [_make_article(title=""), _make_article(title="Apple earnings")]
        result = _deduplicate(articles)
        assert len(result) == 1
        assert result[0]["title"] == "Apple earnings"


class TestNormalise:
    def test_content_snippet_trailer_stripped(self):
        article = _make_article(content="Apple beat estimates this quarter... [+2345 chars]")
        result = _normalise(article, "AAPL", "Apple")
        assert "[+" not in result["content_snippet"]
        assert "Apple beat estimates" in result["content_snippet"]

    def test_content_snippet_capped_at_500_chars(self):
        article = _make_article(content="x" * 600)
        result = _normalise(article, "AAPL", "Apple")
        assert len(result["content_snippet"]) <= 500

    def test_null_fields_default_to_empty_string(self):
        article = _make_article(description=None, content=None)
        result = _normalise(article, "AAPL", "Apple")
        assert result["description"] == ""
        assert result["content_snippet"] == ""

    def test_source_name_extracted(self):
        result = _normalise(_make_article(), "AAPL", "Apple")
        assert result["source"] == "Reuters"


class TestTitleKey:
    def test_strips_punctuation(self):
        assert _title_key("Apple's Q1 Earnings!") == "apples q1 earnings"

    def test_lowercases(self):
        assert _title_key("APPLE EARNINGS") == "apple earnings"


# ─────────────────────────────────────────────────────────────────────────────
# Reddit stub (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class TestRedditClient:
    def test_fetch_posts_not_implemented(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
        monkeypatch.setenv("REDDIT_USER_AGENT", "agent")
        with patch("praw.Reddit"):
            client = RedditClient()
            with pytest.raises(NotImplementedError):
                client.fetch_posts("wallstreetbets")
