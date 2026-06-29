"""Unit tests for src/ingestion/gdelt.py.  No real BigQuery calls are made."""
from __future__ import annotations

import sys
from datetime import date
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Inject a mock google-cloud-bigquery before any import of the module under
# test, so tests pass even when the library is not installed.
# ---------------------------------------------------------------------------

def _make_mock_bq() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_google, mock_google_cloud, mock_bq) with correct hierarchy."""
    mock_bq           = MagicMock(name="google.cloud.bigquery")
    mock_google_cloud = MagicMock(name="google.cloud")
    mock_google       = MagicMock(name="google")
    mock_google_cloud.bigquery = mock_bq
    mock_google.cloud          = mock_google_cloud
    return mock_google, mock_google_cloud, mock_bq


@pytest.fixture()
def mock_bq(monkeypatch):
    """Patch sys.modules so 'from google.cloud import bigquery' returns a mock."""
    mock_google, mock_google_cloud, mock_bq_mod = _make_mock_bq()
    modules = {
        "google":               mock_google,
        "google.cloud":         mock_google_cloud,
        "google.cloud.bigquery": mock_bq_mod,
    }
    with patch.dict(sys.modules, modules):
        yield mock_bq_mod


@pytest.fixture()
def creds(monkeypatch):
    """Set GOOGLE_APPLICATION_CREDENTIALS to a fake path."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/fake/service-account.json")


def _make_row(
    *,
    date_int: int = 20240315143022,
    url: str = "https://reuters.com/tech/apple-earnings",
    source: str = "Reuters",
    organizations: str = "APPLE INC;WALL STREET",
    v2_organizations: str = "apple inc,#100,#9,#1;wall street,#200,#11,#1",
    v2_all_names: str = "Apple Inc,5;Tim Cook,3",
    extras: str = "<PAGE_TITLE>Apple Beats Q3 Earnings Estimates</PAGE_TITLE>",
) -> MagicMock:
    row = MagicMock()
    row.DATE               = date_int
    row.DocumentIdentifier = url
    row.SourceCommonName   = source
    row.Organizations      = organizations
    row.V2Organizations    = v2_organizations
    row.V2AllNames         = v2_all_names
    row.Extras             = extras
    return row


def _make_client(mock_bq, rows: list, creds):  # noqa: ANN001
    """Construct a GDELTClient backed by the mock BQ module."""
    # Re-import after sys.modules is patched so the lazy import inside __init__
    # resolves to our mock.
    from src.ingestion.gdelt import GDELTClient  # noqa: PLC0415
    mock_bq.Client.return_value.query.return_value.result.return_value = rows
    return GDELTClient()


# ── Instantiation and auth ─────────────────────────────────────────────────────

class TestGDELTClientInit:
    def test_raises_auth_error_when_credentials_not_set(self, mock_bq, monkeypatch):
        # Import first so load_dotenv() inside the module has already run, then
        # remove the credential path so the runtime check in __init__ fails.
        from src.ingestion.gdelt import GDELTAuthError, GDELTClient
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        with pytest.raises(GDELTAuthError, match="GOOGLE_APPLICATION_CREDENTIALS"):
            GDELTClient()

    def test_raises_gdelt_error_when_bq_not_installed(self, creds):
        # Setting sys.modules entry to None is the Python standard sentinel for
        # "this import has permanently failed" — 'from google.cloud import bigquery'
        # will then raise ImportError regardless of whether the package is installed.
        from src.ingestion.gdelt import GDELTClient, GDELTError
        with patch.dict(sys.modules, {"google.cloud.bigquery": None}):
            with pytest.raises(GDELTError, match="google-cloud-bigquery is not installed"):
                GDELTClient()

    def test_raises_auth_error_when_client_creation_fails(self, mock_bq, creds):
        mock_bq.Client.side_effect = RuntimeError("credentials file not found")
        from src.ingestion.gdelt import GDELTAuthError, GDELTClient
        with pytest.raises(GDELTAuthError, match="Failed to initialise BigQuery client"):
            GDELTClient()

    def test_instantiation_succeeds_with_valid_credentials(self, mock_bq, creds):
        from src.ingestion.gdelt import GDELTClient
        client = GDELTClient()
        mock_bq.Client.assert_called_once()
        assert client is not None

    def test_project_arg_forwarded_to_bq_client(self, mock_bq, creds):
        from src.ingestion.gdelt import GDELTClient
        GDELTClient(project="my-gcp-project")
        _, kwargs = mock_bq.Client.call_args
        assert kwargs.get("project") == "my-gcp-project"

    def test_project_falls_back_to_env_var(self, mock_bq, creds, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        from src.ingestion.gdelt import GDELTClient
        GDELTClient()
        _, kwargs = mock_bq.Client.call_args
        assert kwargs.get("project") == "env-project"


# ── fetch_for_ticker ───────────────────────────────────────────────────────────

class TestFetchForTicker:
    def test_returns_normalised_records(self, mock_bq, creds):
        row = _make_row()
        client = _make_client(mock_bq, [row], creds)
        results = client.fetch_for_ticker(
            "AAPL", "Apple", date(2024, 3, 1), date(2024, 3, 31)
        )
        assert len(results) == 1
        r = results[0]
        assert r["ticker"]       == "AAPL"
        assert r["company"]      == "Apple"
        assert r["title"]        == "Apple Beats Q3 Earnings Estimates"
        assert r["source"]       == "Reuters"
        assert r["url"]          == "https://reuters.com/tech/apple-earnings"
        assert r["published_at"] == "2024-03-15T14:30:22Z"

    def test_empty_result_from_bigquery(self, mock_bq, creds):
        client = _make_client(mock_bq, [], creds)
        results = client.fetch_for_ticker(
            "AAPL", "Apple", date(2024, 3, 1), date(2024, 3, 31)
        )
        assert results == []

    def test_raises_gdelt_error_on_query_failure(self, mock_bq, creds):
        from src.ingestion.gdelt import GDELTClient, GDELTError
        mock_bq.Client.return_value.query.return_value.result.side_effect = (
            RuntimeError("quota exceeded")
        )
        client = GDELTClient()
        with pytest.raises(GDELTError, match="BigQuery query failed for AAPL"):
            client.fetch_for_ticker(
                "AAPL", "Apple", date(2024, 3, 1), date(2024, 3, 31)
            )

    def test_correct_date_integer_range_passed_as_params(self, mock_bq, creds):
        client = _make_client(mock_bq, [], creds)
        client.fetch_for_ticker(
            "AAPL", "Apple", date(2024, 3, 1), date(2024, 3, 31)
        )
        call_kwargs = mock_bq.Client.return_value.query.call_args
        job_config  = call_kwargs[1]["job_config"]

        # Extract parameter values from the mock ScalarQueryParameter calls.
        param_calls = mock_bq.ScalarQueryParameter.call_args_list
        params = {c[0][0]: c[0][2] for c in param_calls}

        assert params["from_int"]  == 20240301_000000
        assert params["to_int"]    == 20240331_999999

    def test_company_name_lowercased_in_pattern(self, mock_bq, creds):
        client = _make_client(mock_bq, [], creds)
        client.fetch_for_ticker(
            "AAPL", "Apple Inc", date(2024, 1, 1), date(2024, 1, 31)
        )
        param_calls = mock_bq.ScalarQueryParameter.call_args_list
        params = {c[0][0]: c[0][2] for c in param_calls}
        assert params["name_pattern"] == "%apple inc%"
        # ticker_pattern was removed — it caused false-positive floods
        # (e.g. %googl% matched all "google" org names, %meta% matched "meta platforms")
        assert "ticker_pattern" not in params

    def test_deduplicates_rows_with_same_url(self, mock_bq, creds):
        row1 = _make_row(url="https://example.com/story")
        row2 = _make_row(url="https://example.com/story")  # duplicate URL
        row3 = _make_row(url="https://example.com/other")
        client = _make_client(mock_bq, [row1, row2, row3], creds)
        results = client.fetch_for_ticker(
            "AAPL", "Apple", date(2024, 3, 1), date(2024, 3, 31)
        )
        urls = [r["url"] for r in results]
        assert len(results) == 2
        assert urls.count("https://example.com/story") == 1

    def test_output_schema_contains_all_required_keys(self, mock_bq, creds):
        """All keys expected by SentimentScorer.score() must be present."""
        row    = _make_row()
        client = _make_client(mock_bq, [row], creds)
        result = client.fetch_for_ticker(
            "AAPL", "Apple", date(2024, 3, 1), date(2024, 3, 31)
        )[0]
        required = {"ticker", "company", "title", "description",
                    "content_snippet", "source", "published_at", "url"}
        assert required.issubset(result.keys())

    def test_description_and_content_snippet_are_empty_strings(self, mock_bq, creds):
        row    = _make_row()
        client = _make_client(mock_bq, [row], creds)
        result = client.fetch_for_ticker(
            "AAPL", "Apple", date(2024, 3, 1), date(2024, 3, 31)
        )[0]
        assert result["description"]     == ""
        assert result["content_snippet"] == ""


# ── _normalise_row ─────────────────────────────────────────────────────────────

class TestNormaliseRow:
    def _call(self, **overrides):
        from src.ingestion.gdelt import _normalise_row  # noqa: PLC0415
        row = _make_row(**overrides)
        return _normalise_row(row, "AAPL", "Apple")

    def test_extracts_page_title_from_extras(self):
        r = self._call(extras="<PAGE_TITLE>  Apple Revenue Surges  </PAGE_TITLE>")
        assert r["title"] == "Apple Revenue Surges"

    def test_page_title_extraction_is_case_insensitive(self):
        r = self._call(extras="<page_title>Apple News</page_title>")
        assert r["title"] == "Apple News"

    def test_falls_back_to_url_slug_when_no_page_title(self):
        r = self._call(
            extras="",
            url="https://example.com/apple-earnings-q3-2024.html",
        )
        assert r["title"] == "Apple Earnings Q3 2024"

    def test_handles_none_extras(self):
        row = _make_row()
        row.Extras = None
        from src.ingestion.gdelt import _normalise_row
        result = _normalise_row(row, "AAPL", "Apple")
        # Should not raise; title falls back to URL slug.
        assert isinstance(result["title"], str)

    def test_source_is_unknown_when_source_common_name_empty(self):
        r = self._call(source="")
        assert r["source"] == "unknown"

    def test_source_is_unknown_when_source_common_name_none(self):
        row = _make_row()
        row.SourceCommonName = None
        from src.ingestion.gdelt import _normalise_row
        result = _normalise_row(row, "AAPL", "Apple")
        assert result["source"] == "unknown"

    def test_published_at_converts_date_integer(self):
        r = self._call(date_int=20240315143022)
        assert r["published_at"] == "2024-03-15T14:30:22Z"

    def test_url_is_document_identifier(self):
        r = self._call(url="https://wsj.com/articles/apple-story")
        assert r["url"] == "https://wsj.com/articles/apple-story"

    def test_ticker_and_company_passed_through(self):
        row = _make_row()
        from src.ingestion.gdelt import _normalise_row
        result = _normalise_row(row, "MSFT", "Microsoft")
        assert result["ticker"]  == "MSFT"
        assert result["company"] == "Microsoft"


# ── _gdelt_date_to_iso ─────────────────────────────────────────────────────────

class TestGdeltDateToIso:
    def _call(self, s: str) -> str:
        from src.ingestion.gdelt import _gdelt_date_to_iso
        return _gdelt_date_to_iso(s)

    def test_full_14_digit_timestamp(self):
        assert self._call("20240315143022") == "2024-03-15T14:30:22Z"

    def test_midnight_timestamp(self):
        assert self._call("20240101000000") == "2024-01-01T00:00:00Z"

    def test_8_digit_date_only(self):
        assert self._call("20240315") == "2024-03-15T00:00:00Z"

    def test_empty_string_returns_empty(self):
        assert self._call("") == ""

    def test_whitespace_only_returns_empty(self):
        assert self._call("   ") == ""

    def test_integer_converted_to_string_works(self):
        # Row.DATE may be a Python int from BigQuery's INT64 type.
        from src.ingestion.gdelt import _gdelt_date_to_iso
        assert _gdelt_date_to_iso(str(20240315143022)) == "2024-03-15T14:30:22Z"

    def test_longer_than_14_digits_uses_first_14(self):
        # Extra digits (if any) are silently ignored — only first 14 are used.
        assert self._call("20240315143022999") == "2024-03-15T14:30:22Z"
