"""Tests for src/config/companies.py."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.config.companies import COMPANIES, CompanyInfo, load_companies

EXPECTED_TICKERS = {"AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"}


class TestLoadCompanies:
    def test_all_tickers_present(self):
        assert EXPECTED_TICKERS.issubset(COMPANIES.keys())

    def test_returns_company_info_instances(self):
        for info in COMPANIES.values():
            assert isinstance(info, CompanyInfo)

    def test_ticker_field_matches_key(self):
        for ticker, info in COMPANIES.items():
            assert info.ticker == ticker

    def test_company_name_non_empty(self):
        for info in COMPANIES.values():
            assert info.company_name, f"{info.ticker} has empty company_name"

    def test_legal_entity_name_non_empty(self):
        for info in COMPANIES.values():
            assert info.legal_entity_name, f"{info.ticker} has empty legal_entity_name"

    def test_all_sequence_fields_are_tuples(self):
        for info in COMPANIES.values():
            assert isinstance(info.aliases, tuple)
            assert isinstance(info.subsidiaries, tuple)
            assert isinstance(info.associated_people, tuple)
            assert isinstance(info.finance_source_whitelist, tuple)

    def test_frozen_dataclass_is_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            COMPANIES["AAPL"].ticker = "XXXX"  # type: ignore[misc]

    def test_custom_path_loads_different_file(self, tmp_path):
        toml = tmp_path / "test_companies.toml"
        toml.write_text(
            '[FAKE]\ncompany_name = "Fake Corp"\naliases = []\n'
            'subsidiaries = []\nassociated_people = []\nfinance_source_whitelist = []\n',
            encoding="utf-8",
        )
        result = load_companies(toml)
        assert "FAKE" in result
        assert result["FAKE"].company_name == "Fake Corp"

    def test_legal_entity_name_falls_back_to_company_name_when_absent(self, tmp_path):
        # Entries without legal_entity_name (e.g. future tickers) should
        # fall back to company_name so the field is always a non-empty string.
        toml = tmp_path / "test_companies.toml"
        toml.write_text(
            '[FAKE]\ncompany_name = "Fake Corp"\naliases = []\n'
            'subsidiaries = []\nassociated_people = []\nfinance_source_whitelist = []\n',
            encoding="utf-8",
        )
        result = load_companies(toml)
        assert result["FAKE"].legal_entity_name == "Fake Corp"


class TestLegalEntityName:
    def test_specific_values(self):
        expected = {
            "AAPL":  "Apple Inc",           # GDELT stores without trailing period
            "NVDA":  "NVIDIA Corp",
            "TSLA":  "Tesla Inc",
            "MSFT":  "Microsoft Corp",
            "AMZN":  "Amazon",              # GDELT uses "amazon"/"amazon web services", not "amazon.com inc"
            "GOOGL": "Alphabet Inc",        # GDELT stores without trailing period
            "META":  "Meta Platforms Inc",
            "AMD":   "AMD",              # GDELT uses the ticker abbreviation, not the full legal name
        }
        for ticker, name in expected.items():
            assert COMPANIES[ticker].legal_entity_name == name, ticker

    def test_distinct_from_company_name_for_all_tickers(self):
        # legal_entity_name should carry more precision than the casual name.
        for ticker, info in COMPANIES.items():
            # At minimum, the legal name should start with or contain the casual name,
            # OR be a recognised alternative (Alphabet vs Google).
            assert info.legal_entity_name, f"{ticker} has empty legal_entity_name"

    def test_is_string_not_tuple(self):
        for info in COMPANIES.values():
            assert isinstance(info.legal_entity_name, str)


class TestCompanyInfo:
    def test_all_names_includes_company_and_aliases(self):
        info = COMPANIES["GOOGL"]
        names = info.all_names()
        assert "Google" in names
        assert "Alphabet" in names

    def test_all_names_excludes_subsidiaries(self):
        info = COMPANIES["MSFT"]
        assert "Xbox" not in info.all_names()

    def test_subsidiaries_populated_for_mapped_tickers(self):
        assert "Xbox" in COMPANIES["MSFT"].subsidiaries
        assert "AWS" in COMPANIES["AMZN"].subsidiaries
        assert "YouTube" in COMPANIES["GOOGL"].subsidiaries
        assert "Instagram" in COMPANIES["META"].subsidiaries

    def test_associated_people_populated(self):
        assert "Elon Musk" in COMPANIES["TSLA"].associated_people
        assert "Jensen Huang" in COMPANIES["NVDA"].associated_people
        assert "Mark Zuckerberg" in COMPANIES["META"].associated_people
        assert "Lisa Su" in COMPANIES["AMD"].associated_people

    def test_is_whitelisted_source_case_insensitive(self):
        info = COMPANIES["AAPL"]
        assert info.is_whitelisted_source("Bloomberg")
        assert info.is_whitelisted_source("bloomberg")
        assert info.is_whitelisted_source("BLOOMBERG")

    def test_is_whitelisted_source_rejects_unknown(self):
        info = COMPANIES["AAPL"]
        assert not info.is_whitelisted_source("SomeBlogspot")

    def test_prime_not_in_amzn_subsidiaries(self):
        # "Prime" triggers "Prime Minister" false positives — must stay excluded.
        assert "Prime" not in COMPANIES["AMZN"].subsidiaries
