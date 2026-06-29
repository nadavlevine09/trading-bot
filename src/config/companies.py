"""
Company reference table loader.

Usage
-----
    from src.config.companies import COMPANIES, CompanyInfo

    info: CompanyInfo = COMPANIES["AAPL"]
    info.company_name          # "Apple"        — short form for NewsAPI queries
    info.legal_entity_name     # "Apple Inc."   — SEC/GDELT form for BQ queries
    info.subsidiaries          # ("Beats",)
    info.associated_people     # ("Tim Cook",)

The module-level ``COMPANIES`` dict is loaded once at import time from
``config/companies.toml`` (relative to the repo root).  Call
``load_companies(custom_path)`` to load a different file in tests.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "config" / "companies.toml"


@dataclass(frozen=True)
class CompanyInfo:
    ticker: str
    company_name: str
    legal_entity_name: str
    aliases: tuple[str, ...]
    subsidiaries: tuple[str, ...]
    associated_people: tuple[str, ...]
    finance_source_whitelist: tuple[str, ...]

    def all_names(self) -> tuple[str, ...]:
        """All name forms: company_name + aliases (not subsidiaries)."""
        return (self.company_name, *self.aliases)

    def is_whitelisted_source(self, source: str) -> bool:
        """Case-insensitive check against finance_source_whitelist."""
        lower = source.lower()
        return any(w.lower() == lower for w in self.finance_source_whitelist)


def load_companies(path: Path = _DEFAULT_PATH) -> dict[str, CompanyInfo]:
    """
    Parse ``companies.toml`` and return a ticker → CompanyInfo mapping.

    Each top-level TOML key is treated as a ticker symbol.  Missing optional
    fields (aliases, subsidiaries, associated_people, finance_source_whitelist)
    default to empty tuples.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    result: dict[str, CompanyInfo] = {}
    for ticker, data in raw.items():
        if not isinstance(data, dict):
            continue
        result[ticker] = CompanyInfo(
            ticker=ticker,
            company_name=data["company_name"],
            legal_entity_name=data.get("legal_entity_name") or data["company_name"],
            aliases=tuple(data.get("aliases", [])),
            subsidiaries=tuple(data.get("subsidiaries", [])),
            associated_people=tuple(data.get("associated_people", [])),
            finance_source_whitelist=tuple(data.get("finance_source_whitelist", [])),
        )
    return result


# Pre-loaded singleton — safe to import directly.
COMPANIES: dict[str, CompanyInfo] = load_companies()
