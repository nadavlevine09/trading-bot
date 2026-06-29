"""NewsAPI free-tier headline ingestion."""
from __future__ import annotations

import difflib
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

_log = logging.getLogger(__name__)

# Similarity threshold above which two titles are treated as duplicates.
# 0.85 catches syndicated re-prints and minor headline rewrites.
_DEDUP_THRESHOLD = 0.85

# NewsAPI free tier caps at 100 results per request (no further pagination).
_MAX_PAGE_SIZE = 100

# Subsidiary / brand aliases per ticker.  Used in two places:
#   1. The NewsAPI query — OR-joining subsidiary names widens the fetch to
#      articles that mention only the brand (e.g. "Xbox" without "Microsoft").
#   2. The relevance filter — subsidiary names count as valid company mentions
#      so Xbox/Azure/AWS articles are not dropped as off-topic.
#
# Only include proper branded ORG names here.  Generic English words
# ("Prime", "Windows") are excluded: spaCy NER cannot reliably distinguish
# "Amazon Prime" from "Prime Minister", or "Windows" the OS from the common noun.
# Those ambiguous cases are handled by including the parent company name itself.
SUBSIDIARY_MAP: dict[str, list[str]] = {
    "MSFT":  ["Microsoft", "Xbox", "Azure"],
    "AMZN":  ["Amazon", "AWS"],
    "GOOGL": ["Google", "Alphabet", "YouTube"],
    "META":  ["Meta", "Facebook", "Instagram", "WhatsApp"],
}

# ── spaCy NER (lazy-loaded; False = unavailable, NER step is skipped) ──────────

_NLP: "Any" = None


def _get_nlp() -> "Any":
    """Load en_core_web_sm once per process and cache it module-wide."""
    global _NLP
    if _NLP is None:
        try:
            import spacy  # noqa: PLC0415
            _NLP = spacy.load("en_core_web_sm")
            _log.debug("spaCy en_core_web_sm loaded.")
        except (ImportError, OSError) as exc:
            _log.warning(
                "spaCy NER unavailable (%s) — relevance filter falls back to "
                "word-boundary matching only.",
                exc,
            )
            _NLP = False  # sentinel: skip NER
    return _NLP


def _org_matches(entity_text: str, name: str) -> bool:
    """
    True when *name* appears as a whole word within *entity_text*.

    One-directional by design: the known company name must be a complete
    sub-phrase of the extracted entity, not the other way around.

    Examples:
        "Apple Inc."   vs "Apple"         → True   (entity is a longer form)
        "Apple"        vs "Apple"         → True   (exact)
        "Goldman"      vs "Goldman Sachs" → False  (partial — entity is shorter)
        "Appleton"     vs "Apple"         → False  (word boundary prevents match)
    """
    return bool(re.search(r"\b" + re.escape(name) + r"\b", entity_text, re.IGNORECASE))


class RateLimitError(Exception):
    """HTTP 429 — daily request quota exhausted (100 req/day on free tier)."""


class NewsAPIError(Exception):
    """Any other NewsAPI failure (bad key, server error, unexpected status)."""


class NewsAPIClient:
    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["NEWSAPI_KEY"]
        self._session = requests.Session()
        self._session.headers["X-Api-Key"] = self.api_key

    # ── Primary entry point ────────────────────────────────────────────────────

    def fetch_for_ticker(
        self,
        ticker: str,
        company_name: str,
        days_back: int = 7,
    ) -> list[dict[str, Any]]:
        """
        Fetch, normalise, and deduplicate recent headlines for a stock.

        Args:
            ticker:       Exchange symbol, e.g. "AAPL".
            company_name: Human-readable name used in the search, e.g. "Apple".
            days_back:    How many calendar days of history to fetch (max 30 on
                          the free tier).

        Returns:
            List of article dicts with keys:
              ticker, company, title, description, content_snippet,
              source, published_at, url
        """
        now = datetime.now(timezone.utc)
        from_date = (now - timedelta(days=days_back)).date()
        to_date = now.date()

        subs = SUBSIDIARY_MAP.get(ticker, [])

        # Build a deduplicated OR-query: primary name + ticker + any subsidiaries.
        # Quoting each term ensures phrase-level matching in the NewsAPI engine.
        seen_terms: set[str] = set()
        query_parts: list[str] = []
        for term in [company_name, ticker] + subs:
            if term not in seen_terms:
                seen_terms.add(term)
                query_parts.append(f'"{term}"')
        query = " OR ".join(query_parts)

        raw = self.fetch_headlines(query=query, from_date=from_date, to_date=to_date)
        raw = _drop_removed(raw)
        relevant = [a for a in raw if _is_relevant(a, ticker, company_name, extra_terms=subs)]
        unique = _deduplicate(relevant)

        _log.info(
            "%s: %d fetched → %d relevant → %d after dedup",
            ticker, len(raw), len(relevant), len(unique),
        )
        return [_normalise(a, ticker, company_name) for a in unique]

    # ── Low-level fetch ────────────────────────────────────────────────────────

    def fetch_headlines(
        self,
        query: str,
        from_date: date | None = None,
        to_date: date | None = None,
        page_size: int = _MAX_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Hit /v2/everything and return the raw article list from NewsAPI.

        Raises:
            RateLimitError: HTTP 429 (daily quota exhausted).
            NewsAPIError:   Any other API or network failure.
        """
        params: dict[str, Any] = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(page_size, _MAX_PAGE_SIZE),
        }
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        try:
            resp = self._session.get(
                f"{self.BASE_URL}/everything",
                params=params,
                timeout=10,
            )
        except requests.exceptions.RequestException as exc:
            raise NewsAPIError(f"Network error: {exc}") from exc

        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After", "unknown")
            raise RateLimitError(
                f"NewsAPI rate limit hit (free tier: 100 req/day). "
                f"Retry-After: {retry}s"
            )
        if resp.status_code == 401:
            raise NewsAPIError("Invalid or missing NEWSAPI_KEY.")
        if not resp.ok:
            raise NewsAPIError(
                f"NewsAPI HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        if data.get("status") != "ok":
            raise NewsAPIError(
                f"NewsAPI error — {data.get('code')}: {data.get('message')}"
            )

        articles = data.get("articles", [])
        _log.debug("Raw results: %d (total available: %d)", len(articles), data.get("totalResults", 0))
        return articles


# ── Module-level helpers (no state, easy to test in isolation) ─────────────────

def _drop_removed(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter articles NewsAPI marks as [Removed] (title or content = '[Removed]')."""
    return [
        a for a in articles
        if (a.get("title") or "").strip() not in ("[Removed]", "", None)
        and a.get("url") != "https://removed.com"
    ]


def _is_relevant(
    article: dict[str, Any],
    ticker: str,
    company_name: str,
    extra_terms: list[str] | tuple[str, ...] = (),
) -> bool:
    """
    Return True if this article is plausibly about the given company.

    Two-stage check, with a consistent title-only restriction for subsidiary
    aliases in both stages:

    **Stage 1 — spaCy NER (primary)**
    Run once on title + description.  For each ORG entity:
    - ``company_name``: matched anywhere (title or description).
    - ``extra_terms`` (subsidiary aliases): matched only when the entity's
      start position falls within the title portion of the haystack.
      Description-only subsidiary ORGs are excluded — the same reason as
      the string fallback: a PlayStation article whose description says
      "unlike Xbox…" should not be attributed to MSFT.

    **Stage 2 — word-boundary string match (fallback)**
    - ``ticker`` and ``company_name``: checked in title + description.
    - ``extra_terms`` (subsidiary aliases): checked in **title only**.
    """
    title     = article.get("title") or ""
    title_end = len(title)
    haystack  = " ".join([title, article.get("description") or ""])

    # Stage 1: spaCy NER — single pass, offset-gated for extra_terms.
    nlp = _get_nlp()
    if nlp:
        doc = nlp(haystack)
        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue
            # company_name: anywhere in title or description
            if _org_matches(ent.text, company_name):
                return True
            # subsidiary aliases: title only (entity must start before description)
            if ent.start_char < title_end:
                for name in extra_terms:
                    if _org_matches(ent.text, name):
                        return True

    # Stage 2: word-boundary string fallback.
    # Primary terms: match anywhere in title or description.
    if any(
        re.search(r"\b" + re.escape(term) + r"\b", haystack, re.IGNORECASE)
        for term in (ticker, company_name)
    ):
        return True
    # Subsidiary aliases: title only — description-only mentions pull in
    # gaming-adjacent or unrelated articles that reference the brand in passing.
    return any(
        re.search(r"\b" + re.escape(term) + r"\b", title, re.IGNORECASE)
        for term in extra_terms
    )


def _deduplicate(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove near-identical articles caused by syndication or minor headline rewrites.
    Comparison is on normalised title strings; O(n²) but n ≤ 100 on the free tier.
    """
    seen: list[str] = []
    unique: list[dict[str, Any]] = []

    for article in articles:
        key = _title_key(article.get("title") or "")
        if not key:
            continue
        if any(
            difflib.SequenceMatcher(None, key, s).ratio() >= _DEDUP_THRESHOLD
            for s in seen
        ):
            continue
        seen.append(key)
        unique.append(article)

    return unique


def _title_key(title: str) -> str:
    """Lowercase and strip punctuation for similarity comparison."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _normalise(article: dict[str, Any], ticker: str, company: str) -> dict[str, Any]:
    """Map a raw NewsAPI article to the internal headline schema."""
    raw_content = article.get("content") or ""
    # Free tier truncates with " [+XXXX chars]" — strip the trailer.
    content_snippet = re.sub(r"\s*\[.*?chars\]\s*$", "", raw_content).strip()

    return {
        "ticker": ticker,
        "company": company,
        "title": (article.get("title") or "").strip(),
        "description": (article.get("description") or "").strip(),
        "content_snippet": content_snippet[:500],
        "source": (article.get("source") or {}).get("name") or "unknown",
        "published_at": article.get("publishedAt") or "",
        "url": article.get("url") or "",
    }
