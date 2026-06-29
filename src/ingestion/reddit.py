"""Reddit headline ingestion via PRAW (secondary source)."""
from __future__ import annotations

import os
from typing import Any

import praw


class RedditClient:
    def __init__(self) -> None:
        self.reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )

    def fetch_posts(
        self,
        subreddit: str,
        limit: int = 100,
        time_filter: str = "day",
    ) -> list[dict[str, Any]]:
        """Return top post dicts from a subreddit within the time window."""
        raise NotImplementedError
