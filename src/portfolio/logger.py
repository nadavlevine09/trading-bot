"""Structured JSON logging for every trade decision."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from .models import TradeDecision

_log = logging.getLogger("portfolio.decisions")


def _serialise(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class DecisionLogger:
    def log(self, decision: TradeDecision) -> None:
        record = {
            "ts": decision.timestamp.isoformat(),
            "action": decision.action,
            "ticker": decision.ticker,
            "qty": decision.qty,
            "price": decision.price,
            "reason": decision.reason,
            "portfolio": decision.portfolio_snapshot,
        }
        _log.info(json.dumps(record, default=_serialise))
