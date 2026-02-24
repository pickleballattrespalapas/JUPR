from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BadgeCandidate:
    badge_id: str
    player_id: int
    club_id: str
    context_type: str
    context_id: str | None
    match_id: str | None
    value_json: dict[str, Any] = field(default_factory=dict)
    value_num: float | None = None


@dataclass
class BadgeEvaluationContext:
    club_id: str
    league_id: str | None
    as_of: datetime | None
    ctx: Any
    facts: pd.DataFrame
    matches: pd.DataFrame
