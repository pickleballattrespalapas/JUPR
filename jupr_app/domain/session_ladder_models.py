from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal


class SessionState(str, Enum):
    DRAFT = "draft"
    ROSTER_OPEN = "roster_open"
    SEEDED_LOCKED = "seeded_locked"
    ROUND_1_ACTIVE = "round_1_active"
    ROUND_1_CLOSED = "round_1_closed"
    ROUND_2_ACTIVE = "round_2_active"
    ROUND_2_CLOSED = "round_2_closed"
    ROUND_3_ACTIVE = "round_3_active"
    ROUND_3_CLOSED = "round_3_closed"
    COMPLETED = "completed"
    PUBLISHED = "published"


SESSION_STATE_VALUES = tuple(state.value for state in SessionState)
RosterStatus = Literal["EXPECTED", "CHECKED_IN", "NO_SHOW", "WALK_IN"]
CourtPodState = Literal["planned", "in_progress", "complete", "void"]


@dataclass(frozen=True)
class SessionRow:
    id: str
    club_id: str
    league_id: str
    season_id: str | None
    session_starts_at: datetime
    session_ends_at: datetime | None
    courts_available: int
    players_per_court: int
    state: SessionState
    created_by: str
    updated_by: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SessionRosterEntryRow:
    id: str
    club_id: str
    session_id: str
    player_id: int
    status: RosterStatus
    rating_snapshot: Decimal
    seed_order: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class CourtPodRow:
    id: str
    club_id: str
    session_id: str
    round_number: int
    court_number: int
    state: CourtPodState
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class CourtPodPlayerRow:
    id: str
    club_id: str
    session_id: str
    court_pod_id: str
    player_id: int
    player_label: str | None
    player_order: int
    created_at: datetime


@dataclass(frozen=True)
class GameRow:
    id: str
    club_id: str
    session_id: str
    court_pod_id: str
    game_number: int
    team_a_player_ids: tuple[int, int]
    team_b_player_ids: tuple[int, int]
    score_a: int | None
    score_b: int | None
    edited_by: str | None
    created_at: datetime
    updated_at: datetime
