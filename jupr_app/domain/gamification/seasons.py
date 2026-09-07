"""Admin-defined club seasons for badge calculations.

Dates are inclusive in the season's configured timezone. Stable season IDs,
not editable names or calendar years, identify awards. No season is invented
when configuration is absent. Storage and admin controls supply badge_seasons
on the evaluation context when the reactivation release is enabled.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from jupr_app.data.paged_reads import read_all_rows

from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext


@dataclass(frozen=True)
class BadgeSeason:
    id: str
    club_id: str
    name: str
    start_date: date
    end_date: date
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.club_id.strip() or not self.name.strip():
            raise ValueError("A season needs an ID, club, and name.")
        if self.end_date < self.start_date or self.end_date == date.max:
            raise ValueError("The season end date must be on or after its start date.")
        ZoneInfo(self.timezone)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> BadgeSeason:
        return cls(
            id=str(row.get("id") or "").strip(),
            club_id=str(row.get("club_id") or "").strip(),
            name=str(row.get("name") or "").strip(),
            start_date=date.fromisoformat(str(row["start_date"])),
            end_date=date.fromisoformat(str(row["end_date"])),
            timezone=str(row.get("timezone") or "UTC"),
        )

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_date, tz=self.timezone).tz_convert("UTC")

    @property
    def end_exclusive(self) -> pd.Timestamp:
        # Calendar-day arithmetic preserves local midnight across DST changes.
        return pd.Timestamp(self.end_date + timedelta(days=1), tz=self.timezone).tz_convert("UTC")

    @property
    def context_id(self) -> str:
        return f"badge-season:{self.id}"

    def evidence(self) -> dict[str, str]:
        return {
            "season_id": self.id,
            "season_name": self.name,
            "season_start": self.start_date.isoformat(),
            "season_end": self.end_date.isoformat(),
            "season_timezone": self.timezone,
        }


def validate_badge_seasons(
    rows: Iterable[Mapping[str, Any]], *, club_id: str
) -> list[BadgeSeason]:
    seasons = [BadgeSeason.from_row(row) for row in rows if str(row.get("club_id")) == str(club_id)]
    if len({season.id for season in seasons}) != len(seasons):
        raise ValueError("Season IDs must be unique within the club.")
    seasons.sort(key=lambda season: season.start)
    for previous, current in zip(seasons, seasons[1:]):
        if current.start < previous.end_exclusive:
            raise ValueError("Badge seasons for the same club cannot overlap.")
    return seasons


def evaluation_time(ctx: BadgeEvaluationContext) -> pd.Timestamp:
    return pd.to_datetime(ctx.as_of, utc=True) if ctx.as_of is not None else pd.Timestamp.now(tz="UTC")


def season_match_groups(
    ctx: BadgeEvaluationContext, *, completed_only: bool = False
) -> Iterator[tuple[BadgeSeason, int, pd.DataFrame]]:
    """Group eligible match facts using only this club's configured seasons."""
    raw_seasons = getattr(ctx.ctx, "badge_seasons", None)
    if raw_seasons is None and getattr(ctx.ctx, "supabase", None) is not None:
        raw_seasons = read_all_rows(lambda: ctx.ctx.supabase.table("badge_seasons").select("*").eq("club_id", ctx.club_id), order="start_date")
        ctx.ctx.badge_seasons = raw_seasons
    if raw_seasons is None or ctx.facts.empty:
        return
    seasons = validate_badge_seasons(raw_seasons, club_id=ctx.club_id)
    facts = ctx.facts.copy()
    if "club_id" in facts:
        facts = facts[facts["club_id"].astype(str) == str(ctx.club_id)].copy()
    facts["date_dt"] = pd.to_datetime(facts["date_dt"], utc=True, errors="coerce")
    facts = facts.dropna(subset=["date_dt", "player_id", "match_id"])
    facts = facts.sort_values(["date_dt", "match_id"]).drop_duplicates(["player_id", "match_id"])
    cutoff = evaluation_time(ctx)
    facts = facts[facts["date_dt"] <= cutoff]
    for season in seasons:
        if completed_only and cutoff < season.end_exclusive:
            continue
        selected = facts[(facts["date_dt"] >= season.start) & (facts["date_dt"] < season.end_exclusive)]
        for player_id, group in selected.groupby("player_id"):
            yield season, int(player_id), group
