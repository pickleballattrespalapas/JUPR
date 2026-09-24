from __future__ import annotations

import os

from fastapi import HTTPException


TEAM_LEAGUES_FEATURE_FLAG = "JUPR_ENABLE_TEAM_LEAGUES"
TRUTHY = {"1", "true", "yes", "y", "on"}
LOCAL_TEST_ENVIRONMENTS = {"local", "test", "development", "dev"}


def team_leagues_enabled() -> bool:
    configured = os.getenv(TEAM_LEAGUES_FEATURE_FLAG, "").strip().lower()
    if configured:
        return configured in TRUTHY
    return os.getenv("JUPR_ENV", "").strip().lower() in LOCAL_TEST_ENVIRONMENTS


def require_team_leagues_enabled_or_403(club_id: str | None = None) -> None:
    # Public routes check the global flag before resolving a slug, then check
    # the resolved club. Admin routes already have a club id at entry.
    outside_production_scope = (
        os.getenv("JUPR_ENV", "").strip().lower() == "production"
        and club_id is not None
        and club_id != "tres_palapas"
    )
    if not team_leagues_enabled() or outside_production_scope:
        raise HTTPException(
            status_code=403,
            detail="Team leagues are temporarily unavailable.",
        )
