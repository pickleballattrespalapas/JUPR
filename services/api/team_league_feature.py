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


def require_team_leagues_enabled_or_403() -> None:
    if not team_leagues_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Team leagues are disabled. Enable "
                f"{TEAM_LEAGUES_FEATURE_FLAG} only on the isolated staging "
                "candidate."
            ),
        )
