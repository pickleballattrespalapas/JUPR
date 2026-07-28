from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jupr_app.domain.match_processing import (
    build_active_league_metadata_expectations,
)


ROOT = Path(__file__).resolve().parents[1]


def _metadata(**overrides):
    row = {
        "id": 7,
        "club_id": "club",
        "league_name": "Open",
        "k_factor": 32,
        "status": "active",
        "is_active": True,
        "ended_at": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_active_league_snapshot_contains_full_lifecycle_cas() -> None:
    result = build_active_league_metadata_expectations(
        _metadata(),
        club_id="club",
        league_names={"Open", "Overall", "POPUP"},
        default_k_factor=24,
    )

    assert result == [
        {
            "league_name": "Open",
            "expected": {
                "id": 7,
                "club_id": "club",
                "league_name": "Open",
                "k_factor": 32,
                "status": "active",
                "is_active": True,
                "ended_at": None,
            },
        }
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "ended", "is_active": False},
        {"status": "active", "ended_at": "2026-07-27T10:00:00Z"},
        {"status": "", "is_active": False},
    ],
)
def test_inactive_official_league_is_rejected_before_planning(
    overrides,
) -> None:
    with pytest.raises(ValueError, match="no longer an active"):
        build_active_league_metadata_expectations(
            _metadata(**overrides),
            club_id="club",
            league_names={"Open"},
            default_k_factor=32,
        )


def test_singles_and_doubles_forward_the_same_league_guard_snapshot() -> None:
    singles = (
        ROOT / "jupr_app/domain/singles_match_processing.py"
    ).read_text()
    direct = (
        ROOT / "jupr_app/services/direct_match_entry_service.py"
    ).read_text()
    admin_singles = (
        ROOT / "jupr_app/services/admin_singles_match_service.py"
    ).read_text()
    migration = (
        ROOT
        / "supabase/migrations/20260727211500_direct_match_active_league_guard.sql"
    ).read_text()

    assert "build_active_league_metadata_expectations" in singles
    assert "league_metadata_expectations" in singles
    assert "df_meta=df_meta" in direct
    assert "df_meta=df_meta" in admin_singles
    assert "JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE" in migration
    assert "v_match_format = 'singles'" in migration
