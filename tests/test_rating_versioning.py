from __future__ import annotations

from pathlib import Path

from jupr_app.domain.matches.persistence import build_match_row
from jupr_app.domain.rating_policy import (
    RATING_ALGORITHM_VERSION,
    RATING_PARAMETER_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "20261109000000_rating_calculation_versioning.sql"


def test_match_row_carries_replayable_rating_metadata() -> None:
    row = build_match_row(
        club_id="club",
        dt_val="2026-09-06T00:00:00+00:00",
        league_name="League A",
        pids=(1, 2, 3, 4),
        scores=(11, 8),
        stored_elo_delta=8,
        match_type="League",
        week_tag="Week 1",
        start_ratings=(1200, 1200, 1200, 1200),
        end_ratings=(1208, 1208, 1192, 1192),
        context={},
        rating_scope="overall_and_league",
        overall_k_factor=32,
        league_k_factor=24,
    )

    assert row["rating_algorithm_version"] == RATING_ALGORITHM_VERSION
    assert row["rating_parameter_version"] == RATING_PARAMETER_VERSION
    assert row["rating_parameters"]["overall_k_factor"] == 32.0
    assert row["rating_parameters"]["league_k_factor"] == 24.0
    assert row["rating_parameters"]["winner_must_gain"] is True
    assert row["rating_parameters"]["loser_may_gain_for_outperformance"] is True


def test_versioning_migration_is_server_only_and_backfills_matches() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "rating_algorithm_version" in sql
    assert "rating_parameter_version" in sql
    assert "rating_parameters" in sql
    assert "enable row level security" in sql
    assert "revoke all on table public.rating_calculation_versions from anon, authenticated" in sql
    assert "update public.matches" in sql
    assert "create trigger stamp_inserted_match_rating_calculation_version" in sql
    assert "create trigger stamp_replayed_match_rating_calculation_version" in sql


def test_public_faq_promises_the_actual_flat_policy() -> None:
    faq = (ROOT / "apps" / "web" / "app" / "faq" / "page.tsx").read_text(encoding="utf-8")

    assert "same calculation rules apply from day one" in faq
    assert "no provisional period" in faq
    assert "It does not use recency, reliability" in faq
    assert "may move more quickly at first" not in faq
    assert "consistency over time" not in faq
