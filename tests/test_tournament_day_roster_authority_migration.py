from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase/migrations/20261108018000_tournament_day_roster_authority.sql"
)


def _sql() -> str:
    assert MIGRATION.exists()
    return MIGRATION.read_text(encoding="utf-8").lower()


def _function_body(sql: str, function_name: str) -> str:
    match = re.search(
        rf"create\s+or\s+replace\s+function\s+public\.{function_name}\b"
        rf"(?P<header>.*?)\bas\s+\$(?P<tag>[a-z_]*)\$"
        rf"(?P<body>.*?)\$(?P=tag)\$;",
        sql,
        flags=re.DOTALL,
    )
    assert match, f"missing function {function_name}"
    return f"{match.group('header')}\n{match.group('body')}"


def test_draw_readiness_uses_roster_and_registration_not_tracking_state() -> None:
    readiness = _function_body(
        _sql(), "assert_tournament_day_live_draw_ready"
    )

    for structural_table in (
        "tournament_registration_days",
        "tournament_event_options",
        "tournament_event_draws",
        "tournament_teams",
        "tournament_games",
        "players",
        "tournament_registrations",
    ):
        assert structural_table in readiness
    assert "registration_count <> 1" in readiness
    assert "active', 'approved', 'confirmed', 'registered" in readiness
    assert "roster_player_duplicate" in readiness
    assert "round_robin_roster" in readiness

    for informational_dependency in (
        "tournament_registration_check_ins",
        "tournament_commerce_orders",
        "attendance_status",
        "waiver_verified",
        "payment_status",
    ):
        assert informational_dependency not in readiness


def test_atomic_court_scheduler_helper_uses_same_roster_authority() -> None:
    ready = _function_body(_sql(), "tournament_day_live_players_ready")

    assert "cardinality" in ready
    assert "select distinct participant.player_id" in ready
    assert "from public.players" in ready
    assert "player.club_id::text = p_club_id" in ready
    assert "from public.tournament_registrations" in ready
    assert "registration_count <> 1" in ready
    for informational_dependency in (
        "tournament_registration_check_ins",
        "tournament_commerce_orders",
        "attendance_status",
        "waiver_verified",
        "payment_status",
    ):
        assert informational_dependency not in ready


def test_claim_guard_allows_tracking_edits_but_fences_identity_mutation() -> None:
    guard = _function_body(
        _sql(), "guard_tournament_check_in_during_player_claim"
    )

    assert "attendance_status" not in guard
    assert "waiver_verified" not in guard
    assert "attendee_identity_key" in guard
    assert "approved_substitute_player_id" in guard
    assert "approved_substitute_name" in guard
    assert "tournament_day_live_participant_claims" in guard
    assert "claim.released_at is null" in guard
    assert "run.state in ('active', 'paused')" in guard
    assert "changing check-in identity or substitute assignment" in guard


def test_replaced_helpers_remain_service_role_only() -> None:
    sql = _sql()
    for helper in (
        "assert_tournament_day_live_draw_ready",
        "tournament_day_live_players_ready",
        "guard_tournament_check_in_during_player_claim",
    ):
        body = _function_body(sql, helper)
        assert "security invoker" in body
        assert "set search_path = ''" in body
        assert re.search(
            rf"revoke\s+execute\s+on\s+function\s+public\.{helper}\([^;]*\)\s+"
            rf"from\s+public,\s*anon,\s*authenticated",
            sql,
            flags=re.DOTALL,
        )
        assert re.search(
            rf"grant\s+execute\s+on\s+function\s+public\.{helper}\([^;]*\)\s+"
            rf"to\s+service_role",
            sql,
            flags=re.DOTALL,
        )
