from __future__ import annotations

import pytest

from jupr_app.services.admin_tournament_draw_service import (
    cancel_admin_tournament_empty_draw,
    cancel_admin_tournament_empty_event,
)
from tests.test_admin_match_log_service import FakeSupabase


UPDATED = "2026-08-25T12:00:00Z"


def _tables() -> dict[str, list[dict]]:
    return {
        "tournaments": [
            {
                "id": "tour-1",
                "club_id": "club",
                "name": "Summer Classic",
                "status": "PUBLISHED",
                "updated_at": UPDATED,
            }
        ],
        "tournament_event_options": [
            {
                "id": "event-empty",
                "tournament_id": "tour-1",
                "event_family_label": "Women's Doubles",
                "division_name": "Women's 3.0",
                "gender_restriction": "WOMEN",
                "enabled": True,
                "status": "active",
            }
        ],
        "tournament_event_draws": [
            {
                "id": "draw-empty",
                "tournament_id": "tour-1",
                "event_option_id": "event-empty",
                "name": "Women's 3.0",
                "status": "active",
                "updated_at": UPDATED,
            }
        ],
        "tournament_registration_selections": [],
        "tournament_registration_team_links": [],
        "tournament_registration_team_members": [],
        "tournament_teams": [],
        "tournament_games": [],
        "tournament_podium": [],
        "tournament_day_live_draws": [],
        "player_badges": [],
        "admin_activity_log": [],
    }


def test_cancel_empty_draw_retains_row_and_excludes_it_from_closeout(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables()

    result = cancel_admin_tournament_empty_draw(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-empty",
        expected_draw_updated_at=UPDATED,
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="CANCEL EMPTY DRAW",
        allow_non_atomic_test_adapter=True,
    )

    assert result["draw"]["status"] == "cancelled"
    assert len(tables["tournament_event_draws"]) == 1
    assert tables["admin_activity_log"][-1]["action_type"] == "cancel_empty_tournament_draw_admin"


def test_cancel_empty_draw_refuses_any_badge_history(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables()
    tables["player_badges"] = [
        {
            "id": "badge-1",
            "club_id": "club",
            "context_type": "tournament",
            "context_id": "tour-1:draw:draw-empty:podium:1",
            "revoked_at": UPDATED,
        }
    ]

    with pytest.raises(ValueError, match="player_badges=1"):
        cancel_admin_tournament_empty_draw(
            FakeSupabase(tables),
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-empty",
            expected_draw_updated_at=UPDATED,
            actor_email="director@example.com",
            actor_role="club_owner",
            confirmation_text="CANCEL EMPTY DRAW",
            allow_non_atomic_test_adapter=True,
        )


def test_cancel_unused_event_disables_but_never_deletes_configuration(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables()
    tables["tournament_event_draws"] = []

    result = cancel_admin_tournament_empty_event(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        event_option_id="event-empty",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="CANCEL EMPTY EVENT",
        allow_non_atomic_test_adapter=True,
    )

    assert result["event_option"]["enabled"] is False
    assert result["event_option"]["status"] == "cancelled"
    assert len(tables["tournament_event_options"]) == 1


def test_cancel_empty_event_refuses_registration_or_partner_evidence(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables()
    tables["tournament_event_draws"] = []
    tables["tournament_registration_team_links"] = [
        {
            "id": "link-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-empty",
        }
    ]

    with pytest.raises(ValueError, match="team_links=1"):
        cancel_admin_tournament_empty_event(
            FakeSupabase(tables),
            club_id="club",
            tournament_id="tour-1",
            event_option_id="event-empty",
            actor_email="director@example.com",
            actor_role="club_owner",
            confirmation_text="CANCEL EMPTY EVENT",
            allow_non_atomic_test_adapter=True,
        )


def test_deployed_empty_event_cancel_requires_atomic_rpc(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    tables = _tables()

    with pytest.raises(PermissionError, match="atomic database RPC"):
        cancel_admin_tournament_empty_event(
            FakeSupabase(tables),
            club_id="club",
            tournament_id="tour-1",
            event_option_id="event-empty",
            actor_email="director@example.com",
            actor_role="club_owner",
            confirmation_text="CANCEL EMPTY EVENT",
            atomic=False,
        )
