from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jupr_app.domain.live_beta_engine import create_round_robin_event, update_round_robin_score
from jupr_app.services.public_live_service import (
    is_public_live_session_row,
    public_live_session_detail,
    public_live_sessions_from_rows,
)


def _live_row(event: dict, *, status: str = "active", session_key: str = "session-1") -> dict:
    return {
        "club_id": "tres_palapas",
        "session_key": session_key,
        "title": event.get("name"),
        "status": status,
        "created_at": "2026-07-02T10:00:00+00:00",
        "updated_at": "2026-07-02T10:05:00+00:00",
        "last_seen_at": "2026-07-02T10:05:00+00:00",
        "expires_at": "2026-07-02T20:00:00+00:00",
        "state": {
            "version": 1,
            "session_key": session_key,
            "event_name": event.get("name"),
            "event_type": event.get("type"),
            "page_state": {
                "event_name": event.get("name"),
                "event": event,
                "admin_roster_rows": [{"private_admin_only": True}],
            },
            "widget_state": {"jupr_live_admin_state_secret_widget": "hidden"},
        },
    }


def test_public_live_detail_projects_scores_without_raw_recovery_state():
    event = create_round_robin_event(
        name="Thursday JUPR Live",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
    )
    first_match_id = event["rounds"][0]["matches"][0]["id"]
    update_round_robin_score(event, first_match_id, 11, 8)

    detail = public_live_session_detail(_live_row(event))

    assert detail["session_key"] == "session-1"
    assert detail["title"] == "Thursday JUPR Live"
    assert detail["event_type"] == "round_robin"
    assert detail["rounds"][0]["matches"][0]["score_a"] == 11
    assert detail["rounds"][0]["matches"][0]["score_b"] == 8
    assert detail["standings"]
    assert "state" not in detail
    assert "page_state" not in detail
    assert "widget_state" not in detail


def test_public_live_sessions_filters_abandoned_and_expired_rows():
    now = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
    event = create_round_robin_event(
        name="Filter Test",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
    )
    active = _live_row(event, session_key="active")
    abandoned = _live_row(event, status="abandoned", session_key="abandoned")
    expired = _live_row(event, session_key="expired")
    expired["expires_at"] = (now - timedelta(minutes=1)).isoformat()

    assert is_public_live_session_row(active, now=now)
    assert not is_public_live_session_row(abandoned, now=now)
    assert not is_public_live_session_row(expired, now=now)

    summaries = public_live_sessions_from_rows([active, abandoned, expired], limit=10)
    assert [row["session_key"] for row in summaries] == ["active"]
