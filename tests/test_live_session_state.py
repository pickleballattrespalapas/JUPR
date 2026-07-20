from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jupr_app.domain.live_session_repo import (
    abandon_expired_live_sessions,
    is_live_session_expired,
    is_missing_live_sessions_table_error,
    is_restorable_live_session,
)
from jupr_app.services.live_session_state import (
    build_live_state_payload,
    hydrate_page_state_from_live_state,
    hydrate_widget_state_from_live_state,
)


def test_build_live_state_payload_curates_page_and_widget_state():
    page_state = {
        "event_name": "Thursday Night",
        "type_label": "Round Robin",
        "participant_count": 8,
        "event": {
            "type": "round_robin",
            "rounds": [
                {"number": 1, "matches": [{"id": "r1m1", "scoreA": 11, "scoreB": 8}]}
            ],
        },
        "not_recoverable": "do not persist this",
    }
    st_session_state = {
        "jupr_live_admin_state_rr_r1m1_a": 11,
        "jupr_live_admin_state_rr_r1m1_b": 8,
        "unrelated_session_value": "ignored",
    }

    payload = build_live_state_payload(
        page_state,
        club_id="tres_palapas",
        session_key="session-1",
        config_state_key="jupr_live_admin_state",
        st_session_state=st_session_state,
    )

    assert payload["version"] == 1
    assert payload["club_id"] == "tres_palapas"
    assert payload["session_key"] == "session-1"
    assert payload["event_name"] == "Thursday Night"
    assert payload["event_type"] == "round_robin"
    assert payload["page_state"]["event"]["rounds"][0]["matches"][0]["scoreA"] == 11
    assert "not_recoverable" not in payload["page_state"]
    assert payload["widget_state"] == {
        "jupr_live_admin_state_rr_r1m1_a": 11,
        "jupr_live_admin_state_rr_r1m1_b": 8,
    }


def test_hydrate_live_state_restores_page_and_widget_state():
    live_state = {
        "page_state": {
            "event_name": "Recovered Event",
            "type_label": "League / Ladder",
            "participant_count": 12,
            "not_recoverable": "ignored",
        },
        "widget_state": {
            "jupr_live_admin_state_lg_m1_a": 15,
            "jupr_live_admin_state_lg_m1_b": 13,
            "other_key": "ignored",
        },
    }
    page_state: dict = {}
    session_state: dict = {}

    hydrate_page_state_from_live_state(page_state, live_state)
    hydrate_widget_state_from_live_state(
        session_state,
        live_state,
        config_state_key="jupr_live_admin_state",
    )

    assert page_state == {
        "event_name": "Recovered Event",
        "type_label": "League / Ladder",
        "participant_count": 12,
    }
    assert session_state == {
        "jupr_live_admin_state_lg_m1_a": 15,
        "jupr_live_admin_state_lg_m1_b": 13,
    }


def test_live_session_restorable_status_and_expiration():
    now = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
    active_future = {
        "status": "active",
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }
    active_past = {
        "status": "active",
        "expires_at": (now - timedelta(seconds=1)).isoformat(),
    }
    abandoned_future = {
        "status": "abandoned",
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }

    assert is_restorable_live_session(active_future, now=now)
    assert is_live_session_expired(active_past, now=now)
    assert not is_restorable_live_session(active_past, now=now)
    assert not is_restorable_live_session(abandoned_future, now=now)


def test_missing_live_sessions_table_error_detection():
    class FakePostgrestError(Exception):
        code = "PGRST205"
        message = "Could not find the table 'public.live_sessions' in the schema cache"

    assert is_missing_live_sessions_table_error(FakePostgrestError("live_sessions missing"))


def test_expired_cleanup_is_explicitly_club_scoped():
    class QuerySpy:
        def __init__(self):
            self.filters = []

        def update(self, _payload):
            return self

        def eq(self, key, value):
            self.filters.append((key, value))
            return self

        def lt(self, key, value):
            self.filters.append((key, value))
            return self

        def execute(self):
            return type("Response", (), {"data": []})()

    query = QuerySpy()
    supabase = type("Supabase", (), {"table": lambda self, _name: query})()

    abandon_expired_live_sessions(
        supabase,
        club_id="club-a",
        now_iso="2026-07-19T12:00:00+00:00",
    )

    assert ("club_id", "club-a") in query.filters


def test_expired_cleanup_rejects_empty_club_scope():
    supabase = type("Supabase", (), {"table": lambda self, _name: None})()

    with pytest.raises(ValueError, match="club_id is required"):
        abandon_expired_live_sessions(supabase, club_id="")
