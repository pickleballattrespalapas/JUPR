from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain.admin.roles import (
    ROLE_CLUB_OWNER,
    ROLE_READ_ONLY,
    ROLE_SCOREKEEPER,
    ROLE_SUPER_ADMIN,
)
from jupr_app.domain.admin_activity_log import (
    build_activity_payload,
    can_view_admin_activity,
    write_admin_activity_log,
)


class _InsertProbe:
    def __init__(self):
        self.rows: list[dict] = []

    def insert(self, payload):
        self.rows.append(dict(payload))
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


class _SupabaseProbe:
    def __init__(self):
        self.probe = _InsertProbe()

    def table(self, _name: str):
        return self.probe


class _FailingSupabase:
    def table(self, _name: str):
        raise RuntimeError("missing table")


def test_build_activity_payload_sets_expected_shape():
    payload = build_activity_payload(
        club_id="club-1",
        actor_email="Admin@Example.com",
        actor_role=ROLE_SCOREKEEPER,
        action_type="match_edit",
        entity_type="match",
        entity_id="42",
        before_json={"score_t1": 11},
        after_json={"score_t1": 9},
        note="Corrected entry",
        source_page="match_log",
        flagged_for_review=True,
    )

    assert payload["club_id"] == "club-1"
    assert payload["actor_email"] == "admin@example.com"
    assert payload["actor_role"] == ROLE_SCOREKEEPER
    assert payload["action_type"] == "match_edit"
    assert payload["entity_type"] == "match"
    assert payload["entity_id"] == "42"
    assert payload["flagged_for_review"] is True
    assert "effective_role" not in payload
    assert "role_source" not in payload


def test_build_activity_payload_nests_effective_role_context():
    payload = build_activity_payload(
        club_id="club-1",
        actor_email="Admin@Example.com",
        actor_role=ROLE_SUPER_ADMIN,
        action_type="match_edit",
        entity_type="match",
        entity_id="42",
        after_json={"score_t1": 9},
        effective_role=ROLE_SCOREKEEPER,
        role_source="super_admin_view_as",
    )
    assert payload["after_json"]["_audit_context"]["effective_role"] == ROLE_SCOREKEEPER
    assert payload["after_json"]["_audit_context"]["role_source"] == "super_admin_view_as"


def test_write_admin_activity_log_avoids_unknown_top_level_view_as_keys():
    supabase = _SupabaseProbe()
    payload = build_activity_payload(
        club_id="club-1",
        actor_email="Admin@Example.com",
        actor_role=ROLE_SUPER_ADMIN,
        action_type="match_edit",
        entity_type="match",
        entity_id="42",
        effective_role=ROLE_SCOREKEEPER,
        role_source="super_admin_view_as",
    )
    write_admin_activity_log(supabase, payload)
    inserted = supabase.probe.rows[0]
    assert "effective_role" not in inserted
    assert "role_source" not in inserted


def test_can_view_admin_activity_is_limited_to_owner_and_super_admin():
    assert can_view_admin_activity(ROLE_SUPER_ADMIN) is True
    assert can_view_admin_activity(ROLE_CLUB_OWNER) is True
    assert can_view_admin_activity(ROLE_SCOREKEEPER) is False
    assert can_view_admin_activity(ROLE_READ_ONLY) is False


def test_write_admin_activity_log_degrades_gracefully():
    ok_result = write_admin_activity_log(_SupabaseProbe(), {"club_id": "club"})
    fail_result = write_admin_activity_log(_FailingSupabase(), {"club_id": "club"})

    assert ok_result.ok is True
    assert fail_result.ok is False
    assert fail_result.warning is not None
