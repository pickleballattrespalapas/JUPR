from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.services.admin_social_submission_service import (
    list_admin_social_submissions,
    moderate_admin_social_submission,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None
        self.update_payload = None
        self.insert_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def update(self, payload):
        self.update_payload = dict(payload or {})
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            table.extend(dict(row) for row in rows)
            return SimpleNamespace(data=rows)
        matched = list(table)
        for key, expected in self.filters:
            matched = [row for row in matched if str(row.get(key)) == str(expected)]
        if self.update_payload is not None:
            for row in matched:
                row.update(self.update_payload)
            return SimpleNamespace(data=matched)
        if self.order_key:
            matched = sorted(
                matched,
                key=lambda row: str(row.get(self.order_key) or ""),
                reverse=self.order_desc,
            )
        if self.limit_value is not None:
            matched = matched[: self.limit_value]
        return SimpleNamespace(data=matched)


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "live_events": [
                {
                    "id": "event-pending",
                    "club_id": "club",
                    "name": "Friday Social",
                    "event_type": "round_robin",
                    "event_date": "2026-07-17",
                    "status": "pending",
                    "result_mode": "social_unrated",
                    "submission_mode": "public",
                    "submitted_by_name": "Alex",
                    "summary_json": {"participant_count": 8, "match_count": 12},
                    "raw_event_json": {"rounds": [{"number": 1}]},
                    "created_at": "2026-07-17T10:00:00Z",
                    "updated_at": "2026-07-17T10:01:00Z",
                },
                {
                    "id": "event-saved",
                    "club_id": "club",
                    "name": "Thursday Social",
                    "event_type": "round_robin",
                    "event_date": "2026-07-16",
                    "status": "saved",
                    "result_mode": "social_unrated",
                    "submission_mode": "admin",
                    "submitted_by_name": "Owner",
                    "summary_json": {"participant_count": 4, "match_count": 3},
                    "raw_event_json": {},
                    "created_at": "2026-07-16T10:00:00Z",
                    "updated_at": "2026-07-16T10:01:00Z",
                },
                {
                    "id": "other-club",
                    "club_id": "other",
                    "name": "Private Other Club Event",
                    "status": "pending",
                    "result_mode": "social_unrated",
                    "submitted_by_name": "Other",
                    "summary_json": {},
                    "raw_event_json": {},
                    "updated_at": "2026-07-17T11:00:00Z",
                },
                {
                    "id": "rated-event",
                    "club_id": "club",
                    "name": "Rated Event",
                    "status": "pending",
                    "result_mode": "rated",
                    "submitted_by_name": "Owner",
                    "summary_json": {},
                    "raw_event_json": {},
                    "updated_at": "2026-07-17T12:00:00Z",
                },
            ],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


@pytest.fixture(autouse=True)
def enable_admin_tools(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")


def test_social_submission_list_is_read_only_and_club_scoped() -> None:
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)

    payload = list_admin_social_submissions(supabase, club_id="club", status="pending", limit=500)

    assert payload["read_only"] is True
    assert payload["summary"] == {"returned_count": 1, "limit": 100}
    assert [row["id"] for row in payload["submissions"]] == ["event-pending"]
    assert payload["submissions"][0]["raw_event_json"]["rounds"][0]["number"] == 1
    assert supabase.storage == before


def test_approve_social_submission_requires_current_status_and_writes_audit() -> None:
    supabase = FakeSupabase()

    payload = moderate_admin_social_submission(
        supabase,
        club_id="club",
        event_id="event-pending",
        action="approve",
        expected_status="pending",
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="APPROVE SOCIAL SUBMISSION",
    )

    row = next(row for row in supabase.storage["live_events"] if row["id"] == "event-pending")
    assert payload["submission"]["status"] == "saved"
    assert row["status"] == "saved"
    assert row["moderated_by"] == "owner@example.com"
    assert row["rejection_reason"] is None
    audit = supabase.storage["admin_activity_log"][0]
    assert audit["action_type"] == "approve_club_social_submission"
    assert audit["flagged_for_review"] is True
    assert "raw_event_json" not in audit["before_json"]
    assert "raw_event_json" not in audit["after_json"]["submission"]


def test_reject_social_submission_requires_reason_before_write() -> None:
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)

    with pytest.raises(ValueError, match="rejection reason"):
        moderate_admin_social_submission(
            supabase,
            club_id="club",
            event_id="event-pending",
            action="reject",
            expected_status="pending",
            actor_email="owner@example.com",
            actor_role="club_owner",
            rejection_reason="",
            confirmation_text="REJECT SOCIAL SUBMISSION",
        )

    assert supabase.storage == before


def test_social_submission_stale_status_is_rejected_without_write() -> None:
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)

    with pytest.raises(ValueError, match="status changed"):
        moderate_admin_social_submission(
            supabase,
            club_id="club",
            event_id="event-saved",
            action="reject",
            expected_status="pending",
            actor_email="owner@example.com",
            actor_role="club_owner",
            rejection_reason="Duplicate submission",
            confirmation_text="REJECT SOCIAL SUBMISSION",
        )

    assert supabase.storage == before


def test_social_submission_cross_club_target_is_rejected_without_write() -> None:
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)

    with pytest.raises(ValueError, match="not found for this club"):
        moderate_admin_social_submission(
            supabase,
            club_id="club",
            event_id="other-club",
            action="approve",
            expected_status="pending",
            actor_email="owner@example.com",
            actor_role="club_owner",
            confirmation_text="APPROVE SOCIAL SUBMISSION",
        )

    assert supabase.storage == before
