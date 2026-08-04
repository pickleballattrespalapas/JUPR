from types import SimpleNamespace

import pytest

from jupr_app.domain import tournament_partner_service as svc
from jupr_app.domain.tournament_registration_compiler import compile_tournament_registration_state


class _FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.limit_n = None
        self._insert_payload = None
        self._update_payload = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def limit(self, value):
        self.limit_n = int(value)
        return self

    def insert(self, payload):
        self._insert_payload = payload
        return self

    def update(self, payload):
        self._update_payload = dict(payload)
        return self

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self._insert_payload is not None:
            rows = self._insert_payload if isinstance(self._insert_payload, list) else [self._insert_payload]
            inserted = [dict(row) for row in rows]
            table.extend(inserted)
            return SimpleNamespace(data=inserted)

        matched = []
        for row in table:
            if all(str(row.get(column)) == str(value) for column, value in self.filters):
                matched.append(row)

        if self._update_payload is not None:
            for row in matched:
                row.update(self._update_payload)
            data = [dict(row) for row in matched]
        else:
            data = [dict(row) for row in matched]

        if self.limit_n is not None:
            data = data[: self.limit_n]
        return SimpleNamespace(data=data)


class _FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return _FakeQuery(self.storage, name)


def _storage():
    return {
        "tournament_registrations": [
            {"id": "reg_mary", "tournament_id": "tour-1", "display_name": "Mary Bauman", "player_id": 101},
            {"id": "reg_elizabeth", "tournament_id": "tour-1", "display_name": "Elizabeth Whelan", "player_id": 102},
            {"id": "reg_alice", "tournament_id": "tour-1", "display_name": "Alice", "player_id": 103},
        ],
        "tournament_registration_selections": [
            {
                "id": "sel_mary",
                "tournament_id": "tour-1",
                "registration_id": "reg_mary",
                "event_option_id": "event-wd-35",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Elizabeth whalen",
            },
            {
                "id": "sel_elizabeth",
                "tournament_id": "tour-1",
                "registration_id": "reg_elizabeth",
                "event_option_id": "event-wd-35",
                "partner_mode": "NEEDS_PARTNER",
            },
            {
                "id": "sel_alice",
                "tournament_id": "tour-1",
                "registration_id": "reg_alice",
                "event_option_id": "event-wd-35",
                "partner_mode": "NEEDS_PARTNER",
            },
        ],
        "tournament_registration_partner_requests": [],
        "tournament_registration_team_links": [],
        "tournament_registration_team_members": [],
    }


def test_mary_requests_elizabeth_and_elizabeth_accepts_cancels_competing_requests():
    storage = _storage()
    supabase = _FakeSupabase(storage)
    accepted = svc.create_partner_request(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        source="NEEDS_PARTNER_LIST",
    )
    competing = svc.create_partner_request(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_alice",
        target_selection_id="sel_elizabeth",
        source="NEEDS_PARTNER_LIST",
    )

    link = svc.accept_partner_request(
        supabase,
        request_id=accepted["id"],
        accepted_by_selection_id="sel_elizabeth",
    )

    assert link["status"] == "CONFIRMED"
    assert len(storage["tournament_registration_team_links"]) == 1
    members = storage["tournament_registration_team_members"]
    assert {row["selection_id"] for row in members} == {"sel_mary", "sel_elizabeth"}
    assert {row["player_order"] for row in members} == {1, 2}
    request_by_id = {row["id"]: row for row in storage["tournament_registration_partner_requests"]}
    assert request_by_id[accepted["id"]]["status"] == "ACCEPTED"
    assert request_by_id[competing["id"]]["status"] == "CANCELLED"


def test_mary_cannot_request_herself():
    supabase = _FakeSupabase(_storage())
    with pytest.raises(ValueError, match="cannot request themselves"):
        svc.create_partner_request(
            supabase,
            tournament_id="tour-1",
            event_option_id="event-wd-35",
            requester_selection_id="sel_mary",
            target_selection_id="sel_mary",
            source="NEEDS_PARTNER_LIST",
        )


def test_mary_cannot_request_elizabeth_if_elizabeth_already_confirmed():
    storage = _storage()
    storage["tournament_registration_team_members"].append(
        {
            "id": "member_existing",
            "team_link_id": "team_existing",
            "tournament_id": "tour-1",
            "event_option_id": "event-wd-35",
            "selection_id": "sel_elizabeth",
            "registration_id": "reg_elizabeth",
            "player_id": 102,
            "player_order": 1,
            "status": "ACTIVE",
        }
    )
    supabase = _FakeSupabase(storage)

    with pytest.raises(ValueError, match="already on a confirmed team"):
        svc.create_partner_request(
            supabase,
            tournament_id="tour-1",
            event_option_id="event-wd-35",
            requester_selection_id="sel_mary",
            target_selection_id="sel_elizabeth",
            source="NEEDS_PARTNER_LIST",
        )


def test_admin_can_confirm_two_compatible_selections():
    storage = _storage()
    supabase = _FakeSupabase(storage)

    link = svc.admin_confirm_partner_link(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        selection1_id="sel_mary",
        selection2_id="sel_elizabeth",
        admin_user_id="admin@example.com",
    )

    assert link["status"] == "ADMIN_CONFIRMED"
    assert len(storage["tournament_registration_team_links"]) == 1
    assert len(storage["tournament_registration_team_members"]) == 2
    audit_request = storage["tournament_registration_partner_requests"][0]
    assert audit_request["status"] == "ADMIN_CONFIRMED"
    assert audit_request["source"] == "ADMIN_RECONCILIATION"

    state = compile_tournament_registration_state(
        tournament={"id": "tour-1"},
        settings={},
        days=[{"id": "day-1", "enabled": True}],
        event_options=[{"id": "event-wd-35", "registration_day_id": "day-1", "partner_required": True}],
        registrations=storage["tournament_registrations"],
        selections=[{**row, "registration_day_id": "day-1"} for row in storage["tournament_registration_selections"]],
        partner_requests=storage["tournament_registration_partner_requests"],
        partner_links=storage["tournament_registration_team_links"],
        team_members=storage["tournament_registration_team_members"],
    )
    confirmed = [entry for roster in state["event_rosters"] for entry in roster["entries"] if entry["status"] == "ADMIN_CONFIRMED"]
    assert len(confirmed) == 1
    assert [member["display_name"] for member in confirmed[0]["members"]] == ["Mary Bauman", "Elizabeth Whelan"]


def test_legacy_free_text_partner_name_alone_cannot_create_confirmed_team():
    storage = _storage()
    supabase = _FakeSupabase(storage)

    request = svc.create_partner_request(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_display_name_snapshot="Elizabeth whalen",
        source="LEGACY_TEXT_MATCH",
    )

    assert request["status"] == "PENDING"
    assert request["target_selection_id"] is None
    assert storage["tournament_registration_team_links"] == []
    assert storage["tournament_registration_team_members"] == []


def test_admin_can_replace_and_then_remove_an_event_partner():
    storage = _storage()
    supabase = _FakeSupabase(storage)
    original = svc.admin_confirm_partner_link(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        selection1_id="sel_mary",
        selection2_id="sel_elizabeth",
        admin_user_id="admin@example.com",
    )

    replaced = svc.admin_replace_partner_link(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        selection_id="sel_mary",
        partner_selection_id="sel_alice",
        admin_user_id="admin@example.com",
    )

    assert replaced["outcome"] == "paired"
    assert replaced["partner_selection_id"] == "sel_alice"
    assert next(row for row in storage["tournament_registration_team_links"] if row["id"] == original["id"])["status"] == "CANCELLED"
    active_links = [row for row in storage["tournament_registration_team_links"] if row["status"] == "ADMIN_CONFIRMED"]
    assert len(active_links) == 1
    assert {active_links[0]["selection1_id"], active_links[0]["selection2_id"]} == {"sel_mary", "sel_alice"}
    modes = {row["id"]: row.get("partner_mode") for row in storage["tournament_registration_selections"]}
    assert modes["sel_elizabeth"] == "NEEDS_PARTNER"
    assert modes["sel_mary"] == "HAS_PARTNER"
    assert modes["sel_alice"] == "HAS_PARTNER"

    removed = svc.admin_replace_partner_link(
        supabase,
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        selection_id="sel_mary",
        partner_selection_id=None,
        unpaired_mode="NONE",
        admin_user_id="admin@example.com",
    )

    assert removed["outcome"] == "unpaired"
    modes = {row["id"]: row.get("partner_mode") for row in storage["tournament_registration_selections"]}
    assert modes["sel_mary"] == "NONE"
    assert modes["sel_alice"] == "NEEDS_PARTNER"
    assert not [row for row in storage["tournament_registration_team_links"] if row["status"] in {"CONFIRMED", "ADMIN_CONFIRMED"}]


def test_admin_partner_assignment_rejects_two_entries_from_same_registration():
    storage = _storage()
    storage["tournament_registration_selections"].append(
        {
            "id": "sel_mary_second",
            "tournament_id": "tour-1",
            "registration_id": "reg_mary",
            "event_option_id": "event-wd-35",
            "partner_mode": "NEEDS_PARTNER",
        }
    )
    supabase = _FakeSupabase(storage)

    with pytest.raises(ValueError, match="another entry from itself"):
        svc.admin_replace_partner_link(
            supabase,
            tournament_id="tour-1",
            event_option_id="event-wd-35",
            selection_id="sel_mary",
            partner_selection_id="sel_mary_second",
            admin_user_id="admin@example.com",
        )
