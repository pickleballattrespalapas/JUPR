from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.services.admin_tournament_checkin_service import (
    StaleTournamentCheckInError,
    TournamentCheckInIdempotencyConflictError,
    build_admin_tournament_checkin_snapshot,
    update_admin_tournament_checkin,
)


class FakeQuery:
    def __init__(self, client, table_name: str):
        self.client = client
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key: str, value: object):
        self.filters.append((key, value))
        return self

    def order(self, key: str, desc: bool = False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value: int):
        self.limit_value = int(value)
        return self

    def execute(self):
        if self.table_name in self.client.failed_tables:
            raise RuntimeError(f"{self.table_name} unavailable")
        rows = [dict(row) for row in self.client.tables.get(self.table_name, [])]
        for key, expected in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(expected)]
        if self.order_key:
            rows.sort(
                key=lambda row: str(row.get(self.order_key) or ""),
                reverse=self.order_desc,
            )
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return SimpleNamespace(data=rows)


class FakeRpc:
    def __init__(self, client, name: str, params: dict):
        self.client = client
        self.name = name
        self.params = dict(params)

    def execute(self):
        self.client.rpc_calls.append((self.name, self.params))
        if self.name != "admin_upsert_tournament_registration_check_in":
            raise RuntimeError(f"unsupported RPC {self.name}")
        rows = self.client.tables["tournament_registration_check_ins"]
        operation_current = next(
            (
                row
                for row in rows
                if str(row.get("last_operation_key"))
                == str(self.params["p_operation_key"])
            ),
            None,
        )
        if operation_current is not None and (
            str(operation_current.get("tournament_id"))
            != str(self.params["p_tournament_id"])
            or str(operation_current.get("registration_id"))
            != str(self.params["p_registration_id"])
            or str(operation_current.get("registration_day_id"))
            != str(self.params["p_registration_day_id"])
        ):
            raise RuntimeError(
                "JUPR_CHECK_IN_IDEMPOTENCY_CONFLICT: operation key is already bound to a different attendance row."
            )
        current = next(
            (
                row
                for row in rows
                if str(row.get("tournament_id")) == str(self.params["p_tournament_id"])
                and str(row.get("registration_id")) == str(self.params["p_registration_id"])
                and str(row.get("registration_day_id"))
                == str(self.params["p_registration_day_id"])
            ),
            None,
        )
        if current is not None and str(current.get("last_operation_key")) == str(
            self.params["p_operation_key"]
        ):
            requested_fingerprint = {
                "attendance_status": self.params["p_attendance_status"],
                "waiver_verified": self.params["p_waiver_verified"],
                "approved_substitute_player_id": self.params[
                    "p_approved_substitute_player_id"
                ],
                "notes": self.params.get("p_notes"),
            }
            current_fingerprint = {
                "attendance_status": current["attendance_status"],
                "waiver_verified": current["waiver_verified"],
                "approved_substitute_player_id": current.get(
                    "approved_substitute_player_id"
                ),
                "notes": current.get("notes"),
            }
            if requested_fingerprint != current_fingerprint:
                raise RuntimeError(
                    "JUPR_CHECK_IN_IDEMPOTENCY_CONFLICT: operation key was reused with a different request."
                )
            return SimpleNamespace(
                data={
                    "ok": True,
                    "check_in": dict(current),
                    "attendee_identity_changed": False,
                    "attendance_reset": False,
                    "idempotent_replay": True,
                }
            )
        expected = self.params.get("p_expected_updated_at")
        if current is None and expected is not None:
            return SimpleNamespace(data={"ok": False, "code": "CHECK_IN_STALE"})
        if current is not None and str(current.get("updated_at")) != str(expected):
            return SimpleNamespace(data={"ok": False, "code": "CHECK_IN_STALE"})

        registration = next(
            row
            for row in self.client.tables["tournament_registrations"]
            if str(row.get("id")) == str(self.params["p_registration_id"])
        )
        if str(registration.get("status") or "").strip().upper() not in {
            "ACTIVE",
            "APPROVED",
            "CONFIRMED",
            "REGISTERED",
        }:
            raise RuntimeError(
                "JUPR_CHECK_IN_INACTIVE: registration is no longer active."
            )
        substitute_id = self.params.get("p_approved_substitute_player_id")
        substitute = next(
            (
                row
                for row in self.client.tables["players"]
                if str(row.get("id")) == str(substitute_id)
                and str(row.get("club_id")) == str(self.params["p_club_id"])
                and row.get("active") is True
            ),
            None,
        )
        if substitute_id is not None and substitute is None:
            raise RuntimeError(
                "JUPR_CHECK_IN_SUBSTITUTE_INVALID: approved substitute must be an active player in this club."
            )
        if substitute is not None:
            next_identity = f"player:{int(substitute['id'])}"
            substitute_name = str(substitute.get("name") or "").strip()
        elif registration.get("player_id") is not None:
            next_identity = f"player:{int(registration['player_id'])}"
            substitute_name = None
        else:
            profile_parts = [
                str(registration.get(key) or "").strip().lower()
                for key in ("display_name", "first_name", "last_name", "email")
            ]
            next_identity = "registration:" + ":".join(
                [str(registration.get("id") or ""), *profile_parts]
            )
            substitute_name = None
        identity_changed = (
            current is not None
            and current.get("attendee_identity_key") != next_identity
        )
        payload = {
            "id": str((current or {}).get("id") or "check-in-1"),
            "tournament_id": self.params["p_tournament_id"],
            "registration_id": self.params["p_registration_id"],
            "registration_day_id": self.params["p_registration_day_id"],
            "attendance_status": (
                "EXPECTED"
                if identity_changed
                else self.params["p_attendance_status"]
            ),
            "checked_in": (
                False
                if identity_changed
                else self.params["p_attendance_status"] == "CHECKED_IN"
            ),
            "waiver_verified": False
            if identity_changed
            else self.params["p_waiver_verified"],
            "attendee_identity_key": next_identity,
            "approved_substitute_player_id": substitute_id,
            "approved_substitute_name": substitute_name,
            "notes": self.params.get("p_notes"),
            "updated_by": self.params.get("p_updated_by"),
            "last_operation_key": self.params.get("p_operation_key"),
            "updated_at": "2026-08-15T12:01:00Z",
        }
        if current is None:
            rows.append(payload)
        else:
            current.clear()
            current.update(payload)
        return SimpleNamespace(
            data={
                "ok": True,
                "check_in": dict(payload),
                "attendee_identity_changed": identity_changed,
                "attendance_reset": identity_changed,
                "idempotent_replay": False,
            }
        )


class FakeSupabase:
    def __init__(self, tables: dict[str, list[dict]], *, failed_tables=()):
        self.tables = deepcopy(tables)
        self.failed_tables = set(failed_tables)
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, name: str):
        return FakeQuery(self, name)

    def rpc(self, name: str, params: dict):
        return FakeRpc(self, name, params)


class RegistrationMutatesBeforeRpc(FakeSupabase):
    def __init__(self, tables: dict[str, list[dict]], *, next_status: str):
        super().__init__(tables)
        self.next_status = next_status

    def rpc(self, name: str, params: dict):
        registration = next(
            row
            for row in self.tables["tournament_registrations"]
            if str(row.get("id")) == str(params["p_registration_id"])
        )
        registration["status"] = self.next_status
        return super().rpc(name, params)


def checkin_tables() -> dict[str, list[dict]]:
    return {
        "tournaments": [
            {
                "id": "tour-1",
                "club_id": "club-1",
                "name": "Summer Classic",
                "status": "PUBLISHED",
                "start_date": "2026-08-20",
                "end_date": "2026-08-20",
            }
        ],
        "tournament_registration_settings": [
            {
                "id": "settings-1",
                "tournament_id": "tour-1",
                "registration_status": "closed",
                "timezone": "America/Chicago",
            }
        ],
        "tournament_registration_days": [
            {
                "id": "day-1",
                "tournament_id": "tour-1",
                "label": "Thursday",
                "event_date": "2026-08-20",
                "enabled": True,
                "sort_order": 1,
                "court_count": 4,
                "court_labels": ["1", "2", "3", "4"],
                "court_open_time": "08:00",
                "court_close_time": "18:00",
            }
        ],
        "tournament_event_options": [
            {
                "id": "event-1",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "scheduled_day_ids": ["day-1"],
                "label": "Women's 3.5",
                "event_family_label": "Women's doubles",
                "division_name": "3.5",
                "event_type": "DOUBLES",
                "partner_required": True,
                "team_allow_substitutes": True,
                "enabled": True,
                "sort_order": 1,
            }
        ],
        "tournament_registrations": [
            {
                "id": "reg-1",
                "tournament_id": "tour-1",
                "player_id": 1,
                "display_name": "Alex Original",
                "status": "confirmed",
                "payment_status": "unpaid",
                "updated_at": "2026-08-15T10:00:00Z",
            },
            {
                "id": "reg-2",
                "tournament_id": "tour-1",
                "player_id": 2,
                "display_name": "Blair Partner",
                "status": "confirmed",
                "payment_status": "paid",
                "updated_at": "2026-08-15T10:00:00Z",
            },
            {
                "id": "reg-3",
                "tournament_id": "tour-1",
                "player_id": 3,
                "display_name": "Casey Needs Partner",
                "status": "confirmed",
                "payment_status": "waived",
                "updated_at": "2026-08-15T10:00:00Z",
            },
            {
                "id": "reg-4",
                "tournament_id": "tour-1",
                "player_id": 4,
                "display_name": "Dana Free Text",
                "status": "confirmed",
                "payment_status": "paid",
                "updated_at": "2026-08-15T10:00:00Z",
            },
        ],
        "tournament_registration_selections": [
            {
                "id": "sel-1",
                "tournament_id": "tour-1",
                "registration_id": "reg-1",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Blair Partner",
            },
            {
                "id": "sel-2",
                "tournament_id": "tour-1",
                "registration_id": "reg-2",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Alex Original",
            },
            {
                "id": "sel-3",
                "tournament_id": "tour-1",
                "registration_id": "reg-3",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                "partner_mode": "NEEDS_PARTNER",
            },
            {
                "id": "sel-4",
                "tournament_id": "tour-1",
                "registration_id": "reg-4",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Unlinked Guest",
            },
        ],
        "tournament_registration_team_links": [
            {
                "id": "link-1",
                "tournament_id": "tour-1",
                "event_option_id": "event-1",
                "registration1_id": "reg-1",
                "registration2_id": "reg-2",
                "selection1_id": "sel-1",
                "selection2_id": "sel-2",
                "status": "CONFIRMED",
            }
        ],
        "tournament_registration_team_members": [
            {
                "id": "member-1",
                "team_link_id": "link-1",
                "tournament_id": "tour-1",
                "event_option_id": "event-1",
                "selection_id": "sel-1",
                "registration_id": "reg-1",
                "player_id": 1,
                "player_order": 1,
                "status": "ACTIVE",
            },
            {
                "id": "member-2",
                "team_link_id": "link-1",
                "tournament_id": "tour-1",
                "event_option_id": "event-1",
                "selection_id": "sel-2",
                "registration_id": "reg-2",
                "player_id": 2,
                "player_order": 2,
                "status": "ACTIVE",
            },
        ],
        "tournament_commerce_orders": [
            {
                "id": "order-1",
                "club_id": "club-1",
                "tournament_id": "tour-1",
                "registration_id": "reg-1",
                "status": "OPEN",
                "payment_status": "PAID",
            }
        ],
        "tournament_registration_check_ins": [
            {
                "id": "check-in-1",
                "tournament_id": "tour-1",
                "registration_id": "reg-1",
                "registration_day_id": "day-1",
                "attendance_status": "CHECKED_IN",
                "checked_in": True,
                "waiver_verified": True,
                "attendee_identity_key": "player:1",
                "approved_substitute_player_id": None,
                "approved_substitute_name": None,
                "notes": "Approved by TD",
                "updated_by": "admin@example.com",
                "last_operation_key": "00000000-0000-4000-8000-000000000001",
                "updated_at": "2026-08-15T12:00:00Z",
            }
        ],
        "tournament_event_draws": [],
        "tournament_teams": [],
        "players": [
            {"id": 1, "club_id": "club-1", "name": "Alex Original", "active": True},
            {"id": 2, "club_id": "club-1", "name": "Blair Partner", "active": True},
            {"id": 3, "club_id": "club-1", "name": "Casey Needs Partner", "active": True},
            {"id": 4, "club_id": "club-1", "name": "Dana Free Text", "active": True},
            {"id": 10, "club_id": "club-1", "name": "Sam Substitute", "active": True},
            {"id": 99, "club_id": "other", "name": "Wrong Club", "active": True},
        ],
    }


def test_snapshot_is_operational_and_keeps_offline_payment_authoritative() -> None:
    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(checkin_tables()), club_id="club-1", tournament_id="tour-1"
    )

    assert snapshot["ok"] is True
    assert snapshot["summary"] == {
        "expected": 4,
        "checked_in": 1,
        "absent": 0,
        "not_checked_in": 3,
        "unresolved": 2,
    }
    assert snapshot["day_scope"]["selected_day_id"] == "day-1"
    assert {row["attendance_status"] for row in snapshot["registrants"]} == {
        "EXPECTED",
        "CHECKED_IN",
    }
    alex = next(card for card in snapshot["registrants"] if card["registration_id"] == "reg-1")
    assert alex["payment"]["status"] == "PAID"
    assert alex["payment"]["source"] == "offline_payment_tracking"
    assert alex["attendee"]["name"] == "Alex Original"
    assert alex["original_registrant"]["name"] == "Alex Original"
    assert alex["waiver"]["subject"] == "attending_player"
    assert alex["check_in"]["identity_current"] is True


def test_snapshot_fails_closed_after_registration_attendee_identity_changes() -> None:
    tables = checkin_tables()
    state = tables["tournament_registration_check_ins"][0]
    state.update(
        {
            "attendee_identity_key": "player:1",
            "approved_substitute_player_id": None,
            "approved_substitute_name": None,
        }
    )
    client = FakeSupabase(tables)

    before = build_admin_tournament_checkin_snapshot(
        client, club_id="club-1", tournament_id="tour-1"
    )
    before_card = next(
        card for card in before["registrants"] if card["registration_id"] == "reg-1"
    )
    assert before_card["check_in"]["checked_in"] is True
    assert before_card["waiver"]["verified"] is True

    registration = client.tables["tournament_registrations"][0]
    registration.update(
        {
            "player_id": 20,
            "display_name": "Alex Replacement",
            "email": "alex.replacement@example.com",
            "updated_at": "2026-08-15T12:30:00Z",
        }
    )

    after = build_admin_tournament_checkin_snapshot(
        client, club_id="club-1", tournament_id="tour-1"
    )
    after_card = next(
        card for card in after["registrants"] if card["registration_id"] == "reg-1"
    )
    assert after["summary"]["checked_in"] == 0
    assert after_card["check_in"]["checked_in"] is False
    assert after_card["waiver"]["verified"] is False
    assert after_card["check_in"]["identity_current"] is False
    assert after_card["check_in"]["requires_reconfirmation"] is True
    assert "ATTENDEE_IDENTITY_STALE" in {
        blocker["code"] for blocker in after_card["blockers"]
    }


def test_snapshot_fails_closed_when_saved_substitute_is_no_longer_active() -> None:
    tables = checkin_tables()
    tables["tournament_registration_check_ins"][0].update(
        {
            "attendee_identity_key": "player:10",
            "approved_substitute_player_id": 10,
            "approved_substitute_name": "Sam Substitute",
        }
    )
    substitute = next(player for player in tables["players"] if player["id"] == 10)
    substitute["active"] = False

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    card = next(
        row for row in snapshot["registrants"] if row["registration_id"] == "reg-1"
    )
    assert card["check_in"]["checked_in"] is False
    assert card["waiver"]["verified"] is False
    assert card["check_in"]["identity_current"] is False
    assert "APPROVED_SUBSTITUTE_INVALID" in {
        blocker["code"] for blocker in card["blockers"]
    }


def test_snapshot_refreshes_substitute_name_without_invalidating_same_player() -> None:
    tables = checkin_tables()
    tables["tournament_registration_check_ins"][0].update(
        {
            "attendee_identity_key": "player:10",
            "approved_substitute_player_id": 10,
            "approved_substitute_name": "Sam Substitute",
        }
    )
    substitute = next(player for player in tables["players"] if player["id"] == 10)
    substitute["name"] = "Sam Updated"

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    card = next(
        row for row in snapshot["registrants"] if row["registration_id"] == "reg-1"
    )
    assert card["attendee"]["name"] == "Sam Updated"
    assert card["check_in"]["checked_in"] is False
    assert card["waiver"]["verified"] is False
    assert card["check_in"]["identity_current"] is False


def test_snapshot_disables_assignment_when_atomic_policy_cannot_be_proven() -> None:
    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(checkin_tables()), club_id="club-1", tournament_id="tour-1"
    )

    card = next(
        row for row in snapshot["registrants"] if row["registration_id"] == "reg-1"
    )
    assert card["substitution"]["event_policy_allows"] is True
    assert card["substitution"]["allowed"] is False
    assert card["substitution"]["blocker"]["code"] == (
        "SUBSTITUTE_ASSIGNMENT_ATOMICITY_UNAVAILABLE"
    )
    assert card["check_in"]["checked_in"] is True
    assert card["waiver"]["verified"] is True


def test_snapshot_fails_closed_for_duplicate_saved_attendee_identity() -> None:
    tables = checkin_tables()
    tables["tournament_registration_check_ins"][0].update(
        {
            "attendee_identity_key": "player:10",
            "approved_substitute_player_id": 10,
            "approved_substitute_name": "Sam Substitute",
        }
    )
    tables["tournament_registration_check_ins"].append(
        {
            "id": "check-in-2",
            "tournament_id": "tour-1",
            "registration_id": "reg-2",
            "registration_day_id": "day-1",
            "attendance_status": "CHECKED_IN",
            "checked_in": True,
            "waiver_verified": True,
            "attendee_identity_key": "player:10",
            "approved_substitute_player_id": 10,
            "approved_substitute_name": "Sam Substitute",
            "updated_by": "admin@example.com",
            "last_operation_key": "00000000-0000-4000-8000-000000000002",
            "updated_at": "2026-08-15T12:02:00Z",
        }
    )

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    duplicate_cards = [
        row
        for row in snapshot["registrants"]
        if row["registration_id"] in {"reg-1", "reg-2"}
    ]
    assert all(row["check_in"]["checked_in"] is False for row in duplicate_cards)
    assert all(row["waiver"]["verified"] is False for row in duplicate_cards)
    assert all(
        "DUPLICATE_ATTENDEE_IDENTITY"
        in {blocker["code"] for blocker in row["blockers"]}
        for row in duplicate_cards
    )


def test_snapshot_exposes_unlinked_and_needs_partner_participants() -> None:
    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(checkin_tables()), club_id="club-1", tournament_id="tour-1"
    )

    assert {row["kind"] for row in snapshot["unresolved_participants"]} == {
        "NEEDS_PARTNER",
        "UNLINKED_FREE_TEXT_PARTNER",
    }
    confirmed = next(card for card in snapshot["registrants"] if card["registration_id"] == "reg-1")
    assert confirmed["events"][0]["team_state"] == "CONFIRMED_LINK"
    assert confirmed["events"][0]["partner_name"] == "Blair Partner"
    assert snapshot["registration_follow_up"] == []


def test_snapshot_uses_authoritative_draw_roster_for_expected_and_partner_blockers() -> None:
    tables = checkin_tables()
    tables["tournament_event_draws"] = [
        {
            "id": "draw-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "name": "Women's 3.5 draw",
            "status": "ACTIVE",
        }
    ]
    tables["tournament_teams"] = [
        {
            "id": "team-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            # The draw relation, not this legacy projection, owns event scope.
            "event_option_id": "stale-event",
            "player1_id": 1,
            "player2_id": 2,
        }
    ]
    tables["tournament_registration_check_ins"].append(
        {
            "id": "check-in-excluded",
            "tournament_id": "tour-1",
            "registration_id": "reg-3",
            "registration_day_id": "day-1",
            "attendance_status": "CHECKED_IN",
            "checked_in": True,
            "waiver_verified": True,
            "attendee_identity_key": "player:3",
            "updated_at": "2026-08-15T12:03:00Z",
        }
    )

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    assert snapshot["summary"] == {
        "expected": 2,
        "checked_in": 1,
        "absent": 0,
        "not_checked_in": 1,
        "unresolved": 0,
    }
    assert {row["registration_id"] for row in snapshot["registrants"]} == {
        "reg-1",
        "reg-2",
    }
    assert snapshot["unresolved_participants"] == []
    assert {
        (row["registration_id"], row["selection_id"])
        for row in snapshot["registration_follow_up"]
    } == {("reg-3", "sel-3"), ("reg-4", "sel-4")}
    assert {
        row["title"] for row in snapshot["registration_follow_up"]
    } == {"Registered but not rostered"}
    partner_item = next(
        row for row in snapshot["completed_items"] if row["code"] == "PARTNER_TEAMS"
    )
    assert partner_item["status"] == "COMPLETE"


def test_snapshot_scopes_each_selection_to_its_own_draw_roster() -> None:
    tables = checkin_tables()
    tables["tournament_event_options"].append(
        {
            "id": "event-2",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "scheduled_day_ids": ["day-1"],
            "label": "Open doubles",
            "event_family_label": "Open doubles",
            "division_name": "4.0",
            "event_type": "DOUBLES",
            "partner_required": True,
            "team_allow_substitutes": False,
            "enabled": True,
            "sort_order": 2,
        }
    )
    tables["tournament_registration_selections"].append(
        {
            "id": "sel-1-event-2",
            "tournament_id": "tour-1",
            "registration_id": "reg-1",
            "registration_day_id": "day-1",
            "event_option_id": "event-2",
            "partner_mode": "NEEDS_PARTNER",
        }
    )
    tables["tournament_event_draws"] = [
        {
            "id": "draw-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "name": "Women's 3.5 draw",
            "status": "ACTIVE",
        },
        {
            "id": "draw-2",
            "tournament_id": "tour-1",
            "event_option_id": "event-2",
            "name": "Open 4.0 draw",
            "status": "ACTIVE",
        },
    ]
    tables["tournament_teams"] = [
        {
            "id": "team-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "event_option_id": "event-1",
            "player1_id": 1,
            "player2_id": 2,
        },
        {
            "id": "team-2",
            "tournament_id": "tour-1",
            "draw_id": "draw-2",
            "event_option_id": "event-2",
            "player1_id": 3,
            "player2_id": 4,
        },
    ]

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    alex = next(
        row for row in snapshot["registrants"] if row["registration_id"] == "reg-1"
    )
    assert {row["event_option_id"] for row in alex["events"]} == {"event-1"}
    assert snapshot["unresolved_participants"] == []
    assert {
        (row["registration_id"], row["selection_id"])
        for row in snapshot["registration_follow_up"]
    } >= {("reg-1", "sel-1-event-2")}


def test_snapshot_applies_draw_authority_per_event() -> None:
    tables = checkin_tables()
    tables["tournament_event_options"].append(
        {
            "id": "event-2",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "scheduled_day_ids": ["day-1"],
            "label": "Open singles",
            "event_family_label": "Singles",
            "division_name": "Open",
            "event_type": "SINGLES",
            "partner_required": False,
            "team_allow_substitutes": False,
            "enabled": True,
            "sort_order": 2,
        }
    )
    casey_selection = next(
        row
        for row in tables["tournament_registration_selections"]
        if row["registration_id"] == "reg-3"
    )
    casey_selection.update({"event_option_id": "event-2", "partner_mode": "NONE"})
    tables["tournament_event_draws"] = [
        {
            "id": "draw-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "name": "Women's 3.5 draw",
            "status": "ACTIVE",
        }
    ]
    tables["tournament_teams"] = [
        {
            "id": "team-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "player1_id": 1,
            "player2_id": 2,
        }
    ]

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    assert {row["registration_id"] for row in snapshot["registrants"]} == {
        "reg-1",
        "reg-2",
        "reg-3",
    }
    assert {row["registration_id"] for row in snapshot["registration_follow_up"]} == {
        "reg-4"
    }


def test_snapshot_blocks_empty_draw_roster_and_counts_no_unrostered_entries() -> None:
    tables = checkin_tables()
    tables["tournament_event_draws"] = [
        {
            "id": "draw-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "name": "Women's 3.5 draw",
            "status": "ACTIVE",
        }
    ]

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    assert snapshot["summary"]["expected"] == 0
    assert snapshot["registrants"] == []
    assert len(snapshot["registration_follow_up"]) == 4
    assert snapshot["readiness"]["draws"]["status"] == "BLOCKED"
    assert "DRAW_ROSTER_EMPTY" in {
        row["code"] for row in snapshot["readiness"]["draws"]["blockers"]
    }


def test_snapshot_blocks_draw_player_without_exact_registration_selection() -> None:
    tables = checkin_tables()
    tables["tournament_event_draws"] = [
        {
            "id": "draw-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "name": "Women's 3.5 draw",
            "status": "ACTIVE",
        }
    ]
    tables["tournament_teams"] = [
        {
            "id": "team-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "player1_id": 1,
            "player2_id": 99,
        }
    ]

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    assert {row["registration_id"] for row in snapshot["registrants"]} == {"reg-1"}
    blocker_codes = {
        row["code"] for row in snapshot["readiness"]["draws"]["blockers"]
    }
    assert "DRAW_ROSTER_PLAYER_UNKNOWN" in blocker_codes
    assert "DRAW_ROSTER_REGISTRATION_UNRESOLVED" in blocker_codes


def test_snapshot_honors_explicit_draw_day_and_hidden_primary_ops_boundary() -> None:
    tables = checkin_tables()
    tables["tournament_registration_days"].append(
        {
            "id": "day-2",
            "tournament_id": "tour-1",
            "label": "Friday",
            "event_date": "2026-08-21",
            "enabled": True,
            "sort_order": 2,
            "court_count": 2,
            "court_labels": ["1", "2"],
        }
    )
    tables["tournament_event_options"][0]["scheduled_day_ids"] = ["day-1", "day-2"]
    tables["tournament_event_draws"] = [
        {
            "id": "draw-day-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "registration_day_id": "day-1",
            "name": "Thursday draw",
            "status": "ACTIVE",
            "draw_kind": "STANDARD",
            "hidden_from_primary_ops": False,
        },
        {
            "id": "draw-day-2-hidden",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "registration_day_id": "day-2",
            "name": "Hidden Friday draw",
            "status": "ACTIVE",
            "draw_kind": "STANDARD",
            "hidden_from_primary_ops": True,
        },
    ]
    tables["tournament_teams"] = [
        {
            "id": "team-day-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-day-1",
            "player1_id": 1,
            "player2_id": 2,
        },
        {
            "id": "team-day-2-hidden",
            "tournament_id": "tour-1",
            "draw_id": "draw-day-2-hidden",
            "player1_id": 3,
            "player2_id": 4,
        },
    ]

    friday = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-2",
    )

    assert friday["summary"]["expected"] == 4
    assert friday["registration_follow_up"] == []
    assert "DRAW_MISSING" in {
        row["code"] for row in friday["readiness"]["draws"]["blockers"]
    }


def test_snapshot_scopes_players_unresolved_and_readiness_to_selected_day() -> None:
    tables = checkin_tables()
    tables["tournament_registration_days"].append(
        {
            "id": "day-2",
            "tournament_id": "tour-1",
            "label": "Friday",
            "event_date": "2026-08-21",
            "enabled": True,
            "sort_order": 2,
            "court_count": 2,
            "court_labels": ["1", "2"],
            "court_open_time": "09:00",
            "court_close_time": "14:00",
        }
    )
    tables["tournament_event_options"].append(
        {
            "id": "event-2",
            "tournament_id": "tour-1",
            "registration_day_id": "day-2",
            "scheduled_day_ids": ["day-2"],
            "label": "Friday singles",
            "event_family_label": "Singles",
            "division_name": "Open",
            "event_type": "SINGLES",
            "partner_required": False,
            "team_allow_substitutes": False,
            "enabled": True,
            "sort_order": 1,
        }
    )
    casey_selection = next(
        row
        for row in tables["tournament_registration_selections"]
        if row["registration_id"] == "reg-3"
    )
    casey_selection.update(
        {"registration_day_id": "day-2", "event_option_id": "event-2"}
    )
    tables["tournament_event_draws"].append(
        {
            "id": "draw-2",
            "tournament_id": "tour-1",
            "event_option_id": "event-2",
            "name": "Friday singles draw",
            "status": "ACTIVE",
        }
    )
    tables["tournament_teams"].append(
        {
            "id": "team-2",
            "tournament_id": "tour-1",
            "draw_id": "draw-2",
            "event_option_id": "event-2",
            "player1_id": 3,
            "player2_id": None,
        }
    )

    thursday = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    friday = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-2",
    )

    assert thursday["day_scope"]["selected_day_id"] == "day-1"
    assert [row["id"] for row in thursday["day_scope"]["available_days"]] == [
        "day-1",
        "day-2",
    ]
    assert {row["registration_id"] for row in thursday["registrants"]} == {
        "reg-1",
        "reg-2",
        "reg-4",
    }
    assert {row["registration_id"] for row in friday["registrants"]} == {"reg-3"}
    assert thursday["unresolved_participants"][0]["registration_id"] == "reg-4"
    assert friday["unresolved_participants"] == []
    assert friday["readiness"]["draws"]["status"] == "COMPLETE"


def test_multiday_event_membership_uses_scheduled_days_and_deduplicates_cards() -> None:
    tables = checkin_tables()
    tables["tournament_registration_days"].append(
        {
            "id": "day-2",
            "tournament_id": "tour-1",
            "label": "Friday",
            "event_date": "2026-08-21",
            "enabled": True,
            "sort_order": 2,
            "court_count": 2,
            "court_labels": ["1", "2"],
            "court_open_time": "09:00",
            "court_close_time": "14:00",
        }
    )
    tables["tournament_event_options"][0]["scheduled_day_ids"] = [
        "day-1",
        "day-2",
    ]
    tables["tournament_event_options"].extend(
        [
            {
                "id": "event-2",
                "tournament_id": "tour-1",
                "registration_day_id": "day-2",
                "scheduled_day_ids": ["day-2"],
                "label": "Friday bonus singles",
                "event_family_label": "Singles",
                "division_name": "Open",
                "event_type": "SINGLES",
                "partner_required": False,
                "team_allow_substitutes": False,
                "enabled": True,
                "sort_order": 2,
            },
            {
                "id": "event-day-1-unresolved",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "scheduled_day_ids": ["day-1"],
                "label": "Thursday unresolved doubles",
                "event_family_label": "Doubles",
                "division_name": "Open",
                "event_type": "DOUBLES",
                "partner_required": True,
                "team_allow_substitutes": False,
                "enabled": True,
                "sort_order": 3,
            },
        ]
    )
    tables["tournament_registration_selections"].append(
        {
            "id": "sel-1-day-2-bonus",
            "tournament_id": "tour-1",
            "registration_id": "reg-1",
            "registration_day_id": "day-2",
            "event_option_id": "event-2",
            "partner_mode": "NONE",
        }
    )
    for selection in tables["tournament_registration_selections"]:
        if selection["registration_id"] in {"reg-3", "reg-4"}:
            selection["event_option_id"] = "event-day-1-unresolved"
            selection["registration_day_id"] = "day-1"

    day_one = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    day_two = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-2",
    )

    assert sum(row["registration_id"] == "reg-1" for row in day_one["registrants"]) == 1
    assert sum(row["registration_id"] == "reg-1" for row in day_two["registrants"]) == 1
    day_two_alex = next(
        row for row in day_two["registrants"] if row["registration_id"] == "reg-1"
    )
    assert {event["event_option_id"] for event in day_two_alex["events"]} == {
        "event-1",
        "event-2",
    }
    assert day_two_alex["attendance_status"] == "EXPECTED"
    assert day_two["unresolved_participants"] == []
    assert {
        row["registration_id"] for row in day_one["unresolved_participants"]
    } == {"reg-3", "reg-4"}


def test_snapshot_rejects_disabled_or_foreign_day() -> None:
    tables = checkin_tables()
    tables["tournament_registration_days"][0]["enabled"] = False
    with pytest.raises(ValueError, match="enabled event day"):
        build_admin_tournament_checkin_snapshot(
            FakeSupabase(tables),
            club_id="club-1",
            tournament_id="tour-1",
            registration_day_id="day-1",
        )


def test_snapshot_fails_closed_for_malformed_scheduled_days() -> None:
    tables = checkin_tables()
    tables["tournament_event_options"][0]["scheduled_day_ids"] = {
        "day": "day-1"
    }

    with pytest.raises(ValueError, match="scheduled days are malformed"):
        build_admin_tournament_checkin_snapshot(
            FakeSupabase(tables),
            club_id="club-1",
            tournament_id="tour-1",
            registration_day_id="day-1",
        )

    tables = checkin_tables()
    tables["tournament_registration_days"][0]["tournament_id"] = "tour-other"
    with pytest.raises(ValueError, match="enabled event day"):
        build_admin_tournament_checkin_snapshot(
            FakeSupabase(tables),
            club_id="club-1",
            tournament_id="tour-1",
            registration_day_id="day-1",
        )


def test_snapshot_never_infers_absent_from_not_checked_in() -> None:
    tables = checkin_tables()
    tables["tournament_registration_check_ins"][0].update(
        {"attendance_status": "ABSENT", "checked_in": False}
    )

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    assert snapshot["summary"]["expected"] == 4
    assert snapshot["summary"]["checked_in"] == 0
    assert snapshot["summary"]["absent"] == 1
    assert snapshot["summary"]["not_checked_in"] == 3
    statuses = {
        row["registration_id"]: row["attendance_status"]
        for row in snapshot["registrants"]
    }
    assert statuses["reg-1"] == "ABSENT"
    assert statuses["reg-2"] == "EXPECTED"


def test_confirmed_link_is_not_complete_when_partner_registration_is_inactive() -> None:
    tables = checkin_tables()
    tables["tournament_registrations"][1]["status"] = "cancelled"

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    alex = next(card for card in snapshot["registrants"] if card["registration_id"] == "reg-1")
    assert alex["events"][0]["team_state"] == "UNRESOLVED"
    assert "PARTNER_REGISTRATION_INACTIVE" in {
        blocker["code"] for blocker in alex["blockers"]
    }


def test_snapshot_ignores_legacy_court_hours_and_truthfully_reports_staffing_review() -> None:
    tables = checkin_tables()
    tables["tournament_registration_settings"][0]["timezone"] = ""
    tables["tournament_registration_days"][0]["court_open_time"] = None
    tables["tournament_registration_days"][0]["court_close_time"] = None

    snapshot = build_admin_tournament_checkin_snapshot(
        FakeSupabase(tables), club_id="club-1", tournament_id="tour-1"
    )

    assert snapshot["readiness"]["schedule"]["status"] == "BLOCKED"
    schedule_blocker_codes = {
        row["code"] for row in snapshot["readiness"]["schedule"]["blockers"]
    }
    assert "TIMEZONE_MISSING" in schedule_blocker_codes
    assert "COURT_TIME_MISSING" not in schedule_blocker_codes
    assert snapshot["readiness"]["staffing"]["status"] == "NEEDS_REVIEW"
    assert snapshot["readiness"]["staffing"]["source"] == "no_authoritative_staffing_record"


def test_snapshot_fails_closed_when_draw_readiness_cannot_be_loaded() -> None:
    with pytest.raises(RuntimeError, match="draw readiness"):
        build_admin_tournament_checkin_snapshot(
            FakeSupabase(
                checkin_tables(), failed_tables={"tournament_event_draws"}
            ),
            club_id="club-1",
            tournament_id="tour-1",
        )


def test_update_uses_sql_cas_and_resets_attendance_when_attendee_changes(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = FakeSupabase(checkin_tables())
    client.tables["tournament_registration_check_ins"][0][
        "attendee_identity_key"
    ] = "player:999"

    result = update_admin_tournament_checkin(
        client,
        club_id="club-1",
        tournament_id="tour-1",
        registration_id="reg-1",
        registration_day_id="day-1",
        expected_updated_at="2026-08-15T12:00:00Z",
        attendance_status="CHECKED_IN",
        operation_key="00000000-0000-4000-8000-000000000101",
        waiver_verified=True,
        approved_substitute_player_id=None,
        approved_substitute_name=None,
        notes="Original player attending",
        actor_email="admin@example.com",
        actor_role="club_owner",
    )

    assert result["attendance_reset"] is True
    assert result["check_in"]["checked_in"] is False
    assert result["check_in"]["waiver_verified"] is False
    assert client.rpc_calls[0][0] == "admin_upsert_tournament_registration_check_in"


def test_update_refuses_registration_excluded_from_authoritative_draw_roster(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    tables = checkin_tables()
    tables["tournament_event_draws"] = [
        {
            "id": "draw-1",
            "tournament_id": "tour-1",
            "event_option_id": "event-1",
            "name": "Women's 3.5 draw",
            "status": "ACTIVE",
        }
    ]
    tables["tournament_teams"] = [
        {
            "id": "team-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "player1_id": 1,
            "player2_id": 2,
        }
    ]
    client = FakeSupabase(tables)

    with pytest.raises(ValueError, match="not mapped to an authoritative"):
        update_admin_tournament_checkin(
            client,
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-3",
            registration_day_id="day-1",
            expected_updated_at=None,
            attendance_status="CHECKED_IN",
            operation_key="00000000-0000-4000-8000-000000000109",
            waiver_verified=True,
            approved_substitute_player_id=None,
            approved_substitute_name=None,
            notes=None,
            actor_email="admin@example.com",
            actor_role="club_owner",
        )

    assert client.rpc_calls == []


def test_update_rejects_stale_version(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    with pytest.raises(StaleTournamentCheckInError, match="changed"):
        update_admin_tournament_checkin(
            FakeSupabase(checkin_tables()),
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-1",
            registration_day_id="day-1",
            expected_updated_at="2026-08-15T11:59:00Z",
            attendance_status="EXPECTED",
            operation_key="00000000-0000-4000-8000-000000000102",
            waiver_verified=False,
            approved_substitute_player_id=None,
            approved_substitute_name=None,
            notes="",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )


def test_update_replays_same_operation_without_timestamp_bump(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = FakeSupabase(checkin_tables())

    result = update_admin_tournament_checkin(
        client,
        club_id="club-1",
        tournament_id="tour-1",
        registration_id="reg-1",
        registration_day_id="day-1",
        expected_updated_at="2026-08-15T11:59:00Z",
        attendance_status="CHECKED_IN",
        operation_key="00000000-0000-4000-8000-000000000001",
        waiver_verified=True,
        approved_substitute_player_id=None,
        approved_substitute_name=None,
        notes="Approved by TD",
        actor_email="admin@example.com",
        actor_role="club_owner",
    )

    assert result["idempotent_replay"] is True
    assert result["check_in"]["updated_at"] == "2026-08-15T12:00:00Z"


def test_update_rejects_operation_key_reuse_for_different_request(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    with pytest.raises(
        TournamentCheckInIdempotencyConflictError,
        match="different attendance request",
    ):
        update_admin_tournament_checkin(
            FakeSupabase(checkin_tables()),
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-1",
            registration_day_id="day-1",
            expected_updated_at="2026-08-15T12:00:00Z",
            attendance_status="ABSENT",
            operation_key="00000000-0000-4000-8000-000000000001",
            waiver_verified=True,
            approved_substitute_player_id=None,
            approved_substitute_name=None,
            notes="Approved by TD",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )


def test_update_rejects_operation_key_reuse_for_different_registration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    with pytest.raises(
        TournamentCheckInIdempotencyConflictError,
        match="different attendance request",
    ):
        update_admin_tournament_checkin(
            FakeSupabase(checkin_tables()),
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-2",
            registration_day_id="day-1",
            expected_updated_at=None,
            attendance_status="EXPECTED",
            operation_key="00000000-0000-4000-8000-000000000001",
            waiver_verified=False,
            approved_substitute_player_id=None,
            approved_substitute_name=None,
            notes="",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )


def test_update_rejects_non_uuid_operation_key_before_rpc(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = FakeSupabase(checkin_tables())
    with pytest.raises(ValueError, match="UUID operation key"):
        update_admin_tournament_checkin(
            client,
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-1",
            registration_day_id="day-1",
            expected_updated_at="2026-08-15T12:00:00Z",
            attendance_status="EXPECTED",
            operation_key="not-a-uuid",
            waiver_verified=False,
            approved_substitute_player_id=None,
            approved_substitute_name=None,
            notes="",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )
    assert client.rpc_calls == []


def test_update_rejects_substitute_when_event_policy_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    tables = checkin_tables()
    tables["tournament_event_options"][0]["team_allow_substitutes"] = False
    with pytest.raises(ValueError, match="Every selected event"):
        update_admin_tournament_checkin(
            FakeSupabase(tables),
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-1",
            registration_day_id="day-1",
            expected_updated_at="2026-08-15T12:00:00Z",
            attendance_status="EXPECTED",
            operation_key="00000000-0000-4000-8000-000000000103",
            waiver_verified=False,
            approved_substitute_player_id=10,
            approved_substitute_name=None,
            notes="",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )


def test_update_rejects_name_only_substitute_without_calling_rpc(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = FakeSupabase(checkin_tables())

    with pytest.raises(ValueError, match="Select an active club player"):
        update_admin_tournament_checkin(
            client,
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-1",
            registration_day_id="day-1",
            expected_updated_at="2026-08-15T12:00:00Z",
            attendance_status="CHECKED_IN",
            operation_key="00000000-0000-4000-8000-000000000104",
            waiver_verified=True,
            approved_substitute_player_id=None,
            approved_substitute_name="Name Only Guest",
            notes="",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )

    assert client.rpc_calls == []


def test_update_disables_substitute_even_when_selected_event_policy_allows(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = FakeSupabase(checkin_tables())

    with pytest.raises(ValueError, match="atomic eligibility and uniqueness"):
        update_admin_tournament_checkin(
            client,
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-1",
            registration_day_id="day-1",
            expected_updated_at="2026-08-15T12:00:00Z",
            attendance_status="EXPECTED",
            operation_key="00000000-0000-4000-8000-000000000105",
            waiver_verified=False,
            approved_substitute_player_id=10,
            approved_substitute_name=None,
            notes="",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )

    assert client.rpc_calls == []


def test_update_rpc_revalidates_positive_registration_status_under_lock(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")

    for status in ("WAITLIST", "PENDING"):
        client = RegistrationMutatesBeforeRpc(checkin_tables(), next_status=status)
        with pytest.raises(ValueError, match="can be checked in"):
            update_admin_tournament_checkin(
                client,
                club_id="club-1",
                tournament_id="tour-1",
                registration_id="reg-1",
                registration_day_id="day-1",
                expected_updated_at="2026-08-15T12:00:00Z",
                attendance_status="CHECKED_IN",
                operation_key=(
                    "00000000-0000-4000-8000-000000000106"
                    if status == "WAITLIST"
                    else "00000000-0000-4000-8000-000000000107"
                ),
                waiver_verified=True,
                approved_substitute_player_id=None,
                approved_substitute_name=None,
                notes="",
                actor_email="admin@example.com",
                actor_role="club_owner",
            )

        assert client.rpc_calls


def test_update_rejects_client_substitute_name_even_with_player_id(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = FakeSupabase(checkin_tables())

    with pytest.raises(ValueError, match="atomic eligibility and uniqueness"):
        update_admin_tournament_checkin(
            client,
            club_id="club-1",
            tournament_id="tour-1",
            registration_id="reg-1",
            registration_day_id="day-1",
            expected_updated_at="2026-08-15T12:00:00Z",
            attendance_status="CHECKED_IN",
            operation_key="00000000-0000-4000-8000-000000000108",
            waiver_verified=True,
            approved_substitute_player_id=10,
            approved_substitute_name="Spoofed Browser Name",
            notes="",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )

    assert client.rpc_calls == []
