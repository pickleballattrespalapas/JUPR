from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_registration_repo import (
    ADMIN_SELECTION_UPDATE_RPC,
    StaleTournamentRegistrationSelectionError,
    update_admin_registration_selection,
)
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase as TableFakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


_DEFAULT_RPC_RESULT = object()


class _FakeRpcQuery:
    def __init__(self, client, name: str, params: dict):
        self.client = client
        self.name = str(name)
        self.params = dict(params or {})

    def execute(self):
        self.client.rpc_calls.append((self.name, self.params))
        if self.client.rpc_error is not None:
            raise self.client.rpc_error
        if self.client.rpc_result is not _DEFAULT_RPC_RESULT:
            return SimpleNamespace(data=self.client.rpc_result)
        if self.name != ADMIN_SELECTION_UPDATE_RPC:
            raise RuntimeError(f"unsupported fake RPC: {self.name}")

        rows = self.client.tables.setdefault("tournament_registration_selections", [])
        selection = next(
            (
                row
                for row in rows
                if str(row.get("tournament_id")) == str(self.params.get("p_tournament_id"))
                and str(row.get("id")) == str(self.params.get("p_selection_id"))
            ),
            None,
        )
        if selection is None:
            return SimpleNamespace(data={"ok": False, "code": "SELECTION_NOT_FOUND"})
        if str(selection.get("updated_at") or "") != str(self.params.get("p_expected_updated_at") or ""):
            return SimpleNamespace(
                data={
                    "ok": False,
                    "code": "SELECTION_WRITE_CONFLICT",
                    "reason": "stale_version",
                }
            )

        selection.update(dict(self.params.get("p_patch") or {}))
        self.client.rpc_update_counter += 1
        selection["updated_at"] = f"2026-03-03T00:00:{self.client.rpc_update_counter:02d}Z"
        return SimpleNamespace(data={"ok": True, "selection": dict(selection)})


class FakeSupabase(TableFakeSupabase):
    """File-local RPC-capable fake; shared table fakes remain unchanged."""

    def __init__(self, tables, *, rpc_error=None, rpc_result=_DEFAULT_RPC_RESULT):
        super().__init__(tables)
        self.rpc_error = rpc_error
        self.rpc_result = rpc_result
        self.rpc_calls: list[tuple[str, dict]] = []
        self.rpc_update_counter = 0

    def rpc(self, name, params):
        return _FakeRpcQuery(self, name, params)


def tournament_tables():
    return {
        "tournaments": [
            {
                "club_id": "club",
                "id": "tour_1",
                "name": "Spring Classic",
                "status": "PUBLISHED",
                "start_date": "2026-04-10",
                "end_date": "2026-04-12",
                "created_at": "2026-03-01T00:00:00Z",
                "updated_at": "2026-03-02T00:00:00Z",
            },
            {
                "club_id": "club",
                "id": "tour_archived",
                "name": "Old Classic",
                "status": "ARCHIVED",
                "created_at": "2025-03-01T00:00:00Z",
            },
        ],
        "tournament_registration_settings": [
            {
                "id": "regset_1",
                "tournament_id": "tour_1",
                "registration_slug": "spring-classic",
                "registration_status": "open",
                "waitlist_enabled": True,
                "partner_board_enabled": True,
            }
        ],
        "tournament_registration_days": [
            {"id": "day_1", "tournament_id": "tour_1", "label": "Friday", "event_date": "2026-04-10", "enabled": True, "sort_order": 1}
        ],
        "tournament_event_options": [
            {
                "id": "event_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "label": "3.5",
                "event_family_label": "Gender Doubles",
                "division_name": "3.5",
                "event_type": "DOUBLES",
                "gender_restriction": "ANY",
                "skill_label": "Open",
                "partner_required": False,
                "partner_board_enabled": True,
                "event_format_default": "round_robin",
                "scoring_default": "one_game_to_11",
                "status": "open",
                "enabled": True,
                "sort_order": 1,
            },
            {
                "id": "event_2",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "label": "4.0",
                "event_family_label": "Gender Doubles",
                "division_name": "4.0",
                "event_type": "DOUBLES",
                "gender_restriction": "ANY",
                "skill_label": "Open",
                "partner_required": False,
                "partner_board_enabled": True,
                "event_format_default": "round_robin",
                "scoring_default": "one_game_to_11",
                "status": "open",
                "enabled": True,
                "sort_order": 2,
            },
        ],
        "tournament_registrations": [
            {
                "id": "registration_1",
                "tournament_id": "tour_1",
                "display_name": "Alex Example",
                "email": "alex@example.com",
                "phone": "555-0100",
                "doubles_skill": 3.5,
                "singles_skill": 3.5,
                "gender": "Men",
                "age": 35,
                "status": "confirmed",
                "payment_status": "paid",
                "notes": "Original note",
                "wants_partner_board_contact": True,
                "submitted_at": "2026-03-03T00:00:00Z",
            }
        ],
        "tournament_registration_selections": [
            {
                "id": "selection_1",
                "tournament_id": "tour_1",
                "registration_id": "registration_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": True,
                "created_at": "2026-03-03T00:00:00Z",
                "updated_at": "2026-03-03T00:00:00Z",
            }
        ],
        "tournament_event_draws": [
            {"id": "draw_1", "tournament_id": "tour_1", "registration_day_id": "day_1", "event_option_id": "event_1", "name": "3.5 Draw", "status": "active", "team_count": 2}
        ],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 1, "player1_id": 1, "player2_id": 2, "source": "REGISTRATION"},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 2, "player1_id": 3, "player2_id": 4, "source": "REGISTRATION"},
        ],
        "tournament_games": [
            {"id": "game_1", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "rr_round_number": 1, "rr_slot_number": 1, "team1_id": "team_1", "team2_id": "team_2", "score_team1": 11, "score_team2": 7, "winner_team_id": "team_1", "status": "complete"}
        ],
        "tournament_podium": [
            {"id": "podium_1", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 1, "team_id": "team_1", "award_label": "Gold"}
        ],
        "admin_activity_log": [],
    }


def _install_auth(monkeypatch):
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_tournament_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/tournaments/admin/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["status"] == "guarded_off"
    assert payload["tournaments_endpoint"] is None
    assert payload["registration_export_endpoint"] is None
    assert payload["broadcast_preview_endpoint"] is None


def test_admin_tournament_status_advertises_reporting_endpoints(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)

    response = TestClient(app).get("/admin/clubs/club/tournaments/admin/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["registration_export_endpoint"].endswith(
        "/registrations/export.csv"
    )
    assert payload["broadcast_preview_endpoint"].endswith(
        "/registrations/broadcast-preview"
    )


def test_admin_tournament_list_contract(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_admin_list"
    assert payload["count"] == 1
    assert payload["tournaments"][0]["id"] == "tour_1"
    assert payload["tournaments"][0]["registration_count"] == 1
    assert payload["tournaments"][0]["selection_count"] == 1


def test_admin_tournament_detail_contract(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_admin_detail"
    assert payload["tournament"]["registration_status"] == "open"
    assert payload["summary"]["registrations"] == 1
    assert payload["summary"]["selections"] == 1
    assert payload["summary"]["by_registration_status"] == {"confirmed": 1}
    assert payload["summary"]["by_payment_status"] == {"paid": 1}
    assert payload["registrations"][0]["display_name"] == "Alex Example"
    assert payload["registrations"][0]["notes"] == "Original note"
    assert payload["registrations"][0]["created_at"] == "2026-03-03T00:00:00Z"
    assert payload["event_options"][0]["division_name"] == "3.5"
    assert payload["selections"][0]["event_label"] == "Gender Doubles / 3.5"


def test_admin_tournament_ops_snapshot_contract(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/ops",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_ops_snapshot"
    assert payload["summary"] == {
        "draws": 1,
        "teams": 2,
        "games": 1,
        "podium": 1,
        "rating_children": 0,
        "completed_games": 1,
    }
    assert payload["draws"][0]["id"] == "draw_1"
    assert payload["teams"][0]["team_number"] == 1
    assert payload["games"][0]["winner_team_id"] == "team_1"
    assert payload["podium"][0]["award_label"] == "Gold"


def test_admin_tournament_registration_update_contract(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "registration_status": "waitlist",
            "payment_status": "refunded",
            "notes": "Refunded after withdrawal.",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_registration_update"
    assert payload["registration"]["registration_status"] == "waitlist"
    assert payload["registration"]["payment_status"] == "refunded"
    assert payload["registration"]["notes"] == "Refunded after withdrawal."
    assert tables["tournament_registrations"][0]["status"] == "waitlist"
    assert tables["tournament_registrations"][0]["payment_status"] == "refunded"
    assert tables["admin_activity_log"][0]["action_type"] == "update_tournament_registration_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_selection_update_contract(monkeypatch):
    tables = tournament_tables()
    original_updated_at = tables["tournament_registration_selections"][0]["updated_at"]
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_2",
            "partner_mode": "NEEDS_PARTNER",
            "partner_note": "Still looking for a partner.",
            "expected_updated_at": original_updated_at,
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_selection_update"
    assert payload["selection"]["event_option_id"] == "event_2"
    assert payload["selection"]["event_label"] == "Gender Doubles / 4.0"
    assert payload["selection"]["partner_mode"] == "NEEDS_PARTNER"
    assert payload["selection"]["partner_note"] == "Still looking for a partner."
    assert payload["selection"]["updated_at"] != original_updated_at
    assert tables["tournament_registration_selections"][0]["event_option_id"] == "event_2"
    assert tables["tournament_registration_selections"][0]["registration_day_id"] == "day_1"
    assert tables["admin_activity_log"][0]["action_type"] == "update_tournament_registration_selection_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True
    assert len(supabase.rpc_calls) == 1
    rpc_name, rpc_params = supabase.rpc_calls[0]
    assert rpc_name == ADMIN_SELECTION_UPDATE_RPC
    assert rpc_params["p_tournament_id"] == "tour_1"
    assert rpc_params["p_selection_id"] == "selection_1"
    assert rpc_params["p_expected_updated_at"] == original_updated_at
    assert rpc_params["p_patch"]["event_option_id"] == "event_2"
    assert "updated_at" not in rpc_params["p_patch"]


def test_admin_tournament_selection_update_requires_version_token(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_note": "No version token.",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 422


def test_admin_tournament_selection_update_rejects_stale_version_without_mutation(monkeypatch):
    tables = tournament_tables()
    before = dict(tables["tournament_registration_selections"][0])
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_note": "Stale edit.",
            "expected_updated_at": "2026-03-02T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 409
    assert "changed after it was loaded" in response.json()["detail"]
    assert tables["tournament_registration_selections"][0] == before
    assert tables["admin_activity_log"] == []


def test_admin_selection_repo_maps_stable_database_conflict_marker():
    tables = tournament_tables()
    before = dict(tables["tournament_registration_selections"][0])
    supabase = FakeSupabase(
        tables,
        rpc_error=Exception(
            {
                "code": "P0001",
                "message": "JUPR_SELECTION_WRITE_CONFLICT: relationship_changed",
            }
        ),
    )

    with pytest.raises(StaleTournamentRegistrationSelectionError, match="Refresh and try again"):
        update_admin_registration_selection(
            supabase,
            tournament_id="tour_1",
            selection_id="selection_1",
            payload={"partner_note": "Concurrent edit."},
            expected_updated_at="2026-03-03T00:00:00Z",
        )

    assert tables["tournament_registration_selections"][0] == before
    assert len(supabase.rpc_calls) == 1


@pytest.mark.parametrize(
    ("marker", "expected_detail"),
    [
        ("JUPR_SELECTION_INVALID_TARGET", "Registration selection target is invalid."),
        ("JUPR_SELECTION_INVALID_PATCH", "Registration selection update is invalid."),
        ("JUPR_RELATION_SELECTION_NOT_FOUND", "Registration selection not found for this tournament."),
    ],
)
def test_admin_selection_api_maps_database_validation_markers_to_safe_400(
    monkeypatch,
    marker,
    expected_detail,
):
    tables = tournament_tables()
    before = dict(tables["tournament_registration_selections"][0])
    supabase = FakeSupabase(
        tables,
        rpc_error=Exception({"code": "P0001", "message": f"{marker}: private database detail"}),
    )
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_note": "Database-validated edit.",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail
    assert "private database detail" not in response.text
    assert tables["tournament_registration_selections"][0] == before
    assert tables["admin_activity_log"] == []


def test_admin_selection_repo_resolves_version_for_legacy_streamlit_caller():
    tables = tournament_tables()
    supabase = FakeSupabase(tables)

    updated = update_admin_registration_selection(
        supabase,
        tournament_id="tour_1",
        selection_id="selection_1",
        payload={"partner_note": "Legacy admin edit."},
    )

    assert updated["partner_note"] == "Legacy admin edit."
    assert len(supabase.rpc_calls) == 1
    assert supabase.rpc_calls[0][1]["p_expected_updated_at"] == "2026-03-03T00:00:00Z"


def test_admin_selection_repo_fails_closed_when_rpc_is_unavailable():
    tables = tournament_tables()
    before = dict(tables["tournament_registration_selections"][0])
    supabase = FakeSupabase(tables, rpc_error=Exception("RPC is unavailable"))

    with pytest.raises(RuntimeError, match="Registration selection update failed"):
        update_admin_registration_selection(
            supabase,
            tournament_id="tour_1",
            selection_id="selection_1",
            payload={"partner_note": "Must not fall back."},
            expected_updated_at="2026-03-03T00:00:00Z",
        )

    assert tables["tournament_registration_selections"][0] == before
    assert len(supabase.rpc_calls) == 1


def test_admin_selection_repo_rejects_malformed_rpc_response_without_mutation():
    tables = tournament_tables()
    before = dict(tables["tournament_registration_selections"][0])
    supabase = FakeSupabase(tables, rpc_result=[])

    with pytest.raises(RuntimeError, match="invalid response"):
        update_admin_registration_selection(
            supabase,
            tournament_id="tour_1",
            selection_id="selection_1",
            payload={"partner_note": "Must not be accepted."},
            expected_updated_at="2026-03-03T00:00:00Z",
        )

    assert tables["tournament_registration_selections"][0] == before


def test_admin_tournament_selection_update_rejects_closed_destination(monkeypatch):
    tables = tournament_tables()
    tables["tournament_event_options"][1]["status"] = "closed"
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_2",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Selected division is not open for registration."
    assert tables["tournament_registration_selections"][0]["event_option_id"] == "event_1"


def test_admin_tournament_selection_update_rejects_duplicate_family(monkeypatch):
    tables = tournament_tables()
    tables["tournament_event_options"].append(
        {
            **tables["tournament_event_options"][1],
            "id": "event_3",
            "division_name": "4.5",
            "label": "4.5",
            "sort_order": 3,
        }
    )
    tables["tournament_registration_selections"].append(
        {
            **tables["tournament_registration_selections"][0],
            "id": "selection_2",
            "event_option_id": "event_2",
        }
    )
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_3",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert "Choose only one division" in response.json()["detail"]
    assert tables["tournament_registration_selections"][0]["event_option_id"] == "event_1"


def test_admin_tournament_selection_update_clears_stale_partner_identity(monkeypatch):
    tables = tournament_tables()
    selection = tables["tournament_registration_selections"][0]
    selection.update(
        {
            "partner_mode": "HAS_PARTNER",
            "partner_name": "Legacy Partner",
            "partner_email": "legacy@example.com",
            "partner_phone": "555-0199",
            "partner_dupr_id": "legacy-dupr",
            "partner_skill": 3.5,
            "partner_age": 40,
        }
    )
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_mode": "NEEDS_PARTNER",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 200
    updated = tables["tournament_registration_selections"][0]
    assert updated["partner_mode"] == "NEEDS_PARTNER"
    assert updated["partner_name"] is None
    assert updated["partner_email"] is None
    assert updated["partner_phone"] is None
    assert updated["partner_dupr_id"] is None
    assert updated["partner_skill"] is None
    assert updated["partner_age"] is None


def test_admin_tournament_selection_update_rejects_free_text_partner_creation(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_mode": "HAS_PARTNER",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert "canonical partner-link workflow" in response.json()["detail"]
    assert tables["tournament_registration_selections"][0]["partner_mode"] == "NEEDS_PARTNER"


def test_admin_tournament_selection_update_enforces_skill_eligibility(monkeypatch):
    tables = tournament_tables()
    tables["tournament_event_options"][1]["skill_label"] = "3.5"
    tables["tournament_registrations"][0]["doubles_skill"] = 4.0
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_2",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert "above the 3.5 division cap" in response.json()["detail"]
    assert tables["tournament_registration_selections"][0]["event_option_id"] == "event_1"


def test_admin_tournament_selection_update_enforces_gender_eligibility(monkeypatch):
    tables = tournament_tables()
    tables["tournament_event_options"][1]["gender_restriction"] = "WOMEN"
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_2",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert "limited to women's registrations" in response.json()["detail"]
    assert tables["tournament_registration_selections"][0]["event_option_id"] == "event_1"


def test_admin_tournament_selection_update_blocks_pending_partner_request_change(monkeypatch):
    tables = tournament_tables()
    tables["tournament_registration_partner_requests"] = [
        {
            "id": "request_1",
            "tournament_id": "tour_1",
            "requester_selection_id": "selection_1",
            "target_selection_id": None,
            "status": "PENDING",
        }
    ]
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_mode": "NONE",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert "pending partner request" in response.json()["detail"]
    assert tables["tournament_registration_selections"][0]["partner_mode"] == "NEEDS_PARTNER"


def test_admin_tournament_selection_update_blocks_confirmed_team_change(monkeypatch):
    tables = tournament_tables()
    tables["tournament_registration_team_members"] = [
        {
            "id": "member_1",
            "tournament_id": "tour_1",
            "selection_id": "selection_1",
            "status": "ACTIVE",
        }
    ]
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_mode": "NONE",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400
    assert "confirmed partner team" in response.json()["detail"]
    assert tables["tournament_registration_selections"][0]["partner_mode"] == "NEEDS_PARTNER"


def test_admin_tournament_bulk_registration_update_contract(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/bulk",
        headers={"Authorization": "Bearer local"},
        json={
            "registration_ids": ["registration_1"],
            "registration_status": "cancelled",
            "payment_status": "refunded",
            "append_note": "Bulk cancellation.",
            "confirmation_text": "BULK UPDATE REGISTRATIONS",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_registration_bulk_update"
    assert payload["updated_count"] == 1
    assert payload["registration_ids"] == ["registration_1"]
    assert payload["registrations"][0]["registration_status"] == "cancelled"
    assert payload["registrations"][0]["payment_status"] == "refunded"
    assert tables["tournament_registrations"][0]["status"] == "cancelled"
    assert tables["tournament_registrations"][0]["payment_status"] == "refunded"
    assert tables["tournament_registrations"][0]["notes"] == "Original note\nBulk cancellation."
    assert tables["admin_activity_log"][0]["action_type"] == "bulk_update_tournament_registrations_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_registration_filtered_csv_export_contract(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/export.csv",
        params={"partner_mode": "NEEDS_PARTNER", "search": "alex"},
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["cache-control"] == "private, no-store, max-age=0"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-jupr-export-row-count"] == "1"
    assert "registration_id,selection_id" in response.text
    assert "registration_1,selection_1" in response.text
    assert "Alex Example" in response.text
    assert "Gender Doubles / 3.5" in response.text


def test_admin_tournament_broadcast_preview_is_dry_run_only(monkeypatch):
    tables = tournament_tables()
    tables["tournament_registrations"].append(
        {
            "id": "registration_cancelled",
            "tournament_id": "tour_1",
            "display_name": "Cancelled Player",
            "email": "cancelled@example.com",
            "status": "cancelled",
            "payment_status": "refunded",
            "submitted_at": "2026-03-02T00:00:00Z",
        }
    )
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/broadcast-preview",
        headers={"Authorization": "Bearer local"},
        json={
            "subject": "Court update",
            "message": "Check in at 8:00.",
            "include_cancelled": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store, max-age=0"
    assert response.headers["x-content-type-options"] == "nosniff"
    payload = response.json()
    assert payload["mode"] == "tournament_broadcast_preview"
    assert payload["dry_run"] is True
    assert payload["send_available"] is False
    assert payload["recipient_count"] == 1
    assert payload["recipients"][0]["email"] == "alex@example.com"
    assert "cancelled@example.com" not in payload["recipient_csv"]
    assert payload["preview"]["subject"] == "Spring Classic: Court update"
    assert "Check in at 8:00." in payload["preview"]["text"]
    assert payload["warnings"] == ["Preview only. This endpoint never sends email."]


def test_admin_tournament_registration_export_rejects_cross_club(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).get(
        "/admin/clubs/another-club/tournaments/admin/tournaments/tour_1/registrations/export.csv",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "tournament not found"


def test_admin_tournament_registration_export_requires_manage_permission(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(
            email="readonly@example.com",
            user_id="user-readonly",
        ),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="read_only"),
    )

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/export.csv",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient permission"
    assert tables["admin_activity_log"][-1]["action_type"] == "admin_tournament_denied"
