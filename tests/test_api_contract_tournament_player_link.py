from types import SimpleNamespace

import pytest
from postgrest.exceptions import APIError

from tests.test_api_contract_admin_tournament_registration_import import (
    _client, tournament_registration_import_tables,
)
from tests.test_admin_match_log_service import FakeSupabase
from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_snapshot

LINK_URL = "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/reg_1/link-player"
BODY = {"player_id": 1, "expected_updated_at": "2026-10-31T12:00:00Z", "confirmation_text": "LINK PLAYER"}


def test_ops_snapshot_lists_only_unlinked_confirmed_draw_entrants(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = tournament_registration_import_tables(missing_player=True)
    tables["players"] = [{"id": 1, "club_id": "club", "name": "Alex Example", "active": True},
                         {"id": 8, "club_id": "club", "name": "Alex (merged into 1)", "active": False}]
    tables["tournament_event_draws"].append({"id": "draw_2", "tournament_id": "tour_1", "event_option_id": "other", "registration_day_id": "day_1"})
    tables["tournament_registrations"][0]["updated_at"] = BODY["expected_updated_at"]
    result = get_admin_tournament_ops_snapshot(FakeSupabase(tables), club_id="club", tournament_id="tour_1", draw_id="draw_1")
    assert result["unlinked_registrations"] == [{"id": "reg_1", "display_name": "Alex Example", "updated_at": BODY["expected_updated_at"], "draw_ids": ["draw_1"]}]
    assert {row["id"] for row in result["players"]} == {1, 8}, "Historical team labels still need retained merged profiles"
    assert "email" not in result["unlinked_registrations"][0]
    result = get_admin_tournament_ops_snapshot(FakeSupabase(tables), club_id="club", tournament_id="tour_1", draw_id="draw_2")
    assert result["unlinked_registrations"] == []
    tables["tournament_registration_selections"][0]["partner_mode"] = "NEEDS_PARTNER"
    result = get_admin_tournament_ops_snapshot(FakeSupabase(tables), club_id="club", tournament_id="tour_1", draw_id="draw_1")
    assert result["unlinked_registrations"] == [], "Players still looking for partners are not import blockers"


def test_combined_draw_lists_unlinked_secondary_registrant_and_fingerprints_membership(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = tournament_registration_import_tables(missing_player=True)
    tables["tournament_registrations"][0]["updated_at"] = BODY["expected_updated_at"]
    tables["tournament_registration_selections"][0]["event_option_id"] = "event_2"
    database = FakeSupabase(tables)
    before = get_admin_tournament_ops_snapshot(database, club_id="club", tournament_id="tour_1", draw_id="draw_1")
    assert before["unlinked_registrations"] == []

    tables["tournament_event_draws"][0]["source_event_option_ids"] = ["event_1", "event_2"]
    after = get_admin_tournament_ops_snapshot(database, club_id="club", tournament_id="tour_1", draw_id="draw_1")

    assert after["unlinked_registrations"] == [{
        "id": "reg_1", "display_name": "Alex Example", "updated_at": BODY["expected_updated_at"],
        "draw_ids": ["draw_1"],
    }]
    assert before["state_fingerprint"] and after["state_fingerprint"]
    assert before["state_fingerprint"] != after["state_fingerprint"]
    assert tables["tournament_registration_selections"][0]["event_option_id"] == "event_2"
    assert "email" not in after["unlinked_registrations"][0]


@pytest.mark.parametrize("entry_event,entry_day,visible", [
    ("event_1", "day_1", True),
    ("event_2", "day_1", True),
    ("event_2", "another_day", False),
    ("unrelated_event", "day_1", False),
])
def test_combined_draw_profile_link_scope_keeps_day_and_source_boundaries(
    monkeypatch, entry_event, entry_day, visible,
):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = tournament_registration_import_tables(missing_player=True)
    tables["tournament_event_draws"][0]["source_event_option_ids"] = ["event_1", "event_2"]
    tables["tournament_registration_selections"][0].update(
        event_option_id=entry_event, registration_day_id=entry_day,
    )

    result = get_admin_tournament_ops_snapshot(FakeSupabase(tables), club_id="club", tournament_id="tour_1", draw_id="draw_1")

    assert [row["id"] for row in result["unlinked_registrations"]] == (["reg_1"] if visible else [])


@pytest.mark.parametrize("rpc_result,status", [
    ({"ok": True, "player_id": 1}, 200),
    ({"ok": True, "player_id": 1, "idempotent_replay": True}, 200),
    ({"ok": False, "code": "STALE", "message": "Reload the draw."}, 409),
    ({"ok": False, "code": "DUPLICATE", "message": "Player already registered."}, 400),
    ({"ok": False, "code": "PLAYER", "message": "Choose a profile from this club."}, 400),
])
def test_link_route_uses_atomic_rpc_and_preserves_safe_errors(monkeypatch, rpc_result, status):
    client = _client(monkeypatch, tournament_registration_import_tables())
    calls = []
    def rpc(_self, name, payload):
        calls.append((name, payload))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=rpc_result))
    monkeypatch.setattr(FakeSupabase, "rpc", rpc, raising=False)
    response = client.post(LINK_URL, headers={"Authorization": "Bearer local"}, json=BODY)
    assert response.status_code == status
    assert calls == [("admin_link_tournament_registration_player_v1", {
        "p_club_id": "club", "p_tournament_id": "tour_1", "p_registration_id": "reg_1",
        "p_player_id": 1, "p_expected_updated_at": BODY["expected_updated_at"],
        "p_actor_email": "admin@example.com", "p_actor_role": "club_owner",
    })]
    if status != 200:
        assert response.json()["detail"] == rpc_result["message"]


def test_link_route_explains_existing_registration_unique_constraint(monkeypatch):
    client = _client(monkeypatch, tournament_registration_import_tables())
    def rpc(*_args, **_kwargs):
        raise APIError({"code": "23505", "message": 'duplicate key violates unique constraint "uq_tournament_registrations_tournament_player"', "details": None, "hint": None})
    monkeypatch.setattr(FakeSupabase, "rpc", rpc, raising=False)
    response = client.post(LINK_URL, headers={"Authorization": "Bearer local"}, json=BODY)
    assert response.status_code == 400
    assert "already has a registration" in response.json()["detail"]


@pytest.mark.parametrize("case,status", [("confirmation", 400), ("disabled", 403), ("staging_gate", 403), ("unauthorized", 403), ("invalid_player", 422), ("invalid_version", 400)])
def test_link_route_rejects_before_rpc(monkeypatch, case, status):
    client = _client(monkeypatch, tournament_registration_import_tables())
    def no_rpc(*_args, **_kwargs):
        pytest.fail("Rejected requests must not mutate any player links")
    monkeypatch.setattr(FakeSupabase, "rpc", no_rpc, raising=False)
    body = dict(BODY)
    if case == "confirmation":
        body["confirmation_text"] = ""
    elif case == "disabled":
        monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "0")
    elif case == "staging_gate":
        monkeypatch.setenv("JUPR_ENV", "staging")
        monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS", "0")
    elif case == "unauthorized":
        monkeypatch.setattr("services.api.admin_tournament_routes.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="viewer"))
    elif case == "invalid_player":
        body["player_id"] = 0
    else:
        body["expected_updated_at"] = "invalid"
    response = client.post(LINK_URL, headers={"Authorization": "Bearer local"}, json=body)
    assert response.status_code == status
