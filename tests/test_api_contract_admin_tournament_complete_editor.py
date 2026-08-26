from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_registration_repo import (
    ADMIN_SELECTION_DELETE_RPC,
    REGISTRATION_SCHEMA_CONTRACT_MIGRATIONS,
    REGISTRATION_SCHEMA_REQUIRED_COLUMNS,
)
from jupr_app.services.admin_tournament_service import (
    update_admin_tournament_registration,
)
from tests.conftest import require_api_dependency
from tests.test_api_contract_admin_tournament import (
    FakeSupabase,
    tournament_tables,
)

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


_DEFAULT_DELETE_RESULT = object()


class _DeleteRpcQuery:
    def __init__(self, client, params: dict):
        self.client = client
        self.params = dict(params)

    def execute(self):
        self.client.rpc_calls.append((ADMIN_SELECTION_DELETE_RPC, self.params))
        if self.client.delete_rpc_error is not None:
            raise self.client.delete_rpc_error
        if self.client.delete_rpc_result is not _DEFAULT_DELETE_RESULT:
            return SimpleNamespace(data=self.client.delete_rpc_result)
        rows = self.client.tables.setdefault("tournament_registration_selections", [])
        selection = next(
            (
                row
                for row in rows
                if str(row.get("tournament_id"))
                == str(self.params.get("p_tournament_id"))
                and str(row.get("id")) == str(self.params.get("p_selection_id"))
            ),
            None,
        )
        if selection is None:
            return SimpleNamespace(data={"ok": False, "code": "SELECTION_NOT_FOUND"})
        if str(selection.get("updated_at") or "") != str(
            self.params.get("p_expected_updated_at") or ""
        ):
            return SimpleNamespace(
                data={"ok": False, "code": "SELECTION_WRITE_CONFLICT"}
            )
        deleted = dict(selection)
        rows.remove(selection)
        return SimpleNamespace(data={"ok": True, "selection": deleted})


class CompleteEditorFakeSupabase(FakeSupabase):
    def __init__(
        self,
        tables,
        *,
        delete_rpc_error=None,
        delete_rpc_result=_DEFAULT_DELETE_RESULT,
    ):
        super().__init__(tables)
        self.delete_rpc_error = delete_rpc_error
        self.delete_rpc_result = delete_rpc_result

    def rpc(self, name, params):
        if str(name) == ADMIN_SELECTION_DELETE_RPC:
            return _DeleteRpcQuery(self, params)
        return super().rpc(name, params)


def _tables() -> dict:
    tables = tournament_tables()
    registration = tables["tournament_registrations"][0]
    registration.update(
        {
            "first_name": "Alex",
            "last_name": "Example",
            "dupr_id": "dupr-alex",
            "age_bracket": "35+",
            "updated_at": "2026-03-03T00:00:00Z",
        }
    )
    tables["players"] = [
        {"id": 10, "club_id": "club", "name": "Alex Linked"},
        {"id": 11, "club_id": "another-club", "name": "Wrong Club"},
    ]
    tables["tournament_admin_operations"] = []
    return tables


def _install(monkeypatch, supabase) -> TestClient:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS", "1"
    )
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr(
        "services.api.main.create_client", lambda _url, _credential: supabase
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(
            email="admin@example.com", user_id="user-1"
        ),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )
    return TestClient(app)


def _detail(client: TestClient) -> dict:
    response = client.get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1",
        headers={"Authorization": "Bearer local"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _enable_guarded_staging(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-registration")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")


def test_registration_editor_round_trip_null_clears_and_waived(monkeypatch) -> None:
    tables = _tables()
    client = _install(monkeypatch, CompleteEditorFakeSupabase(tables))

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "first_name": "Alexis",
            "last_name": "Example",
            "display_name": "Alexis Example",
            "email": "ALEXIS@example.com",
            "phone": None,
            "player_id": 10,
            "gender": "Women",
            "age": 36,
            "age_bracket": None,
            "dupr_id": None,
            "doubles_skill": 4.25,
            "singles_skill": 4.0,
            "wants_partner_board_contact": False,
            "registration_status": "confirmed",
            "payment_status": "waived",
            "notes": None,
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE REGISTRATION",
            "source": "next_tournament_registration_detail",
        },
    )

    assert response.status_code == 200, response.text
    row = tables["tournament_registrations"][0]
    assert row["email"] == "alexis@example.com"
    assert row["phone"] is None
    assert row["age_bracket"] is None
    assert row["dupr_id"] is None
    assert row["notes"] is None
    assert row["player_id"] == 10
    assert row["payment_status"] == "waived"
    assert response.json()["registration"]["doubles_skill"] == 4.25
    assert tables["admin_activity_log"][-1]["action_type"] == (
        "update_tournament_registration_admin"
    )


@pytest.mark.parametrize(
    ("patch", "detail"),
    [
        ({"email": "not-an-email"}, "valid registration email"),
        ({"age": 4}, "between 5 and 120"),
        ({"doubles_skill": "nan"}, "finite number"),
        ({"player_id": 11}, "does not belong to this club"),
    ],
)
def test_registration_editor_rejects_invalid_identity_and_numbers(
    monkeypatch, patch, detail
) -> None:
    tables = _tables()
    before = dict(tables["tournament_registrations"][0])
    client = _install(monkeypatch, CompleteEditorFakeSupabase(tables))

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            **patch,
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 400, response.text
    assert detail in response.json()["detail"]
    assert tables["tournament_registrations"][0] == before


@pytest.mark.parametrize(
    ("patch", "detail"),
    [
        ({"age": True}, "Age must be a whole number"),
        ({"doubles_skill": True}, "Doubles Skill must be a number"),
    ],
)
def test_registration_service_rejects_boolean_numeric_values(
    monkeypatch, patch, detail
) -> None:
    tables = _tables()
    before = dict(tables["tournament_registrations"][0])
    supabase = CompleteEditorFakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    with pytest.raises(ValueError, match=detail):
        update_admin_tournament_registration(
            supabase,
            club_id="club",
            tournament_id="tour_1",
            registration_id="registration_1",
            patch=patch,
            expected_updated_at="2026-03-03T00:00:00Z",
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="SAVE REGISTRATION",
            dry_run=True,
        )

    assert tables["tournament_registrations"][0] == before
    assert tables["admin_activity_log"] == []


def test_registration_editor_rejects_duplicate_email_during_preflight(
    monkeypatch,
) -> None:
    tables = _tables()
    tables["tournament_registrations"].append(
        {
            "id": "registration_2",
            "tournament_id": "tour_1",
            "display_name": "Other Player",
            "email": "Other@Example.com",
            "status": "confirmed",
            "payment_status": "paid",
            "updated_at": "2026-03-03T00:00:00Z",
        }
    )
    client = _install(monkeypatch, CompleteEditorFakeSupabase(tables))

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "email": "other@example.com",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 400, response.text
    assert "already uses that email" in response.json()["detail"]
    assert tables["admin_activity_log"] == []


@pytest.mark.parametrize("email", [None, "", "   "])
def test_registration_editor_email_cannot_be_cleared(monkeypatch, email) -> None:
    tables = _tables()
    before = dict(tables["tournament_registrations"][0])
    client = _install(monkeypatch, CompleteEditorFakeSupabase(tables))

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "email": email,
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 400, response.text
    assert "email is required" in response.json()["detail"]
    assert tables["tournament_registrations"][0] == before


def test_registration_editor_requires_row_version(monkeypatch) -> None:
    client = _install(monkeypatch, CompleteEditorFakeSupabase(_tables()))

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "phone": "555-0199",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 422


def test_imported_registration_allows_non_identity_edit_with_unchanged_guarded_fields(
    monkeypatch,
) -> None:
    tables = _tables()
    registration = tables["tournament_registrations"][0]
    registration["player_id"] = 10
    tables["tournament_teams"] = [
        {
            "id": "team_1",
            "tournament_id": "tour_1",
            "registration_day_id": "day_1",
            "event_option_id": "event_1",
            "source": "REGISTRATION",
        }
    ]
    client = _install(monkeypatch, CompleteEditorFakeSupabase(tables))

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "phone": "555-0199",
            "player_id": 10,
            "registration_status": "confirmed",
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 200, response.text
    assert registration["phone"] == "555-0199"


def test_selection_create_supports_complete_manual_partner(monkeypatch) -> None:
    tables = _tables()
    tables["tournament_registration_selections"] = []
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)
    fingerprint = _detail(client)["state_fingerprint"]

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1/selections",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_1",
            "partner_mode": "HAS_PARTNER",
            "partner_name": "Blair Partner",
            "partner_email": "blair@example.com",
            "partner_phone": "555-0120",
            "partner_dupr_id": "dupr-blair",
            "partner_skill": 3.75,
            "partner_age": 34,
            "partner_gender": "Men",
            "partner_note": "Manual partner verified by staff.",
            "show_on_partner_board": True,
            "expected_state_fingerprint": fingerprint,
            "confirmation_text": "SAVE SELECTION",
            "source": "next_tournament_registration_detail",
        },
    )

    assert response.status_code == 200, response.text
    selection = tables["tournament_registration_selections"][0]
    assert selection["registration_day_id"] == "day_1"
    assert selection["partner_mode"] == "HAS_PARTNER"
    assert selection["partner_email"] == "blair@example.com"
    assert selection["partner_gender"] == "Men"
    assert selection["show_on_partner_board"] is False
    assert tables["admin_activity_log"][-1]["action_type"] == (
        "create_tournament_registration_selection_admin"
    )


@pytest.mark.parametrize(
    ("event_option_id", "detail"),
    [
        ("event_1", "already has an entry"),
        ("event_2", "Choose only one division"),
    ],
)
def test_selection_create_rejects_exact_and_family_duplicates(
    monkeypatch, event_option_id, detail
) -> None:
    tables = _tables()
    client = _install(monkeypatch, CompleteEditorFakeSupabase(tables))
    fingerprint = _detail(client)["state_fingerprint"]

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1/selections",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": event_option_id,
            "partner_mode": "NONE",
            "expected_state_fingerprint": fingerprint,
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 400, response.text
    assert detail in response.json()["detail"]
    assert len(tables["tournament_registration_selections"]) == 1


def test_selection_update_supports_complete_manual_partner(monkeypatch) -> None:
    tables = _tables()
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_mode": "HAS_PARTNER",
            "partner_name": "Blair Partner",
            "partner_email": "blair@example.com",
            "partner_phone": None,
            "partner_dupr_id": "dupr-blair",
            "partner_skill": 3.75,
            "partner_age": 34,
            "partner_gender": "Women",
            "partner_note": None,
            "show_on_partner_board": True,
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 200, response.text
    selection = tables["tournament_registration_selections"][0]
    assert selection["partner_mode"] == "HAS_PARTNER"
    assert selection["partner_name"] == "Blair Partner"
    assert selection["partner_phone"] is None
    assert selection["partner_gender"] == "Women"
    assert selection["partner_note"] is None
    assert selection["show_on_partner_board"] is False
    assert supabase.rpc_calls[-1][1]["p_patch"]["partner_gender"] == "Women"


@pytest.mark.parametrize("operation", ["create", "update"])
def test_partner_board_requires_registration_consent(
    monkeypatch, operation
) -> None:
    tables = _tables()
    tables["tournament_registrations"][0]["wants_partner_board_contact"] = False
    if operation == "create":
        tables["tournament_registration_selections"] = []
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)
    if operation == "create":
        response = client.post(
            "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1/selections",
            headers={"Authorization": "Bearer local"},
            json={
                "event_option_id": "event_1",
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": True,
                "expected_state_fingerprint": _detail(client)["state_fingerprint"],
                "confirmation_text": "SAVE SELECTION",
            },
        )
    else:
        response = client.patch(
            "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
            headers={"Authorization": "Bearer local"},
            json={
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": True,
                "expected_updated_at": "2026-03-03T00:00:00Z",
                "confirmation_text": "SAVE SELECTION",
            },
        )

    assert response.status_code == 400, response.text
    assert "consent is required" in response.json()["detail"]
    assert tables["admin_activity_log"] == []


def test_partner_board_with_registration_consent_is_allowed(monkeypatch) -> None:
    tables = _tables()
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)

    response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "partner_mode": "NEEDS_PARTNER",
            "show_on_partner_board": True,
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 200, response.text
    assert tables["tournament_registration_selections"][0][
        "show_on_partner_board"
    ] is True


def test_selection_delete_uses_cas_rpc_and_audits(monkeypatch) -> None:
    tables = _tables()
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)

    response = client.request(
        "DELETE",
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "REMOVE SELECTION",
            "source": "next_tournament_registration_detail",
        },
    )

    assert response.status_code == 200, response.text
    assert tables["tournament_registration_selections"] == []
    assert supabase.rpc_calls[-1] == (
        ADMIN_SELECTION_DELETE_RPC,
        {
            "p_tournament_id": "tour_1",
            "p_selection_id": "selection_1",
            "p_expected_updated_at": "2026-03-03T00:00:00Z",
        },
    )
    assert tables["admin_activity_log"][-1]["action_type"] == (
        "delete_tournament_registration_selection_admin"
    )


@pytest.mark.parametrize(
    ("lock_kind", "detail"),
    [
        ("confirmed_link", "confirmed partner team"),
        ("pending_request", "pending partner request"),
        ("imported_draw", "imported into a draw"),
    ],
)
def test_selection_delete_prechecks_relationships_and_draw_without_rpc(
    monkeypatch, lock_kind, detail
) -> None:
    tables = _tables()
    if lock_kind == "confirmed_link":
        tables["tournament_registration_team_links"] = [
            {
                "id": "link_1",
                "tournament_id": "tour_1",
                "selection1_id": "selection_1",
                "selection2_id": "selection_2",
                "status": "CONFIRMED",
            }
        ]
    elif lock_kind == "pending_request":
        tables["tournament_registration_partner_requests"] = [
            {
                "id": "request_1",
                "tournament_id": "tour_1",
                "requester_selection_id": "selection_1",
                "target_selection_id": "selection_2",
                "status": "PENDING",
            }
        ]
    else:
        tables["tournament_teams"] = [
            {
                "id": "team_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "source": "REGISTRATION",
                "source_selection_id": "selection_1",
            }
        ]
    before = list(tables["tournament_registration_selections"])
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)

    response = client.request(
        "DELETE",
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "REMOVE SELECTION",
        },
    )

    assert response.status_code == 400, response.text
    assert detail in response.json()["detail"]
    assert tables["tournament_registration_selections"] == before
    assert supabase.rpc_calls == []
    assert tables["admin_activity_log"] == []


def test_guarded_selection_create_replays_without_duplicate_insert(monkeypatch) -> None:
    tables = _tables()
    tables["tournament_registration_selections"] = []
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)
    fingerprint = _detail(client)["state_fingerprint"]
    _enable_guarded_staging(monkeypatch)
    request = {
        "event_option_id": "event_1",
        "partner_mode": "NONE",
        "expected_state_fingerprint": fingerprint,
        "confirmation_text": "SAVE SELECTION",
    }

    first = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1/selections",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    replay = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1/selections",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert first.json()["idempotent_replay"] is False
    assert replay.json()["idempotent_replay"] is True
    assert len(tables["tournament_registration_selections"]) == 1
    assert len(tables["tournament_admin_operations"]) == 1
    operation = tables["tournament_admin_operations"][0]
    assert operation["entity_type"] == "tournament_registration_selection"
    assert operation["entity_id"] == "registration_1:event_1"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_selection_create_intent",
        "create_tournament_registration_selection_admin",
        "tournament_registration_selection_create_completion",
    ]


def test_guarded_selection_delete_replays_without_second_rpc(monkeypatch) -> None:
    tables = _tables()
    supabase = CompleteEditorFakeSupabase(tables)
    client = _install(monkeypatch, supabase)
    _enable_guarded_staging(monkeypatch)
    request = {
        "expected_updated_at": "2026-03-03T00:00:00Z",
        "confirmation_text": "REMOVE SELECTION",
    }

    first = client.request(
        "DELETE",
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    replay = client.request(
        "DELETE",
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert first.json()["idempotent_replay"] is False
    assert replay.json()["idempotent_replay"] is True
    assert len(
        [call for call in supabase.rpc_calls if call[0] == ADMIN_SELECTION_DELETE_RPC]
    ) == 1
    assert len(tables["tournament_admin_operations"]) == 1
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_selection_delete_intent",
        "delete_tournament_registration_selection_admin",
        "tournament_registration_selection_delete_completion",
    ]


@pytest.mark.parametrize(
    ("code", "status_code", "detail"),
    [
        ("SELECTION_WRITE_CONFLICT", 409, "changed after it was loaded"),
        ("SELECTION_RELATIONSHIP_LOCKED", 400, "active partner relationship"),
        ("SELECTION_IMPORTED_TO_DRAW", 400, "imported into a draw"),
        ("SELECTION_NOT_FOUND", 400, "not found"),
    ],
)
def test_selection_delete_rpc_codes_fail_closed(
    monkeypatch, code, status_code, detail
) -> None:
    tables = _tables()
    before = list(tables["tournament_registration_selections"])
    supabase = CompleteEditorFakeSupabase(
        tables, delete_rpc_result={"ok": False, "code": code}
    )
    client = _install(monkeypatch, supabase)

    response = client.request(
        "DELETE",
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_updated_at": "2026-03-03T00:00:00Z",
            "confirmation_text": "REMOVE SELECTION",
        },
    )

    assert response.status_code == status_code, response.text
    assert detail in response.json()["detail"]
    assert tables["tournament_registration_selections"] == before
    assert tables["admin_activity_log"] == []


def test_selection_partner_gender_is_a_required_schema_column() -> None:
    assert REGISTRATION_SCHEMA_REQUIRED_COLUMNS[
        "tournament_registration_selections"
    ] == ("partner_gender",)
    assert (
        "supabase/migrations/20260807150000_tournament_complete_registration_editor.sql"
        in REGISTRATION_SCHEMA_CONTRACT_MIGRATIONS
    )


def test_admin_status_exposes_selection_create_and_delete_capabilities(
    monkeypatch,
) -> None:
    client = _install(monkeypatch, CompleteEditorFakeSupabase(_tables()))

    response = client.get(
        "/admin/clubs/club/tournaments/admin/status",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["selection_create_endpoint"] == (
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}"
        "/registrations/{registration_id}/selections"
    )
    assert response.json()["selection_delete_endpoint"] == (
        "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}"
        "/selections/{selection_id}"
    )


def test_disabled_admin_status_hides_selection_create_and_delete_capabilities(
    monkeypatch,
) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", raising=False)
    from jupr_app.services.admin_tournament_service import (
        build_admin_tournament_status,
    )

    status = build_admin_tournament_status(None, club_id="club")

    assert status["selection_create_endpoint"] is None
    assert status["selection_delete_endpoint"] is None
