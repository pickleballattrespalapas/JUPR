from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_api_contract_admin_weekly_recap import _install_env, weekly_recap_tables
from tests.test_admin_match_log_service import FakeQuery, FakeSupabase

from fastapi import HTTPException
from fastapi.testclient import TestClient

from jupr_app.domain.admin.staff_policy import operator_request_authorized
from jupr_app.services import admin_weekly_recap_service as service
from services.api.main import app


DELETE_URL = "/admin/clubs/club/weekly-recap/recaps/2026-07-06"
DELETE_BODY = {
    "expected_recap_id": "recap-1",
    "expected_row_version": 1,
    "confirmation_text": "DELETE RECAP",
}


def delete(client, body=None, url=DELETE_URL):
    return client.request(
        "DELETE", url, headers={"Authorization": "Bearer local"},
        json=DELETE_BODY if body is None else body,
    )


@pytest.fixture
def workspace(monkeypatch):
    db = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, db)
    return db, TestClient(app)


def test_delete_removes_only_reviewed_draft_and_audits_before_after(workspace):
    db, client = workspace
    before = deepcopy(db.tables["weekly_recaps"][0])
    others = [
        {**before, "id": "other-club", "club_id": "another-club"},
        {**before, "id": "other-draft", "week_start": "2026-07-13"},
        {**before, "id": "published", "week_start": "2026-06-29", "status": "published"},
    ]
    db.tables["weekly_recaps"].extend(others)
    response = delete(client)
    assert response.status_code == 200
    assert response.json()["deleted_recap_id"] == "recap-1"
    assert db.tables["weekly_recaps"] == others
    audit = db.tables["admin_activity_log"][-1]
    assert audit["action_type"] == "delete_weekly_recap_draft_admin"
    assert audit["before_json"]["recap"]["final_json"] == before["final_json"]
    assert audit["after_json"]["recap"] is None
    assert delete(client).status_code == 409


@pytest.mark.parametrize("role", ["administrator", "club_owner", "super_admin"])
def test_administrator_roles_can_delete_and_list_exposes_capability(monkeypatch, workspace, role):
    _, client = workspace
    monkeypatch.setattr("services.api.admin_weekly_recap_routes.resolve_admin_role", lambda **_: SimpleNamespace(role=role))
    listed = client.get("/admin/clubs/club/weekly-recap/recaps", headers={"Authorization": "Bearer local"})
    assert listed.json()["can_delete_drafts"] is True
    assert delete(client).status_code == 200


@pytest.mark.parametrize("role", ["operator", "organizer", "scorekeeper", "read_only", "__unassigned__"])
def test_non_administrator_cannot_delete(monkeypatch, workspace, role):
    db, client = workspace
    def resolved_role(**_):
        operator_request_authorized.set(True)
        return SimpleNamespace(role=role)

    monkeypatch.setattr("services.api.admin_weekly_recap_routes.resolve_admin_role", resolved_role)
    response = delete(client)
    if role == "operator":
        listed = client.get("/admin/clubs/club/weekly-recap/recaps", headers={"Authorization": "Bearer local"})
        assert listed.status_code == 200
        assert listed.json()["can_delete_drafts"] is False
    assert response.status_code == 403
    assert len(db.tables["weekly_recaps"]) == 1


@pytest.mark.parametrize("patch", [
    {"expected_recap_id": "replaced-draft"},
    {"expected_row_version": 2},
])
def test_stale_delete_does_not_remove_current_draft(workspace, patch):
    db, client = workspace
    response = delete(client, {**DELETE_BODY, **patch})
    assert response.status_code == 409
    assert len(db.tables["weekly_recaps"]) == 1


@pytest.mark.parametrize("key", ["expected_recap_id", "expected_row_version"])
def test_delete_requires_exact_reviewed_record(workspace, key):
    db, client = workspace
    body = {k: v for k, v in DELETE_BODY.items() if k != key}
    assert delete(client, body).status_code == 422
    assert len(db.tables["weekly_recaps"]) == 1


def test_delete_requires_confirmation(workspace):
    db, client = workspace
    assert delete(client, {**DELETE_BODY, "confirmation_text": ""}).status_code == 400
    assert len(db.tables["weekly_recaps"]) == 1


def test_published_recap_cannot_be_deleted(workspace):
    db, client = workspace
    db.tables["weekly_recaps"][0]["status"] = "published"
    assert delete(client).status_code == 409
    assert db.tables["weekly_recaps"][0]["status"] == "published"


@pytest.mark.parametrize("race_patch", [
    {"status": "published"},
    {"row_version": 2, "edits_json": {"looking_ahead": ["New edit"]}},
    {"id": "replacement-draft"},
])
def test_atomic_delete_survives_publish_edit_or_recreation_race(monkeypatch, workspace, race_patch):
    db, client = workspace
    execute = FakeQuery.execute

    def race(query):
        if query.table_name == "weekly_recaps" and query.delete_mode:
            db.tables["weekly_recaps"][0].update(race_patch)
        return execute(query)

    monkeypatch.setattr(FakeQuery, "execute", race)
    assert delete(client).status_code == 409
    assert len(db.tables["weekly_recaps"]) == 1
    for key, value in race_patch.items():
        assert db.tables["weekly_recaps"][0][key] == value


def test_required_audit_intent_failure_does_not_delete(monkeypatch, workspace):
    db, client = workspace
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    db.tables["__failed_insert_tables__"] = {"admin_activity_log"}
    response = delete(client)
    assert response.status_code == 500
    assert "nothing was changed" in response.json()["detail"]
    assert len(db.tables["weekly_recaps"]) == 1


def test_completion_audit_failure_reports_uncertain_result(monkeypatch, workspace):
    db, client = workspace
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    execute = FakeQuery.execute

    def fail_completion(query):
        result = execute(query)
        if query.table_name == "weekly_recaps" and query.delete_mode:
            db.tables["__failed_insert_tables__"] = {"admin_activity_log"}
        return result

    monkeypatch.setattr(FakeQuery, "execute", fail_completion)
    response = delete(client)
    assert response.status_code == 500
    assert "Reload before retrying" in response.json()["detail"]
    assert db.tables["weekly_recaps"] == []
    assert db.tables["admin_activity_log"][0]["action_type"] == "delete_weekly_recap_draft_admin_intent"


def test_delete_respects_staging_write_guard(monkeypatch, workspace):
    db, client = workspace
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    assert delete(client).status_code == 403
    assert len(db.tables["weekly_recaps"]) == 1


def test_delete_is_available_in_open_staging(monkeypatch, workspace):
    db, client = workspace
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS", "1")
    response = delete(client)
    assert response.status_code == 200, response.json()
    assert db.tables["weekly_recaps"] == []


def test_delete_requires_authentication(monkeypatch, workspace):
    db, client = workspace

    def unauthenticated(_):
        raise HTTPException(status_code=401, detail="Sign in required")

    monkeypatch.setattr("services.api.admin_weekly_recap_routes.authenticate_bearer", unauthenticated)
    assert delete(client).status_code == 401
    assert len(db.tables["weekly_recaps"]) == 1


def test_existing_save_or_publish_cannot_recreate_deleted_draft(workspace):
    db, _ = workspace
    db.tables["weekly_recaps"] = []
    with pytest.raises(service.StaleWeeklyRecapStateError, match="deleted"):
        service._upsert_recap_row(
            db, club_id="club", week_start="2026-07-06",
            expected_row_version=1, payload={"status": "published"},
        )
    assert db.tables["weekly_recaps"] == []


@pytest.fixture
def recap_calculation(monkeypatch):
    monkeypatch.setattr(service, "_recap_context", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(service, "compute_weekly_recap", lambda *_args, **_kwargs: {"numbers": {"matches": 5}, "spotlight": []})
    monkeypatch.setattr(service, "get_spotlight_candidates", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(service, "_candidates_for_row", lambda *_args, **_kwargs: {})


def mutate_recap(client, action, *, expected_id="recap-1", expected_version=1):
    body = {
        "week_start": "2026-07-06", "week_end": "2026-07-12",
        "expected_recap_id": expected_id, "expected_row_version": expected_version,
        "edits_json": {"looking_ahead": ["New edit"]},
        "confirmation_text": f"{action.upper()} RECAP",
    }
    if action == "generate":
        url, method = "/admin/clubs/club/weekly-recap/generate", "POST"
    elif action == "save":
        url, method = DELETE_URL, "PATCH"
    else:
        url, method = f"{DELETE_URL}/publish", "POST"
        body["action"] = action
    return client.request(method, url, headers={"Authorization": "Bearer local"}, json=body)


@pytest.mark.parametrize("action", ["generate", "save", "publish", "unpublish"])
@pytest.mark.parametrize("expected_id", ["deleted-draft-A", None])
def test_old_tab_cannot_write_recreated_recap_even_at_same_version(workspace, recap_calculation, action, expected_id):
    db, client = workspace
    db.tables["weekly_recaps"][0]["id"] = "recreated-draft-B"
    if action == "unpublish":
        db.tables["weekly_recaps"][0]["status"] = "published"
    before = deepcopy(db.tables["weekly_recaps"])
    response = mutate_recap(client, action, expected_id=expected_id)
    assert response.status_code == 409, response.json()
    assert db.tables["weekly_recaps"] == before


@pytest.mark.parametrize("action", ["generate", "save", "publish", "unpublish"])
def test_current_recap_identity_and_version_keep_existing_mutations_working(workspace, recap_calculation, action):
    db, client = workspace
    if action == "unpublish":
        db.tables["weekly_recaps"][0]["status"] = "published"
    response = mutate_recap(client, action)
    assert response.status_code == 200, response.json()
    assert response.json()["recap"]["id"] == "recap-1"
    assert response.json()["recap"]["status"] == ("published" if action == "publish" else "draft")


@pytest.mark.parametrize("action", ["generate", "save", "publish", "unpublish"])
def test_replacement_between_fetch_and_atomic_update_cannot_be_overwritten(monkeypatch, workspace, recap_calculation, action):
    db, client = workspace
    if action == "unpublish":
        db.tables["weekly_recaps"][0]["status"] = "published"
    replacement = {**deepcopy(db.tables["weekly_recaps"][0]), "id": "replacement-draft-B", "final_json": {"looking_ahead": ["Keep B"]}}
    execute = FakeQuery.execute

    def replace_before_update(query):
        if query.table_name == "weekly_recaps" and query.update_payload is not None:
            db.tables["weekly_recaps"] = [deepcopy(replacement)]
        return execute(query)

    monkeypatch.setattr(FakeQuery, "execute", replace_before_update)
    response = mutate_recap(client, action)
    assert response.status_code == 409, response.json()
    assert db.tables["weekly_recaps"] == [replacement]


@pytest.mark.parametrize("action", ["generate", "save", "publish", "unpublish"])
def test_deleted_recap_is_not_recreated_by_stale_mutation(workspace, recap_calculation, action):
    db, client = workspace
    db.tables["weekly_recaps"] = []
    response = mutate_recap(client, action)
    assert response.status_code == 409, response.json()
    assert db.tables["weekly_recaps"] == []


def test_fresh_generation_still_works_without_expected_id(workspace, recap_calculation):
    db, client = workspace
    db.tables["weekly_recaps"] = []
    response = mutate_recap(client, "generate", expected_id=None, expected_version=None)
    assert response.status_code == 200, response.json()
    assert len(db.tables["weekly_recaps"]) == 1
    assert response.json()["recap"]["row_version"] == 1
