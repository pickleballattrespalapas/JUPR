from copy import deepcopy

import pytest

from tests.test_api_contract_admin_tournament_setup import FakeSupabase, install_env
from fastapi.testclient import TestClient
from services.api.main import app
from jupr_app.services.admin_tournament_setup_service import update_admin_tournament_setup_settings
from jupr_app.services.public_tournament_registration_service import build_public_tournament_registration_page


HEADERS = {"Authorization": "Bearer local"}
BASE = "/admin/clubs/club/tournaments/setup/tournaments/t1"


def generated_configuration():
    days = [{"id": "day1", "tournament_id": "t1", "label": "Day 1", "date": "2026-10-31", "enabled": True, "sort_order": 1}]
    events = [
        {
            "id": f"mixed-{skill}", "tournament_id": "t1", "registration_day_id": "day1",
            "scheduled_day_ids": ["day1"], "event_family_label": "Mixed Doubles",
            "division_name": f"Mixed {skill}", "label": f"Mixed {skill}",
            "event_type": "MIXED_DOUBLES", "gender_restriction": "MIXED",
            "partner_required": True, "public_partner_board": True,
            "skill_label": skill, "skill_mode": "STANDARD", "age_mode": "ALL_AGES",
            "event_format_default": "ROUND_ROBIN_PLUS_PLAYOFF", "scoring_default": "GAME_TO_15",
            "status": "draft", "enabled": True, "sort_order": index + 1,
            "capacity_teams": 16, "price_usd": 20,
        }
        for index, skill in enumerate(["2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
    ]
    return {"days": days, "event_options": events}


def publish_generated(client):
    config = generated_configuration()
    generated = client.put(BASE + "/draft", headers=HEADERS, json={
        **config, "saved_step": "events", "confirmation_text": "SAVE SETUP DRAFT",
    })
    assert generated.status_code == 200, generated.text
    # The user never visits or saves the individual Divisions tab.
    published = client.post(BASE + "/publish", headers=HEADERS, json={
        **config, "confirmation_text": "PUBLISH SETUP",
    })
    assert published.status_code == 200, published.text


def open_registration(client):
    return client.patch(BASE + "/settings", headers=HEADERS, json={
        "registration_status": "open", "confirmation_text": "SAVE SETUP",
    })


def test_generate_publish_open_makes_all_six_divisions_selectable(monkeypatch):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    client = TestClient(app)
    publish_generated(client)
    assert {event["status"] for event in sb.storage["tournament_event_options"]} == {"draft"}
    assert sb.storage["tournament_registration_settings"][0]["registration_status"] == "draft"
    public_before = build_public_tournament_registration_page(sb, club_id="club", tournament_id="t1")
    assert public_before["registration_open"] is False
    assert public_before["events"] == []

    opened = open_registration(client)
    assert opened.status_code == 200, opened.text
    assert opened.json()["opened_division_count"] == 6
    page = build_public_tournament_registration_page(sb, club_id="club", tournament_id="t1")
    assert page["registration_open"] is True
    assert len(page["events"]) == 6
    assert all(event["selectable"] for event in page["events"])
    assert {event["price_usd"] for event in page["events"]} == {20}
    audit = sb.storage["admin_activity_log"][-1]
    assert {row["status"] for row in audit["before_json"]["divisions"]} == {"draft"}
    assert {row["status"] for row in audit["after_json"]["value"]["divisions"]} == {"open"}
    again = open_registration(client)
    assert again.status_code == 200
    assert again.json()["opened_division_count"] == 0


def test_existing_open_tournament_can_repair_drafts_without_publishing_private_edits(monkeypatch):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    client = TestClient(app)
    publish_generated(client)
    settings = sb.storage["tournament_registration_settings"][0]
    settings.update({"registration_status": "open", "venue_address": "Keep this venue", "rules_markdown": "Keep these rules"})
    private_draft = deepcopy(settings["builder_draft_json"])
    private_draft["divisions"].append({"id": "not-published", "status": "draft"})
    settings["builder_draft_json"] = private_draft
    before = deepcopy(private_draft)
    detail = client.get(BASE, headers=HEADERS).json()
    assert detail["registration_readiness"]["available_division_count"] == 0
    assert detail["registration_readiness"]["ready_division_count"] == 6

    assert open_registration(client).status_code == 200
    assert settings["venue_address"] == "Keep this venue"
    assert settings["rules_markdown"] == "Keep these rules"
    assert settings["builder_draft_json"] == before
    assert len(sb.storage["tournament_event_options"]) == 6
    assert client.get(BASE, headers=HEADERS).json()["registration_readiness"]["available_division_count"] == 6


@pytest.mark.parametrize("blocked", ["no_divisions", "disabled_divisions", "disabled_day", "missing_day", "unpublished_tournament"])
def test_open_rejects_unavailable_setup_without_writes(monkeypatch, blocked):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    client = TestClient(app)
    publish_generated(client)
    if blocked == "no_divisions":
        sb.storage["tournament_event_options"] = []
    elif blocked == "disabled_divisions":
        for event in sb.storage["tournament_event_options"]:
            event["enabled"] = False
    elif blocked == "disabled_day":
        sb.storage["tournament_registration_days"][0]["enabled"] = False
    elif blocked == "missing_day":
        sb.storage["tournament_registration_days"] = []
    else:
        sb.storage["tournaments"][0]["status"] = "DRAFT"
    before = deepcopy(sb.storage)
    response = open_registration(client)
    assert response.status_code == 400, response.text
    assert sb.storage == before


def test_open_preserves_disabled_individually_closed_and_other_tournament_divisions(monkeypatch):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    client = TestClient(app)
    publish_generated(client)
    events = sb.storage["tournament_event_options"]
    events[0]["enabled"] = False
    events[1]["status"] = "closed"
    events[2]["status"] = "confirmed"
    events[3]["status"] = "tentative"
    events.append({**events[4], "id": "other-event", "tournament_id": "t2"})
    untouched = deepcopy([events[0], events[1], events[2], events[3], events[6]])
    response = open_registration(client)
    assert response.status_code == 200, response.text
    assert response.json()["opened_division_count"] == 2
    assert [events[0], events[1], events[2], events[3], events[6]] == untouched


def test_close_then_reopen_restores_divisions_saved_while_closed(monkeypatch):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    client = TestClient(app)
    publish_generated(client)
    assert open_registration(client).status_code == 200
    closed = client.patch(BASE + "/settings", headers=HEADERS, json={
        "registration_status": "closed", "confirmation_text": "SAVE SETUP",
    })
    assert closed.status_code == 200
    page = build_public_tournament_registration_page(sb, club_id="club", tournament_id="t1")
    assert page["registration_open"] is False
    assert not any(event["selectable"] for event in page["events"])
    for event in sb.storage["tournament_event_options"]:
        event["status"] = "closed"
    assert open_registration(client).json()["opened_division_count"] == 6


def test_preflight_never_opens_divisions(monkeypatch):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    publish_generated(TestClient(app))
    before = deepcopy(sb.storage)
    result = update_admin_tournament_setup_settings(
        sb, club_id="club", tournament_id="t1", patch={"registration_status": "open"},
        actor_email="admin@example.com", actor_role="club_owner", confirmation_text="SAVE SETUP", dry_run=True,
    )
    assert result["write_count"] == 0
    assert sb.storage == before


def test_division_write_failure_does_not_open_tournament(monkeypatch):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    publish_generated(TestClient(app))
    original_table = sb.table

    def table(name):
        query = original_table(name)
        if name == "tournament_event_options":
            execute = query.execute

            def fail_update():
                if query.update_payload is not None:
                    raise RuntimeError("simulated division write failure")
                return execute()

            query.execute = fail_update
        return query

    monkeypatch.setattr(sb, "table", table)
    response = open_registration(TestClient(app, raise_server_exceptions=False))
    assert response.status_code >= 400
    assert sb.storage["tournament_registration_settings"][0]["registration_status"] == "draft"
    assert all(event["status"] == "draft" for event in sb.storage["tournament_event_options"])


@pytest.mark.parametrize("stale", [False, True])
def test_guarded_open_requires_current_published_state(monkeypatch, stale):
    sb = FakeSupabase()
    install_env(monkeypatch, sb)
    client = TestClient(app)
    publish_generated(client)
    fingerprint = client.get(BASE, headers=HEADERS).json()["state_fingerprint"]
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "local-test-key")
    response = client.patch(BASE + "/settings", headers=HEADERS, json={
        "registration_status": "open", "confirmation_text": "SAVE SETUP",
        "expected_state_fingerprint": "stale" if stale else fingerprint,
    })
    if stale:
        assert response.status_code == 409, response.text
        assert sb.storage["tournament_registration_settings"][0]["registration_status"] == "draft"
        assert all(event["status"] == "draft" for event in sb.storage["tournament_event_options"])
    else:
        assert response.status_code == 200, response.text
        assert response.json()["opened_division_count"] == 6
        assert sb.storage["tournament_admin_operations"][0]["status"] == "completed"
