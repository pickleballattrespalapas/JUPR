from __future__ import annotations

from copy import deepcopy

import pytest

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.domain.tournament_registration_repo import (
    PUBLIC_REGISTRATION_EDIT_RPC,
    TournamentRegistrationEditConflictError,
    TournamentRegistrationImportedDrawError,
)
from jupr_app.services import public_tournament_registration_edit_service as edit_service
from jupr_app.services.public_tournament_registration_edit_service import (
    PublicRegistrationEditUnavailableError,
    build_public_tournament_registration_edit_page,
    request_public_tournament_registration_edit_link,
    submit_public_tournament_registration_edit,
)
from jupr_app.services.public_tournament_registration_service import submit_public_tournament_registration
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage


def _registered_supabase(monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "test-registration-edit-secret-32bytes")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://next.example.com")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "phone": "555-0100",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=result["registration_id"],
        email="alex@example.com",
        secret="test-registration-edit-secret-32bytes",
    )
    supabase.rpc_calls.clear()
    return supabase, storage, result["registration_id"], token


def _edit_versions(storage):
    registration = storage["tournament_registrations"][0]
    return {
        "expected_updated_at": registration["updated_at"],
        "expected_selection_versions": [
            {"id": row["id"], "updated_at": row["updated_at"]}
            for row in storage["tournament_registration_selections"]
            if row["registration_id"] == registration["id"]
        ],
    }


def test_registration_edit_page_verifies_token_and_hydrates_registration(monkeypatch) -> None:
    supabase, _storage, registration_id, token = _registered_supabase(monkeypatch)

    payload = build_public_tournament_registration_edit_page(
        supabase,
        club_id="club-1",
        edit_token=token,
        registration_slug="tres-open",
    )

    assert payload["edit_token_valid"] is True
    assert payload["registration"]["id"] == registration_id
    assert payload["registration"]["email"] == "alex@example.com"
    assert payload["registration"]["phone"] == "555-0100"
    assert payload["selections"][0]["event_option_id"] == "event1"
    assert "phone" in payload["registration"]
    assert "admin_notes" not in payload["tournament"]
    assert "internal_seed_notes" not in payload["events"][0]


def test_registration_edit_submit_updates_existing_registration_and_locks_email(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)
    storage["tournament_registrations"][0]["age_bracket"] = "50+"
    selection_id = storage["tournament_registration_selections"][0]["id"]

    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Alexis",
            "last_name": "Rivera",
            "display_name": "Alexis R",
            "email": "evil@example.com",
            "phone": "555-9999",
                "doubles_skill": 4.25,
                "wants_partner_board_contact": True,
                "terms_accepted": True,
            "selections": [
                {
                    "event_option_id": "event1",
                    "partner_mode": "NEEDS_PARTNER",
                    "show_on_partner_board": True,
                    "partner_note": "Looking for a steady partner",
                }
            ],
        },
    )

    assert result["ok"] is True
    assert result["registration_id"] == registration_id
    assert result["confirmation_delivery"] == {"status": "dry_run", "delivered": False}
    assert result["confirmation_token"]
    assert result["email_delivery"]["status"] == "dry_run"
    assert "provider_message_id" not in str(result)
    assert "to_email" not in str(result)
    assert supabase.rpc_calls[0][0] == PUBLIC_REGISTRATION_EDIT_RPC
    registrations = storage["tournament_registrations"]
    assert len(registrations) == 1
    assert registrations[0]["display_name"] == "Alexis R"
    assert registrations[0]["email"] == "alex@example.com"
    assert registrations[0]["phone"] == "555-9999"
    assert registrations[0]["age_bracket"] == "50+"
    selections = storage["tournament_registration_selections"]
    assert len(selections) == 1
    assert selections[0]["id"] == selection_id
    assert selections[0]["partner_mode"] == "NEEDS_PARTNER"
    assert selections[0]["show_on_partner_board"] is True


def test_linked_registration_edit_preserves_self_reported_singles_when_official_is_missing(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)
    player = storage["players"][0]
    assert player.get("singles_rating") is None

    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "doubles_skill": 4.0,
            "singles_skill": 3.5,
            "terms_accepted": True,
            "selections": [
                {"event_option_id": "event1", "partner_mode": "NONE"}
            ],
        },
    )

    assert result["registration_id"] == registration_id
    assert storage["tournament_registrations"][0]["doubles_skill"] == 4.0
    assert storage["tournament_registrations"][0]["singles_skill"] == 3.5
    assert player.get("singles_rating") is None


def test_linked_registration_edit_uses_official_singles_rating_added_before_save(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)
    edit_page = build_public_tournament_registration_edit_page(
        supabase,
        club_id="club-1",
        edit_token=token,
        registration_slug="tres-open",
    )
    assert edit_page["registration"]["singles_skill"] is None

    storage["players"][0]["singles_rating"] = 1400
    storage["players"][0]["singles_matches_played"] = 1
    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "doubles_skill": 4.0,
            "singles_skill": 3.0,
            "terms_accepted": True,
            "selections": [
                {"event_option_id": "event1", "partner_mode": "NONE"}
            ],
        },
    )

    assert result["registration_id"] == registration_id
    assert storage["tournament_registrations"][0]["singles_skill"] == 3.5
    assert storage["players"][0]["singles_rating"] == 1400


def test_registration_edit_rejects_stale_versions_without_mutation(monkeypatch) -> None:
    supabase, storage, _registration_id, token = _registered_supabase(monkeypatch)
    before = deepcopy(storage)
    versions = _edit_versions(storage)
    versions["expected_updated_at"] = "2020-01-01T00:00:00Z"

    with pytest.raises(TournamentRegistrationEditConflictError, match="Refresh"):
        submit_public_tournament_registration_edit(
            supabase,
            club_id="club-1",
            edit_token=token,
            payload={
                **versions,
                "tournament_id": "t1",
                "first_name": "Changed",
                "last_name": "Rivera",
                "email": "alex@example.com",
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )

    assert storage == before
    assert supabase.rpc_calls == []


def test_registration_edit_switches_division_in_place_within_event_family(monkeypatch) -> None:
    supabase, storage, _registration_id, token = _registered_supabase(monkeypatch)
    selection_id = storage["tournament_registration_selections"][0]["id"]
    replacement_event = {
        **deepcopy(storage["tournament_event_options"][0]),
        "id": "event2",
        "sort_order": 2,
        "label": "Intermediate Doubles",
        "division_name": "Intermediate",
    }
    storage["tournament_event_options"].append(replacement_event)

    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "terms_accepted": True,
            "selections": [
                {
                    "id": selection_id,
                    "event_option_id": "event2",
                    "registration_day_id": "day1",
                    "partner_mode": "NONE",
                }
            ],
        },
    )

    assert result["ok"] is True
    saved_selections = storage["tournament_registration_selections"]
    assert len(saved_selections) == 1
    assert saved_selections[0]["id"] == selection_id
    assert saved_selections[0]["event_option_id"] == "event2"


def test_registration_edit_save_survives_confirmation_delivery_failure(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)

    monkeypatch.setattr(
        edit_service,
        "build_registration_confirmation_delivery",
        lambda *_args, **_kwargs: {
            "confirmation_available": True,
            "confirmation_token": "confirmation-token",
            "email_delivery": {"status": "failed", "message": "Registration was saved, but email failed."},
        },
    )

    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "first_name": "Saved",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert result["registration_id"] == registration_id
    assert result["confirmation_delivery"] == {"status": "failed", "delivered": False}
    assert storage["tournament_registrations"][0]["first_name"] == "Saved"


def test_registration_edit_replay_reports_completed_without_resending(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)
    delivery_calls: list[bool] = []
    monkeypatch.setattr(
        edit_service,
        "save_registration",
        lambda *_args, **_kwargs: {
            "registration_id": registration_id,
            "submitted_at": "2026-07-01T00:00:00Z",
            "updated_at": "2026-07-01T00:00:00Z",
            "selection_count": 1,
            "idempotent_replay": True,
        },
    )

    def replay_delivery(*_args, **kwargs):
        delivery_calls.append(bool(kwargs.get("send_email")))
        return {
            "confirmation_available": True,
            "confirmation_token": "replay-confirmation-token",
            "email_delivery": {
                "status": "already_completed",
                "message": "Registration was already saved.",
            },
        }

    monkeypatch.setattr(
        edit_service,
        "build_registration_confirmation_delivery",
        replay_delivery,
    )

    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "terms_accepted": True,
            "selections": [
                {"event_option_id": "event1", "partner_mode": "NONE"}
            ],
        },
    )

    assert result["confirmation_delivery"] == {
        "status": "already_completed",
        "delivered": False,
    }
    assert result["confirmation_token"] == "replay-confirmation-token"
    assert delivery_calls == [False]


def test_registration_edit_rpc_failure_leaves_registration_unchanged(monkeypatch) -> None:
    supabase, storage, _registration_id, token = _registered_supabase(monkeypatch)
    storage["_fail_public_registration_edit_rpc"] = True
    before = deepcopy(storage)

    with pytest.raises(RuntimeError, match="without changing"):
        submit_public_tournament_registration_edit(
            supabase,
            club_id="club-1",
            edit_token=token,
            payload={
                **_edit_versions(storage),
                "tournament_id": "t1",
                "first_name": "Must not persist",
                "last_name": "Rivera",
                "email": "alex@example.com",
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )

    assert storage == before


@pytest.mark.parametrize(
    "team_source", ["REGISTRATION", "REGISTRATION_COMBINED_RATING"]
)
def test_imported_draw_blocks_get_and_post_without_mutation(
    monkeypatch, team_source: str
) -> None:
    supabase, storage, _registration_id, token = _registered_supabase(monkeypatch)
    second_event = {
        **deepcopy(storage["tournament_event_options"][0]),
        "id": "event2",
        "sort_order": 2,
        "label": "Open Singles",
        "event_family_label": "Singles",
        "division_name": "Open Singles",
        "event_type": "SINGLES",
    }
    storage["tournament_event_options"].append(second_event)
    storage["tournament_registration_selections"].append(
        {
            **deepcopy(storage["tournament_registration_selections"][0]),
            "id": "sel-second-event",
            "event_option_id": "event2",
            "sort_order": 2,
        }
    )
    storage["tournament_event_draws"].append(
        {
            "id": "draw-1",
            "tournament_id": "t1",
            "registration_day_id": "day1",
            "event_option_id": "event2",
        }
    )
    storage["tournament_teams"].append(
        {
            "id": "team-imported",
            "tournament_id": "t1",
            "draw_id": "draw-1",
            "registration_day_id": None,
            "event_option_id": None,
            "source": team_source,
            "source_selection_id": "sel-second-event",
        }
    )
    before = deepcopy(storage)
    sends = {"count": 0}
    monkeypatch.setattr(
        edit_service,
        "send_tournament_registration_edit_email",
        lambda **_kwargs: sends.__setitem__("count", sends["count"] + 1),
    )

    link_result = request_public_tournament_registration_edit_link(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        registration_slug="tres-open",
        email="alex@example.com",
    )
    assert link_result["accepted"] is True
    assert sends["count"] == 0

    with pytest.raises(TournamentRegistrationImportedDrawError, match="imported into a draw"):
        build_public_tournament_registration_edit_page(
            supabase,
            club_id="club-1",
            edit_token=token,
            registration_slug="tres-open",
        )

    with pytest.raises(TournamentRegistrationImportedDrawError, match="imported into a draw"):
        submit_public_tournament_registration_edit(
            supabase,
            club_id="club-1",
            edit_token=token,
            payload={
                **_edit_versions(storage),
                "tournament_id": "t1",
                "first_name": "Changed",
                "last_name": "Rivera",
                "email": "alex@example.com",
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )

    assert storage == before
    assert supabase.rpc_calls == []


def test_registration_excluded_from_imported_draw_remains_editable(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)
    storage["tournament_event_draws"].append(
        {
            "id": "draw-1",
            "tournament_id": "t1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
        }
    )
    storage["tournament_teams"].append(
        {
            "id": "other-imported-team",
            "tournament_id": "t1",
            "draw_id": "draw-1",
            "registration_day_id": "stale-day",
            "event_option_id": "stale-event",
            "source": "REGISTRATION",
            "player1_id": 9001,
            "player2_id": 9002,
        }
    )

    page = build_public_tournament_registration_edit_page(
        supabase,
        club_id="club-1",
        edit_token=token,
        registration_slug="tres-open",
    )
    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Alexis",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "terms_accepted": True,
            "selections": [
                {
                    "event_option_id": "event1",
                    "partner_mode": "NEEDS_PARTNER",
                }
            ],
        },
    )

    assert page["registration"]["id"] == registration_id
    assert result["ok"] is True
    assert storage["tournament_registrations"][0]["first_name"] == "Alexis"


def test_registration_edit_rejects_wrong_club(monkeypatch) -> None:
    supabase, _storage, _registration_id, token = _registered_supabase(monkeypatch)

    try:
        build_public_tournament_registration_edit_page(
            supabase,
            club_id="other-club",
            edit_token=token,
            registration_slug="tres-open",
        )
    except ValueError as exc:
        assert "different club" in str(exc)
    else:
        raise AssertionError("Expected wrong-club edit link rejection")


def test_registration_edit_link_request_sends_email_without_exposing_match(monkeypatch) -> None:
    supabase, _storage, _registration_id, _token = _registered_supabase(monkeypatch)
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://next.example.com")
    captured: dict[str, str] = {}

    def fake_send(**kwargs):
        captured.update({key: str(value) for key, value in kwargs.items()})
        return {"status": "dry_run", "provider_message_id": "dry_run", "to_email": kwargs["registered_email"]}

    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", fake_send)

    payload = request_public_tournament_registration_edit_link(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        registration_slug="tres-open",
        email="alex@example.com",
    )

    assert payload == {
        "ok": True,
        "mode": "registration_edit_link_request",
        "accepted": True,
        "message": "If that email matches a registration, we’ll send the edit link there.",
    }
    assert captured["registered_email"] == "alex@example.com"
    assert captured["tournament_name"] == "Tres Palapas Open"
    assert captured["edit_url"].startswith("https://next.example.com/clubs/tres-palapas/tournament-registration/edit?")
    assert "edit_token=" in captured["edit_url"]
    assert "edit_token" not in str(payload)


def test_registration_edit_link_exact_retry_does_not_resend(monkeypatch) -> None:
    supabase, _storage, _registration_id, _token = _registered_supabase(monkeypatch)
    calls = {"send": 0}

    def fake_send(**_kwargs):
        calls["send"] += 1
        return {"status": "dry_run"}

    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", fake_send)
    kwargs = {
        "club_id": "club-1",
        "club_slug": "tres-palapas",
        "registration_slug": "tres-open",
        "email": "alex@example.com",
        "idempotency_key": "edit-link-replay-1",
    }

    first = request_public_tournament_registration_edit_link(supabase, **kwargs)
    replay = request_public_tournament_registration_edit_link(supabase, **kwargs)

    assert first == replay
    assert first["accepted"] is True
    assert calls["send"] == 1


def test_registration_edit_link_request_missing_email_does_not_enumerate(monkeypatch) -> None:
    supabase, _storage, _registration_id, _token = _registered_supabase(monkeypatch)
    calls = {"send": 0}

    def fake_send(**_kwargs):
        calls["send"] += 1
        return {}

    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", fake_send)

    payload = request_public_tournament_registration_edit_link(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        registration_slug="tres-open",
        email="missing@example.com",
    )

    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert calls["send"] == 0


def test_registration_edit_link_request_honeypot_is_silent(monkeypatch) -> None:
    supabase, _storage, _registration_id, _token = _registered_supabase(monkeypatch)
    calls = {"send": 0}
    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", lambda **_kwargs: calls.__setitem__("send", calls["send"] + 1))

    payload = request_public_tournament_registration_edit_link(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        registration_slug="tres-open",
        email="alex@example.com",
        website="bot field",
    )

    assert payload["ok"] is True
    assert calls["send"] == 0


def test_public_edit_api_requires_explicit_stable_secret(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_REGISTRATION_EDIT_SECRET", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "rotatable-service-role-key")

    with pytest.raises(PublicRegistrationEditUnavailableError, match="temporarily unavailable"):
        request_public_tournament_registration_edit_link(
            FakeSupabase(fake_storage()),
            club_id="club-1",
            club_slug="tres-palapas",
            registration_slug="tres-open",
            email="alex@example.com",
        )


def test_registration_edit_preserves_existing_closed_division_but_cannot_add_it(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)
    storage["tournament_event_options"][0]["status"] = "draft"
    storage["tournament_event_options"][0]["enabled"] = False

    edit_page = build_public_tournament_registration_edit_page(
        supabase,
        club_id="club-1",
        edit_token=token,
        registration_slug="tres-open",
    )
    preserved_event = next(event for event in edit_page["events"] if event["id"] == "event1")
    assert preserved_event["selectable"] is False

    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            **_edit_versions(storage),
            "tournament_id": "t1",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    assert result["registration_id"] == registration_id
    assert storage["tournament_registration_selections"][0]["event_option_id"] == "event1"

    with pytest.raises(ValueError, match="no longer open"):
        submit_public_tournament_registration(
            supabase,
            club_id="club-1",
            payload={
                "registration_slug": "tres-open",
                "first_name": "New",
                "email": "new@example.com",
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )


def test_registration_edit_locks_player_link_and_revalidates_eligibility(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "test-registration-edit-secret-32bytes")
    storage = fake_storage()
    storage["players"] = [
        {"id": 10, "club_id": "club-1", "name": "Alex Rivera", "email": "alex@example.com", "rating": 1200, "active": True, "inactive_at": None},
        {"id": 11, "club_id": "club-1", "name": "Other Player", "email": "other@example.com", "rating": 1200, "active": True, "inactive_at": None},
    ]
    supabase = FakeSupabase(storage)
    created = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "player_id": 10,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    # Initial public intake cannot establish a trusted player link. Simulate the
    # staff-reviewed link that a later edit token is allowed to preserve.
    storage["tournament_registrations"][0]["player_id"] = 10
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=created["registration_id"],
        email="alex@example.com",
        secret="test-registration-edit-secret-32bytes",
    )

    edit_page = build_public_tournament_registration_edit_page(
        supabase,
        club_id="club-1",
        edit_token=token,
        registration_slug="tres-open",
    )
    assert [player["id"] for player in edit_page["players"]] == ["10"]

    with pytest.raises(ValueError, match="cannot be changed"):
        submit_public_tournament_registration_edit(
            supabase,
            club_id="club-1",
            edit_token=token,
            payload={
                **_edit_versions(storage),
                "tournament_id": "t1",
                "first_name": "Alex",
                "email": "alex@example.com",
                "player_id": 11,
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )


def test_registration_edit_rejects_slug_for_another_open_tournament(monkeypatch) -> None:
    supabase, storage, _registration_id, token = _registered_supabase(monkeypatch)
    storage["tournament_registration_settings"][0]["registration_status"] = "closed"
    storage["tournaments"].append(
        {"id": "t2", "club_id": "club-1", "name": "Other Open", "status": "ACTIVE", "created_at": "2026-01-02T00:00:00Z"}
    )
    storage["tournament_registration_settings"].append(
        {
            "id": "rs2",
            "tournament_id": "t2",
            "registration_slug": "other-open",
            "registration_status": "open",
            "builder_draft_json": {"published_at": "2026-08-01T00:00:00Z"},
        }
    )

    with pytest.raises(ValueError, match="different tournament"):
        build_public_tournament_registration_edit_page(
            supabase,
            club_id="club-1",
            edit_token=token,
            registration_slug="other-open",
        )

    with pytest.raises(ValueError, match="different tournament"):
        submit_public_tournament_registration_edit(
            supabase,
            club_id="club-1",
            edit_token=token,
            payload={
                **_edit_versions(storage),
                "tournament_id": "t1",
                "registration_slug": "other-open",
                "first_name": "Alex",
                "last_name": "Rivera",
                "email": "alex@example.com",
                "doubles_skill": 4.0,
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )
