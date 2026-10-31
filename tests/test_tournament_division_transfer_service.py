from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_registration_repo import (
    PUBLIC_REGISTRATION_EDIT_RPC, StaleTournamentRegistrationSelectionError,
    TournamentRegistrationEditConflictError, _raise_division_transfer_failure,
)
from jupr_app.services.admin_tournament_service import update_admin_tournament_selection
from jupr_app.services.public_tournament_registration_edit_service import (
    build_public_tournament_registration_edit_page, submit_public_tournament_registration_edit,
)
from jupr_app.services.tournament_division_transfer_service import PARTNER_FIELDS
from tests.test_public_tournament_registration_edit_service import (
    _confirmed_reciprocal_registration, _edit_versions, _registered_supabase,
)


def _team(monkeypatch, *, other_day=False):
    supabase, storage, registration_id, token = _confirmed_reciprocal_registration(monkeypatch)
    storage["tournament_registration_team_links"][0]["updated_at"] = "2026-10-16T10:00:00Z"
    source = storage["tournament_event_options"][0]
    source["price_usd"] = 25
    target = {**source, "id": "higher-division", "division_name": "4.0", "label": "Women's Doubles 4.0", "skill_label": "4.0"}
    if other_day:
        original = next(row for row in storage["tournament_registration_days"] if row["id"] == source["registration_day_id"])
        storage["tournament_registration_days"].append({**original, "id": "another-day"})
        target["registration_day_id"] = "another-day"
    storage["tournament_event_options"].append(target)
    page = build_public_tournament_registration_edit_page(supabase, club_id="club-1", edit_token=token)
    payload = {**page["registration"], **_edit_versions(storage), "tournament_id": "t1", "terms_accepted": True,
               "selections": deepcopy(page["selections"])}
    payload["selections"][0]["event_option_id"] = target["id"]
    payload["selections"][0]["registration_day_id"] = target["registration_day_id"]
    return supabase, storage, token, payload


def test_public_transfer_uses_atomic_rpc_with_original_identity_and_partner_fields(monkeypatch):
    supabase, storage, token, payload = _team(monkeypatch, other_day=True)
    before = deepcopy(storage)
    rpc_calls = []

    def rpc(name, params):
        rpc_calls.append((name, params))
        # SQL integration tests cover atomic mutation. This contract test checks
        # the token-bound player's validated request reaches that single boundary.
        return SimpleNamespace(execute=lambda: SimpleNamespace(data={"ok": True, "registration_id": payload["id"],
            "updated_at": "2026-10-16T12:00:00Z", "selection_count": 1}))

    monkeypatch.setattr(supabase, "rpc", rpc)
    result = submit_public_tournament_registration_edit(supabase, club_id="club-1", edit_token=token, payload=payload)
    assert result["ok"]
    assert len(rpc_calls) == 1 and rpc_calls[0][0] == PUBLIC_REGISTRATION_EDIT_RPC
    row = rpc_calls[0][1]["p_selections"][0]
    assert row["id"] == before["tournament_registration_selections"][0]["id"]
    assert row["event_option_id"] == "higher-division"
    assert row["registration_day_id"] == "another-day"
    assert {key: row.get(key) for key in PARTNER_FIELDS} == {
        key: before["tournament_registration_selections"][0].get(key) for key in PARTNER_FIELDS}
    assert row["division_transfer_context"] == payload["selections"][0]["division_transfer_context"]
    assert "payment_status" not in rpc_calls[0][1]["p_registration_patch"]
    for table in ("tournament_registration_selections", "tournament_registration_team_links", "tournament_registrations"):
        assert storage[table] == before[table]


def test_public_single_entry_can_change_days_without_recreating_selection(monkeypatch):
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)
    source = storage["tournament_event_options"][0]
    day = next(row for row in storage["tournament_registration_days"] if row["id"] == source["registration_day_id"])
    storage["tournament_registration_days"].append({**day, "id": "later-day"})
    storage["tournament_event_options"].append({**source, "id": "later-division", "registration_day_id": "later-day"})
    page = build_public_tournament_registration_edit_page(supabase, club_id="club-1", edit_token=token)
    selection_id = page["selections"][0]["id"]
    page["selections"][0].update(event_option_id="later-division", registration_day_id="later-day")
    result = submit_public_tournament_registration_edit(
        supabase, club_id="club-1", edit_token=token,
        payload={**page["registration"], **_edit_versions(storage), "tournament_id": "t1", "terms_accepted": True,
                 "selections": page["selections"]},
    )
    assert result["ok"]
    entries = [row for row in storage["tournament_registration_selections"] if row["registration_id"] == registration_id]
    assert len(entries) == 1 and entries[0]["id"] == selection_id
    assert entries[0]["event_option_id"] == "later-division"
    assert entries[0]["registration_day_id"] == "later-day"


@pytest.mark.parametrize("change,match", [
    ("stale", "team changed"), ("other_family", "same event"), ("fee", "different entry fees"),
    ("partner_entry", "another entry"), ("primary_entry", "another entry"),
    ("pending", "pending partner request"), ("partner_cancelled", "cancelled"),
    ("partner_rating", "above the 4 division cap"), ("replace_partner", "keeps your confirmed partner"),
])
def test_public_transfer_rejections_are_read_only(monkeypatch, change, match):
    supabase, storage, token, payload = _team(monkeypatch, other_day=True)
    source = storage["tournament_registration_selections"][0]
    target = storage["tournament_event_options"][-1]
    if change == "stale":
        storage["tournament_registrations"][1]["updated_at"] = "2026-10-16T12:00:00Z"
    elif change == "other_family":
        target["event_family_label"] = "Mixed Doubles"
    elif change == "fee":
        target["price_usd"] = 50
    elif change in {"partner_entry", "primary_entry"}:
        storage["tournament_registration_selections"].append({**source, "id": "duplicate", "event_option_id": target["id"],
            "registration_id": "reg-partner" if change == "partner_entry" else source["registration_id"]})
        if change == "primary_entry":
            payload.update(_edit_versions(storage))
    elif change == "pending":
        storage["tournament_registration_partner_requests"].append({"id": "request", "tournament_id": "t1",
            "requester_selection_id": "sel-partner", "status": "PENDING"})
    elif change == "partner_cancelled":
        storage["tournament_registrations"][1]["status"] = "cancelled"
    elif change == "partner_rating":
        storage["tournament_registrations"][1]["doubles_skill"] = 5.0
    else:
        payload["selections"][0]["partner_email"] = "someone-else@example.com"
    before = deepcopy(storage)
    with pytest.raises(ValueError, match=match):
        submit_public_tournament_registration_edit(supabase, club_id="club-1", edit_token=token, payload=payload)
    assert not supabase.rpc_calls
    for table in before:
        assert storage[table] == before[table]


def test_admin_confirmed_transfer_preflight_preserves_team_and_passes_versions(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase, storage, _token, payload = _team(monkeypatch, other_day=True)
    selection = storage["tournament_registration_selections"][0]
    before = deepcopy(storage)
    result = update_admin_tournament_selection(
        supabase, club_id="club-1", tournament_id="t1", selection_id=selection["id"],
        expected_updated_at=selection["updated_at"], actor_email="admin@example.com", actor_role="club_owner",
        confirmation_text="SAVE SELECTION", dry_run=True,
        patch={"event_option_id": "higher-division", "division_transfer_context": payload["selections"][0]["division_transfer_context"]},
    )
    assert result["patch"] == {"event_option_id": "higher-division", "registration_day_id": "another-day",
        "division_transfer_context": payload["selections"][0]["division_transfer_context"]}
    for table in before:
        assert storage[table] == before[table]
    assert not supabase.rpc_calls


def test_admin_confirmed_transfer_requires_partner_snapshot(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase, storage, _token, _payload = _team(monkeypatch)
    selection = storage["tournament_registration_selections"][0]
    with pytest.raises(StaleTournamentRegistrationSelectionError, match="team changed"):
        update_admin_tournament_selection(
            supabase, club_id="club-1", tournament_id="t1", selection_id=selection["id"],
            expected_updated_at=selection["updated_at"], actor_email="admin@example.com", actor_role="club_owner",
            confirmation_text="SAVE SELECTION", dry_run=True, patch={"event_option_id": "higher-division"},
        )
    assert not supabase.rpc_calls


@pytest.mark.parametrize("note", [None, "", "Moving up to 3.0", "2.0 level"])
def test_admin_confirmed_transfer_allows_public_note_edits(monkeypatch, note):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase, storage, _token, payload = _team(monkeypatch, other_day=True)
    selection = storage["tournament_registration_selections"][0]
    selection["partner_note"] = "2.0 level"
    before = deepcopy(storage)
    result = update_admin_tournament_selection(
        supabase, club_id="club-1", tournament_id="t1", selection_id=selection["id"],
        expected_updated_at=selection["updated_at"], actor_email="admin@example.com", actor_role="club_owner",
        confirmation_text="SAVE SELECTION", dry_run=True,
        patch={"event_option_id": "higher-division", "partner_note": note,
               "division_transfer_context": payload["selections"][0]["division_transfer_context"]},
    )
    assert result["patch"] == {
        "event_option_id": "higher-division", "registration_day_id": "another-day",
        "partner_note": note or "",
        "division_transfer_context": payload["selections"][0]["division_transfer_context"],
    }
    assert not supabase.rpc_calls
    for table in before:
        assert storage[table] == before[table]


@pytest.mark.parametrize("partner_patch", [
    {"partner_mode": "NONE"}, {"partner_name": "Someone else"},
    {"partner_email": "someone-else@example.com"}, {"partner_phone": "555-1234"},
    {"partner_dupr_id": "different-id"}, {"partner_skill": 6.5},
    {"partner_age": 25}, {"partner_gender": "Men"},
])
def test_admin_confirmed_transfer_still_rejects_partner_changes(monkeypatch, partner_patch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase, storage, _token, payload = _team(monkeypatch)
    selection = storage["tournament_registration_selections"][0]
    before = deepcopy(storage)
    with pytest.raises(ValueError, match="keeps the confirmed partner"):
        update_admin_tournament_selection(
            supabase, club_id="club-1", tournament_id="t1", selection_id=selection["id"],
            expected_updated_at=selection["updated_at"], actor_email="admin@example.com", actor_role="club_owner",
            confirmation_text="SAVE SELECTION", dry_run=True,
            patch={"event_option_id": "higher-division", "partner_note": "",
                   "division_transfer_context": payload["selections"][0]["division_transfer_context"],
                   **partner_patch},
        )
    assert not supabase.rpc_calls
    for table in before:
        assert storage[table] == before[table]


@pytest.mark.parametrize("reason,error,match", [
    ("DIVISION_FULL", ValueError, "full"),
    ("DIVISION_COMMERCE_REVIEW_REQUIRED", ValueError, "event-specific pricing"),
    ("STALE_TEAM_VERSION", TournamentRegistrationEditConflictError, "team changed"),
])
def test_atomic_transfer_failures_remain_actionable_in_public_edit(reason, error, match):
    with pytest.raises(error, match=match):
        _raise_division_transfer_failure(
            {"ok": False, "code": "SELECTION_WRITE_CONFLICT", "reason": reason}, public_edit=True,
        )
