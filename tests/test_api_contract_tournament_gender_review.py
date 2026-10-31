from copy import deepcopy

import pytest

from jupr_app.domain.tournament_gender_review import REVIEW_TABLE
from jupr_app.domain.tournament_registration_repo import StaleTournamentRegistrationSelectionError
from jupr_app.services.admin_tournament_service import (
    get_admin_tournament_detail, review_admin_tournament_gender_eligibility,
)
from jupr_app.services.admin_tournament_registration_import_service import import_admin_tournament_registrations_to_draw
from jupr_app.services import admin_tournament_registration_import_service as import_service
from jupr_app.services.public_tournament_registration_service import submit_public_tournament_registration
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage
from tests.test_api_contract_admin_tournament_registration_import import tournament_registration_import_tables, _client


@pytest.mark.parametrize("restriction,player_gender,partner_gender", [
    ("MIXED", "Women", "Non-binary"), ("MIXED", "Non-binary", "Women"),
    ("MIXED", "Women", "Women"), ("MIXED", "Men", "Men"),
    ("MEN", "Women", "Men"), ("WOMEN", "Women", "Men"),
    ("MIXED", "Other", "Men"),
])
def test_public_submission_preserves_gender_and_keeps_review_private(monkeypatch, restriction, player_gender, partner_gender):
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    storage = fake_storage()
    storage["tournament_event_options"][0].update(
        event_type="MIXED_DOUBLES", partner_required=True, gender_restriction=restriction,
        skill_mode="OPEN", skill_label="Open",
    )
    db = FakeSupabase(storage)
    result = submit_public_tournament_registration(db, club_id="club-1", payload={
        "registration_slug": "tres-open", "first_name": "Fixture", "last_name": "Player",
        "email": "player@example.invalid", "gender": player_gender, "age": 36,
        "doubles_skill": 3, "terms_accepted": True,
        "selections": [{"event_option_id": "event1", "partner_mode": "HAS_PARTNER",
            "partner_name": "Fixture Partner", "partner_email": "partner@example.invalid",
            "partner_gender": partner_gender, "partner_age": 35, "partner_skill": 3,
            "gender_review": {"status": "APPROVED"}}],
    })
    registration = next(row for row in storage["tournament_registrations"] if row["id"] == result["registration_id"])
    assert registration["gender"] == player_gender
    selection = next(row for row in storage["tournament_registration_selections"] if row["registration_id"] == result["registration_id"])
    assert selection["partner_gender"] == partner_gender
    assert "gender_review" not in result
    detail = get_admin_tournament_detail(db, club_id="club-1", tournament_id=registration["tournament_id"])
    row = next(row for row in detail["selections"] if row["id"] == selection["id"])
    assert row["gender_review"]["status"] == "PENDING"
    assert "snapshot" not in row["gender_review"]


@pytest.fixture
def reviewed_pair(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = tournament_registration_import_tables()
    tables["tournament_event_options"][0].update(event_type="MIXED_DOUBLES", gender_restriction="MIXED")
    tables["tournament_registrations"][0]["gender"] = "Women"
    tables["tournament_registrations"][1]["gender"] = "Non-binary"
    return tables, FakeSupabase(tables)


def _detail(db):
    return get_admin_tournament_detail(db, club_id="club", tournament_id="tour_1")


def _review(db, selection_id="sel_1"):
    return next(row["gender_review"] for row in _detail(db)["selections"] if row["id"] == selection_id)


def _decide(db, decision="APPROVED", **overrides):
    return review_admin_tournament_gender_eligibility(db, **{
        "club_id": "club", "tournament_id": "tour_1", "selection_id": "sel_1", "decision": decision,
        "expected_review_version": _review(db)["review_version"], "actor_email": "admin@example.invalid",
        "actor_role": "administrator", **overrides,
    })


def _import(db, **overrides):
    return import_admin_tournament_registrations_to_draw(db, club_id="club", tournament_id="tour_1",
        draw_id="draw_1", actor_email="admin@example.invalid", actor_role="administrator",
        confirmation_text="IMPORT REGISTRATIONS", **{"dry_run": True, **overrides})


@pytest.mark.parametrize("import_mode", ["REPLACE", "APPEND"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_confirmed_pair_imports_without_second_gender_approval(reviewed_pair, import_mode, dry_run):
    tables, db = reviewed_pair
    before = deepcopy(tables)
    assert _review(db)["status"] == "PENDING"

    result = _import(db, import_mode=import_mode, dry_run=dry_run)

    assert result["ok"] is True
    assert result["updated_count"] == 1
    assert [(team["player1_id"], team["player2_id"]) for team in result["teams"]] == [(1, 2)]
    assert tables.get(REVIEW_TABLE, []) == before.get(REVIEW_TABLE, [])
    assert tables["tournament_registrations"] == before["tournament_registrations"]
    if dry_run:
        assert tables["tournament_teams"] == before["tournament_teams"]
    else:
        assert any(team.get("player1_id") == 1 and team.get("player2_id") == 2 for team in tables["tournament_teams"])
        assert len(tables["tournament_teams"]) == (2 if import_mode == "APPEND" else 1)


def test_explicit_approval_applies_to_pair_and_remains_importable(reviewed_pair):
    tables, db = reviewed_pair
    assert _review(db)["fingerprint"] == _review(db, "sel_2")["fingerprint"]
    assert _import(db)["ok"] is True
    _decide(db)
    assert _review(db)["status"] == _review(db, "sel_2")["status"] == "APPROVED"
    assert _import(db)["ok"] is True
    assert len(tables[REVIEW_TABLE]) == 1
    assert tables[REVIEW_TABLE][0]["reviewed_by"] == "admin@example.invalid"


@pytest.mark.parametrize("declined", [False, True])
def test_atomic_import_accepts_pending_but_never_writes_declined_pair(monkeypatch, reviewed_pair, declined):
    tables, db = reviewed_pair
    version = "2026-10-31T21:00:00Z"
    tables["tournament_event_draws"][0]["updated_at"] = version
    writes = []
    monkeypatch.setattr(import_service, "write_admin_tournament_draw_teams_atomic",
                        lambda _db, **kwargs: writes.append(kwargs) or kwargs["rows"])
    if declined:
        _decide(db, "DECLINED")
        with pytest.raises(ValueError, match="administrator declined"):
            _import(db, dry_run=False, atomic=True, expected_draw_updated_at=version)
        assert writes == []
    else:
        result = _import(db, dry_run=False, atomic=True, expected_draw_updated_at=version)
        assert result["updated_count"] == 1
        assert len(writes) == 1
        assert writes[0]["expected_draw_updated_at"] == version
        assert [(row["player1_id"], row["player2_id"]) for row in writes[0]["rows"]] == [(1, 2)]


def test_declined_or_changed_pair_cannot_reuse_approval(reviewed_pair):
    tables, db = reviewed_pair
    old_version = _review(db)["review_version"]
    _decide(db, "DECLINED")
    before = deepcopy(tables)
    with pytest.raises(ValueError, match="administrator declined"):
        _import(db, dry_run=False)
    assert tables["tournament_teams"] == before["tournament_teams"]
    with pytest.raises(StaleTournamentRegistrationSelectionError):
        _decide(db, expected_review_version=old_version)
    _decide(db)
    tables["tournament_registrations"][1]["gender"] = "Women"
    assert _review(db)["status"] == "PENDING"
    assert _import(db)["ok"] is True


def test_only_admin_in_own_club_can_review(reviewed_pair):
    tables, db = reviewed_pair
    before = deepcopy(tables)
    with pytest.raises(PermissionError, match="administrator"):
        _decide(db, actor_role="operator")
    with pytest.raises(ValueError, match="not found"):
        _decide(db, club_id="another-club")
    assert tables.get(REVIEW_TABLE, []) == before.get(REVIEW_TABLE, [])


def test_regular_pair_does_not_need_review(reviewed_pair):
    tables, db = reviewed_pair
    tables["tournament_registrations"][1]["gender"] = "Men"
    assert _review(db) is None
    assert _import(db)["ok"] is True


def test_admin_route_saves_private_decision_and_requires_confirmation(monkeypatch, reviewed_pair):
    tables, db = reviewed_pair
    client = _client(monkeypatch, tables)
    monkeypatch.setattr("services.api.main.create_client", lambda *_args: db)
    url = "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/sel_1/gender-review"
    payload = {"expected_review_version": _review(db)["review_version"], "decision": "APPROVED"}
    assert client.post(url, headers={"Authorization": "Bearer local"}, json=payload).status_code == 400
    payload["confirmation_text"] = "REVIEW GENDER ELIGIBILITY"
    response = client.post(url, headers={"Authorization": "Bearer local"}, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["gender_review"]["status"] == "APPROVED"
    assert client.post(url, headers={"Authorization": "Bearer local"}, json=payload).status_code == 409
