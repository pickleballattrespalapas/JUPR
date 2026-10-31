from __future__ import annotations

from copy import deepcopy

import pytest

from jupr_app.services import admin_tournament_registration_import_service as service
from jupr_app.services.admin_tournament_team_service import replace_admin_tournament_draw_teams
from jupr_app.services.admin_tournament_service import get_admin_tournament_detail, review_admin_tournament_gender_eligibility
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_public_tournament_registration_service import FakeSupabase as RegistrationSupabase
from tests.test_api_contract_admin_tournament_registration_import import (
    tournament_registration_import_tables,
)


def combined_tables():
    tables = tournament_registration_import_tables()
    draw = tables["tournament_event_draws"][0]
    draw.update(source_event_option_ids=["event_1", "event_2"], updated_at="2026-10-31T12:00:00Z")
    event = tables["tournament_event_options"][0]
    event.update(
        enabled=True, status="OPEN", registration_day_id="day_1",
        scheduled_day_ids=["day_1"], event_family_label="Men's doubles",
        gender_restriction="MEN", label="Men's doubles", skill_label="4.0",
    )
    tables["tournament_event_options"].append({**event, "id": "event_2", "skill_label": "4.5"})
    for index, player_id in enumerate((4, 5), start=1):
        tables["tournament_registrations"].append({
            "id": f"reg_{player_id}", "tournament_id": "tour_1", "status": "confirmed",
            "player_id": player_id, "display_name": f"Player {player_id}",
            "email": f"player{player_id}@example.com",
        })
        tables["tournament_registration_selections"].append({
            "id": f"sel_{player_id}", "tournament_id": "tour_1",
            "registration_id": f"reg_{player_id}", "registration_day_id": "day_1",
            "event_option_id": "event_2", "partner_mode": "HAS_PARTNER",
        })
        tables["tournament_registration_team_members"].append({
            "id": f"member_{player_id}", "team_link_id": "link_2", "tournament_id": "tour_1",
            "event_option_id": "event_2", "selection_id": f"sel_{player_id}",
            "registration_id": f"reg_{player_id}", "player_id": player_id,
            "player_order": index, "status": "ACTIVE",
        })
    tables["tournament_registration_team_links"].append({
        "id": "link_2", "tournament_id": "tour_1", "event_option_id": "event_2",
        "registration1_id": "reg_4", "registration2_id": "reg_5",
        "selection1_id": "sel_4", "selection2_id": "sel_5",
        "player1_id": 4, "player2_id": 5, "status": "CONFIRMED",
    })
    return tables


def import_teams(monkeypatch, tables, **kwargs):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    return service.import_admin_tournament_registrations_to_draw(
        FakeSupabase(tables), club_id="club", tournament_id="tour_1", draw_id="draw_1",
        actor_email="admin@example.com", actor_role="club_owner",
        confirmation_text="IMPORT REGISTRATIONS",
        expected_draw_updated_at="2026-10-31T12:00:00Z", **kwargs,
    )


def test_combined_import_preserves_source_divisions_and_confirmed_pairs(monkeypatch):
    tables = combined_tables()
    source_tables = (
        "tournament_registrations", "tournament_registration_selections",
        "tournament_registration_team_links", "tournament_registration_team_members",
    )
    before = {name: deepcopy(tables[name]) for name in source_tables}
    checked_selection_ids = set()
    monkeypatch.setattr(service, "require_no_declined_gender_reviews_for_import", lambda _db, **kwargs: checked_selection_ids.update(kwargs["selection_ids"]))

    result = import_teams(monkeypatch, tables)

    assert result["updated_count"] == 2
    assert {(team["player1_id"], team["player2_id"]) for team in result["teams"]} == {(1, 2), (4, 5)}
    assert {(team["source_event_option_id"], team["source_selection_id"]) for team in result["teams"]} == {
        ("event_1", "sel_1"), ("event_2", "sel_4"),
    }
    assert {team["event_option_id"] for team in result["teams"]} == {"event_1"}
    assert {team["registration_day_id"] for team in result["teams"]} == {"day_1"}
    assert checked_selection_ids == {"sel_1", "sel_2", "sel_4", "sel_5"}
    assert {name: tables[name] for name in source_tables} == before
    assert {team["source_event_option_id"] for team in tables["tournament_teams"]} == {"event_1", "event_2"}


def test_combined_import_atomic_payload_retains_original_selection_and_event(monkeypatch):
    tables = combined_tables()
    writes = []
    monkeypatch.setattr(service, "write_admin_tournament_draw_teams_atomic", lambda _db, **kwargs: writes.append(kwargs) or kwargs["rows"])

    result = import_teams(monkeypatch, tables, atomic=True)

    assert len(writes) == 1
    assert writes[0]["replace"] is True
    assert writes[0]["expected_draw_updated_at"] == "2026-10-31T12:00:00Z"
    assert {row["source_selection_id"] for row in writes[0]["rows"]} == {"sel_1", "sel_4"}
    assert {row["source_event_option_id"] for row in writes[0]["rows"]} == {"event_1", "event_2"}
    assert result["teams"][1]["source_event_option_id"] == "event_2"


@pytest.mark.parametrize("declined_selection", [None, "sel_1", "sel_4"])
def test_combined_import_does_not_require_second_gender_approval(monkeypatch, declined_selection):
    tables = combined_tables()
    for event in tables["tournament_event_options"]:
        event.update(event_type="MIXED_DOUBLES", gender_restriction="MIXED", event_family_label="Mixed doubles")
    for registration in tables["tournament_registrations"]:
        registration["gender"] = "Women"
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    db = RegistrationSupabase(tables)
    detail = get_admin_tournament_detail(db, club_id="club", tournament_id="tour_1")
    reviews = {row["id"]: row["gender_review"] for row in detail["selections"] if row.get("gender_review")}
    assert set(reviews) == {"sel_1", "sel_2", "sel_4", "sel_5"}
    assert all(review["status"] == "PENDING" for review in reviews.values())
    if declined_selection:
        review_admin_tournament_gender_eligibility(
            db, club_id="club", tournament_id="tour_1", selection_id=declined_selection,
            decision="DECLINED", expected_review_version=reviews[declined_selection]["review_version"],
            actor_email="admin@example.com", actor_role="club_owner",
        )
        before = deepcopy(tables["tournament_teams"])
        with pytest.raises(ValueError, match="administrator declined"):
            import_teams(monkeypatch, tables)
        assert tables["tournament_teams"] == before
    else:
        result = import_teams(monkeypatch, tables)
        assert result["updated_count"] == 2
        assert {team["source_event_option_id"] for team in result["teams"]} == {"event_1", "event_2"}


def test_combined_append_preserves_existing_teams_and_skips_exact_pairs(monkeypatch):
    tables = combined_tables()
    tables["tournament_teams"] = [{
        "id": "saved_manual", "tournament_id": "tour_1", "draw_id": "draw_1",
        "event_option_id": "event_1", "registration_day_id": "day_1",
        "team_number": 7, "player1_id": 2, "player2_id": 1,
        "source": "MANUAL", "notes": "Director's saved pair", "seed": 1,
    }]
    saved_manual = deepcopy(tables["tournament_teams"][0])

    result = import_teams(monkeypatch, tables, import_mode="APPEND")

    assert result["updated_count"] == 1
    assert result["teams"][0]["team_number"] == 8
    assert result["teams"][0]["source_event_option_id"] == "event_2"
    assert tables["tournament_teams"][0] == saved_manual
    assert len(tables["tournament_teams"]) == 2
    after = deepcopy(tables["tournament_teams"])

    repeated = import_teams(monkeypatch, tables, import_mode="APPEND")

    assert repeated["updated_count"] == 0
    assert repeated["write_count"] == 0
    assert tables["tournament_teams"] == after
    assert len(tables["admin_activity_log"]) == 1


def test_combined_append_blocks_existing_conflicting_pair(monkeypatch):
    tables = combined_tables()
    tables["tournament_teams"][0].update(player1_id=1, player2_id=9)
    before = deepcopy(tables["tournament_teams"])
    with pytest.raises(ValueError, match="already exist"):
        import_teams(monkeypatch, tables, import_mode="APPEND")
    assert tables["tournament_teams"] == before


@pytest.mark.parametrize("change", [
    {"enabled": False}, {"registration_day_id": "other_day", "scheduled_day_ids": ["other_day"]},
    {"event_family_label": "Women's doubles"}, {"eligibility_mode": "COMBINED_RATING_CAP"},
    {"competition_format": "FOUR_PLAYER_TEAM"},
])
def test_combined_import_rechecks_source_compatibility_before_writing(monkeypatch, change):
    tables = combined_tables()
    tables["tournament_event_options"][1].update(change)
    before = deepcopy(tables["tournament_teams"])

    with pytest.raises(ValueError):
        import_teams(monkeypatch, tables)

    assert tables["tournament_teams"] == before
    assert tables["admin_activity_log"] == []


def test_combined_import_rejects_missing_source_division(monkeypatch):
    tables = combined_tables()
    tables["tournament_event_options"].pop()
    with pytest.raises(ValueError):
        import_teams(monkeypatch, tables)
    assert tables["tournament_teams"][0]["id"] == "old_team"


def test_combined_import_refuses_a_team_link_spanning_source_divisions(monkeypatch):
    tables = combined_tables()
    tables["tournament_registration_team_members"][-1]["event_option_id"] = "event_1"
    with pytest.raises(ValueError, match="could not be resolved"):
        import_teams(monkeypatch, tables)
    assert tables["tournament_teams"][0]["id"] == "old_team"


def test_combined_import_refuses_player_appearing_in_both_divisions(monkeypatch):
    tables = combined_tables()
    tables["tournament_registration_team_links"][-1]["player1_id"] = 1
    tables["tournament_registration_team_members"][-2]["player_id"] = 1
    tables["tournament_registrations"][-2]["player_id"] = 1
    with pytest.raises(ValueError, match="Duplicate player IDs"):
        import_teams(monkeypatch, tables)
    assert tables["tournament_teams"][0]["id"] == "old_team"


def test_superseded_source_draw_cannot_be_reimported(monkeypatch):
    tables = combined_tables()
    tables["tournament_event_draws"][0]["combined_into_draw_id"] = "other_draw"
    with pytest.raises(ValueError, match="combined into another"):
        import_teams(monkeypatch, tables)


def test_combined_singles_import_preserves_each_original_division(monkeypatch):
    tables = combined_tables()
    for event in tables["tournament_event_options"]:
        event.update(event_type="SINGLES", event_family_label="Men's singles", partner_required=False)

    result = import_teams(monkeypatch, tables, dry_run=True)

    assert result["updated_count"] == 4
    assert {row["player1_id"] for row in result["teams"]} == {1, 2, 4, 5}
    assert all(row["player2_id"] is None for row in result["teams"])
    assert {row["source_selection_id"] for row in result["teams"]} == {"sel_1", "sel_2", "sel_4", "sel_5"}
    assert {row["source_event_option_id"] for row in result["teams"]} == {"event_1", "event_2"}
    assert all(row["event_option_id"] == "event_1" for row in result["teams"])
    assert tables["tournament_teams"][0]["id"] == "old_team"


@pytest.mark.parametrize("player_pair,expected_source,expected_selection", [
    ((5, 4), "event_2", "sel_4"),
    ((4, 9), "event_1", None),
])
def test_team_editor_preserves_provenance_only_for_unchanged_pair(
    monkeypatch, player_pair, expected_source, expected_selection,
):
    tables = combined_tables()
    tables["players"] = [{"id": player_id, "club_id": "club"} for player_id in (4, 5, 9)]
    tables["tournament_teams"] = [{
        "id": "moved_team", "tournament_id": "tour_1", "draw_id": "draw_1",
        "event_option_id": "event_1", "registration_day_id": "day_1", "team_number": 1,
        "player1_id": 4, "player2_id": 5, "source": "REGISTRATION",
        "source_event_option_id": "event_2", "source_selection_id": "sel_4",
    }]
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    result = replace_admin_tournament_draw_teams(
        FakeSupabase(tables), club_id="club", tournament_id="tour_1", draw_id="draw_1",
        actor_email="admin@example.com", actor_role="club_owner", confirmation_text="SAVE TEAMS",
        expected_draw_updated_at="2026-10-31T12:00:00Z", dry_run=True,
        teams=[{
            "team_number": 1, "player1_id": player_pair[0], "player2_id": player_pair[1],
            "seed": 2, "notes": "Reviewed seeding", "source_event_option_id": "forged_event",
            "source_selection_id": "forged_selection",
        }],
    )

    team = result["teams"][0]
    assert team["source_event_option_id"] == expected_source
    assert team["source_selection_id"] == expected_selection
    assert team["source"] == ("REGISTRATION" if expected_selection else "MANUAL")
    assert team["seed"] == 2
    assert team["notes"] == "Reviewed seeding"
