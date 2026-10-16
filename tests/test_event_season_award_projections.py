"""Exercise season History through the real public award and podium projections."""
import json

import pytest

from jupr_app.services import event_season_service as history_service
from tests.test_public_tournament_registration_service import FakeSupabase
from tests.test_public_tournament_results_service import _results_storage


def linked_history(monkeypatch, kind, tables, sources):
    tables["pcs_event_series"] = [{"id": "series", "club_id": "club-1", "name": "Annual competition"}]
    tables["pcs_event_editions"] = [
        {"series_id": "series", "club_id": "club-1", "event_kind": kind, "source_id": sid,
         "label": str(2026 + position), "position": position}
        for position, sid in enumerate(sources)
    ]
    monkeypatch.setattr(history_service, "load_source", lambda db, club, event_kind, sid:
                        sources.get(sid) if club == "club-1" and event_kind == kind else None)
    return FakeSupabase(tables)


@pytest.mark.parametrize("selected", ["old", "next"])
def test_interclub_history_keeps_club_honors_with_their_season(monkeypatch, selected):
    sources = {sid: {"event": {"details": {"name": sid}, "opened_at": "2026-01-01"},
                     "setup": {"name": sid}, "complete": sid == "old", "public": True, "fingerprint": "a" * 32}
               for sid in ["old", "next"]}
    base = {"season_id": "old", "club_id": "club-1", "entry_id": None, "division": "3.5", "title": "Champion",
            "recipient_type": "club", "recipient_name": "Season-winning club", "season_name": "2026",
            "earned_at": "2026-09-30", "email": "private@example.invalid"}
    tables = {"pcs_public_interclub_awards": [
        {**base, "id": "division", "award_key": "division_champion"},
        {**base, "id": "cup", "award_key": "club_cup_champion"},
        {**base, "id": "player", "award_key": "division_champion", "recipient_type": "player", "recipient_name": "Finals player"},
        {**base, "id": "different-season", "award_key": "club_cup_champion", "season_id": "unrelated"},
    ]}
    db = linked_history(monkeypatch, "interclub", tables, sources)
    history = history_service.event_history(db, club_id="club-1", kind="interclub", source_id=selected)
    old = next(season for season in history["seasons"] if season["source_id"] == "old")
    assert {award["id"] for award in old["honors"]} == {"division", "cup"}
    assert {award["title"] for award in old["honors"]} == {"3.5 Division Champion", "League Champion"}
    assert all(award["recipient"] == "Season-winning club" for award in old["honors"])
    assert next(season for season in history["seasons"] if season["source_id"] == "next")["honors"] == []
    assert "private@example.invalid" not in json.dumps(history)


@pytest.fixture
def tournament_history(monkeypatch):
    tables = _results_storage()
    tables["tournaments"][0]["status"] = "COMPLETED"
    tables["tournament_lifecycle_receipts"] = [{"id": "receipt", "tournament_id": "t1", "action": "complete"}]
    # Two recipients can share the same placing; both need distinct award cards.
    tables["tournament_podium"][1]["placement"] = 1
    tables["tournament_event_draws"].append({"id": "team-draw", "tournament_id": "t1", "event_option_id": "team-event",
                                              "draw_kind": "TEAM_PARENT", "status": "published", "name": "Team Cup"})
    tables["tournament_event_options"].append({**tables["tournament_event_options"][0], "id": "team-event", "competition_format": "FOUR_PLAYER_TEAM"})
    tables["tournament_four_player_teams"] = [{"id": "four-player-team", "tournament_id": "t1", "draw_id": "team-draw",
                                               "name": "Team champions", "status": "CONFIRMED", "eligibility_state": "ELIGIBLE"}]
    tables["tournament_four_player_podium"] = [{"tournament_id": "t1", "draw_id": "team-draw", "team_id": "four-player-team",
                                                "placement": 1, "published_at": "2026-09-30"}]
    sources = {
        "t1": {"event": tables["tournaments"][0], "complete": True, "public": True, "fingerprint": "a" * 32},
        "next": {"event": {"name": "2027 Open", "status": "ACTIVE"}, "complete": False, "public": True, "fingerprint": "b" * 32},
    }
    return linked_history(monkeypatch, "tournament", tables, sources), tables, sources


@pytest.mark.parametrize("selected", ["t1", "next"])
@pytest.mark.parametrize("admin", [False, True])
def test_tournament_history_includes_standard_and_team_podiums(tournament_history, selected, admin):
    db, _tables, _sources = tournament_history
    history = history_service.event_history(db, club_id="club-1", kind="tournament", source_id=selected, slug="club", admin=admin)
    honors = next(season for season in history["seasons"] if season["source_id"] == "t1")["honors"]
    assert {award["recipient"] for award in honors} == {"Alex Ace", "Blair Backhand", "Team champions"}
    assert len({award["id"] for award in honors}) == len(honors) == 3
    assert next(season for season in history["seasons"] if season["source_id"] == "next")["honors"] == []
    assert "private@example.com" not in json.dumps(history)


@pytest.mark.parametrize("hidden", ["unfinished", "unpublished", "hidden_archive"])
def test_tournament_history_respects_completion_and_explicit_visibility(tournament_history, hidden):
    db, tables, sources = tournament_history
    if hidden == "unfinished":
        sources["t1"]["complete"] = False
    else:
        sources["t1"]["public"] = False
        if hidden == "hidden_archive":
            tables["tournaments"][0]["status"] = "ARCHIVED"
        else:
            tables["tournament_registration_settings"][0]["builder_draft_json"] = {}
    history = history_service.event_history(db, club_id="club-1", kind="tournament", source_id="next", slug="club")
    assert not any(season["honors"] for season in history["seasons"])


def test_team_tournament_history_omits_unpublished_podiums_and_draws(tournament_history):
    db, tables, _sources = tournament_history
    tables["tournament_four_player_podium"][0]["published_at"] = None
    hidden_draw = {**tables["tournament_event_draws"][-1], "id": "draft-team-draw", "status": "draft"}
    tables["tournament_event_draws"].append(hidden_draw)
    history = history_service.event_history(db, club_id="club-1", kind="tournament", source_id="t1", slug="club")
    assert {award["recipient"] for award in history["seasons"][-1]["honors"]} == {"Alex Ace", "Blair Backhand"}
