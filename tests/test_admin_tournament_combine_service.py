from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.services.admin_tournament_combine_service import combine_admin_tournament_draw_divisions
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_state_fingerprint
from tests.test_admin_match_log_service import FakeSupabase


VERSION = "2026-10-31T10:00:00Z"


def combine_tables():
    event = {"tournament_id": "tournament", "registration_day_id": "day", "event_family_label": "Mixed doubles", "event_type": "MIXED_DOUBLES", "gender_restriction": "MIXED", "partner_required": True, "competition_format": "STANDARD", "enabled": True, "status": "open", "updated_at": VERSION}
    return {
        "tournaments": [{"id": "tournament", "club_id": "club", "status": "PUBLISHED"}],
        "tournament_registration_days": [{"id": "day", "tournament_id": "tournament", "enabled": True}],
        "tournament_event_options": [{**event, "id": "event-40", "division_name": "4.0"}, {**event, "id": "event-45", "division_name": "4.5"}],
        "tournament_event_draws": [{"id": f"draw-{division}", "tournament_id": "tournament", "registration_day_id": "day", "event_option_id": f"event-{division}", "name": division, "status": "draft", "updated_at": VERSION} for division in ("40", "45")],
        "tournament_teams": [{"id": f"team-{division}", "tournament_id": "tournament", "draw_id": f"draw-{division}", "registration_day_id": "day", "event_option_id": f"event-{division}", "team_number": 1, "player1_id": index * 2 + 1, "player2_id": index * 2 + 2, "source": "REGISTRATION", "source_selection_id": f"selection-{division}", "updated_at": VERSION} for index, division in enumerate(("40", "45"))],
        "tournament_registration_selections": [{"id": f"selection-{division}", "tournament_id": "tournament", "event_option_id": f"event-{division}", "registration_day_id": "day", "registration_id": f"registration-{division}"} for division in ("40", "45")],
        "tournament_registrations": [], "tournament_games": [], "tournament_podium": [],
        "tournament_day_live_draws": [], "tournament_day_live_queue": [], "admin_activity_log": [],
        "players": [], "matches": [], "player_badges": [], "tournament_team_match_games": [],
    }


class CombineSupabase(FakeSupabase):
    def __init__(self, tables):
        super().__init__(tables)
        self.tables = tables
        self.rpc_calls = []
        self.rpc_error = None

    def rpc(self, name, params):
        self.rpc_calls.append((name, deepcopy(params)))

        def execute():
            if self.rpc_error:
                raise RuntimeError(self.rpc_error)
            target = next(row for row in self.tables["tournament_event_draws"] if row["id"] == params["p_target_draw_id"])
            target.update(name=params["p_name"], source_event_option_ids=list(params["p_source_event_option_ids"]), updated_at="2026-10-31T10:01:00Z")
            return SimpleNamespace(data={"ok": True, "draw": deepcopy(target)})

        return SimpleNamespace(execute=execute)


def combine(db, **overrides):
    options = dict(club_id="club", tournament_id="tournament", draw_id="draw-40", event_option_ids=["event-45", "event-40"], name="Mixed 4.0 / 4.5", actor_email="admin@example.com", actor_role="club_owner", confirmation_text="COMBINE DIVISIONS", expected_state_fingerprint=get_admin_tournament_ops_state_fingerprint(db, club_id="club", tournament_id="tournament"))
    options.update(overrides)
    return combine_admin_tournament_draw_divisions(db, **options)


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")


def test_preview_describes_merged_roster_without_mutation():
    db = CombineSupabase(combine_tables())
    before = deepcopy(db.tables)
    result = combine(db, dry_run=True)
    assert result["write_count"] == 0
    assert result["moved_team_count"] == 1
    assert result["combined_source_draw_ids"] == ["draw-45"]
    assert result["draw"]["source_event_option_ids"] == ["event-40", "event-45"]
    assert db.tables == before
    assert db.rpc_calls == []


def test_combine_passes_exact_versions_to_atomic_rpc_and_audits_original_draws():
    db = CombineSupabase(combine_tables())
    registrations = deepcopy(db.tables["tournament_registration_selections"])
    result = combine(db)
    assert result["draw"]["name"] == "Mixed 4.0 / 4.5"
    name, params = db.rpc_calls[0]
    assert name == "admin_combine_tournament_draw_divisions_cas"
    assert params["p_club_id"] == "club"
    assert params["p_target_draw_id"] == "draw-40"
    assert params["p_source_event_option_ids"] == ["event-40", "event-45"]
    for key, ids in (("p_expected_draws", ["draw-40", "draw-45"]), ("p_expected_teams", ["team-40", "team-45"]), ("p_expected_events", ["event-40", "event-45"])):
        assert params[key] == [{"id": value, "updated_at": VERSION} for value in ids]
    assert db.tables["tournament_registration_selections"] == registrations
    assert db.tables["admin_activity_log"][0]["action_type"] == "combine_tournament_draw_divisions_admin"
    assert len(db.tables["admin_activity_log"][0]["before_json"]["draws"]) == 2


@pytest.mark.parametrize("overrides", [{"expected_state_fingerprint": "0" * 64}, {"club_id": "other-club"}, {"confirmation_text": ""}, {"event_option_ids": ["event-45"]}, {"event_option_ids": ["event-40", "event-40"]}, {"event_option_ids": ["event-40", "missing"]}, {"name": " "}])
def test_invalid_or_unreviewed_request_never_calls_rpc(overrides):
    db = CombineSupabase(combine_tables())
    with pytest.raises((ValueError, StaleTournamentAdminStateError)):
        combine(db, **overrides)
    assert db.rpc_calls == []


@pytest.mark.parametrize("patch", [{"event_family_label": "Men's doubles"}, {"registration_day_id": "other-day"}, {"scheduled_day_ids": ["other-day"]}, {"enabled": False}, {"competition_format": "FOUR_PLAYER_TEAM"}, {"eligibility_mode": "COMBINED_RATING_CAP"}, {"skill_mode": "COMBINED_RATING_CAP"}])
def test_incompatible_source_is_rejected_before_rpc(patch):
    db = CombineSupabase(combine_tables())
    db.tables["tournament_event_options"][1].update(patch)
    with pytest.raises(ValueError):
        combine(db)
    assert db.rpc_calls == []


@pytest.mark.parametrize("table", ["tournament_games", "tournament_podium", "tournament_day_live_draws", "tournament_day_live_queue"])
@pytest.mark.parametrize("draw_id", ["draw-40", "draw-45"])
def test_any_source_participation_blocks_combination(table, draw_id):
    db = CombineSupabase(combine_tables())
    db.tables[table].append({"id": "played", "tournament_id": "tournament", "draw_id": draw_id})
    with pytest.raises(ValueError, match="before"):
        combine(db)
    assert db.rpc_calls == []


@pytest.mark.parametrize("patch", [{"pool_config": {"version": 1}}, {"status": "PUBLISHED"}, {"draw_kind": "TEAM_PARENT"}, {"registration_day_id": "other-day"}])
def test_source_draw_cannot_have_pools_played_status_or_different_day(patch):
    db = CombineSupabase(combine_tables())
    db.tables["tournament_event_draws"][1].update(patch)
    with pytest.raises(ValueError):
        combine(db)
    assert db.rpc_calls == []


def test_duplicate_player_across_divisions_is_rejected():
    db = CombineSupabase(combine_tables())
    db.tables["tournament_teams"][1]["player2_id"] = 1
    with pytest.raises(ValueError, match="more than one"):
        combine(db)
    assert db.rpc_calls == []


def test_existing_combined_membership_cannot_be_removed():
    db = CombineSupabase(combine_tables())
    db.tables["tournament_event_draws"][0]["source_event_option_ids"] = ["event-40", "event-50"]
    with pytest.raises(ValueError, match="Keep every division"):
        combine(db)
    assert db.rpc_calls == []


def test_another_combined_draw_cannot_be_absorbed():
    db = CombineSupabase(combine_tables())
    db.tables["tournament_event_draws"][1]["source_event_option_ids"] = ["event-45", "event-50"]
    with pytest.raises(ValueError, match="existing combined draw"):
        combine(db)
    assert db.rpc_calls == []


@pytest.mark.parametrize("marker,exception", [("JUPR_TOURNAMENT_COMBINE_TEAMS_STALE", StaleTournamentAdminStateError), ("JUPR_TOURNAMENT_COMBINE_DRAW_IN_USE", ValueError), ("connection unavailable", RuntimeError)])
def test_atomic_race_or_failure_cannot_claim_success(marker, exception):
    db = CombineSupabase(combine_tables())
    db.rpc_error = marker
    with pytest.raises(exception):
        combine(db)
    assert db.tables["admin_activity_log"] == []


def test_unavailable_live_evidence_fails_closed():
    db = CombineSupabase(combine_tables())
    original = db.table
    def table(name):
        if name == "tournament_day_live_draws":
            raise RuntimeError("live evidence unavailable")
        return original(name)
    db.table = table
    with pytest.raises(RuntimeError, match="live evidence unavailable"):
        combine(db)
    assert db.rpc_calls == []
