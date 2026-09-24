from types import SimpleNamespace

from jupr_app.domain.tournament_registration_confirmation_tokens import (
    build_registration_confirmation_token,
)
from jupr_app.services.public_tournament_team_service import (
    build_public_four_player_team_setup_recovery,
)


class _Query:
    def __init__(self, rows):
        self.rows = list(rows)
        self.filters = []
        self.bound = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.bound = int(value)
        return self

    def execute(self):
        rows = [
            row
            for row in self.rows
            if all(str(row.get(key)) == str(value) for key, value in self.filters)
        ]
        if self.bound is not None:
            rows = rows[: self.bound]
        return SimpleNamespace(data=rows)


class _Supabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return _Query(self.tables.get(name, []))


def test_team_setup_recovery_uses_registration_team_and_operation_truth(
    monkeypatch,
):
    monkeypatch.setenv(
        "JUPR_REGISTRATION_CONFIRMATION_SECRET",
        "team-recovery-unit-test-secret",
    )
    token = build_registration_confirmation_token(
        tournament_id="tournament-1",
        registration_id="registration-1",
        email="captain@example.com",
    )
    supabase = _Supabase(
        {
            "tournaments": [
                {
                    "id": "tournament-1",
                    "club_id": "club-1",
                    "name": "Summer Teams",
                }
            ],
            "tournament_registrations": [
                {
                    "id": "registration-1",
                    "tournament_id": "tournament-1",
                    "display_name": "Captain",
                    "email": "captain@example.com",
                    "gender": "Men",
                    "status": "CONFIRMED",
                }
            ],
            "tournament_registration_selections": [
                {
                    "id": "selection-1",
                    "tournament_id": "tournament-1",
                    "registration_id": "registration-1",
                    "event_option_id": "event-complete",
                },
                {
                    "id": "selection-2",
                    "tournament_id": "tournament-1",
                    "registration_id": "registration-1",
                    "event_option_id": "event-required",
                },
            ],
            "tournament_event_options": [
                {
                    "id": "event-complete",
                    "tournament_id": "tournament-1",
                    "registration_day_id": "day-1",
                    "label": "Team 3.5",
                    "event_family_label": "Mixed Team",
                    "division_name": "3.5",
                    "competition_format": "FOUR_PLAYER_TEAM",
                    "sort_order": 1,
                },
                {
                    "id": "event-required",
                    "tournament_id": "tournament-1",
                    "registration_day_id": "day-1",
                    "label": "Team 4.0",
                    "event_family_label": "Mixed Team",
                    "division_name": "4.0",
                    "competition_format": "FOUR_PLAYER_TEAM",
                    "sort_order": 2,
                },
            ],
            "tournament_four_player_teams": [
                {
                    "id": "team-1",
                    "tournament_id": "tournament-1",
                    "event_option_id": "event-complete",
                    "captain_registration_id": "registration-1",
                    "name": "Safe Team",
                    "status": "FORMING",
                    "eligibility_state": "NOT_REQUIRED",
                    "version": 1,
                    "creation_fingerprint": "must-not-leak",
                }
            ],
            "tournament_four_player_team_members": [
                {
                    "id": "member-1",
                    "team_id": "team-1",
                    "tournament_id": "tournament-1",
                    "slot": "MAN_1",
                    "invited_email": "captain@example.com",
                    "display_name_snapshot": "Captain",
                    "status": "ACCEPTED",
                    "invitation_version": 1,
                    "invitation_token_hash": "must-not-leak",
                }
            ],
            "tournament_team_operations": [
                {
                    "operation_key": "must-not-leak",
                    "request_fingerprint": "must-not-leak",
                    "tournament_id": "tournament-1",
                    "surface": "registration",
                    "action": "four_player_team_create",
                    "entity_id": "event-complete",
                    "actor": "captain@example.com",
                    "status": "COMPLETED",
                }
            ],
        }
    )

    result = build_public_four_player_team_setup_recovery(
        supabase,
        club_id="club-1",
        confirmation_token=token,
    )

    assert [event["setup_state"] for event in result["events"]] == [
        "COMPLETE",
        "SETUP_REQUIRED",
    ]
    assert result["events"][0]["team"]["name"] == "Safe Team"
    rendered = repr(result)
    assert "must-not-leak" not in rendered
    assert "operation_key" not in rendered
    assert "request_fingerprint" not in rendered


def test_solo_confirmation_follows_assignment_without_exposing_other_teams(monkeypatch):
    import pytest

    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "team-recovery-unit-test-secret")
    token = build_registration_confirmation_token(
        tournament_id="t1", registration_id="solo", email="solo@example.invalid",
    )
    tables = {
        "tournaments": [{"id": "t1", "club_id": "club1"}],
        "tournament_registrations": [{"id": "solo", "tournament_id": "t1", "email": "solo@example.invalid", "status": "CONFIRMED"}],
        "tournament_registration_selections": [{"registration_id": "solo", "tournament_id": "t1", "event_option_id": "e1"}],
        "tournament_event_options": [{"id": "e1", "tournament_id": "t1", "competition_format": "FOUR_PLAYER_TEAM"}],
        "tournament_four_player_teams": [{"id": "team1", "tournament_id": "t1", "event_option_id": "e1", "captain_registration_id": "captain", "status": "CONFIRMED", "name": "Assigned Team"}],
        "tournament_four_player_team_members": [],
    }
    supabase = _Supabase(tables)

    def state():
        return build_public_four_player_team_setup_recovery(supabase, club_id="club1", confirmation_token=token)["events"][0]

    assert state()["setup_state"] == "SETUP_REQUIRED"
    member = {"id": "m1", "tournament_id": "t1", "team_id": "team1", "registration_id": "solo", "slot": "MAN_2", "status": "ACCEPTED", "display_name_snapshot": "Solo Player", "invited_email": "solo@example.invalid", "invitation_token_hash": "must-not-leak"}
    tables["tournament_four_player_team_members"].append(member)
    assigned = state()
    assert assigned["setup_state"] == "COMPLETE"
    assert assigned["team"]["name"] == "Assigned Team"
    assert "must-not-leak" not in repr(assigned)
    assert "solo@example.invalid" not in repr(assigned)
    for status in ["REMOVED", "DECLINED", "CANCELLED"]:
        member["status"] = status
        assert state()["setup_state"] == "SETUP_REQUIRED"
    member.update(status="INVITED", registration_id=None, invited_email=" SOLO@example.invalid ")
    assert state()["setup_state"] == "COMPLETE"
    member["registration_id"] = "another-registration"
    assert state()["setup_state"] == "SETUP_REQUIRED", "Email cannot override an existing registration link"
    member["registration_id"] = "solo"
    member["tournament_id"] = "other-tournament"
    assert state()["setup_state"] == "SETUP_REQUIRED"
    member["tournament_id"] = "t1"
    tables["tournament_four_player_teams"][0]["event_option_id"] = "unselected-event"
    assert state()["setup_state"] == "SETUP_REQUIRED"
    tables["tournament_four_player_teams"][0]["event_option_id"] = "e1"
    tables["tournament_four_player_teams"][0]["status"] = "WITHDRAWN"
    assert state()["setup_state"] == "SETUP_REQUIRED"
    with pytest.raises(ValueError, match="different club"):
        build_public_four_player_team_setup_recovery(supabase, club_id="foreign", confirmation_token=token)
