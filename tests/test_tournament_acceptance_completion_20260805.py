from __future__ import annotations

from pathlib import Path

from jupr_app.domain.tournament_age_policy import build_age_split_preview
from jupr_app.domain.tournament_registration_repo import (
    normalize_registration_configuration_payload,
    upsert_registration_settings,
)

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read_web(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def read_root(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class _Response:
    def __init__(self, data):
        self.data = data


class _UpsertQuery:
    def __init__(self):
        self.payload = None

    def upsert(self, payload, **_kwargs):
        self.payload = dict(payload)
        return self

    def execute(self):
        return _Response([dict(self.payload or {})])


class _UpsertClient:
    def __init__(self):
        self.query = _UpsertQuery()

    def table(self, name: str):
        assert name == "tournament_registration_settings"
        return self.query


def test_venue_inventory_is_stable_and_day_subsets_are_canonical() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    venue = read_web("app/admin/tournaments/setup/TournamentVenueModel.ts")
    builder = read_web("app/admin/tournament-setup/tournamentSetupBuilder.ts")
    repo = read_root("jupr_app/domain/tournament_registration_repo.py")
    migration = read_root("supabase/migrations/20260805043000_tournament_venue_inventory.sql")

    assert "Venue address" in panel
    assert "Directions to the venue (optional)" in panel
    assert 'aria-label="Total venue courts"' in panel
    assert "This read-only count is derived from the court inventory" in panel
    assert "Add court" in panel
    assert "Only Remove court deletes it" in panel
    assert "Use all venue courts" in panel
    assert "Which courts are available?" in panel
    assert "venue_courts_json" in venue
    assert "available_court_ids" in venue
    assert 'title: court.title == null ? "" : String(court.title)' in venue
    assert "settingsWithVenueCourts" in venue
    assert "withVenueCourtAvailability" in venue
    assert "available_court_ids" in builder
    assert '"available_court_ids"' in repo
    assert "venue_address text" in migration
    assert "venue_directions text" in migration
    assert "venue_courts_json jsonb" in migration
    assert "available_court_ids jsonb" in migration


def test_registration_settings_preserve_blank_optional_court_titles() -> None:
    client = _UpsertClient()
    row = upsert_registration_settings(
        client,
        {
            "id": "settings-1",
            "tournament_id": "tournament-1",
            "venue_address": "123 Main Street",
            "venue_directions": "Use the south gate.",
            "venue_courts_json": [
                {"id": "venue-court-1", "title": "Championship Court"},
                {"id": "venue-court-2", "title": ""},
            ],
        },
    )

    assert row["venue_courts_json"] == [
        {"id": "venue-court-1", "title": "Championship Court"},
        {"id": "venue-court-2", "title": ""},
    ]
    assert row["venue_address"] == "123 Main Street"
    assert row["venue_directions"] == "Use the south gate."


def test_day_payload_preserves_exact_available_court_subset() -> None:
    days, events = normalize_registration_configuration_payload(
        tournament_id="tournament-1",
        days=[
            {
                "id": "day-1",
                "label": "Friday",
                "event_date": "2026-08-07",
                "available_court_ids": ["venue-court-3", "venue-court-1", "venue-court-3"],
                "court_labels": ["Court 3", "Championship Court"],
            }
        ],
        event_options=[
            {
                "id": "division-1",
                "registration_day_id": "day-1",
                "scheduled_day_ids": ["day-1"],
                "label": "3.5",
                "event_type": "GENDER_DOUBLES",
            }
        ],
    )

    assert events[0]["registration_day_id"] == "day-1"
    assert days[0]["available_court_ids"] == ["venue-court-3", "venue-court-1"]
    assert days[0]["court_count"] == 2


def test_review_humanizes_values_and_hides_raw_payloads_by_default() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    review = read_web("app/admin/tournaments/setup/TournamentReviewValue.tsx")

    assert "ReviewComparisonDisplay" in panel
    assert "ReviewValueDisplay" in panel
    assert 'scheduled_day_ids: "Tournament days"' in review
    assert 'skill_age_rules: "Skill and age rules"' in review
    assert 'venue_courts_json: "Venue courts"' in review
    assert "Current published value" in review
    assert "Proposed draft value" in review
    assert 'status === "Added"' in review
    assert 'status === "Removed"' in review
    assert '? "Changed" : "Unchanged"' in review
    assert "Technical details" in review
    assert 'overflowWrap: "anywhere"' in review
    assert "Intl.DateTimeFormat" in review


def test_split_age_copy_and_bulk_division_presets_are_available() -> None:
    age = read_web("app/admin/tournaments/setup/TournamentAgePolicyEditor.tsx")
    event_dialog = read_web("app/admin/tournaments/setup/TournamentSetupEventFamilyDialog.tsx")
    event_card = read_web("app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx")
    presets = read_web("app/admin/tournaments/setup/TournamentDivisionPresetDialog.tsx")
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    domain = read_root("jupr_app/domain/tournament_age_policy.py")

    assert "Split-age partners (one under / one over)" in age
    assert "Each team must include one player under the threshold" in age
    assert "This does not create separate Under 50 and 50+ divisions" in age
    assert 'structure === "SINGLES" && policy.mode === "SPLIT_AGE"' in event_dialog
    assert "Generate divisions" in event_card
    for skill in ('"3.0"', '"3.5"', '"4.0"', '"4.5"', '"Open"'):
        assert skill in presets
    assert "Existing divisions were detected and left unselected" in presets
    assert "Another selected proposal uses this division name" in presets
    assert "existingNames.has(normalized)" in presets
    assert "team_age_rule: inherited.team_age_rule" in presets
    assert "Save selected divisions" in presets
    assert "persistConfigurationDraft" in panel
    assert "Continue to Divisions" in panel
    assert "Save draft and continue to Divisions" not in panel
    assert "player_age < split_threshold <= partner_age" in domain
    assert "Split-age partners is available only for doubles and team events" in domain


def test_public_registration_projects_and_displays_venue_details() -> None:
    service = read_root("jupr_app/services/public_tournament_registration_service.py")
    api = read_web("lib/tournamentRegistrationApi.ts")
    page = read_web("app/clubs/[clubSlug]/tournament-registration/page.tsx")

    for field in ("location_name", "venue_address", "venue_directions", "timezone"):
        assert f'"{field}"' in service
        assert field in api
    assert ">Venue<" in page
    assert "Open map" in page
    assert "Arrival directions" in page


def test_split_age_preview_is_a_composition_rule_not_a_minimum_bracket() -> None:
    preview = build_age_split_preview(
        policy={
            "mode": "SPLIT_AGE",
            "split_age_threshold": 50,
            "min_teams_per_age_group": 4,
        },
        registrations={
            "registration-1": {"id": "registration-1", "display_name": "Valid Team", "age": 49},
            "registration-2": {"id": "registration-2", "display_name": "Invalid Team", "age": 49},
        },
        selections=[
            {"id": "selection-1", "registration_id": "registration-1", "partner_age": 50},
            {"id": "selection-2", "registration_id": "registration-2", "partner_age": 48},
        ],
        participant_type="GENDER_DOUBLES",
    )

    assert preview["brackets"][0]["label"] == "One under 50 / one 50+"
    assert preview["brackets"][0]["count"] == 1
    assert preview["brackets"][0]["viable"] is True
    assert preview["unassigned_entries"][0]["assignment_issue"] == (
        "Team must include one player under 50 and one player 50+."
    )
    assert not any("below the minimum" in message.lower() for message in preview["recommendations"])
