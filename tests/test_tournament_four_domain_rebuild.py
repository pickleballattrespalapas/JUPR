from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_registration_repo import (
    analyze_registration_publish_impact,
    build_builder_draft_payload,
    publish_registration_configuration,
)
from jupr_app.domain.tournament_age_policy import (
    build_age_split_preview,
    evaluate_age_eligibility,
    normalize_age_policy,
)
from jupr_app.domain.tournament_registration_compiler import (
    evaluate_selection_gender_eligibility,
)
from jupr_app.services.admin_tournament_setup_service import (
    preview_admin_tournament_age_split,
)


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read_web(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def read_root(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class FakeQuery:
    def __init__(self, storage: dict[str, list[dict]], table_name: str):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None
        self.insert_payload = None
        self.upsert_payload = None
        self.update_payload = None
        self.delete_mode = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append(("eq", key, value))
        return self

    def neq(self, key, value):
        self.filters.append(("neq", key, value))
        return self

    def in_(self, key, values):
        self.filters.append(("in", key, {str(value) for value in values or []}))
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def upsert(self, payload, **_kwargs):
        self.upsert_payload = payload
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def delete(self):
        self.delete_mode = True
        return self

    def _scoped(self, rows):
        scoped = list(rows)
        for operation, key, expected in self.filters:
            if operation == "eq":
                scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
            elif operation == "neq":
                scoped = [row for row in scoped if str(row.get(key)) != str(expected)]
            elif operation == "in":
                scoped = [row for row in scoped if str(row.get(key)) in expected]
        return scoped

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        scoped = self._scoped(rows)
        if self.delete_mode:
            self.storage[self.table_name] = [row for row in rows if row not in scoped]
            return SimpleNamespace(data=[dict(row) for row in scoped], count=len(scoped))
        if self.update_payload is not None:
            updated = []
            for row in scoped:
                row.update(self.update_payload)
                updated.append(dict(row))
            return SimpleNamespace(data=updated, count=len(updated))
        if self.insert_payload is not None:
            payloads = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            out = []
            for payload in payloads:
                row = dict(payload)
                rows.append(row)
                out.append(dict(row))
            return SimpleNamespace(data=out, count=len(out))
        if self.upsert_payload is not None:
            payloads = self.upsert_payload if isinstance(self.upsert_payload, list) else [self.upsert_payload]
            out = []
            for payload in payloads:
                row = dict(payload)
                existing = next(
                    (
                        current
                        for current in rows
                        if str(current.get("id") or "") == str(row.get("id") or "")
                        or (
                            self.table_name == "tournament_registration_settings"
                            and str(current.get("tournament_id") or "")
                            == str(row.get("tournament_id") or "")
                        )
                    ),
                    None,
                )
                if existing is None:
                    rows.append(row)
                    out.append(dict(row))
                else:
                    existing.update(row)
                    out.append(dict(existing))
            return SimpleNamespace(data=out, count=len(out))
        if self.order_key:
            scoped = sorted(
                scoped,
                key=lambda row: str(row.get(self.order_key) or ""),
                reverse=self.order_desc,
            )
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in scoped], count=len(scoped))


class FakeSupabase:
    def __init__(self, storage: dict[str, list[dict]]):
        self.storage = storage

    def table(self, name: str):
        return FakeQuery(self.storage, name)


def base_storage() -> dict[str, list[dict]]:
    return {
        "tournaments": [
            {
                "id": "t1",
                "club_id": "club",
                "name": "Four Domain Classic",
                "status": "DRAFT",
                "start_date": "2026-10-01",
                "end_date": "2026-10-02",
            }
        ],
        "tournament_registration_settings": [
            {
                "id": "settings-1",
                "tournament_id": "t1",
                "registration_slug": "four-domain-classic",
                "registration_status": "draft",
            }
        ],
        "tournament_registration_days": [
            {
                "id": "day1",
                "tournament_id": "t1",
                "label": "Day 1",
                "event_date": "2026-10-01",
                "court_count": 10,
                "court_labels": [],
                "enabled": True,
                "sort_order": 1,
            }
        ],
        "tournament_event_options": [
            {
                "id": "event1",
                "tournament_id": "t1",
                "registration_day_id": "day1",
                "scheduled_day_ids": ["day1"],
                "event_family_label": "Gender Doubles",
                "division_name": "3.5 Open",
                "event_type": "GENDER_DOUBLES",
                "gender_restriction": "MEN",
                "skill_label": "3.5",
                "skill_mode": "SKILL_BRACKET",
                "age_label": "All Ages",
                "age_mode": "ALL_AGES",
                "age_rules": {"mode": "ALL_AGES"},
                "capacity_teams": 16,
                "price_usd": 40,
                "waitlist_enabled": True,
                "partner_board_enabled": True,
                "status": "open",
                "enabled": True,
                "sort_order": 1,
            }
        ],
        "tournament_registrations": [],
        "tournament_registration_selections": [],
        "tournament_event_draws": [],
        "tournament_teams": [],
        "tournament_games": [],
        "admin_activity_log": [],
    }


def test_four_domain_navigation_and_workspace_contract() -> None:
    nav = read_web("components/TournamentSetupWizardNav.tsx")
    phase_nav = read_web("components/TournamentPhaseNav.tsx")
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")

    for label in ("Tournament", "Competition", "Commerce", "Review"):
        assert f'label: "{label}"' in nav
    assert 'steps: ["basics", "schedule"]' in nav
    assert 'steps: ["events", "divisions"]' in nav
    assert "Domain {domainDefinition.number} of 4" in panel
    assert 'label: "Tournament"' in phase_nav
    assert 'label: "Competition"' in phase_nav
    assert 'label: "Commerce"' in phase_nav
    assert 'label: "Review"' in phase_nav


def test_venue_is_centralized_and_event_start_times_remain_outside_venue() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    builder = read_web("app/admin/tournament-setup/tournamentSetupBuilder.ts")

    venue_model = read_web("app/admin/tournaments/setup/TournamentVenueModel.ts")
    assert "Store the venue once, maintain a stable court inventory" in panel
    assert "Venue address" in panel
    assert "Directions to the venue (optional)" in panel
    assert "Total venue courts" in panel
    assert "Venue court inventory" in panel
    assert "Fixed tournament date" in panel
    assert "Which courts are available?" in panel
    assert "tournament-level court hours" in panel
    assert "court_open_time: null" in venue_model
    assert "court_close_time: null" in venue_model
    assert "available_court_ids" in venue_model
    assert "venue_courts_json" in venue_model
    assert "FACILITY_COURT_LIMIT = 100" in builder
    schedule = panel.split("function renderSchedule()", 1)[1].split("function renderReview()", 1)[0]
    assert "Courts open" not in schedule
    assert "Courts close" not in schedule


def test_event_policy_owns_four_player_team_and_age_policy() -> None:
    event_dialog = read_web("app/admin/tournaments/setup/TournamentSetupEventFamilyDialog.tsx")
    division_dialog = read_web("app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx")
    age_editor = read_web("app/admin/tournaments/setup/TournamentAgePolicyEditor.tsx")
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")

    assert '<option value="FOUR_PLAYER_TEAM">Four-player team</option>' in event_dialog
    assert "Four-player team rules" in event_dialog
    assert "TournamentAgePolicyEditor" in event_dialog
    assert "Event age policy" in event_dialog
    assert 'type EventStructure = "SINGLES" | "GENDER_DOUBLES" | "MIXED_DOUBLES" | "FOUR_PLAYER_TEAM"' in event_dialog
    eligibility_section = division_dialog.split("Division eligibility", 1)[1].split("Capacity", 1)[0]
    assert 'value="COMBINED_RATING_CAP"' in eligibility_section
    assert 'competition_format).toUpperCase() === "FOUR_PLAYER_TEAM"' in eligibility_section
    assert "Inherit from parent event" in division_dialog
    assert "Override for this division" in division_dialog
    assert "AUTO_AGE_SPLIT" in age_editor
    assert "Minimum entries per resulting bracket" in age_editor
    assert "Candidate age brackets" in age_editor
    assert "Team age rule" in age_editor
    assert "Underfilled bracket fallback" in age_editor
    assert "Preview age split" in panel
    assert "No rows were changed" in panel


def test_division_cards_resolve_inherited_defaults_and_commerce_has_presets() -> None:
    division_card = read_web("app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx")
    commerce = read_web("app/admin/tournaments/commerce/TournamentCommercePanel.tsx")

    assert "resolvedDraw" in division_card
    assert "resolvedScoring" in division_card
    assert '" (from event)"' in division_card
    assert '" (division override)"' in division_card
    assert "Add T-shirt sizes" in commerce
    assert "Add meal choices" in commerce
    assert "Add room choices" in commerce
    assert "addVariantPreset" in commerce


def test_review_auto_runs_compares_values_and_guards_forced_changes() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    repo = read_root("jupr_app/domain/tournament_registration_repo.py")
    service = read_root("jupr_app/services/admin_tournament_setup_service.py")

    assert "autoReviewSignatureRef" in panel
    assert "void reviewImpact()" in panel
    review_values = read_web("app/admin/tournaments/setup/TournamentReviewValue.tsx")
    assert "ReviewComparisonDisplay" in panel
    assert "Current published value" in review_values
    assert "Proposed draft value" in review_values
    assert "Technical details" in review_values
    assert 'scheduled_day_ids: "Tournament days"' in review_values
    assert "Force change with registration resolution" in panel
    assert "Manual registration-resolution queue" in panel
    assert "Open registration editor" in panel
    assert "Save Review actions" in panel
    assert "comparablePublishedStateSignature" in panel
    assert "affected_registrations" in repo
    assert "FORCE_CHANGE_WITH_RESOLUTION" in repo
    assert "allowed_block_ids" in repo
    assert "FORCED_RESOLUTION_ACTIONS" in service
    assert "Publish remains blocked" in service


def test_age_preview_contract_is_read_only_and_registered_in_staging() -> None:
    routes = read_root("services/api/admin_tournament_setup_routes.py")
    waves = read_root("scripts/staging_write_waves.py")

    path = "/admin/clubs/{club_id}/tournaments/setup/tournaments/{tournament_id}/age-split-preview"
    assert path in routes
    assert "TournamentAgeSplitPreviewRequest" in routes
    assert "preview_admin_tournament_age_split" in routes
    assert f'("POST", "{path}")' in waves


def test_builder_draft_preserves_published_event_family_baseline() -> None:
    payload = build_builder_draft_payload(
        days=[{"id": "day1"}],
        event_families=[{"event_family": "New draft"}],
        divisions=[{"id": "event1"}],
        published_event_families=[{"event_family": "Published"}],
        published_at="2026-08-04T00:00:00Z",
    )

    assert payload["version"] == 3
    assert payload["event_families"] == [{"event_family": "New draft"}]
    assert payload["published_event_families"] == [{"event_family": "Published"}]
    assert payload["published_at"] == "2026-08-04T00:00:00Z"


def test_age_policy_preview_groups_entries_without_writes(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    storage = base_storage()
    storage["tournament_registrations"] = [
        {"id": "r1", "tournament_id": "t1", "display_name": "Young Team", "email": "young@example.com", "age": 45, "submitted_at": "2026-01-01"},
        {"id": "r2", "tournament_id": "t1", "display_name": "Fifties Team", "email": "fifty@example.com", "age": 54, "submitted_at": "2026-01-02"},
        {"id": "r3", "tournament_id": "t1", "display_name": "Sixties Team", "email": "sixty@example.com", "age": 66, "submitted_at": "2026-01-03"},
    ]
    storage["tournament_registration_selections"] = [
        {"id": "s1", "tournament_id": "t1", "registration_id": "r1", "registration_day_id": "day1", "event_option_id": "event1", "partner_age": 48},
        {"id": "s2", "tournament_id": "t1", "registration_id": "r2", "registration_day_id": "day1", "event_option_id": "event1", "partner_age": 57},
        {"id": "s3", "tournament_id": "t1", "registration_id": "r3", "registration_day_id": "day1", "event_option_id": "event1", "partner_age": 63},
    ]
    supabase = FakeSupabase(storage)
    before = deepcopy(storage)

    result = build_age_split_preview(
        policy={
            "mode": "AUTO_AGE_SPLIT",
            "min_teams_per_age_group": 1,
            "team_age_rule": "YOUNGER",
            "merge_strategy": "CLOSEST",
            "brackets": [
                {"id": "under-50", "label": "Under 50", "min_age": None, "max_age": 49},
                {"id": "50-59", "label": "50-59", "min_age": 50, "max_age": 59},
                {"id": "60-plus", "label": "60+", "min_age": 60, "max_age": None},
            ],
        },
        registrations={row["id"]: row for row in storage["tournament_registrations"]},
        selections=storage["tournament_registration_selections"],
        participant_type="GENDER_DOUBLES",
    )

    assert [row["count"] for row in result["brackets"]] == [1, 1, 1]
    assert all(row["viable"] for row in result["brackets"])
    assert result["pending_entries"] == []
    assert result["unassigned_entries"] == []
    assert storage == before


def test_age_preview_provisionally_groups_needs_partner_then_regroups() -> None:
    policy = {
        "mode": "AUTO_AGE_SPLIT",
        "min_teams_per_age_group": 1,
        "team_age_rule": "YOUNGER",
        "brackets": [
            {"id": "under-50", "label": "Under 50", "max_age": 49},
            {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
            {"id": "65-plus", "label": "65+", "min_age": 65},
        ],
    }
    registrations = {
        "r1": {"id": "r1", "display_name": "Pending Partner", "age": 67},
    }

    provisional = build_age_split_preview(
        policy=policy,
        registrations=registrations,
        selections=[
            {
                "id": "s1",
                "registration_id": "r1",
                "partner_mode": "NEEDS_PARTNER",
                "partner_age": 44,
            },
        ],
        participant_type="GENDER_DOUBLES",
    )
    provisional_entry = provisional["brackets"][2]["entries"][0]
    assert provisional_entry["preferred_age_group"] == "65+"
    assert provisional_entry["effective_age"] == 67
    assert provisional_entry["partner_age"] is None
    assert provisional_entry["pending_partner"] is True
    assert provisional_entry["provisional"] is True
    assert provisional["brackets"][2]["provisional_count"] == 1
    assert provisional["pending_entries"] == []
    assert provisional["unassigned_entries"] == []

    paired = build_age_split_preview(
        policy=policy,
        registrations=registrations,
        selections=[
            {
                "id": "s1",
                "registration_id": "r1",
                "partner_mode": "HAS_PARTNER",
                "partner_age": 44,
            },
        ],
        participant_type="GENDER_DOUBLES",
    )
    paired_entry = paired["brackets"][0]["entries"][0]
    assert paired_entry["preferred_age_group"] == "Under 50"
    assert paired_entry["effective_age"] == 44
    assert paired_entry["pending_partner"] is False
    assert paired_entry["provisional"] is False

    split_age = build_age_split_preview(
        policy={"mode": "SPLIT_AGE", "split_age_threshold": 50},
        registrations=registrations,
        selections=[
            {"id": "s1", "registration_id": "r1", "partner_mode": "NEEDS_PARTNER"},
        ],
        participant_type="GENDER_DOUBLES",
    )
    assert split_age["brackets"][0]["entries"] == []
    assert split_age["pending_entries"][0]["registration_id"] == "r1"
    assert split_age["pending_entries"][0]["provisional"] is True
    assert split_age["unassigned_entries"] == []


def test_pending_partner_age_metadata_requires_a_successful_provisional_result() -> None:
    auto_policy = {
        "mode": "AUTO_AGE_SPLIT",
        "brackets": [
            {"id": "under-50", "label": "Under 50", "max_age": 49},
            {"id": "50-plus", "label": "50+", "min_age": 50},
        ],
    }
    missing_player = evaluate_age_eligibility(
        policy=auto_policy,
        player_age=None,
        partner_age=None,
        participant_type="GENDER_DOUBLES",
        allow_missing_partner_for_preview=True,
    )
    assert missing_player["status"] == "MISSING_DATA"
    assert missing_player["provisional"] is False
    assert missing_player["recompute_when_partner_assigned"] is False

    for team_age_rule in ("OLDER", "AVERAGE"):
        pending = evaluate_age_eligibility(
            policy={
                "mode": "FIXED_AGE_BRACKET",
                "label": "50+",
                "min_age": 50,
                "team_age_rule": team_age_rule,
            },
            player_age=44,
            partner_age=None,
            participant_type="GENDER_DOUBLES",
            allow_missing_partner_for_preview=True,
        )
        assert pending["status"] == "ELIGIBLE"
        assert pending["preferred_age_group"] is None
        assert pending["effective_age"] == 44
        assert pending["provisional"] is True

        fixed_preview = build_age_split_preview(
            policy={
                "mode": "FIXED_AGE_BRACKET",
                "label": "50+",
                "min_age": 50,
                "team_age_rule": team_age_rule,
                "min_teams_per_age_group": 1,
            },
            registrations={
                "r1": {"id": "r1", "display_name": "Pending Partner", "age": 44},
            },
            selections=[
                {"id": "s1", "registration_id": "r1", "partner_mode": "NEEDS_PARTNER"},
            ],
            participant_type="GENDER_DOUBLES",
        )
        assert fixed_preview["brackets"][0]["count"] == 0
        assert fixed_preview["pending_entries"][0]["registration_id"] == "r1"
        assert fixed_preview["pending_entries"][0]["provisional"] is True
        assert fixed_preview["unassigned_entries"] == []
        assert not any("Resolve" in row for row in fixed_preview["recommendations"])

        multi_bracket_preview = build_age_split_preview(
            policy={
                "mode": "AUTO_AGE_SPLIT",
                "team_age_rule": team_age_rule,
                "min_teams_per_age_group": 1,
                "brackets": [
                    {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
                    {"id": "65-plus", "label": "65+", "min_age": 65},
                ],
            },
            registrations={
                "r1": {"id": "r1", "display_name": "Pending Partner", "age": 44},
            },
            selections=[
                {"id": "s1", "registration_id": "r1", "partner_mode": "NEEDS_PARTNER"},
            ],
            participant_type="GENDER_DOUBLES",
        )
        assert [row["count"] for row in multi_bracket_preview["brackets"]] == [0, 0]
        assert multi_bracket_preview["pending_entries"][0]["registration_id"] == "r1"
        assert multi_bracket_preview["unassigned_entries"] == []
        assert not any("Resolve" in row for row in multi_bracket_preview["recommendations"])

    deterministic_failure = evaluate_age_eligibility(
        policy={
            "mode": "FIXED_AGE_BRACKET",
            "label": "50+",
            "min_age": 50,
            "team_age_rule": "YOUNGER",
        },
        player_age=44,
        partner_age=None,
        participant_type="GENDER_DOUBLES",
        allow_missing_partner_for_preview=True,
    )
    assert deterministic_failure["status"] == "INELIGIBLE"
    assert deterministic_failure["provisional"] is False


def test_pending_partner_gender_is_provisional_only_when_player_data_qualifies() -> None:
    pending_selection = {"partner_mode": "NEEDS_PARTNER"}

    men = evaluate_selection_gender_eligibility(
        event={"event_type": "GENDER_DOUBLES", "gender_restriction": "MEN"},
        selection=pending_selection,
        player={"gender": "Men"},
        allow_missing_partner_for_preview=True,
    )
    assert men["status"] == "ELIGIBLE"
    assert men["pending_partner"] is True
    assert men["provisional"] is True
    assert men["recompute_when_partner_assigned"] is True

    mixed = evaluate_selection_gender_eligibility(
        event={"event_type": "MIXED_DOUBLES", "gender_restriction": "MIXED"},
        selection=pending_selection,
        player={"gender": "Women"},
        allow_missing_partner_for_preview=True,
    )
    assert mixed["status"] == "ELIGIBLE"
    assert mixed["provisional"] is True

    missing_player = evaluate_selection_gender_eligibility(
        event={"event_type": "GENDER_DOUBLES", "gender_restriction": "MEN"},
        selection=pending_selection,
        player={"gender": None},
        allow_missing_partner_for_preview=True,
    )
    assert missing_player["status"] == "MISSING_DATA"
    assert missing_player["provisional"] is False

    wrong_player = evaluate_selection_gender_eligibility(
        event={"event_type": "GENDER_DOUBLES", "gender_restriction": "MEN"},
        selection=pending_selection,
        player={"gender": "Women"},
        allow_missing_partner_for_preview=True,
    )
    assert wrong_player["status"] == "INELIGIBLE"
    assert wrong_player["provisional"] is False

    paired_missing = evaluate_selection_gender_eligibility(
        event={"event_type": "GENDER_DOUBLES", "gender_restriction": "MEN"},
        selection={"partner_mode": "HAS_PARTNER"},
        player={"gender": "Men"},
        allow_missing_partner_for_preview=True,
    )
    assert paired_missing["status"] == "MISSING_DATA"
    assert paired_missing["provisional"] is False

    paired_wrong = evaluate_selection_gender_eligibility(
        event={"event_type": "GENDER_DOUBLES", "gender_restriction": "MEN"},
        selection={"partner_mode": "HAS_PARTNER"},
        player={"gender": "Men"},
        partner={"gender": "Women"},
        allow_missing_partner_for_preview=True,
    )
    assert paired_wrong["status"] == "INELIGIBLE"
    assert paired_wrong["provisional"] is False


def test_split_age_partner_rule_requires_one_under_and_one_at_or_above() -> None:
    policy = {
        "mode": "SPLIT_AGE",
        "split_age_threshold": 50,
        "min_teams_per_age_group": 1,
    }
    registrations = {
        "r1": {"id": "r1", "display_name": "Valid Team", "age": 49},
        "r2": {"id": "r2", "display_name": "Both Under", "age": 49},
        "r3": {"id": "r3", "display_name": "Both Over", "age": 50},
    }
    selections = [
        {"id": "s1", "registration_id": "r1", "partner_age": 50},
        {"id": "s2", "registration_id": "r2", "partner_age": 48},
        {"id": "s3", "registration_id": "r3", "partner_age": 55},
    ]

    result = build_age_split_preview(
        policy=policy,
        registrations=registrations,
        selections=selections,
        participant_type="GENDER_DOUBLES",
    )

    assert result["policy"]["brackets"] == [
        {
            "id": "split-age-50",
            "label": "One under 50 / one 50+",
            "min_age": None,
            "max_age": None,
        }
    ]
    assert result["brackets"][0]["count"] == 1
    assert result["brackets"][0]["entries"][0]["registration_id"] == "r1"
    assert {row["registration_id"] for row in result["unassigned_entries"]} == {"r2", "r3"}
    assert all("one player under 50" in row["assignment_issue"] for row in result["unassigned_entries"])

    with pytest.raises(ValueError, match="only for doubles and team events"):
        build_age_split_preview(
            policy=policy,
            registrations=registrations,
            selections=selections,
            participant_type="SINGLES",
        )


def test_age_policy_validation_rejects_overlap_and_bad_minimum() -> None:
    with pytest.raises(ValueError, match="[Mm]inimum"):
        normalize_age_policy(
            {
                "mode": "AUTO_AGE_SPLIT",
                "min_teams_per_age_group": 0,
                "brackets": [
                    {"label": "Under 50", "max_age": 49},
                    {"label": "50+", "min_age": 50},
                ],
            }
        )

    with pytest.raises(ValueError, match="overlap"):
        normalize_age_policy(
            {
                "mode": "AUTO_AGE_SPLIT",
                "min_teams_per_age_group": 4,
                "brackets": [
                    {"label": "Under 60", "min_age": 1, "max_age": 59},
                    {"label": "50+", "min_age": 50, "max_age": None},
                ],
            }
        )



def test_review_reverts_only_blocked_fields_and_requires_explicit_completion() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")

    revert_block = panel.split("function keepPublishedValueForBlockedChange", 1)[1].split("function forcedResolutionPlans", 1)[0]
    assert 'item.field === "registration_day_id"' in revert_block
    assert 'item.field === "scheduled_day_ids"' in revert_block
    assert 'item.field === "event_type"' in revert_block
    assert 'item.field === "gender_restriction"' in revert_block
    assert 'item.field === "skill_age_rules"' in revert_block
    assert 'item.field === "capacity_teams"' in revert_block
    assert "Other draft changes were preserved" in revert_block
    assert "nextValue: SetupRecord = { ...currentRow.value }" in revert_block
    assert "publishedFamilyName" in revert_block
    assert "restoredParentRelationship" in revert_block
    assert "nextValue.event_family_label = publishedFamilyName" in revert_block
    assert "mergedFamilyDays" in revert_block
    assert "restored the published parent Event" in revert_block
    assert "persistConfigurationDraft(" in revert_block

    update_block = panel.split("function updateForcedRegistration", 1)[1].split("function forcedResolutionComplete", 1)[0]
    assert "next.resolved = false" in update_block
    assert 'Object.prototype.hasOwnProperty.call(patch, "resolved")' in update_block
    assert 'const noteRequired = action === "OTHER"' in update_block
    assert "I completed and verified this registration action" in panel
    assert "Schedule change — no registration conflict" in panel
    assert "Age-grouping change — no registration conflict" in panel
    assert "Provisional age group:" in panel
    assert "recalculated when a partner is assigned." in panel
    assert "effectiveAgeValue == null" in panel
    assert "Entries awaiting partner-based placement" in panel
    assert "Required eligibility information" in panel
    assert "Complete the required eligibility information before acknowledging this impact" in panel
    assert "I completed and verified this communication action" in panel
    assert "communication_change_acknowledgements" in panel
    service = read_root("jupr_app/services/admin_tournament_setup_service.py")
    assert "def _communication_acknowledgement_summary" in service
    assert '"NOTIFY_AFFECTED"' in service
    assert '"ACKNOWLEDGE_NO_NOTICE"' in service
    assert "communication_change_acknowledgements" in service


def test_age_preview_uses_only_canonical_tournament_event_ids() -> None:
    service = read_root("jupr_app/services/admin_tournament_setup_service.py")
    preview = service.split("def preview_admin_tournament_age_split", 1)[1].split("def _forced_resolution_summary", 1)[0]

    assert "canonical_event_ids" in preview
    assert "list_event_options(supabase, str(tournament_id))" in preview
    assert "list_partner_team_links(supabase, str(tournament_id))" in preview
    assert "linked_partner_registration_lookup(" in preview
    assert "in canonical_event_ids" in preview
    assert 'str(row.get("event_option_id") or "").strip() in event_ids' in preview


def test_admin_age_preview_uses_canonical_partner_link_without_writes(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    storage = base_storage()
    storage["tournament_registrations"] = [
        {"id": "r1", "tournament_id": "t1", "display_name": "Older", "age": 67},
        {"id": "r2", "tournament_id": "t1", "display_name": "Younger", "age": 44},
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "s1",
            "tournament_id": "t1",
            "registration_id": "r1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "NEEDS_PARTNER",
        },
        {
            "id": "s2",
            "tournament_id": "t1",
            "registration_id": "r2",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "NEEDS_PARTNER",
        },
    ]
    storage["tournament_registration_team_links"] = [
        {
            "id": "link-1",
            "tournament_id": "t1",
            "event_option_id": "event1",
            "selection1_id": "s1",
            "selection2_id": "s2",
            "registration1_id": "r1",
            "registration2_id": "r2",
            "status": "CONFIRMED",
        }
    ]
    before = deepcopy(storage)

    result = preview_admin_tournament_age_split(
        FakeSupabase(storage),
        club_id="club",
        tournament_id="t1",
        event_family="Gender Doubles",
        participant_type="GENDER_DOUBLES",
        policy={
            "mode": "AUTO_AGE_SPLIT",
            "min_teams_per_age_group": 1,
            "team_age_rule": "YOUNGER",
            "brackets": [
                {"id": "under-50", "label": "Under 50", "max_age": 49},
                {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
                {"id": "65-plus", "label": "65+", "min_age": 65},
            ],
        },
        event_options=storage["tournament_event_options"],
    )

    assert result["write_count"] == 0
    assert result["total_entries"] == 1
    assert result["unassigned_entries"] == []
    assert [entry["effective_age"] for entry in result["brackets"][0]["entries"]] == [44]
    assert storage == before


def test_auto_review_does_not_loop_after_an_error() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    review = panel.split("async function reviewImpact()", 1)[1].split("async function publishSetup", 1)[0]
    catch_block = review.split("} catch (error) {", 1)[1].split("} finally {", 1)[0]

    assert "autoReviewSignatureRef.current = """ not in catch_block
    assert "Use Refresh review to try again" in catch_block


def test_exhaustive_auto_age_split_is_communication_impact_with_targeted_missing_data() -> None:
    storage = base_storage()
    storage["tournament_registrations"] = [
        {
            "id": "r1",
            "tournament_id": "t1",
            "display_name": "Liam Chen",
            "email": "liam@example.com",
            "status": "confirmed",
            "age": 34,
        },
        {
            "id": "r2",
            "tournament_id": "t1",
            "display_name": "Caleb Nguyen",
            "email": "caleb@example.com",
            "status": "confirmed",
            "age": 61,
        },
        {
            "id": "r3",
            "tournament_id": "t1",
            "display_name": "Samuel Patel",
            "email": "samuel@example.com",
            "status": "confirmed",
            "age": 44,
            "gender": "Men",
            "doubles_skill": 3.2,
        },
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "s1",
            "tournament_id": "t1",
            "registration_id": "r1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_age": 29,
        },
        {
            "id": "s2",
            "tournament_id": "t1",
            "registration_id": "r2",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_age": 48,
        },
        {
            "id": "s3",
            "tournament_id": "t1",
            "registration_id": "r3",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "HAS_PARTNER",
            "partner_name": "Known Partner",
            "partner_gender": "Men",
            "partner_skill": 3.2,
            "partner_age": None,
        },
    ]
    supabase = FakeSupabase(storage)
    draft_event = {
        **storage["tournament_event_options"][0],
        "age_mode": "AUTO_AGE_SPLIT",
        "age_label": "All Ages",
        "age_rules": {
            "mode": "AUTO_AGE_SPLIT",
            "team_age_rule": "YOUNGER",
            "merge_strategy": "CLOSEST",
            "min_teams_per_age_group": 4,
            "brackets": [
                {"id": "under-50", "label": "Under 50", "max_age": 49},
                {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
                {"id": "65-plus", "label": "65+", "min_age": 65},
            ],
        },
    }

    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[draft_event],
    )

    assert not [row for row in impact["blocked_details"] if row["field"] == "skill_age_rules"]
    assert impact["summary"]["communication_impacts"] == 1
    detail = impact["communication_impact_details"][0]
    assert detail["impact_type"] == "AGE_GROUPING_COMMUNICATION"
    assert detail["requires_data_completion"] is True
    assert [row["registration_id"] for row in detail["data_completion_registrations"]] == ["r3"]
    proposed_by_registration = {
        row["registration_id"]: row["proposed_value"]
        for row in detail["affected_registrations"]
    }
    assert proposed_by_registration["r1"]["age_label"] == "Under 50"
    assert proposed_by_registration["r1"]["effective_age"] == 29
    assert proposed_by_registration["r2"]["age_label"] == "Under 50"
    assert proposed_by_registration["r2"]["effective_age"] == 48
    assert proposed_by_registration["r3"]["age_label"] == "Needs age information"
    assert proposed_by_registration["r3"]["assignment_issue_type"] == "MISSING_AGE_DATA"


def test_age_policy_that_excludes_an_existing_team_remains_hard_conflict() -> None:
    storage = base_storage()
    storage["tournament_registrations"] = [
        {
            "id": "r1",
            "tournament_id": "t1",
            "display_name": "Under Age Team",
            "status": "confirmed",
            "age": 34,
        }
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "s1",
            "tournament_id": "t1",
            "registration_id": "r1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_age": 29,
        }
    ]
    supabase = FakeSupabase(storage)
    draft_event = {
        **storage["tournament_event_options"][0],
        "age_mode": "FIXED_AGE_BRACKET",
        "age_label": "50–64",
        "age_rules": {
            "mode": "FIXED_AGE_BRACKET",
            "label": "50–64",
            "min_age": 50,
            "max_age": 64,
            "team_age_rule": "YOUNGER",
        },
    }

    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[draft_event],
    )

    detail = next(row for row in impact["blocked_details"] if row["field"] == "skill_age_rules")
    assert [row["registration_id"] for row in detail["affected_registrations"]] == ["r1"]
    assert detail["affected_registrations"][0]["proposed_value"]["assignment_issue_type"] == "AGE_NOT_ELIGIBLE"
    assert impact["communication_impact_details"] == []


def test_registration_only_schedule_change_is_communication_impact() -> None:
    storage = base_storage()
    storage["tournament_registration_days"].append(
        {
            "id": "day2",
            "tournament_id": "t1",
            "label": "Day 2",
            "event_date": "2026-10-02",
            "court_count": 10,
            "court_labels": [],
            "enabled": True,
            "sort_order": 2,
        }
    )
    storage["tournament_event_options"][0]["scheduled_day_ids"] = ["day1", "day2"]
    storage["tournament_registrations"] = [
        {
            "id": "r1",
            "tournament_id": "t1",
            "display_name": "Alex Player",
            "email": "alex@example.com",
            "status": "confirmed",
            "submitted_at": "2026-01-01",
        }
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "s1",
            "tournament_id": "t1",
            "registration_id": "r1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
        }
    ]
    supabase = FakeSupabase(storage)
    draft_event = {
        **storage["tournament_event_options"][0],
        "scheduled_day_ids": ["day1"],
        "registration_day_id": "day1",
    }

    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[draft_event],
    )

    assert not [row for row in impact["blocked_details"] if row["field"] == "scheduled_day_ids"]
    assert impact["summary"]["communication_impacts"] == 1
    detail = impact["communication_impact_details"][0]
    assert detail["impact_type"] == "SCHEDULE_COMMUNICATION"
    assert detail["current_value"] == ["day1", "day2"]
    assert detail["proposed_value"] == ["day1"]
    assert detail["affected_registrations"][0]["registration_id"] == "r1"
    assert detail["resolution_options"] == [
        "KEEP_PUBLISHED_VALUE",
        "NOTIFY_AFFECTED",
        "ACKNOWLEDGE_NO_NOTICE",
    ]





def test_schedule_change_that_removes_selected_day_is_hard_conflict() -> None:
    storage = base_storage()
    storage["tournament_registration_days"].append(
        {
            "id": "day2",
            "tournament_id": "t1",
            "label": "Day 2",
            "event_date": "2026-10-02",
            "court_count": 10,
            "court_labels": [],
            "enabled": True,
            "sort_order": 2,
        }
    )
    storage["tournament_event_options"][0]["scheduled_day_ids"] = ["day1", "day2"]
    storage["tournament_registrations"] = [
        {"id": "r1", "tournament_id": "t1", "display_name": "Day Two Player", "status": "confirmed"}
    ]
    storage["tournament_registration_selections"] = [
        {"id": "s1", "tournament_id": "t1", "registration_id": "r1", "registration_day_id": "day2", "event_option_id": "event1"}
    ]
    supabase = FakeSupabase(storage)
    draft_event = {
        **storage["tournament_event_options"][0],
        "scheduled_day_ids": ["day1"],
        "registration_day_id": "day1",
    }

    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[draft_event],
    )

    schedule_block = next(row for row in impact["blocked_details"] if row["field"] == "scheduled_day_ids")
    assert schedule_block["forceable"] is True
    assert "selected by existing registrations" in schedule_block["message"]
    assert [row["registration_id"] for row in schedule_block["affected_registrations"]] == ["r1"]
    assert impact["communication_impact_details"] == []


def test_operational_schedule_change_remains_hard_blocked() -> None:
    storage = base_storage()
    storage["tournament_registration_days"].append(
        {
            "id": "day2",
            "tournament_id": "t1",
            "label": "Day 2",
            "event_date": "2026-10-02",
            "court_count": 10,
            "court_labels": [],
            "enabled": True,
            "sort_order": 2,
        }
    )
    storage["tournament_event_options"][0]["scheduled_day_ids"] = ["day1", "day2"]
    storage["tournament_registrations"] = [
        {"id": "r1", "tournament_id": "t1", "display_name": "Alex Player", "status": "confirmed"}
    ]
    storage["tournament_registration_selections"] = [
        {"id": "s1", "tournament_id": "t1", "registration_id": "r1", "registration_day_id": "day1", "event_option_id": "event1"}
    ]
    storage["tournament_event_draws"] = [
        {"id": "draw1", "tournament_id": "t1", "event_option_id": "event1", "registration_day_id": "day1"}
    ]
    supabase = FakeSupabase(storage)
    draft_event = {
        **storage["tournament_event_options"][0],
        "scheduled_day_ids": ["day1"],
        "registration_day_id": "day1",
    }

    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[draft_event],
    )

    schedule_block = next(row for row in impact["blocked_details"] if row["field"] == "scheduled_day_ids")
    assert schedule_block["forceable"] is False
    assert "after draws, teams, or games exist" in schedule_block["message"]
    assert impact["communication_impact_details"] == []


def test_skill_ceiling_expansion_preserves_registrations_and_true_play_down_conflict_is_forceable() -> None:
    storage = base_storage()
    storage["tournament_registrations"] = [
        {
            "id": "r1",
            "tournament_id": "t1",
            "display_name": "Alex Player",
            "email": "alex@example.com",
            "status": "confirmed",
            "submitted_at": "2026-01-01",
            "doubles_skill": 3.8,
            "gender": "Men",
            "age": 35,
        }
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "s1",
            "tournament_id": "t1",
            "registration_id": "r1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "NEEDS_PARTNER",
        }
    ]
    supabase = FakeSupabase(storage)

    expanded = {**storage["tournament_event_options"][0], "skill_label": "4.0"}
    impact = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[expanded],
    )
    assert not [row for row in impact["blocked_details"] if row["field"] == "skill_age_rules"]

    storage["tournament_event_options"][0]["skill_label"] = "4.0"
    storage["tournament_registrations"][0]["doubles_skill"] = 4.2
    lowered = {**storage["tournament_event_options"][0], "skill_label": "3.5"}
    impact_lowered = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[lowered],
    )
    detail = next(row for row in impact_lowered["blocked_details"] if row["field"] == "skill_age_rules")
    assert detail["forceable"] is True
    assert [row["registration_id"] for row in detail["affected_registrations"]] == ["r1"]
    assert "FORCE_CHANGE_WITH_RESOLUTION" in detail["resolution_options"]

    storage["tournament_event_draws"] = [
        {"id": "draw1", "tournament_id": "t1", "event_option_id": "event1", "registration_day_id": "day1"}
    ]
    impact_with_draw = analyze_registration_publish_impact(
        supabase,
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[lowered],
    )
    blocked_with_draw = next(row for row in impact_with_draw["blocked_details"] if row["field"] == "skill_age_rules")
    assert blocked_with_draw["forceable"] is False
    assert "FORCE_CHANGE_WITH_RESOLUTION" not in blocked_with_draw["resolution_options"]



def test_directional_age_groups_prefer_oldest_eligible_group_and_allow_older_players_down() -> None:
    policy = {
        "mode": "AUTO_AGE_SPLIT",
        "team_age_rule": "YOUNGER",
        "merge_strategy": "CLOSEST",
        "min_teams_per_age_group": 1,
        "brackets": [
            {"id": "under-50", "label": "Under 50", "max_age": 49},
            {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
            {"id": "65-plus", "label": "65+", "min_age": 65},
        ],
    }
    registrations = {
        "r65": {"id": "r65", "display_name": "Older Team", "age": 67},
        "r50": {"id": "r50", "display_name": "Middle Team", "age": 67},
        "ru50": {"id": "ru50", "display_name": "Younger Team", "age": 67},
    }
    selections = [
        {"id": "s65", "registration_id": "r65", "partner_age": 68},
        {"id": "s50", "registration_id": "r50", "partner_age": 54},
        {"id": "su50", "registration_id": "ru50", "partner_age": 45},
    ]

    preview = build_age_split_preview(
        policy=policy,
        registrations=registrations,
        selections=selections,
        participant_type="GENDER_DOUBLES",
    )

    by_group = {
        row["label"]: [entry["selection_id"] for entry in row["entries"]]
        for row in preview["brackets"]
    }
    assert by_group == {
        "Under 50": ["su50"],
        "50–64": ["s50"],
        "65+": ["s65"],
    }
    assert preview["unassigned_entries"] == []


def test_fixed_age_group_uses_minimum_age_not_closed_maximum() -> None:
    policy = {
        "mode": "FIXED_AGE_BRACKET",
        "label": "50–64",
        "min_age": 50,
        "max_age": 64,
        "team_age_rule": "YOUNGER",
    }
    registrations = {
        "older": {"id": "older", "display_name": "Older", "age": 67},
        "younger": {"id": "younger", "display_name": "Younger", "age": 49},
    }
    selections = [
        {"id": "older-selection", "registration_id": "older", "partner_age": 68},
        {"id": "younger-selection", "registration_id": "younger", "partner_age": 70},
    ]

    preview = build_age_split_preview(
        policy=policy,
        registrations=registrations,
        selections=selections,
        participant_type="GENDER_DOUBLES",
    )

    assert [row["selection_id"] for row in preview["brackets"][0]["entries"]] == ["older-selection"]
    assert preview["unassigned_entries"][0]["selection_id"] == "younger-selection"
    assert preview["unassigned_entries"][0]["assignment_issue_type"] == "AGE_NOT_ELIGIBLE"


def test_gender_change_blocks_only_registrations_outside_proposed_restriction() -> None:
    storage = base_storage()
    storage["tournament_event_options"][0]["gender_restriction"] = "ANY"
    storage["tournament_registrations"] = [
        {"id": "man", "tournament_id": "t1", "display_name": "Man", "status": "confirmed", "gender": "Men"},
        {"id": "woman", "tournament_id": "t1", "display_name": "Woman", "status": "confirmed", "gender": "Women"},
    ]
    storage["tournament_registration_selections"] = [
        {"id": "sm", "tournament_id": "t1", "registration_id": "man", "registration_day_id": "day1", "event_option_id": "event1", "partner_mode": "NEEDS_PARTNER"},
        {"id": "sw", "tournament_id": "t1", "registration_id": "woman", "registration_day_id": "day1", "event_option_id": "event1", "partner_mode": "NEEDS_PARTNER"},
    ]
    proposed = {**storage["tournament_event_options"][0], "gender_restriction": "WOMEN"}

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    detail = next(row for row in impact["blocked_details"] if row["field"] == "gender_restriction")
    assert [row["registration_id"] for row in detail["affected_registrations"]] == ["man"]
    assert detail["affected_registrations"][0]["proposed_value"]["gender_eligibility"]["status"] == "INELIGIBLE"


def test_skill_ceiling_change_blocks_only_team_that_is_playing_down() -> None:
    storage = base_storage()
    storage["tournament_event_options"][0]["skill_label"] = "4.0"
    storage["tournament_registrations"] = [
        {"id": "valid", "tournament_id": "t1", "display_name": "Valid", "status": "confirmed", "doubles_skill": 3.8},
        {"id": "invalid", "tournament_id": "t1", "display_name": "Invalid", "status": "confirmed", "doubles_skill": 3.8},
    ]
    storage["tournament_registration_selections"] = [
        {"id": "sv", "tournament_id": "t1", "registration_id": "valid", "registration_day_id": "day1", "event_option_id": "event1", "partner_mode": "HAS_PARTNER", "partner_skill": 3.9},
        {"id": "si", "tournament_id": "t1", "registration_id": "invalid", "registration_day_id": "day1", "event_option_id": "event1", "partner_mode": "HAS_PARTNER", "partner_skill": 4.05},
    ]
    proposed = {**storage["tournament_event_options"][0], "skill_label": "3.5"}

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    detail = next(row for row in impact["blocked_details"] if row["field"] == "skill_age_rules")
    assert [row["registration_id"] for row in detail["affected_registrations"]] == ["invalid"]
    proposed_value = detail["affected_registrations"][0]["proposed_value"]
    assert proposed_value["skill_eligibility"]["controlling_rating"] == 4.05
    assert proposed_value["skill_eligibility"]["skill_ceiling_exclusive"] == 4.0


def test_cosmetic_skill_metadata_does_not_turn_age_grouping_into_skill_conflict() -> None:
    storage = base_storage()
    storage["tournament_event_options"][0].update({"skill_label": "3.5+", "skill_mode": "minimum"})
    storage["tournament_registrations"] = [
        {"id": "r1", "tournament_id": "t1", "display_name": "Eligible Team", "status": "confirmed", "age": 67, "doubles_skill": 3.7},
    ]
    storage["tournament_registration_selections"] = [
        {"id": "s1", "tournament_id": "t1", "registration_id": "r1", "registration_day_id": "day1", "event_option_id": "event1", "partner_mode": "HAS_PARTNER", "partner_age": 68, "partner_skill": 3.8},
    ]
    proposed = {
        **storage["tournament_event_options"][0],
        "skill_label": "3.5",
        "skill_mode": "SKILL_BRACKET",
        "eligibility_mode": "STANDARD",
        "age_mode": "AUTO_AGE_SPLIT",
        "age_label": "All Ages",
        "age_rules": {
            "mode": "AUTO_AGE_SPLIT",
            "team_age_rule": "YOUNGER",
            "merge_strategy": "CLOSEST",
            "min_teams_per_age_group": 1,
            "brackets": [
                {"id": "under-50", "label": "Under 50", "max_age": 49},
                {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
                {"id": "65-plus", "label": "65+", "min_age": 65},
            ],
        },
    }

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    assert not [row for row in impact["blocked_details"] if row["field"] == "skill_age_rules"]
    detail = next(row for row in impact["communication_impact_details"] if row["impact_type"] == "AGE_GROUPING_COMMUNICATION")
    assert detail["affected_registrations"][0]["proposed_value"]["age_label"] == "65+"


def test_draw_and_scoring_changes_do_not_create_registration_eligibility_conflicts() -> None:
    storage = base_storage()
    storage["tournament_registrations"] = [
        {"id": "r1", "tournament_id": "t1", "display_name": "Player", "status": "confirmed"},
    ]
    storage["tournament_registration_selections"] = [
        {"id": "s1", "tournament_id": "t1", "registration_id": "r1", "registration_day_id": "day1", "event_option_id": "event1"},
    ]
    proposed = {
        **storage["tournament_event_options"][0],
        "event_format_default": "DOUBLE_ELIM",
        "scoring_default": "GAME_TO_21",
    }

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    assert impact["blocked_details"] == []
    assert impact["communication_impact_details"] == []


def test_review_resolves_linked_partner_profile_for_directional_eligibility() -> None:
    storage = base_storage()
    storage["tournament_event_options"][0].update(
        {
            "skill_label": "4.0",
            "gender_restriction": "ANY",
            "age_mode": "ALL_AGES",
            "age_label": "All Ages",
            "age_rules": {"mode": "ALL_AGES"},
        }
    )
    storage["tournament_registrations"] = [
        {
            "id": "r1",
            "tournament_id": "t1",
            "display_name": "Primary",
            "email": "primary@example.com",
            "status": "confirmed",
            "doubles_skill": 3.8,
            "gender": "Women",
            "age": 60,
        },
        {
            "id": "r2",
            "tournament_id": "t1",
            "display_name": "Partner",
            "email": "partner@example.com",
            "status": "confirmed",
            "doubles_skill": 4.05,
            "gender": "Women",
            "age": 67,
        },
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "s1",
            "tournament_id": "t1",
            "registration_id": "r1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "HAS_PARTNER",
            "partner_email": "partner@example.com",
        }
    ]
    proposed = {
        **storage["tournament_event_options"][0],
        "skill_label": "3.5",
        "gender_restriction": "WOMEN",
        "age_mode": "FIXED_AGE_BRACKET",
        "age_label": "50+",
        "age_rules": {
            "mode": "FIXED_AGE_BRACKET",
            "label": "50+",
            "min_age": 50,
            "team_age_rule": "YOUNGER",
        },
    }

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    detail = next(row for row in impact["blocked_details"] if row["field"] == "skill_age_rules")
    proposed_value = detail["affected_registrations"][0]["proposed_value"]
    assert proposed_value["skill_eligibility"]["controlling_rating"] == 4.05
    assert proposed_value["gender_eligibility"]["status"] == "ELIGIBLE"
    assert proposed_value["age_eligibility"]["status"] == "ELIGIBLE"
    assert proposed_value["partner_age"] == 67


def test_missing_partner_age_does_not_hide_known_directional_age_ineligibility() -> None:
    preview = build_age_split_preview(
        policy={
            "mode": "FIXED_AGE_BRACKET",
            "label": "50+",
            "min_age": 50,
            "team_age_rule": "YOUNGER",
        },
        registrations={
            "r1": {"id": "r1", "display_name": "Young Player", "age": 45},
        },
        selections=[
            {"id": "s1", "registration_id": "r1", "partner_mode": "NEEDS_PARTNER"},
        ],
        participant_type="GENDER_DOUBLES",
    )

    assert preview["unassigned_entries"][0]["assignment_issue_type"] == "AGE_NOT_ELIGIBLE"
    assert "older partner cannot make this team eligible" in preview["unassigned_entries"][0]["assignment_issue"]


def test_public_registration_uses_directional_skill_and_age_boundaries() -> None:
    eligibility = read_web("lib/tournamentRegistrationEligibility.ts")
    skill_helper = read_web("lib/tournamentSkillEligibility.ts")
    form = read_web("app/clubs/[clubSlug]/tournament-registration/TournamentRegistrationForm.tsx")
    edit_form = read_web("app/clubs/[clubSlug]/tournament-registration/edit/EditTournamentRegistrationForm.tsx")
    service = read_root("jupr_app/services/public_tournament_registration_service.py")

    assert "skillEligibilityPolicy" in eligibility
    assert "anchor + 0.5" in skill_helper
    assert 'mode === "MINIMUM"' in skill_helper
    assert "policy.minimum" in eligibility
    assert "rating >= policy.maximumExclusive" in eligibility
    assert "hardMinimumAge" in eligibility
    assert "age < minimumAge" in eligibility
    assert "age: numericValue(contact.age)" in form
    assert "age: numericState(ageDraft)" in edit_form
    assert 'const submittedAge = numberOrNull(formData.get("age"))' in edit_form
    assert "age: submittedAge" in edit_form
    assert '"age_rules": row.get("age_rules")' in service
    assert "_validate_age_eligibility" in service
    assert "evaluate_selection_gender_eligibility" in service


def test_upward_open_skill_event_age_regrouping_never_becomes_skill_conflict() -> None:
    storage = base_storage()
    storage["tournament_event_options"][0].update(
        {
            "skill_label": "3.5+",
            "skill_mode": "minimum",
            "eligibility_mode": "STANDARD",
            "age_mode": "open",
            "age_label": "Open",
            "age_rules": None,
        }
    )
    storage["tournament_registrations"] = [
        {
            "id": "complete",
            "tournament_id": "t1",
            "display_name": "Complete Team",
            "email": "complete@example.com",
            "status": "confirmed",
            "doubles_skill": 4.3,
            "age": 67,
            "gender": "Men",
        },
        {
            "id": "missing",
            "tournament_id": "t1",
            "display_name": "Missing Partner Age",
            "email": "missing@example.com",
            "status": "confirmed",
            "doubles_skill": 4.4,
            "age": 44,
            "gender": "Men",
        },
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "complete-selection",
            "tournament_id": "t1",
            "registration_id": "complete",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "HAS_PARTNER",
            "partner_age": 68,
            "partner_skill": 5.0,
            "partner_gender": "Men",
        },
        {
            "id": "missing-selection",
            "tournament_id": "t1",
            "registration_id": "missing",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "NEEDS_PARTNER",
        },
    ]
    proposed = {
        **storage["tournament_event_options"][0],
        "age_mode": "AUTO_AGE_SPLIT",
        "age_label": "Age groups",
        "age_rules": {
            "mode": "AUTO_AGE_SPLIT",
            "team_age_rule": "YOUNGER",
            "merge_strategy": "CLOSEST",
            "min_teams_per_age_group": 1,
            "brackets": [
                {"id": "under-50", "label": "Under 50", "max_age": 49},
                {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
                {"id": "65-plus", "label": "65+", "min_age": 65},
            ],
        },
    }

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    assert not [row for row in impact["blocked_details"] if row["field"] == "skill_age_rules"]
    detail = next(row for row in impact["communication_impact_details"] if row["impact_type"] == "AGE_GROUPING_COMMUNICATION")
    by_registration = {
        row["registration_id"]: row["proposed_value"]
        for row in detail["affected_registrations"]
    }
    assert by_registration["complete"]["preferred_age_group"] == "65+"
    assert by_registration["complete"]["skill_eligibility"]["status"] == "ELIGIBLE"
    pending = by_registration["missing"]
    assert detail["requires_data_completion"] is False
    assert detail["data_completion_registrations"] == []
    assert pending["preferred_age_group"] == "Under 50"
    assert pending["age_label"] == "Under 50"
    assert pending["effective_age"] == 44
    assert pending["age_eligibility"]["status"] == "ELIGIBLE"
    assert pending["age_eligibility"]["pending_partner"] is True
    assert pending["age_eligibility"]["provisional"] is True
    assert pending["gender_eligibility"]["status"] == "ELIGIBLE"
    assert pending["data_completion_required"] is False


def test_split_age_review_defers_unpaired_entry_without_data_completion_block() -> None:
    storage = base_storage()
    storage["tournament_registrations"] = [
        {
            "id": "pending",
            "tournament_id": "t1",
            "display_name": "Pending Partner",
            "email": "pending@example.com",
            "status": "confirmed",
            "doubles_skill": 3.7,
            "age": 44,
            "gender": "Men",
        },
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "pending-selection",
            "tournament_id": "t1",
            "registration_id": "pending",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "NEEDS_PARTNER",
        },
    ]
    proposed = {
        **storage["tournament_event_options"][0],
        "age_mode": "SPLIT_AGE",
        "age_label": "Split age partners",
        "age_rules": {"mode": "SPLIT_AGE", "split_age_threshold": 50},
    }

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    assert not [row for row in impact["blocked_details"] if row["field"] == "skill_age_rules"]
    detail = next(
        row
        for row in impact["communication_impact_details"]
        if row["impact_type"] == "AGE_GROUPING_COMMUNICATION"
    )
    assert detail["requires_data_completion"] is False
    assert detail["data_completion_registrations"] == []
    assert "partner-pending registration" in detail["message"]
    assert "recalculated when a partner is assigned" in detail["message"]
    review = detail["affected_registrations"][0]["proposed_value"]
    assert review["preferred_age_group"] is None
    assert review["age_eligibility"]["status"] == "ELIGIBLE"
    assert review["age_eligibility"]["provisional"] is True
    assert review["age_eligibility"]["recompute_when_partner_assigned"] is True


def test_age_review_regroups_from_canonical_linked_partner() -> None:
    storage = base_storage()
    storage["tournament_registrations"] = [
        {
            "id": "older",
            "tournament_id": "t1",
            "display_name": "Older Player",
            "email": "older@example.com",
            "status": "confirmed",
            "doubles_skill": 3.2,
            "age": 67,
            "gender": "Men",
        },
        {
            "id": "younger",
            "tournament_id": "t1",
            "display_name": "Younger Partner",
            "email": "younger@example.com",
            "status": "confirmed",
            "doubles_skill": 3.3,
            "age": 44,
            "gender": "Men",
        },
    ]
    storage["tournament_registration_selections"] = [
        {
            "id": "older-selection",
            "tournament_id": "t1",
            "registration_id": "older",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "NEEDS_PARTNER",
            "partner_age": 88,
            "partner_gender": "Women",
            "partner_skill": 6.5,
        },
        {
            "id": "younger-selection",
            "tournament_id": "t1",
            "registration_id": "younger",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "partner_mode": "NEEDS_PARTNER",
            "partner_age": 21,
            "partner_gender": "Women",
            "partner_skill": 6.5,
        },
    ]
    storage["tournament_registration_team_links"] = [
        {
            "id": "link-1",
            "tournament_id": "t1",
            "event_option_id": "event1",
            "selection1_id": "older-selection",
            "selection2_id": "younger-selection",
            "registration1_id": "older",
            "registration2_id": "younger",
            "status": "ADMIN_CONFIRMED",
        }
    ]
    proposed = {
        **storage["tournament_event_options"][0],
        "age_mode": "AUTO_AGE_SPLIT",
        "age_label": "Age groups",
        "age_rules": {
            "mode": "AUTO_AGE_SPLIT",
            "team_age_rule": "YOUNGER",
            "merge_strategy": "CLOSEST",
            "min_teams_per_age_group": 1,
            "brackets": [
                {"id": "under-50", "label": "Under 50", "max_age": 49},
                {"id": "50-64", "label": "50–64", "min_age": 50, "max_age": 64},
                {"id": "65-plus", "label": "65+", "min_age": 65},
            ],
        },
    }

    impact = analyze_registration_publish_impact(
        FakeSupabase(storage),
        tournament_id="t1",
        days=storage["tournament_registration_days"],
        event_options=[proposed],
    )

    detail = next(
        row
        for row in impact["communication_impact_details"]
        if row["impact_type"] == "AGE_GROUPING_COMMUNICATION"
    )
    assert detail["requires_data_completion"] is False
    assert detail["data_completion_registrations"] == []
    proposed_by_registration = {
        row["registration_id"]: row["proposed_value"]
        for row in detail["affected_registrations"]
    }
    assert set(proposed_by_registration) == {"older", "younger"}
    for review in proposed_by_registration.values():
        assert review["preferred_age_group"] == "Under 50"
        assert review["effective_age"] == 44
        assert review["age_eligibility"]["pending_partner"] is False
        assert review["gender_eligibility"]["status"] == "ELIGIBLE"
        assert review["skill_eligibility"]["status"] == "ELIGIBLE"
