from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "apps/web"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_current_chooser_has_separate_past_archive_and_results_navigation() -> None:
    home = read("app/clubs/[clubSlug]/tournaments/page.tsx")
    past = read("app/clubs/[clubSlug]/tournaments/past/page.tsx")
    nav = read("components/PublicTournamentNav.tsx")

    assert "Past tournaments" in home
    assert "/tournaments/past" in home
    assert 'getPublicTournamentResultsIndex(\n    params.clubSlug,\n    "past"' in past
    assert "Officially completed tournaments" in past
    assert '["results", "Live & Results"]' in nav
    assert 'results: `${base}/tournament-results`' in nav


def test_standard_results_page_renders_all_patron_result_surfaces() -> None:
    page = read("app/clubs/[clubSlug]/tournament-results/page.tsx")
    api = read("lib/tournamentResultsApi.ts")

    for label in (
        "Podium and medals",
        "Standings",
        "Playoff bracket",
        "Completed scores",
        "outcome_label",
        "Current tournaments",
        "Past tournaments",
    ):
        assert label in page or label in api
    assert "PublicTournamentNav" in page
    assert 'active="results"' in page
    assert "public_game_key" in api
    assert "public_draw_key" in api
    assert "PublicTournamentTiebreakExplanation" in api
    assert "tiebreak_explanations?: PublicTournamentTiebreakExplanation[]" in api
    assert "round_robin_complete?: boolean" in api
    assert "ranking_policy?:" in api
    assert "description?: string | null" in api
    assert "criteria?: string[]" in api
    assert "retired_teams_eligible?: boolean" in api
    assert "team_ids" not in api
    assert "final_team_ids" not in api
    assert "admin_notes" not in api
    assert "email" not in api


def test_public_tiebreak_explanation_is_compact_collapsed_and_server_authored() -> None:
    page = read("app/clubs/[clubSlug]/tournament-results/page.tsx")

    assert 'data-testid="public-tiebreak-explanation"' in page
    assert "How tied teams were ranked" in page
    assert "<details" in page
    assert "<summary" in page
    assert "<details open" not in page
    assert "explanation.title" in page
    assert "explanation.summary" in page
    assert "step.detail" in page
    assert "tiebreakCriterionLabel" in page
    assert "tiebreakOutcomeLabel" in page
    assert 'tiebreakOutcomeLabel(step.outcome, step.detail)' in page
    assert 'aria-label={`How tied teams were ranked in ${draw.name}`}' in page
    assert 'aria-label={`Tie-break steps for ${explanation.title} in ${draw.name}`}' in page
    assert "rankingPolicyDescription" in page
    assert "rankingCriteria.map(tiebreakCriterionLabel)" in page
    assert "Final round-robin order." in page
    assert "Provisional — this order may change until round-robin play is complete." in page
    assert page.index("<table") < page.index("<details")
    assert page.index("<details") < page.index("Playoff bracket")


def test_current_public_draws_render_as_url_backed_tabs() -> None:
    page = read("app/clubs/[clubSlug]/tournament-results/page.tsx")

    assert 'draw?: string;' in page
    assert 'draw.state === "LIVE" || draw.state === "READY"' in page
    assert 'aria-label="Current tournament draws"' in page
    assert 'aria-current={selected ? "page" : undefined}' in page
    assert "prefetch={false}" in page
    assert "scroll={false}" in page
    assert "draw: publicDrawKey" in page
    assert "selectedCurrentDraw ? <DrawResults draw={selectedCurrentDraw} /> : null" in page
    assert "Completed draws" in page
    assert "Upcoming draws" in page


def test_registration_and_edit_render_multiday_event_once_with_schedule_detail() -> None:
    create = read(
        "app/clubs/[clubSlug]/tournament-registration/TournamentRegistrationForm.tsx"
    )
    edit = read(
        "app/clubs/[clubSlug]/tournament-registration/edit/"
        "EditTournamentRegistrationForm.tsx"
    )
    home = read("app/clubs/[clubSlug]/tournaments/page.tsx")

    assert "scheduledDaysLabel(eventOption, daysById)" in create
    assert "event.registration_day_id === day.id" in create
    assert "scheduledDaysLabel(eventOption, dayById)" in edit
    assert "event.registration_day_id === day.id" in edit
    assert "event.scheduled_day_ids?.length" in home


def test_confirmation_and_secure_edit_use_per_event_actions() -> None:
    confirmation = read(
        "app/clubs/[clubSlug]/tournament-registration/confirmation/page.tsx"
    )
    registration = read("app/clubs/[clubSlug]/tournament-registration/page.tsx")
    edit = read(
        "app/clubs/[clubSlug]/tournament-registration/edit/"
        "EditTournamentRegistrationForm.tsx"
    )

    assert "Edit this event" in confirmation
    assert "+ Add Event" in confirmation
    assert "selection.scheduled_days?.length" in confirmation
    assert 'id="manage-registration"' in registration
    assert "EditLinkRequestForm" in registration
    assert "Registered events" in edit
    assert "Edit event" in edit
    assert "InteractionDialog" in edit
    assert "Apply event changes" in edit
    assert "+ Add Event" in edit
    assert "Changes are staged until" in edit


def test_secure_edit_recomputes_event_eligibility_from_the_live_age_draft() -> None:
    edit = read(
        "app/clubs/[clubSlug]/tournament-registration/edit/"
        "EditTournamentRegistrationForm.tsx"
    )

    assert 'const [ageDraft, setAgeDraft] = useState(String(registration.age ?? ""))' in edit
    assert "age: numericState(ageDraft)" in edit
    assert "[ageDraft, gender, linkedPlayer, doublesSkill, singlesSkill]" in edit
    assert 'name="age" value={ageDraft} onChange={(event) => setAgeDraft(event.target.value)}' in edit
    assert 'const submittedAge = numberOrNull(formData.get("age"))' in edit
    assert "age: submittedAge" in edit
    assert "Enter an age between 1 and 120" in edit
    assert edit.index("const formData = new FormData(event.currentTarget)") < edit.index("const ineligible = selectedIds")
    assert "publicEventEligibilityReason(event, eligibilityProfile)" in edit
    assert "choose another division before saving" in edit


def test_secure_edit_form_remounts_when_token_or_registration_identity_changes() -> None:
    page = read("app/clubs/[clubSlug]/tournament-registration/edit/page.tsx")

    assert 'key={`${editToken}:${data.registration.id}:${data.registration.updated_at}`}' in page
