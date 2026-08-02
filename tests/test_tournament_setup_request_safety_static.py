from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = (ROOT / "apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx").read_text()
BUILDER = (ROOT / "apps/web/app/admin/tournament-setup/tournamentSetupBuilder.ts").read_text()
DIVISION_CARD = (
    ROOT / "apps/web/app/admin/tournament-setup/TournamentSetupDivisionCard.tsx"
).read_text()
BUILDER_UI = (
    ROOT / "apps/web/app/admin/tournament-setup/TournamentSetupBuilder.tsx"
).read_text()
CREATE_PAGE = (
    ROOT / "apps/web/app/admin/tournament-setup/create/page.tsx"
).read_text()
E2E = (ROOT / "apps/web/e2e/tournament-setup-create.spec.ts").read_text()


def test_setup_detail_latest_request_guard_prevents_stale_overwrite() -> None:
    assert "useLatestRequestGuard" in PANEL
    assert "detailRequest.begin()" in PANEL
    assert "detailRequest.isCurrent(generation)" in PANEL
    assert "detailRequest.invalidate()" in PANEL
    assert "setDetail(null)" in PANEL


def test_setup_write_actions_use_latest_request_guard() -> None:
    assert "const actionRequest = useLatestRequestGuard(accessToken);" in PANEL
    assert PANEL.count("const generation = actionRequest.begin();") >= 6
    assert "if (!actionRequest.isCurrent(generation)) return;" in PANEL
    assert "if (actionRequest.isCurrent(generation))" in PANEL


def test_setup_draft_preserves_legacy_shape_and_preview_contract() -> None:
    save_draft = PANEL.split("async function saveDraft", 1)[1].split(
        "async function publishSetup", 1
    )[0]
    assert "const draft = configurationPayload(configuration);" in save_draft
    assert "publishConfigurationPayload(configuration)" not in save_draft
    assert "saved configuration" in PANEL.lower()


def test_setup_publish_and_impact_require_canonical_projection() -> None:
    publish = PANEL.split("async function publishSetup", 1)[1].split(
        "function seedFromPublished", 1
    )[0]
    impact = PANEL.split("async function reviewImpact", 1)[1].split(
        "if (!status?.enabled)", 1
    )[0]
    assert "publishConfigurationPayload(configuration)" in publish
    assert "publishConfigurationPayload(configuration)" in impact
    assert "reviewedDraftSignature" in PANEL
    assert "impactReview.impact_fingerprint" in publish


def test_setup_create_command_is_idempotent_and_persisted() -> None:
    create = CREATE_PAGE.split("async function createTournament", 1)[1]
    assert "createCommandRef" in CREATE_PAGE
    assert "setCreateCommand(command);" in CREATE_PAGE
    assert "createCommandStorageKey(clubId)" in CREATE_PAGE
    assert "command.idempotencyKey" in create
    assert "command.confirmationText" in create
    assert "command.requestFingerprint" in create
    assert "if (loaded)" in create
    assert "setCreateCommand(null);" in create
    assert "readStoredCreateCommand(clubId)" in PANEL
    assert "window.localStorage.setItem(createCommandStorageKey" in PANEL
    assert "Nothing is published and registration remains closed" in PANEL
    assert (
        "creates one protected DRAFT shell and opens it in the builder" in E2E
    )
    assert "expect.poll(() => createWrites).toBe(1)" in E2E


def test_legacy_drafts_are_projected_only_for_impact_and_publish() -> None:
    save_draft = PANEL.split("async function saveDraft", 1)[1].split(
        "async function publishSetup", 1
    )[0]
    publish = PANEL.split("async function publishSetup", 1)[1].split(
        "function seedFromPublished", 1
    )[0]
    impact = PANEL.split("async function reviewImpact", 1)[1].split(
        "if (!status?.enabled)", 1
    )[0]

    assert "configurationPayload(configuration)" in save_draft
    assert "publishConfigurationPayload(configuration)" not in save_draft
    assert "publishConfigurationPayload(configuration)" in publish
    assert "publishConfigurationPayload(configuration)" in impact
    assert "dayIdsByLabel.get(normalizedLookupKey(assignedDay))" in BUILDER
    assert "event_family_label: familyName" in BUILDER
    assert "event_format_default:" in BUILDER
    assert "scoring_default:" in BUILDER
    assert "const projected = projectCanonicalAgeRuleEdits(row);" in BUILDER
    assert 'Object.prototype.hasOwnProperty.call(projected, "scheduled_day_ids")' in BUILDER
    assert "scheduledDayIds.length > 1" in BUILDER
    assert "return next;" in BUILDER


def test_canonical_age_rules_and_legacy_family_defaults_use_pure_builder_helpers() -> None:
    assert 'ageRuleValue(value, "min_teams_per_age_group")' in DIVISION_CARD
    assert 'ageRuleValue(value, "split_age_threshold")' in DIVISION_CARD
    assert "setAgeRuleNumber(value," in DIVISION_CARD
    assert "setEventAgeMode(value, event.target.value)" in DIVISION_CARD
    assert "effectiveParticipantType(value, eventFamilies)" in DIVISION_CARD
    assert "effectiveGenderRestriction(value, eventFamilies)" in DIVISION_CARD
    assert "eventFamilies={configuration.eventFamilies}" in BUILDER_UI
    assert "finiteNumber(ageRuleValue(row, \"min_teams_per_age_group\"))" in BUILDER
    assert "finiteNumber(ageRuleValue(row, \"split_age_threshold\"))" in BUILDER
    assert "clearIncompatibleAgeRuleFields" in BUILDER
