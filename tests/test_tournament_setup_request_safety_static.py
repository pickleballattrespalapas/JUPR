from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = (
    ROOT / "apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx"
).read_text(encoding="utf-8")
BUILDER = (
    ROOT / "apps/web/app/admin/tournament-setup/tournamentSetupBuilder.ts"
).read_text(encoding="utf-8")
BUILDER_UI = (
    ROOT / "apps/web/app/admin/tournament-setup/TournamentSetupBuilder.tsx"
).read_text(encoding="utf-8")
DIVISION_CARD = (
    ROOT / "apps/web/app/admin/tournament-setup/TournamentSetupDivisionCard.tsx"
).read_text(encoding="utf-8")
E2E = (
    ROOT / "apps/web/e2e/tournament-setup.builder.staging.spec.ts"
).read_text(encoding="utf-8")


def test_tournament_setup_waits_for_auth_and_guards_every_request_class() -> None:
    assert "useAuthenticatedAutoLoad(status?.enabled ? accessToken : \"\", loadTournaments)" in PANEL
    assert "const listRequest = useLatestRequestGuard(accessToken, resetWorkspace);" in PANEL
    assert "const detailRequest = useLatestRequestGuard(accessToken);" in PANEL
    assert "const operationRequest = useLatestRequestGuard(accessToken);" in PANEL
    assert "if (!listRequest.isCurrent(generation)) return false;" in PANEL
    assert "if (!detailRequest.isCurrent(generation)) return false;" in PANEL
    assert "if (!operationRequest.isCurrent(generation)) return;" in PANEL


def test_tournament_selection_clears_old_record_before_loading_replacement() -> None:
    selection = PANEL.split("function selectTournament", 1)[1].split(
        "useAuthenticatedAutoLoad", 1
    )[0]
    assert "detailRequest.invalidate();" in selection
    assert "operationRequest.invalidate();" in selection
    assert "clearDetailState();" in selection
    assert "setSelectedId(id);" in selection
    assert selection.index("clearDetailState();") < selection.index("setSelectedId(id);")
    assert "{detail && detailIsCurrent ? <>" in PANEL
    assert "current setup remains visible" not in PANEL


def test_tournament_list_refresh_preserves_dirty_current_setup() -> None:
    load_list = PANEL.split("async function loadTournaments", 1)[1].split(
        "async function loadDetail", 1
    )[0]
    assert "const preserveCurrentEdits =" in load_list
    assert "nextId === selectedId" in load_list
    assert "nextId === loadedDetailId" in load_list
    assert "Boolean(detail)" in load_list
    assert "Unsaved setup edits were preserved." in load_list
    assert 'await page.getByRole("button", { name: "Refresh list" }).click()' in E2E
    assert "expect(detailReads).toBe(1)" in E2E


def test_browser_contract_covers_empty_retry_mobile_payload_and_stale_selection() -> None:
    assert "works at a mobile viewport" in E2E
    assert "preserves payloads" in E2E
    assert "exposes empty and failed list states with a working retry" in E2E
    assert "clears the prior record while a new selection loads" in E2E
    assert "ignores a deferred authenticated response after logout" in E2E
    assert 'expect.poll(() => draftWrites).toBe(1)' in E2E
    assert 'expect.poll(() => listReads).toBe(1)' in E2E
    assert 'getByRole("heading", { name: "Draft summary" })).toHaveCount(0)' in E2E


def test_guided_shell_creation_uses_one_stable_retry_command() -> None:
    create = PANEL.split("async function createTournament", 1)[1].split(
        "async function saveSettings", 1
    )[0]

    assert "globalThis.crypto.randomUUID()" in create
    assert "const command = createCommand ||" in create
    assert "persistCreateCommand(command);" in create
    assert "tournament_id: command.tournamentId" in create
    assert "idempotency_key: command.idempotencyKey" in create
    assert "Retry keeps the same protected request." in create
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
    assert "if (!usesLegacyShape) return projectCanonicalAgeRuleEdits(row);" in BUILDER


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
