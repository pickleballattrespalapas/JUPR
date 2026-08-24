from __future__ import annotations

from collections import Counter
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def _product_sources() -> list[Path]:
    return sorted(
        path
        for root in (WEB / "app", WEB / "components")
        for path in root.rglob("*")
        if path.suffix in {".ts", ".tsx", ".js", ".jsx"}
    )


def test_native_browser_prompts_are_not_product_interaction_surfaces() -> None:
    offenders: list[str] = []
    pattern = re.compile(r"\bwindow\.(?:confirm|alert|prompt)\s*\(")
    for path in _product_sources():
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_confirm_action_requires_an_explicit_completion_result() -> None:
    types = (WEB / "components" / "interaction" / "types.ts").read_text(encoding="utf-8")
    confirm = (WEB / "components" / "ConfirmAction.tsx").read_text(encoding="utf-8")
    provider = (WEB / "components" / "interaction" / "InteractionProvider.tsx").read_text(encoding="utf-8")
    assert "Promise<ActionCompletion>" in types
    assert "Promise<void" not in types
    assert "isActionCompletion" in types
    assert "onConfirm: ActionCallback" in confirm
    assert "openAction(" in confirm
    assert "lifecycle.run" in provider


def test_confirmation_outcome_is_owned_above_consumer_lifetimes() -> None:
    layout = (WEB / "app" / "layout.tsx").read_text(encoding="utf-8")
    confirm = (WEB / "components" / "ConfirmAction.tsx").read_text(encoding="utf-8")
    provider = (WEB / "components" / "interaction" / "InteractionProvider.tsx").read_text(encoding="utf-8")

    assert "<InteractionProvider>" in layout
    assert "useActionLifecycle" not in confirm
    assert "Object.freeze({ ...request, origin })" in provider
    assert "if (activeRef.current) return false" in provider
    assert "restoreFocus={false}" in provider
    assert "function isEligibleFocusTarget" in provider
    assert '!element.matches(":disabled")' in provider
    assert 'element.getAttribute("aria-disabled") !== "true"' in provider
    assert "isEligibleFocusTarget(explicitTarget)" in provider
    assert "isEligibleFocusTarget(origin)" in provider


def test_interaction_provider_behavioral_harness_is_server_gated() -> None:
    route = (WEB / "app" / "__interaction-provider-test" / "page.tsx").read_text(encoding="utf-8")
    playwright_config = (WEB / "playwright.config.ts").read_text(encoding="utf-8")
    behavioral_spec = (WEB / "e2e" / "interaction-provider.local.spec.ts").read_text(encoding="utf-8")

    assert 'process.env.JUPR_INTERACTION_TEST_HARNESS !== "1"' in route
    assert "notFound()" in route
    assert 'JUPR_INTERACTION_TEST_HARNESS: "1"' in playwright_config
    assert "keeps success visible after the originating consumer unmounts" in behavioral_spec
    assert "accepts only one interaction synchronously" in behavioral_spec
    assert "focusTargetId takes precedence over the connected trigger" in behavioral_spec
    assert "falls back to main when the connected trigger becomes disabled" in behavioral_spec
    assert "form dialog falls back to main when its connected trigger becomes disabled" in behavioral_spec
    assert "form dialog success target takes precedence over its connected trigger" in behavioral_spec


def test_shared_dialog_focus_restoration_skips_ineligible_targets() -> None:
    dialog = (WEB / "components" / "interaction" / "InteractionDialog.tsx").read_text(encoding="utf-8")
    form_dialog = (WEB / "components" / "interaction" / "FormDialog.tsx").read_text(encoding="utf-8")

    assert "function isEligibleFocusTarget" in dialog
    assert '!element.matches(":disabled")' in dialog
    assert 'element.getAttribute("aria-disabled") !== "true"' in dialog
    assert "const returnTarget = returnFocusRef?.current ?? null" in dialog
    assert "isEligibleFocusTarget(returnTarget)" in dialog
    assert "isEligibleFocusTarget(rememberedFocusRef.current)" in dialog
    assert ".find((element) => isEligibleFocusTarget(element) && element.tabIndex >= 0)" in dialog
    assert "if (originFocusRef) originFocusRef.current = rememberedFocusRef.current" in dialog
    assert 'restoreFocus={lifecycle.phase !== "success"}' in form_dialog
    assert "originFocusRef={rememberedOriginRef}" in form_dialog
    assert "focusEligibleElement(explicitTarget)" in form_dialog
    assert "focusEligibleElement(rememberedOriginRef.current)" in form_dialog
    assert "lifecycle.reset();\n      onCancel();" not in form_dialog


def test_dedicated_page_actions_keep_typed_persistent_feedback() -> None:
    # Long forms and authentication are intentional dedicated-page exceptions.
    # They still separate success from error explicitly and announce both.
    feedback_contracts = {
        "app/clubs/[clubSlug]/tournament-registration/EditLinkRequestForm.tsx": ("role=\"status\"", "role=\"alert\""),
        "app/support/SupportRequestForm.tsx": ("role=\"status\"", "role=\"alert\""),
        "app/data-corrections/DataCorrectionForm.tsx": ("role=\"status\"", "role=\"alert\""),
        "app/profile-privacy/ProfilePrivacyRequestForm.tsx": ("role=\"status\"", "role=\"alert\""),
    }
    for relative, required_markers in feedback_contracts.items():
        source = (WEB / relative).read_text(encoding="utf-8")
        for marker in required_markers:
            assert marker in source, f"{relative} is missing {marker}"


def test_shared_dialogs_are_the_only_product_modal_primitive() -> None:
    permitted = {
        "apps/web/components/interaction/InteractionDialog.tsx",
    }
    offenders: list[str] = []
    patterns = (
        re.compile(r"<dialog\b"),
        re.compile(r"\brole=[\"']dialog[\"']"),
    )
    for path in _product_sources():
        relative = str(path.relative_to(ROOT))
        if relative in permitted:
            continue
        source = path.read_text(encoding="utf-8")
        if any(pattern.search(source) for pattern in patterns):
            offenders.append(relative)
    assert offenders == []


def test_app_wide_standard_and_complete_inventories_are_checked_in() -> None:
    standard = (ROOT / "docs" / "interaction-standard.md").read_text(encoding="utf-8")
    frontend = (ROOT / "docs" / "audits" / "frontend-interaction-inventory.md").read_text(encoding="utf-8")
    backend = (ROOT / "docs" / "audits" / "backend-guarded-action-inventory.md").read_text(encoding="utf-8")
    report = (ROOT / "docs" / "audits" / "remediation-report.md").read_text(encoding="utf-8")

    for heading in ("### Create", "### Edit", "### Delete", "### Bulk Edit", "### Publish"):
        assert heading in standard
    assert "### Guarded and high-consequence actions" in standard
    frontend_count = len(re.findall(r"^\| A-(?:CA|NS)-\d{3}\s+\|", frontend, flags=re.MULTILINE))
    backend_rows = re.findall(r"^\| BE-\d{3}\s+\|.*$", backend, flags=re.MULTILINE)
    backend_ids = [re.match(r"^\| (BE-\d{3})", row).group(1) for row in backend_rows]
    assert frontend_count == 232
    assert backend_ids == [f"BE-{index:03d}" for index in range(1, 199)]
    assert frontend_count + len(backend_ids) == 430

    method_counts = Counter(re.search(r"`(POST|PATCH|PUT|DELETE) ", row).group(1) for row in backend_rows)
    assert method_counts == {"POST": 159, "PATCH": 30, "PUT": 8, "DELETE": 1}
    assert sum("/admin/" in row for row in backend_rows) == 167
    assert sum("/admin/" not in row for row in backend_rows) == 31
    wave_assignment_count = sum(
        len(row.rstrip(" |").rsplit("|", 1)[-1].strip().split(",")) for row in backend_rows
    )
    assert wave_assignment_count - len(backend_rows) == 12
    assert "| Total audited frontend + backend contracts | **430** |" in report
    assert "| Staging write-wave coverage | **198 / 198** |" in report

    appended_recovery_rows = (
        "| BE-193 | Match Uploader — `POST /admin/clubs/{club_id}/match-uploader/player-operations/{operation_key}/reconcile` — `post_admin_match_uploader_player_batch_reconcile` | `RECONCILE PLAYER BATCH` | — | — | match-player |",
        "| BE-194 | Player Editor — `POST /admin/clubs/{club_id}/players/editor/operations/{operation_key}/reconcile` — `post_admin_player_editor_operation_reconcile` | `RECONCILE PLAYER OPERATION` | — | — | match-player |",
        "| BE-195 | Club Social — `POST /admin/clubs/{club_id}/match-log/social/operations/{operation_key}/reconcile` — `post_admin_match_log_social_operation_reconcile` | `RECONCILE SOCIAL MATCH` | — | — | match-player |",
        "| BE-196 | League Live — `POST /admin/clubs/{club_id}/league-manager/live-operations/{operation_key}/reconcile` — `post_admin_league_live_create_reconcile` | `RECONCILE LIVE SESSION` | — | — | league-live-domain,league-live-submit |",
        "| BE-197 | Tournament Ops — `POST /admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/teams/import-registrations/operations/{operation_reference}/reconcile` — `post_admin_tournament_registration_team_import_reconcile` | `RECONCILE REGISTRATION IMPORT` | retained_request.expected_state_fingerprint,retained_request.expected_draw_updated_at | retained_request.idempotency_key | tournament-operations |",
        "| BE-198 | League Live — `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/retry` — `post_admin_league_live_round_retry` | `RETRY LEAGUE ROUND` | server-retained expected_updated_at,expected_operation_key | server-retained idempotency_key | league-live-submit |",
    )
    for row in appended_recovery_rows:
        assert row in backend
