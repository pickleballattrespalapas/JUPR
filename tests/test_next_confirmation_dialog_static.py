from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"

ACTION_CONFIRMATION_FILES = (
    "app/admin/badges/BadgeDiagnosticsPanel.tsx",
    "app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx",
    "app/admin/jupr-live/JuprLiveAdminPanel.tsx",
    "app/admin/league-manager/GuidedLeagueSettingsEditor.tsx",
    "app/admin/league-manager/LeagueManagerPanel.tsx",
    "app/admin/league-manager/awards/LeagueAwardsPanel.tsx",
    "app/admin/league-manager/live/LeagueLiveRoundPanel.tsx",
    "app/admin/match-canonical-audit/MatchCanonicalAuditPanel.tsx",
    "app/admin/match-log/MatchLogApplyPanel.tsx",
    "app/admin/match-log/MatchLogBulkExcludePanel.tsx",
    "app/admin/match-log/MatchLogQuickReplayPanel.tsx",
    "app/admin/match-log/MatchLogSocialPanel.tsx",
    "app/admin/moneyball/MoneyballPanel.tsx",
    "app/admin/player-updates/PlayerUpdatesPanel.tsx",
    "app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx",
    "app/admin/players/PlayerEditorPanel.tsx",
    "app/admin/replay-history/ReplayHistoryForm.tsx",
    "app/admin/support-requests/SupportRequestsPanel.tsx",
    "app/admin/tools/AdminToolsPanel.tsx",
    "app/admin/tournament-live/TournamentLivePanel.tsx",
    "app/admin/tournament-setup/TournamentSetupPanel.tsx",
    "app/admin/tournaments/TournamentAdminPanel.tsx",
    "app/admin/tournaments/bulk/BulkRegistrationPanel.tsx",
    "app/admin/tournaments/delete-draft/DeleteDraftPanel.tsx",
    "app/admin/tournaments/ops/TournamentOpsPanel.tsx",
    "app/admin/tournaments/status/TournamentStatusPanel.tsx",
    "app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx",
)


def test_shared_confirmation_dialog_is_accessible_and_keeps_the_server_phrase_internal() -> None:
    source = (WEB / "components" / "ConfirmAction.tsx").read_text()

    for required in (
        'export function ConfirmAction',
        'createPortal(',
        'setPortalContainer(document.body)',
        '<dialog',
        'aria-labelledby={titleId}',
        'aria-describedby=',
        'aria-modal="true"',
        'cancelRef.current?.focus()',
        'window.requestAnimationFrame',
        'fallbackFocusRef.current',
        'wasOpenRef.current',
        'restoreFocusAfterDialog',
        'trigger?.isConnected && !trigger.disabled',
        'document.querySelector<HTMLElement>("main")',
        'onCancel={handleDialogCancel}',
        'onConfirm(confirmationText)',
        'if (disabled || busy || submittingRef.current)',
        'submittingRef.current',
        'role="alert"',
        'cancelLabel = "No, go back"',
    ):
        assert required in source


def test_all_admin_action_confirmations_use_dialogs_instead_of_typed_inputs() -> None:
    offenders: list[str] = []

    for relative in ACTION_CONFIRMATION_FILES:
        source = (WEB / relative).read_text()
        assert 'import { ConfirmAction } from "@/components/ConfirmAction";' in source, relative
        assert '<ConfirmAction' in source, relative

        for tag in re.findall(r"<input\b[^>]*>", source, flags=re.IGNORECASE | re.DOTALL):
            if re.search(r"confirm(?:ation)?", tag, flags=re.IGNORECASE):
                normalized_tag = " ".join(tag.split())
                offenders.append(f"{relative}: {normalized_tag}")

    assert offenders == []


def test_password_confirmation_remains_a_password_verification_field() -> None:
    source = (WEB / "app" / "admin" / "reset-password" / "AdminResetPasswordForm.tsx").read_text()
    assert "Confirm new password" in source
    assert 'type="password"' in source
