from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"

def test_shared_confirmation_dialog_is_accessible_and_keeps_the_server_phrase_internal() -> None:
    source = (WEB / "components" / "ConfirmAction.tsx").read_text()
    provider = (WEB / "components" / "interaction" / "InteractionProvider.tsx").read_text()
    dialog = (WEB / "components" / "interaction" / "InteractionDialog.tsx").read_text()
    lifecycle = (WEB / "components" / "interaction" / "useActionLifecycle.ts").read_text()

    for required in (
        'export function ConfirmAction',
        'openAction(',
        'cancelLabel = "No, go back"',
    ):
        assert required in source

    for required in (
        '<InteractionDialog',
        'active.onConfirm(active.confirmationText)',
        'lifecycle.run',
        'lifecycle.recover',
        'Object.freeze({ ...request, origin })',
    ):
        assert required in provider

    for required in (
        'createPortal(',
        'setPortalContainer(document.body)',
        '<dialog',
        'aria-labelledby={titleId}',
        'aria-describedby=',
        'aria-modal="true"',
        'window.requestAnimationFrame',
        'wasOpenRef.current',
        'document.querySelector<HTMLElement>("main")',
        'onCancel={handleCancel}',
    ):
        assert required in dialog

    assert "inFlightRef.current" in lifecycle
    assert 'setPhase("success")' not in lifecycle
    assert "setPhase(result.status)" in lifecycle


def test_all_admin_action_confirmations_use_dialogs_instead_of_typed_inputs() -> None:
    offenders: list[str] = []
    action_confirmation_files = sorted(
        path
        for path in (WEB / "app" / "admin").rglob("*.tsx")
        if "<ConfirmAction" in path.read_text()
    )
    assert action_confirmation_files

    for path in action_confirmation_files:
        relative = str(path.relative_to(WEB))
        source = path.read_text()
        assert 'import { ConfirmAction } from "@/components/ConfirmAction";' in source, relative

        for tag in re.findall(r"<input\b[^>]*>", source, flags=re.IGNORECASE | re.DOTALL):
            if re.search(r"confirm(?:ation)?", tag, flags=re.IGNORECASE):
                normalized_tag = " ".join(tag.split())
                offenders.append(f"{relative}: {normalized_tag}")

    assert offenders == []


def test_password_confirmation_remains_a_password_verification_field() -> None:
    source = (WEB / "app" / "admin" / "reset-password" / "AdminResetPasswordForm.tsx").read_text()
    assert "Confirm new password" in source
    assert 'type="password"' in source
