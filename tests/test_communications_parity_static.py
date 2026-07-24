from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_communications_migration_is_service_role_only_and_stale_guarded() -> None:
    sql = _read("supabase/migrations/20260719182606_communications_outbox_stale_guards.sql")

    assert "add column if not exists row_version" in sql
    assert "send_status in ('pending', 'sending', 'sent', 'skipped', 'error')" in sql
    assert "replace_verified_update_subscription" in sql
    assert "for update" in sql
    assert "replacement_operation_key" in sql
    assert "queue_operation_key" in sql
    assert "revoke all on table public.%I from public, anon, authenticated" in sql
    assert "grant all privileges on table public.%I to service_role" in sql
    assert "revoke execute on function public.replace_verified_update_subscription" in sql


def test_next_communications_ui_has_confirmation_and_uncertain_delivery_guards() -> None:
    panel = _read("apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx")

    for phrase in (
        "QUEUE PLAYER UPDATES",
        "SEND PLAYER UPDATES",
        "RETRY UNCERTAIN EMAILS",
        "DELETE QUEUED UPDATES",
        "REPLACE VERIFIED SUBSCRIBER",
        "UNSUBSCRIBE VERIFIED SUBSCRIBER",
    ):
        assert phrase in panel
    assert "expected_row_version" in panel
    assert "Supabase service role" in panel
    assert "service_role" not in _read("apps/web/lib/adminPlayerUpdatesApi.ts").lower().replace("service_role_configured", "")


def test_communications_ui_exposes_read_only_mode_and_disables_mutations() -> None:
    player_updates = _read(
        "apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx"
    )
    verified_updates = _read(
        "apps/web/app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx"
    )
    weekly_recap = _read(
        "apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx"
    )

    assert "mutations_enabled: boolean" in _read(
        "apps/web/lib/adminPlayerUpdatesApi.ts"
    )
    assert "mutations_enabled: boolean" in _read(
        "apps/web/lib/adminWeeklyRecapApi.ts"
    )
    assert (
        "const mutationControlsDisabled = workspaceControlsDisabled || "
        "!status.mutations_enabled"
    ) in player_updates
    assert "Read-only:" in player_updates
    assert "disabled={mutationControlsDisabled}" in player_updates
    assert "disabled={busy || !status.mutations_enabled}" in verified_updates
    assert "Read-only:" in verified_updates
    assert (
        "const mutationControlsDisabled = busy || !status.mutations_enabled"
    ) in weekly_recap
    assert "saved recaps and unpublished previews remain available" in weekly_recap
    assert "disabled={mutationControlsDisabled" in weekly_recap
    assert (
        weekly_recap.count(
            'disabled={mutationControlsDisabled || selectedRecap.status === "published"}'
        )
        >= 6
    )


def test_admin_recap_has_full_unpublished_preview_and_print_surface() -> None:
    preview = _read("apps/web/app/admin/weekly-recap/AdminWeeklyRecapPreview.tsx")
    panel = _read("apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx")

    assert "Unpublished draft — operator preview only" in preview
    assert "window.print()" in preview
    assert "<NumberStrip recap={recap}" in preview
    assert "Around the Club" in preview
    assert "Tournaments" in preview
    assert "Looking Ahead" in preview
    assert "expected_row_version" in panel


def test_guarded_off_communications_status_cards_do_not_prompt_for_sign_in() -> None:
    for relative_path in (
        "apps/web/app/admin/player-updates/page.tsx",
        "apps/web/app/admin/weekly-recap/page.tsx",
    ):
        page = _read(relative_path)
        assert "Sign in to load" not in page
        assert "Available after admin sign-in" in page
        assert "Guarded off" in page
