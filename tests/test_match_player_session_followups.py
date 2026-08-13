from pathlib import Path


def test_match_uploader_followups_are_present() -> None:
    source = Path("apps/web/app/admin/match-uploader/MatchUploaderForm.tsx").read_text(encoding="utf-8")
    assert "validateRequiredRow" in source
    assert "complete the highlighted player fields" in source
    assert "manualValidationAttempted" in source
    assert "Keep rows with data" in source
    assert "playerOptionsFor" in source
    assert "selectedElsewhere" in source
    assert 'aria-label={`Match ${index + 1} Team 1`}' in source
    assert 'aria-label={`Match ${index + 1} Team 2`}' in source
    assert "Number(value) / 400" in source
    assert "disabled={saving || !accessToken}" in source


def test_confirmation_can_show_completion_result() -> None:
    source = Path("apps/web/components/ConfirmAction.tsx").read_text(encoding="utf-8")
    provider = Path("apps/web/components/interaction/InteractionProvider.tsx").read_text(encoding="utf-8")
    assert "ConfirmActionSuccess" in source
    assert "openAction(" in source
    assert 'completion?.status === "success"' in provider
    assert 'completion.closeLabel ?? "OK"' in provider
    assert "lifecycle.run" in provider


def test_match_log_success_and_replay_history_are_visible() -> None:
    panel = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx").read_text(encoding="utf-8")
    replay = Path("apps/web/app/admin/match-log/MatchLogQuickReplayPanel.tsx").read_text(encoding="utf-8")
    assert "Match edit and replay complete" in panel
    assert "Matches updated" in panel
    assert "Ratings replay" in panel
    assert "throw applyError" in panel
    assert 'href="/admin/replay-history"' in replay
    assert "Open Replay History" in replay
