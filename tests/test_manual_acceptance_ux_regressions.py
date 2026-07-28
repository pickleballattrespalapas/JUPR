from pathlib import Path


def test_match_uploader_manual_acceptance_fixes_are_present() -> None:
    source = Path("apps/web/app/admin/match-uploader/MatchUploaderForm.tsx").read_text(encoding="utf-8")
    assert 'useState<"singles" | "manual" | "round_robin">("manual")' in source
    assert '[newMatchRow(todayIsoDate(), status.week_tag_options[0] || "Week 1")]' in source
    assert "const readyRows = rows.filter" in source
    assert "hasInvalidFilledRows" in source
    assert 'triggerLabel="Remove match"' in source
    assert 'triggerLabel="Remove All"' in source
    assert "SubmissionResultDialog" in source
    assert "Successfully inserted" in source
    assert "Player-update email was not sent in staging." in source
    assert "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS is not enabled" not in source
    assert "matchingPlayers.length === 0" in source
    assert "rowHasEnteredData" in source
    assert "payload.match_write_committed === false" in source
    assert "Review Match Log before retrying" in source


def test_match_log_edit_page_is_compact() -> None:
    workspace = Path("apps/web/app/admin/match-log/MatchLogWorkspace.tsx").read_text(encoding="utf-8")
    panel = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx").read_text(encoding="utf-8")
    assert 'const showsMatchSummary = showsMatchContext && mode !== "edit"' in workspace
    assert 'const showsMatchTable = showsMatchContext && mode !== "edit"' in workspace
    assert "Find a match" in workspace
    assert 'mode === "guided" ? "Match editor"' in panel
    assert '<details data-testid="match-edit-operation-history"' in panel
    assert "Use the filters above to narrow the choices" in panel
