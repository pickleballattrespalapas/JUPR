from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def test_match_uploader_exposes_all_modes_and_post_commit_email_outcome() -> None:
    source = (WEB / "app" / "admin" / "match-uploader" / "MatchUploaderForm.tsx").read_text(encoding="utf-8")

    for label in ("Singles match", "Doubles manual / batch", "Doubles round robin", "Create Players & Continue"):
        assert label in source
    assert "Post-batch player-update email" in source
    assert "match_write_committed" in source
    assert "Outcome unknown: check Match Log before retrying" in source
    assert "Open Player Updates" in source
    assert "status.singles_write_enabled && status.singles_submit_endpoint" in source
    assert "Direct singles entry remains unavailable" in source


def test_player_editor_exposes_reviewed_atomic_merge_and_recovery() -> None:
    source = (WEB / "app" / "admin" / "players" / "PlayerEditorPanel.tsx").read_text(encoding="utf-8")

    assert "preview_fingerprint" in source
    assert "operation_id" in source
    assert "Check merge operation" in source
    assert "Attach replay evidence" in source
    assert "Open tracked Streamlit fallback" in source
    assert "COMPENSATE MERGE" in source
    assert "server-only service role" in source


def test_score_entry_hides_write_form_until_api_is_ready() -> None:
    page = (WEB / "app" / "clubs" / "[clubSlug]" / "admin" / "score-entry" / "page.tsx").read_text(encoding="utf-8")

    assert "getAdminScoreEntryStatus" in page
    assert "readiness.data?.ready" in page
    assert "Score entry is in fallback mode" in page
    assert "Streamlit fallback" in page
