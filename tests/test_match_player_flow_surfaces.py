from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def test_match_uploader_exposes_all_modes_and_post_commit_email_outcome() -> None:
    source = (WEB / "app" / "admin" / "match-uploader" / "MatchUploaderForm.tsx").read_text(encoding="utf-8")

    for label in ("Singles match", "Doubles manual / batch", "Doubles round robin", "Create Players & Continue"):
        assert label in source
    assert "Player-update email:" in source
    assert "match_write_committed" in source
    assert "Retry this unchanged batch; duplicate protection is active." in source
    assert "directMatchIdempotencyKey" in source
    assert "result.auto_player_updates" in source
    assert "status.singles_write_enabled && status.singles_submit_endpoint" in source
    assert "Direct singles entry remains unavailable" in source


def test_match_uploader_has_searchable_players_and_flexible_workflows() -> None:
    source = (WEB / "app" / "admin" / "match-uploader" / "MatchUploaderForm.tsx").read_text(encoding="utf-8")

    assert "function SearchablePlayerInput" in source
    assert "function SearchablePlayerMultiInput" in source
    assert 'list={`${inputId}-options`}' in source
    assert "Starting JUPR" in source
    assert "next_match_uploader_inline_new_player" in source
    assert "Create & add" in source
    assert 'aria-label={`Remove ${name}`}' in source
    assert "player_names: [...court.playerNames]" in source
    assert "namesText" not in source
    for label in (
        "Add 1 Match",
        "Add 5 Matches",
        "Remove All",
        "Add round robin",
        "Team 1 score",
        "Team 2 score",
    ):
        assert label in source


def test_match_uploader_uses_the_club_local_calendar_date() -> None:
    source = (WEB / "app" / "admin" / "match-uploader" / "MatchUploaderForm.tsx").read_text(encoding="utf-8")
    today_source = source[source.index("function todayIsoDate"):source.index("function randomId")]

    assert "getFullYear()" in today_source
    assert "getMonth()" in today_source
    assert "getDate()" in today_source
    assert ".toISOString()" not in today_source


def test_match_uploader_allows_the_first_player_to_be_created_inline() -> None:
    page = (WEB / "app" / "admin" / "match-uploader" / "page.tsx").read_text(encoding="utf-8")

    assert "Create the first player directly" in page
    assert "{status ? (" in page
    assert "status && players.length" not in page


def test_player_editor_exposes_reviewed_atomic_merge_and_recovery() -> None:
    source = (WEB / "app" / "admin" / "players" / "PlayerEditorPanel.tsx").read_text(encoding="utf-8")

    assert "preview_fingerprint" in source
    assert "operation_id" in source
    assert "Check merge operation" in source
    assert "Attach replay evidence" in source
    assert "Open tracked Streamlit fallback" in source
    assert "COMPENSATE MERGE" in source
    assert "server-only service role" in source
    assert 'role={isErrorMessage(message) ? "alert" : "status"}' in source
    assert 'aria-live="polite"' in source
    assert 'position: "sticky"' in source
    assert "Dismiss Player Editor message" in source
    assert 'gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))"' in source


def test_score_entry_hides_write_form_until_api_is_ready() -> None:
    page = (WEB / "app" / "clubs" / "[clubSlug]" / "admin" / "score-entry" / "page.tsx").read_text(encoding="utf-8")

    assert "getAdminScoreEntryStatus" in page
    assert "readiness.data?.ready" in page
    assert "Score entry isn’t ready here" in page
    assert "Open backup score entry" in page
