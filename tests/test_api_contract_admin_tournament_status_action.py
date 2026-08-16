from __future__ import annotations

from pathlib import Path

import pytest

from jupr_app.domain.tournament_registration_repo import archive_tournament
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_tournament import _install_auth
from tests.test_admin_tournament_lifecycle_service import (
    _official_match_for_game,
    _ready_tables,
)

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_streamlit_and_repository_cannot_bypass_guarded_closeout():
    for relative in (
        "jupr_app/ui/pages/tournament_ops.py",
        "jupr_app/ui/pages/tournaments.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '.button("Archive Tournament"' not in source
        assert "Archive is available only from Tournament Closeout" in source

    with pytest.raises(PermissionError, match="guarded Tournament Closeout"):
        archive_tournament(object(), "tour-1")

    tournaments_source = (
        ROOT / "jupr_app/ui/pages/tournaments.py"
    ).read_text(encoding="utf-8")
    status_options = tournaments_source.split(
        "TOURNAMENT_STATUS_OPTIONS =", 1
    )[1].split("\n", 1)[0]
    assert "ARCHIVED" not in status_options
    assert "Archived tournaments may be created only through guarded Tournament Closeout" in tournaments_source


def test_legacy_streamlit_ops_cannot_publish_scores_or_finalize_podium():
    source = (ROOT / "jupr_app/ui/pages/tournament_ops.py").read_text(
        encoding="utf-8"
    )

    assert "submit_match_batch" not in source
    assert "Backfill Tournament Trophies" not in source
    assert "award_tournament_trophies_from_podium" not in source
    assert '.button("Finalize tournament"' not in source
    assert 'update({"status": "COMPLETE"})' not in source
    assert "Legacy score saving is disabled" in source
    assert "Legacy podium finalization is disabled" in source


def test_admin_tournament_archive_is_fail_closed_and_unarchive_remains_guarded(monkeypatch):
    tables, _ = _ready_tables(monkeypatch)
    tables["tournaments"][0]["updated_at"] = "2026-03-02T00:00:00Z"
    tables["matches"] = [
        _official_match_for_game(tables, game)
        for game in tables["tournament_games"]
    ]
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    client = TestClient(app)
    operation_count = len(tables["tournament_admin_operations"])
    audit_count = len(tables["admin_activity_log"])
    archive_response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour-1/status-action",
        headers={"Authorization": "Bearer local"},
        json={"action": "archive", "confirmation_text": "ARCHIVE"},
    )

    assert archive_response.status_code == 403
    assert "ARCHIVE_ATOMIC_COMMIT_UNAVAILABLE" in archive_response.json()["detail"]
    assert tables["tournaments"][0]["status"] == "PUBLISHED"
    assert len(tables["tournament_admin_operations"]) == operation_count
    assert len(tables["admin_activity_log"]) == audit_count

    # Previously archived tournaments can still be restored; that transition
    # does not claim closeout completeness.
    tables["tournaments"][0]["status"] = "ARCHIVED"

    unarchive_response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour-1/status-action",
        headers={"Authorization": "Bearer local"},
        json={"action": "unarchive", "confirmation_text": "UNARCHIVE"},
    )

    assert unarchive_response.status_code == 200
    unarchived = unarchive_response.json()
    assert unarchived["action"] == "unarchive"
    assert unarchived["tournament"]["status"] == "DRAFT"
    assert tables["tournaments"][0]["status"] == "DRAFT"
    status_actions = [
        row["action_type"]
        for row in tables["admin_activity_log"]
        if row["action_type"] in {"archive_tournament_admin", "unarchive_tournament_admin"}
    ]
    assert status_actions == ["unarchive_tournament_admin"]
    assert all(
        row["flagged_for_review"] is True
        for row in tables["admin_activity_log"]
        if row["action_type"] in {"archive_tournament_admin", "unarchive_tournament_admin"}
    )
