from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_registration_repo import archive_tournament
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_tournament import _install_auth
from tests.test_admin_tournament_lifecycle_service import (
    _publish_draw_with_immutable_evidence,
    _ready_tables,
)

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
)
from jupr_app.services.admin_tournament_status_service import (
    apply_admin_tournament_status_action,
)


ROOT = Path(__file__).resolve().parents[1]


class _AtomicTerminalSupabase(FakeSupabase):
    def __init__(self, tables, *, snapshots: list[str]):
        super().__init__(tables)
        self.snapshots = list(snapshots)
        self.rpc_calls: list[tuple[str, dict]] = []

    def rpc(self, name, params):
        self.rpc_calls.append((str(name), dict(params or {})))
        if name == "admin_tournament_completion_snapshot":
            fingerprint = self.snapshots.pop(0)
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(
                    data={"snapshot": {"snapshot_fingerprint": fingerprint}}
                )
            )
        if name == "admin_transition_tournament_terminal_status_cas":
            tournament = dict(self.tables["tournaments"][0])
            tournament["status"] = "COMPLETED"
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(
                    data={
                        "tournament": tournament,
                        "receipt": {
                            "action": "complete",
                            "to_status": "COMPLETED",
                            "operation_key": params["p_operation_key"],
                        },
                    }
                )
            )
        raise AssertionError(f"unexpected RPC {name}")


def _ready_completion_evidence() -> dict:
    return {
        "contract": "jupr:tournament-lifecycle:v1",
        "authority": "server",
        "phase": "ready_to_complete",
        "counts": {},
        "domain_readiness": {"completion": {"ready": True, "blockers": []}},
        "draws": [],
        "warnings": [],
    }


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


def test_admin_tournament_complete_archive_and_unarchive_are_separate_actions(monkeypatch):
    tables, supabase = _ready_tables(monkeypatch)
    tables["tournaments"][0]["updated_at"] = "2026-03-02T00:00:00Z"
    tables["tournaments"][0]["status"] = "ACTIVE"
    _publish_draw_with_immutable_evidence(tables, supabase, "draw-1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    client = TestClient(app)
    complete_response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour-1/status-action",
        headers={"Authorization": "Bearer local"},
        json={"action": "complete", "confirmation_text": "COMPLETE"},
    )

    assert complete_response.status_code == 200, complete_response.text
    completed = complete_response.json()
    assert completed["action"] == "complete"
    assert completed["tournament"]["status"] == "COMPLETED"
    assert completed["lifecycle_receipt"]["to_status"] == "COMPLETED"

    archive_response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour-1/status-action",
        headers={"Authorization": "Bearer local"},
        json={"action": "archive", "confirmation_text": "ARCHIVE"},
    )

    assert archive_response.status_code == 200, archive_response.text
    assert archive_response.json()["tournament"]["status"] == "ARCHIVED"

    unarchive_response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour-1/status-action",
        headers={"Authorization": "Bearer local"},
        json={"action": "unarchive", "confirmation_text": "UNARCHIVE"},
    )

    assert unarchive_response.status_code == 200
    unarchived = unarchive_response.json()
    assert unarchived["action"] == "unarchive"
    assert unarchived["tournament"]["status"] == "COMPLETED"
    assert tables["tournaments"][0]["status"] == "COMPLETED"
    status_actions = [
        row["action_type"]
        for row in tables["admin_activity_log"]
        if row["action_type"] in {"archive_tournament_admin", "unarchive_tournament_admin"}
    ]
    assert status_actions == ["archive_tournament_admin", "unarchive_tournament_admin"]
    assert all(
        row["flagged_for_review"] is True
        for row in tables["admin_activity_log"]
        if row["action_type"] in {"archive_tournament_admin", "unarchive_tournament_admin"}
    )


def test_atomic_completion_rejects_readiness_built_across_changed_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_status_service.require_admin_tournament_completion_readiness",
        lambda *_args, **_kwargs: _ready_completion_evidence(),
    )
    supabase = _AtomicTerminalSupabase(
        {
            "tournaments": [
                {
                    "id": "tour-1",
                    "club_id": "club",
                    "name": "Race fixture",
                    "status": "ACTIVE",
                    "updated_at": "2026-08-25T12:00:00Z",
                }
            ]
        },
        snapshots=["a" * 32, "b" * 32],
    )

    with pytest.raises(StaleTournamentAdminStateError, match="readiness was being reviewed"):
        apply_admin_tournament_status_action(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            action="complete",
            expected_updated_at="2026-08-25T12:00:00Z",
            actor_email="director@example.com",
            actor_role="club_owner",
            confirmation_text="COMPLETE",
            guarded_operation_key="a" * 64,
            request_fingerprint="b" * 64,
            atomic=True,
        )

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_tournament_completion_snapshot",
        "admin_tournament_completion_snapshot",
    ]


def test_atomic_completion_passes_stable_readiness_snapshot_to_terminal_rpc(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_status_service.require_admin_tournament_completion_readiness",
        lambda *_args, **_kwargs: _ready_completion_evidence(),
    )
    supabase = _AtomicTerminalSupabase(
        {
            "tournaments": [
                {
                    "id": "tour-1",
                    "club_id": "club",
                    "name": "Stable fixture",
                    "status": "ACTIVE",
                    "updated_at": "2026-08-25T12:00:00Z",
                }
            ],
            "admin_activity_log": [],
        },
        snapshots=["c" * 32, "c" * 32],
    )

    result = apply_admin_tournament_status_action(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        action="complete",
        expected_updated_at="2026-08-25T12:00:00Z",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="COMPLETE",
        guarded_operation_key="a" * 64,
        request_fingerprint="b" * 64,
        atomic=True,
    )

    transition_params = next(
        params
        for name, params in supabase.rpc_calls
        if name == "admin_transition_tournament_terminal_status_cas"
    )
    assert transition_params["p_snapshot_fingerprint"] == "c" * 32
    assert result["tournament"]["status"] == "COMPLETED"
