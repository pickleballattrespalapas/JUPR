from pathlib import Path


PANEL = Path("apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx")
ROUTES = Path("services/api/admin_league_manager_routes.py")
SERVICE = Path("jupr_app/services/admin_league_live_submit_service.py")
STATUS = Path("jupr_app/services/admin_league_live_service.py")
EVIDENCE = Path("docs/league_live_submit_parity_evidence.md")
MATRIX = Path("docs/next_streamlit_parity_matrix.md")


def test_next_uses_one_fastapi_complete_round_publish_path() -> None:
    panel = PANEL.read_text(encoding="utf-8")
    routes = ROUTES.read_text(encoding="utf-8")
    assert "/match-uploader/batch" not in panel
    assert "/rounds/${encodeURIComponent(String(currentRound))}/submit" in panel
    assert "expected_match_count: matches.length" in panel
    assert "!allSeriesComplete" in panel
    assert "series_key: match.row_id" in panel
    assert "Retry exact league-round publish" in panel
    assert "() => submitRound(confirmationText)" in panel
    assert "const operationReference = error instanceof LeagueLiveRequestError" in panel
    assert "${failureDetail} Retry these exact scores" in panel
    assert "every played game still counts as an official league game" in Path(
        "apps/web/app/admin/league-manager/GuidedLeagueSettingsEditor.tsx"
    ).read_text(encoding="utf-8")
    assert "submit_admin_league_live_round_publish" in routes
    assert "rounds/{round_number}/submit" in routes
    assert "retry_admin_league_live_round_publish" in routes
    assert "rounds/{round_number}/retry" in routes


def test_zero_write_recovery_retries_retained_request_instead_of_reconciling() -> None:
    panel = PANEL.read_text(encoding="utf-8")
    service = SERVICE.read_text(encoding="utf-8")
    assert 'RETRYABLE_PUBLISH_STATUSES = new Set(["intent", "publishing", "retryable"])' in panel
    assert "Retry retained league-round publish" in panel
    assert 'confirmationText="RETRY LEAGUE ROUND"' in panel
    assert "/rounds/${encodeURIComponent(String(round))}/retry" in panel
    assert "RECONCILABLE_PUBLISH_STATUSES.has(operation.status)" in panel
    assert "def retry_admin_league_live_round_publish" in service
    assert 'operation.get("request_json")' in service
    assert 'operation.get("idempotency_key")' in service
    assert 'status not in {"intent", "publishing", "retryable"}' in service
    assert "blockingCurrentRoundPublishOperation" in panel
    assert "a new publish is intentionally blocked" in panel
    assert "comparison_operation_key_matches_retained_plan" in service


def test_publish_is_staging_only_python_authority_with_recovery() -> None:
    service = SERVICE.read_text(encoding="utf-8")
    status = STATUS.read_text(encoding="utf-8")
    for token in (
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
        'runtime == "staging"',
        "ensure_admin_league_live_publish_schema_ready",
        "submit_admin_match_uploader_batch",
        "idempotency_key=f\"league-live:{key}\"",
        "allow_league_live_context=True",
        "match_format=match_format",
        "recover_league_live_round_publish_response_loss_admin",
        "recovery_required",
        "compensate_admin_league_live_round_publish",
        "Do not blindly republish",
    ):
        assert token in service or token in status


def test_manual_evidence_exists_and_matrix_remains_partial() -> None:
    evidence = EVIDENCE.read_text(encoding="utf-8")
    matrix = MATRIX.read_text(encoding="utf-8")
    assert "Deferred manual staging book" in evidence
    assert "No migration or staging write was performed" in evidence
    league_row = next(line for line in matrix.splitlines() if "`league_manager`" in line)
    assert "`Partial`" in league_row
    assert "durable all-match publish" in league_row
