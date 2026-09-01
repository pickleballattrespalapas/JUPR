from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_dedicated_live_routes_do_not_reuse_jupr_live_session_surface() -> None:
    routes = _read("services/api/admin_tournament_live_routes.py")
    installer = _read("services/api/admin_operations_routes.py")
    assert "/tournament-live/tournaments/{tournament_id}/snapshot" in routes
    assert "/draws/{draw_id}/commands" in routes
    assert "/operations/{operation_key}/reconcile" in routes
    assert "install_admin_tournament_live_routes" in installer
    assert "live_sessions" not in routes
    assert "admin_jupr_live" not in routes
    assert "class AdminTournamentLiveGameScore(BaseModel)" in routes
    assert "game_scores: list[AdminTournamentLiveGameScore]" in routes


def test_python_service_reuses_tournament_domain_authority_and_draw_lock() -> None:
    service = _read("jupr_app/services/admin_tournament_live_service.py")
    assert '"authority": "python_fastapi"' in service
    assert 'lock_scope=f"tournament:{tournament_id}:draw:{draw_id}"' in service
    assert "update_admin_tournament_game_score" in service
    assert "generate_admin_tournament_round_robin_games" in service
    assert "generate_admin_tournament_playoff_games" in service
    assert "generate_admin_tournament_draw_podium" in service
    assert "award_admin_tournament_draw_podium" in service
    assert "publish_admin_tournament_draw_matches" in service
    assert 'environment != "staging"' in service
    assert "RECONCILE TOURNAMENT LIVE" in service


def test_private_schema_extends_one_operation_ledger_with_uuid_uniqueness() -> None:
    base = _read("supabase/migrations/20260719203000_tournament_admin_operations.sql").lower()
    live = _read("supabase/migrations/20260719205000_tournament_live_operations.sql").lower()
    assert "enable row level security" in base
    assert "revoke all on table public.tournament_admin_operations from public, anon, authenticated" in base
    assert "client_idempotency_key" in live
    assert "'operations'" in live
    assert "tournament_live" in live
    assert "idx_tournament_admin_operations_client_idempotency" in live
    assert "create unique index" in live
    assert "service_role" in base
    assert "admin_award_tournament_draw_podium_cas" in live
    assert "p_expected_podium" in live
    assert "p_expected_awards" in live
    assert live.index("perform podium.id") < live.index("for update of d")
    assert "block_tournament_draw_change_during_official_publish" in live
    assert "jupr_tournament_official_publish_lock" in live
    assert "'ops_official_publish', 'tournament_live_official_publish'" in live
    assert "operation.status in ('intent', 'mutated', 'recovery_required')" in live
    assert "operation.status in ('intent', 'mutated', 'recovery_required', 'completed', 'failed')" not in live
    assert "admin_apply_tournament_official_rating_plan_cas" in live
    assert "p_publish_plan is distinct from v_operation.request_json #> '{payload,publish_plan}'" in live
    assert "match_payload_projections" in live
    assert "jupr_tournament_official_publish_match_payload_stale" in live
    assert "idx_tournament_admin_operations_active_official_rating" in live
    assert "block_rating_change_during_tournament_official_publish" in live
    assert "before insert or update or delete on public.matches" in live
    assert "before insert or update or delete on public.players" in live
    assert "before insert or update or delete on public.league_ratings" in live
    assert "before insert or update or delete on public.leagues_metadata" in live
    assert "jupr_tournament_official_publish_tournament_stale" in live
    assert "jupr_tournament_official_publish_event_option_stale" in live
    assert "jupr_tournament_official_publish_player_stale" in live
    assert "jupr_tournament_official_publish_league_rating_stale" in live
    assert "get diagnostics v_inserted = row_count" in live
    assert "get diagnostics v_row_count = row_count" in live
    assert "nullif(item.context_id, '')::text" in live
    assert "nullif(item.context_id, '')::uuid" not in live
    assert "p_tournament_id::uuid" in live
    assert "item.tournament_game_id::uuid" in live
    assert "grant execute on function public.admin_apply_tournament_official_rating_plan_cas" in live
    assert "from public, anon, authenticated" in live


def test_responsive_runner_retains_exact_request_and_exposes_audit_evidence() -> None:
    panel = _read("apps/web/app/admin/tournament-live/TournamentLivePanel.tsx")
    css = _read("apps/web/app/admin/tournament-live/TournamentLivePanel.module.css")
    spec = _read("apps/web/e2e/tournament-live.staging.spec.ts")
    assert "jupr_tournament_live_pending_v2" in panel
    assert "state_fingerprint" in panel
    assert "idempotency_key" in panel
    assert "expected_draw_updated_at" in panel
    assert "expected_team_versions" in panel
    assert "expected_source_game_versions" in panel
    assert "snapshot.source_game_versions" in panel
    assert "game_scores?: LiveGameScore[]" in panel
    assert "validateBestOfThreeGameScores" in panel
    assert "Individual game scores" in panel
    assert "tournament-live-game-${gameNumber}-team-a-score" in panel
    assert "source_game_versions" in _read("apps/web/lib/adminTournamentApi.ts")
    assert "publication_rating_game_ids" in _read(
        "apps/web/lib/adminTournamentApi.ts"
    )
    assert "/tournaments/admin/ops/tournaments" not in panel
    assert "const lockedTournamentId = initialTournamentId;" in panel
    assert "Refresh available draws" in panel
    assert "Retry exact retained request" in panel
    assert "Durable operation and audit evidence" in panel
    assert "not JUPR Live" in panel
    assert "mobileGames" in panel
    assert "@media (max-width: 760px)" in css
    assert "min-height: 44px" in css
    assert "JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E" in spec
    assert "finally" in spec


def test_runbook_keeps_matrix_partial_and_documents_stop_conditions() -> None:
    runbook = _read("docs/tournament_live_parity_evidence.md")
    matrix = _read("docs/next_streamlit_parity_matrix.md")
    assert "does not move the parity matrix row to" in runbook
    assert "Stop immediately" in runbook
    assert "| `tournament_live` | 🔴 Tournament Live | Admin | `Partial` — automated-ready/manual-ready:" in matrix
    assert "| `tournament_live` | 🔴 Tournament Live | Admin | `Done`" not in matrix
    assert "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES" in runbook
