from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260728020000_combined_rating_team_tournaments.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def test_migration_is_forward_only_and_service_role_scoped():
    sql = _sql()

    assert "create table if not exists public.tournament_four_player_teams" in sql
    assert "create table if not exists public.tournament_team_match_games" in sql
    assert "enable row level security" in sql
    assert "force row level security" in sql
    assert "from public, anon, authenticated" in sql
    assert "to service_role" in sql
    assert "drop table" not in sql.lower()


def test_combined_rating_is_strict_and_snapshotted_at_close():
    sql = _sql()

    assert "combined_rating_cap" in sql
    assert "v_combined < v_event.combined_rating_cap" in sql
    assert "JUPR_TOURNAMENT_RATING_REVIEW_MATH_INVALID" in sql
    assert "combined_rating_snapshot" in sql
    assert "rating_as_of" in sql
    assert "finalized_at" in sql
    assert "REGISTRATION_CLOSE" in sql
    assert "JUPR_TOURNAMENT_RATING_REVIEW_SELECTION_STALE" in sql
    assert "perform public.refresh_initial_combined_rating_review_v1(" in sql


def test_admin_rating_review_recomputes_authoritative_evidence_and_close_is_immutable():
    sql = _sql()
    function_body = sql.split(
        "create or replace function public.admin_record_tournament_rating_review_cas(",
        1,
    )[1].split(
        "create or replace function public.admin_create_tournament_four_player_team(",
        1,
    )[0]

    assert "perform public.refresh_initial_combined_rating_review_v1(" in function_body
    assert "v_state := v_authoritative.state;" in function_body
    assert "v_player_rating := v_authoritative.player_rating;" in function_body
    assert "v_partner_rating := v_authoritative.partner_rating;" in function_body
    assert "v_combined := v_authoritative.combined_rating;" in function_body
    assert "coalesce(v_authoritative.rating_as_of, clock_timestamp())" in function_body
    assert "JUPR_TOURNAMENT_RATING_REVIEW_FINALIZED_IMMUTABLE" in function_body
    assert "'immutable_replay', true" in function_body
    assert "v_player_id_snapshot := v_authoritative.player_id_snapshot;" in function_body
    assert (
        "v_partner_player_id_snapshot :="
        in function_body
    )


def test_finalized_rating_identity_and_relationship_are_database_immutable():
    sql = _sql()

    assert "guard_finalized_combined_rating_review_v1" in sql
    assert "guard_finalized_combined_rating_relationship_v1" in sql
    assert "guard_finalized_combined_rating_registration_v1" in sql
    assert "JUPR_TOURNAMENT_RATING_FINALIZED_RELATIONSHIP_IMMUTABLE" in sql
    assert "JUPR_TOURNAMENT_RATING_FINALIZED_PLAYER_IDENTITY_IMMUTABLE" in sql
    rpc = sql.split(
        "create or replace function public.admin_write_combined_rating_draw_teams_cas",
        1,
    )[1]
    assert "review.player_id_snapshot = x.player1_id" in rpc
    assert "review.partner_player_id_snapshot = x.player2_id" in rpc


def test_substitutes_are_configurable_and_roster_changes_are_audited():
    sql = _sql()

    assert "team_allow_substitutes boolean not null default false" in sql
    assert "JUPR_TOURNAMENT_TEAM_SUBSTITUTES_DISABLED" in sql
    assert "admin_amend_tournament_four_player_roster_cas" in sql
    assert "p_reason" in sql
    assert "four_player_roster_" in sql
    service = Path(
        "jupr_app/services/admin_tournament_team_competition_service.py"
    ).read_text(encoding="utf-8")
    roster_service = service.split(
        "def amend_four_player_team_roster(", 1
    )[1].split("def replace_team_podium(", 1)[0]
    assert "_deliver_invitations(" in roster_service
    assert 'if normalized_action != "REPLACE":' in roster_service
    roster_sql = sql.split(
        "create or replace function public.admin_amend_tournament_four_player_roster_cas(",
        1,
    )[1].split(
        "create or replace function public.calculate_tournament_team_podium(",
        1,
    )[0]
    assert "lower(coalesce(v_member->>'gender', ''))" in roster_sql


def test_withdrawal_fails_closed_after_match_activity():
    sql = _sql()
    roster_sql = sql.split(
        "create or replace function public.admin_amend_tournament_four_player_roster_cas(",
        1,
    )[1].split(
        "create or replace function public.calculate_tournament_team_podium(",
        1,
    )[0]

    assert "v_action = 'WITHDRAW' and exists" in roster_sql
    assert "tournament_team_lineup_submissions" in roster_sql
    assert "tournament_team_match_games" in roster_sql
    assert "JUPR_TOURNAMENT_TEAM_WITHDRAWAL_LOCKED_BY_MATCH_ACTIVITY" in roster_sql


def test_invitation_acceptance_requires_confirmed_registration_in_sql():
    sql = _sql()
    invitation_sql = sql.split(
        "create or replace function public.server_respond_tournament_four_player_invite(",
        1,
    )[1].split(
        "create or replace function public.admin_reissue_tournament_four_player_invite_cas(",
        1,
    )[0]

    assert (
        "upper(coalesce(registration.status, '')) = 'CONFIRMED'"
        in invitation_sql
    )
    assert "not in (\n       'CANCELLED', 'WITHDRAWN'" not in invitation_sql


def test_match_order_tiebreak_and_playoffs_are_server_enforced():
    sql = _sql()

    for game_code in ("WOMENS", "MENS", "MIXED_1", "MIXED_2", "TIEBREAK"):
        assert game_code in sql
    assert "SKINNY_RELAY" in sql
    assert "TOP_2_FINAL" in sql
    assert "TOP_4_SEMIFINALS_WITH_BRONZE" in sql
    assert "JUPR_TOURNAMENT_TEAM_PLAYOFF_SOURCE_INCOMPLETE" in sql
    assert "calculate_tournament_team_podium" in sql


def test_draw_writes_are_serialized_and_canonical_publish_fails_closed():
    sql = _sql()

    assert "pg_advisory_xact_lock" in sql
    assert "lock_tournament_team_draw" in sql
    assert "enforce_tournament_team_official_match_source" in sql
    assert "JUPR_TOURNAMENT_TEAM_PARENT_MATCH_PUBLISH_FORBIDDEN" in sql
    assert "JUPR_TOURNAMENT_TEAM_RATING_MATCH_SOURCE_INVALID" in sql
    assert "JUPR_TOURNAMENT_TEAM_RATING_MATCH_NOT_FINAL" in sql


def test_reconciliation_requires_reason_and_updates_playoff_dependencies():
    sql = _sql()

    assert "JUPR_TOURNAMENT_TEAM_RECONCILIATION_REASON_REQUIRED" in sql
    assert "resolve_tournament_team_playoff_dependencies" in sql
    assert "official_shape_rejected" in sql
    assert "official_side_swapped" in sql
    assert "excluded_from_ratings" in sql


def test_team_creation_recovers_by_business_identity_without_duplicate_team():
    sql = _sql()

    assert "creation_fingerprint text null" in sql
    assert "p_creation_fingerprint text" in sql
    assert "recovered_by_business_identity" in sql
    assert "JUPR_TOURNAMENT_FOUR_PLAYER_TEAM_CREATE_CONFLICT" in sql
    assert "team.captain_registration_id::text = p_captain_registration_id" in sql


def test_invitation_fresh_and_idempotent_results_use_public_projection():
    sql = _sql()
    function_body = sql.split(
        "create or replace function public.server_respond_tournament_four_player_invite(",
        1,
    )[1].split(
        "create or replace function public.admin_reissue_tournament_four_player_invite_cas(",
        1,
    )[0]

    assert function_body.count("'invitation', jsonb_build_object(") == 2
    idempotent_result = function_body.split(
        "if v_member.status = 'ACCEPTED' and v_action = 'ACCEPT'",
        1,
    )[1].split("if v_member.status <> 'INVITED'", 1)[0]
    final_result = function_body.rsplit("v_result := jsonb_build_object(", 1)[1]
    for result_builder in (idempotent_result, final_result):
        assert "'team', jsonb_build_object(" in result_builder
        assert "'invitation', jsonb_build_object(" in result_builder
        assert "'team', to_jsonb(v_team)" not in result_builder
        assert "'member', to_jsonb(v_member)" not in result_builder
        assert "'operation_key', p_operation_key" not in result_builder
