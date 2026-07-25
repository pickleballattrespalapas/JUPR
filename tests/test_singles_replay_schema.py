from pathlib import Path
from runpy import run_path


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_MATCH_INSERT_COLUMNS = run_path(
    str(ROOT / "jupr_app/domain/matches/persistence.py")
)["OPTIONAL_MATCH_INSERT_COLUMNS"]
MIGRATION = Path(
    "supabase/migrations/20260725181500_singles_replay_recovery.sql"
)
TOURNAMENT_CAS_MIGRATION = Path(
    "supabase/migrations/20260719205000_tournament_live_operations.sql"
)


def test_singles_replay_schema_preserves_legacy_baseline_and_is_service_only() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "singles_replay_baseline jsonb" in sql
    assert "'rating', coalesce(singles_rating, rating, 1200.0)" in sql
    assert "where singles_replay_baseline is null" in sql
    assert "alter column singles_replay_baseline drop default" in sql
    assert "alter column singles_replay_baseline set not null" in sql
    assert "set default '{\"rating\":1200" not in sql
    assert "singles_replay_managed boolean not null default false" in sql
    assert "players_singles_replay_baseline_shape_check" in sql
    assert "matches_singles_replay_managed_shape_check" in sql
    assert "idx_matches_club_singles_replay_order" in sql
    assert "where singles_replay_managed" in sql
    assert "bulk_update_player_singles_stats(rows jsonb)" in sql
    assert "security definer" in sql
    assert "set search_path = public" in sql
    assert (
        "revoke all on function public.bulk_update_player_singles_stats(jsonb)"
        in sql
    )
    assert "from public, anon, authenticated" in sql
    assert (
        "grant execute on function public.bulk_update_player_singles_stats(jsonb)"
        in sql
    )
    assert "to service_role" in sql
    assert "singles_replay_managed" not in OPTIONAL_MATCH_INSERT_COLUMNS
    assert "notify pgrst, 'reload schema'" in sql


def test_new_player_baseline_uses_the_same_seed_as_singles_processing() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "initialize_player_singles_replay_baseline()" in sql
    assert "if new.singles_replay_baseline is null then" in sql
    assert (
        "'rating', coalesce(new.singles_rating, new.rating, 1200.0)"
        in sql
    )
    assert "'wins', coalesce(new.singles_wins, 0)" in sql
    assert "'losses', coalesce(new.singles_losses, 0)" in sql
    assert (
        "'matches_played', coalesce(new.singles_matches_played, 0)"
        in sql
    )
    assert "'last_game_at', new.singles_last_game_at" in sql
    assert (
        "create trigger trg_01_players_initialize_singles_replay_baseline"
        in sql
    )
    assert "before insert on public.players" in sql
    assert (
        "execute function public.initialize_player_singles_replay_baseline()"
        in sql
    )
    assert (
        "revoke all on function "
        "public.initialize_player_singles_replay_baseline()"
        in sql
    )
    assert (
        "grant execute on function "
        "public.initialize_player_singles_replay_baseline()"
        in sql
    )
    assert sql.index("where singles_replay_baseline is null") < sql.index(
        "create trigger trg_01_players_initialize_singles_replay_baseline"
    )


def test_singles_replay_baseline_and_marker_are_immutable() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "protect_player_singles_replay_baseline()" in sql
    assert (
        "new.singles_replay_baseline is distinct from "
        "old.singles_replay_baseline"
    ) in sql
    assert "jupr_singles_replay_baseline_immutable" in sql
    assert "before update on public.players" in sql

    assert "protect_match_singles_replay_contract()" in sql
    assert (
        "new.singles_replay_managed is distinct from "
        "old.singles_replay_managed"
    ) in sql
    assert "jupr_singles_replay_marker_immutable" in sql
    assert "jupr_singles_replay_identity_immutable" in sql
    assert "before update or delete on public.matches" in sql


def test_legacy_singles_rating_changes_and_hard_deletes_are_rejected() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "v_legacy_singles := old.singles_replay_managed is not true" in sql
    assert "jupr_managed_singles_hard_delete_unsupported" in sql
    assert "jupr_legacy_singles_hard_delete_unsupported" in sql
    assert "jupr_legacy_singles_rating_mutation_unsupported" in sql
    for field in (
        "id",
        "club_id",
        "date",
        "match_type",
        "match_format",
        "t1_p1",
        "t1_p2",
        "t2_p1",
        "t2_p2",
        "score_t1",
        "score_t2",
        "rating_scope",
        "rating_bonus_elo",
        "rating_bonus_reason",
        "deleted_at",
    ):
        assert f"new.{field} is distinct from old.{field}" in sql


def test_atomic_tournament_singles_are_marked_from_exact_cas_identity() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    cas_sql = TOURNAMENT_CAS_MIGRATION.read_text(encoding="utf-8").lower()

    transaction_identity = (
        "pg_catalog.current_setting("
        "'jupr.official_publish_operation_key', true)"
    )
    assert transaction_identity in sql
    assert "new.match_format is distinct from 'singles'" in sql
    assert "new.context_type is distinct from 'tournament_game'" in sql
    assert "new.context_id is distinct from new.tournament_game_id::text" in sql
    assert "join public.tournament_games game" in sql
    assert "operation.club_id = new.club_id" in sql
    assert "operation.operation_key = v_operation_key" in sql
    assert "operation.entity_type = 'tournament_event_draw'" in sql
    assert "operation.entity_id = game.draw_id::text" in sql
    assert "'ops_official_publish'" in sql
    assert "'tournament_live_official_publish'" in sql
    assert "operation.status = 'intent'" in sql
    assert (
        "message = "
        "'jupr_tournament_official_publish_singles_replay_identity_invalid'"
        in sql
    )
    assert "new.singles_replay_managed := true" in sql

    trigger = (
        "create trigger "
        "trg_01_matches_atomic_tournament_singles_replay_managed"
    )
    assert trigger in sql
    assert "before insert on public.matches" in sql
    assert (
        "execute function "
        "public.mark_atomic_tournament_singles_replay_managed()"
        in sql
    )
    assert (
        "revoke all on function "
        "public.mark_atomic_tournament_singles_replay_managed()"
        in sql
    )
    assert (
        "grant execute on function "
        "public.mark_atomic_tournament_singles_replay_managed()"
        in sql
    )

    cas_marker = (
        "perform set_config("
        "'jupr.official_publish_operation_key', p_operation_key, true)"
    )
    assert cas_marker in cas_sql
    assert cas_sql.index(cas_marker) < cas_sql.index("insert into public.matches")
