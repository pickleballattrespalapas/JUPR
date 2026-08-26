from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20261108017000_exact_registration_draw_roster_lock.sql"
)


def _sql() -> str:
    return " ".join(MIGRATION.read_text(encoding="utf-8").lower().split())


def test_public_edit_lock_requires_exact_roster_membership() -> None:
    sql = _sql()

    assert (
        "create or replace function public."
        "server_update_public_tournament_registration_edit"
    ) in sql
    assert "team.source_selection_id::text = selection.id" in sql
    assert (
        "upper(coalesce(team.source, '')) in ( "
        "'registration', 'registration_combined_rating' )"
    ) in sql
    assert (
        "v_registration.player_id in (team.player1_id, team.player2_id)"
        in sql
    )
    assert "registration_imported_to_draw" in sql
    assert "pg_get_functiondef" not in sql


def test_public_edit_lock_prefers_authoritative_draw_scope() -> None:
    sql = _sql()

    assert "left join public.tournament_event_draws as draw" in sql
    assert "draw.id = team.draw_id" in sql
    assert (
        "coalesce( draw.registration_day_id::text, "
        "team.registration_day_id::text ) = selection.registration_day_id"
    ) in sql
    assert (
        "coalesce( draw.event_option_id::text, team.event_option_id::text ) "
        "= selection.event_option_id"
    ) in sql


def test_public_edit_lock_replacement_remains_service_role_only() -> None:
    sql = _sql()
    signature = (
        "public.server_update_public_tournament_registration_edit( "
        "text, text, timestamptz, jsonb, jsonb, jsonb )"
    )

    assert "language plpgsql security invoker set search_path = ''" in sql
    assert f"revoke all on function {signature} from public, anon, authenticated" in sql
    assert f"grant execute on function {signature} to service_role" in sql
