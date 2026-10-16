from pathlib import Path


MIGRATION = Path("supabase/migrations/20261109004102_tournament_team_division_transfer.sql")


def _function(path: Path, name: str) -> str:
    source = path.read_text()
    start = source.index(f"create or replace function {name}(")
    end = source.index("$function$;", start) + len("$function$;")
    return source[start:end]


def test_existing_edit_guards_are_preserved_verbatim_as_private_baselines() -> None:
    pairs = [
        (
            Path("supabase/migrations/20260807150000_tournament_complete_registration_editor.sql"),
            "public.admin_update_tournament_registration_selection",
            "private.admin_selection_before_division_transfer",
        ),
        (
            Path("supabase/migrations/20261108017000_exact_registration_draw_roster_lock.sql"),
            "public.server_update_public_tournament_registration_edit",
            "private.public_edit_before_division_transfer",
        ),
    ]
    for original, public_name, private_name in pairs:
        assert _function(MIGRATION, private_name) == _function(original, public_name).replace(public_name, private_name, 1)


def test_all_new_callable_functions_are_invoker_and_server_only() -> None:
    sql = " ".join(MIGRATION.read_text().lower().split())
    signatures = [
        "private.tournament_division_family(jsonb)",
        "private.transfer_tournament_registration_team_division(text,text,text,timestamptz,jsonb)",
        "private.admin_selection_before_division_transfer(text,text,timestamptz,jsonb)",
        "private.public_edit_before_division_transfer(text,text,timestamptz,jsonb,jsonb,jsonb)",
        "public.admin_update_tournament_registration_selection(text,text,timestamptz,jsonb)",
        "public.server_update_public_tournament_registration_edit(text,text,timestamptz,jsonb,jsonb,jsonb)",
    ]
    for signature in signatures:
        assert f"revoke all on function {signature} from public, anon, authenticated" in sql
        assert f"grant execute on function {signature} to service_role" in sql
    for name in [
        "private.transfer_tournament_registration_team_division",
        "public.admin_update_tournament_registration_selection",
        "public.server_update_public_tournament_registration_edit",
    ]:
        body = _function(MIGRATION, name)
        assert "security invoker set search_path = ''" in body


def test_public_save_locks_event_union_before_baseline_and_rolls_back_on_transfer_conflict() -> None:
    body = _function(MIGRATION, "public.server_update_public_tournament_registration_edit")
    assert body.index("order by event.id for update") < body.index("v_result := private.public_edit_before_division_transfer")
    assert "jsonb_array_elements(p_selections)" in body
    assert "jsonb_array_elements(v_baseline_selections)" in body
    assert "raise exception using errcode='PT409',message=v_transfer_result::text" in body
    assert "exception when sqlstate 'PT409'" in body


def test_transfer_preserves_commerce_history_and_updates_all_active_relationships() -> None:
    body = _function(MIGRATION, "private.transfer_tournament_registration_team_division")
    for table in ["tournament_registration_selections", "tournament_registration_partner_requests",
                  "tournament_registration_team_links", "tournament_registration_team_members"]:
        assert f"update public.{table}" in body
    assert "update public.tournament_commerce" not in body
    assert "delete from" not in body
    assert "DIVISION_COMMERCE_REVIEW_REQUIRED" in body
    assert "line.line_snapshot->'component_snapshot'" in body


def test_transfer_parses_legacy_text_age_rules_with_blank_fallback() -> None:
    body = _function(MIGRATION, "private.transfer_tournament_registration_team_division")
    assert "coalesce(nullif(btrim(v_target.age_rules::text),'')::jsonb,'{}'::jsonb)" in body
