from __future__ import annotations

import re
from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260807150000_tournament_complete_registration_editor.sql"
)
LEGACY_EDIT_MIGRATION = Path(
    "supabase/migrations/20260719160821_public_registration_edit_transaction.sql"
)
LEGACY_COMMERCE_MIGRATION = Path(
    "supabase/migrations/20260728010000_tournament_commerce.sql"
)

EDIT_DECLARATION = (
    "create or replace function "
    "public.server_update_public_tournament_registration_edit("
)
CREATE_WITH_COMMERCE_DECLARATION = (
    "create or replace function "
    "public.server_create_public_tournament_registration_with_commerce("
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_block(path: Path, declaration: str) -> str:
    source = _source(path)
    start = source.index(declaration)
    end = source.index("$function$;", start) + len("$function$;")
    return source[start:end]


def _normalized(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip().lower()


def _replace_exact(
    source: str,
    before: str,
    after: str,
    *,
    count: int = 1,
) -> str:
    assert source.count(before) == count
    return source.replace(before, after)


def _legacy_edit_with_partner_gender() -> str:
    source = _function_block(LEGACY_EDIT_MIGRATION, EDIT_DECLARATION)
    source = _replace_exact(
        source,
        "    'partner_age',\n    'partner_note',",
        "    'partner_age',\n    'partner_gender',\n    'partner_note',",
    )
    source = _replace_exact(
        source,
        "    partner_age = desired.partner_age,\n    partner_note =",
        "    partner_age = desired.partner_age,\n"
        "    partner_gender = nullif(btrim(desired.partner_gender), ''),\n"
        "    partner_note =",
    )
    source = _replace_exact(
        source,
        "    partner_age integer,\n    partner_note text,",
        "    partner_age integer,\n"
        "    partner_gender text,\n"
        "    partner_note text,",
        count=2,
    )
    source = _replace_exact(
        source,
        "    partner_skill,\n    partner_age,\n    partner_note,",
        "    partner_skill,\n"
        "    partner_age,\n"
        "    partner_gender,\n"
        "    partner_note,",
    )
    return _replace_exact(
        source,
        "    desired.partner_skill,\n"
        "    desired.partner_age,\n"
        "    nullif(btrim(desired.partner_note), ''),",
        "    desired.partner_skill,\n"
        "    desired.partner_age,\n"
        "    nullif(btrim(desired.partner_gender), ''),\n"
        "    nullif(btrim(desired.partner_note), ''),",
    )


def _legacy_create_with_commerce_with_partner_gender() -> str:
    source = _function_block(
        LEGACY_COMMERCE_MIGRATION,
        CREATE_WITH_COMMERCE_DECLARATION,
    )
    source = _replace_exact(
        source,
        "partner_phone, partner_dupr_id, partner_skill, partner_age,\n"
        "    partner_note",
        "partner_phone, partner_dupr_id, partner_skill, partner_age,\n"
        "    partner_gender, partner_note",
    )
    source = _replace_exact(
        source,
        "    selection.partner_skill,\n"
        "    selection.partner_age,\n"
        "    nullif(btrim(selection.partner_note), ''),",
        "    selection.partner_skill,\n"
        "    selection.partner_age,\n"
        "    nullif(btrim(selection.partner_gender), ''),\n"
        "    nullif(btrim(selection.partner_note), ''),",
    )
    return _replace_exact(
        source,
        "    partner_skill numeric,\n"
        "    partner_age integer,\n"
        "    partner_note text,",
        "    partner_skill numeric,\n"
        "    partner_age integer,\n"
        "    partner_gender text,\n"
        "    partner_note text,",
    )


def test_public_edit_rpc_only_adds_partner_gender_persistence() -> None:
    replacement = _function_block(MIGRATION, EDIT_DECLARATION)

    assert _normalized(replacement) == _normalized(
        _legacy_edit_with_partner_gender()
    )


def test_public_create_with_commerce_only_adds_partner_gender_persistence() -> None:
    replacement = _function_block(
        MIGRATION,
        CREATE_WITH_COMMERCE_DECLARATION,
    )

    assert _normalized(replacement) == _normalized(
        _legacy_create_with_commerce_with_partner_gender()
    )


def test_public_rpc_replacements_fail_closed_and_remain_service_role_only() -> None:
    sql = _normalized(_source(MIGRATION))
    edit = _normalized(_function_block(MIGRATION, EDIT_DECLARATION))
    create = _normalized(
        _function_block(MIGRATION, CREATE_WITH_COMMERCE_DECLARATION)
    )

    assert (
        "to_regprocedure( "
        "'public.server_update_public_tournament_registration_edit"
        "(text,text,timestamp with time zone,jsonb,jsonb,jsonb)' ) is null"
    ) in sql
    assert (
        "to_regprocedure( "
        "'public.server_create_public_tournament_registration_with_commerce"
        "(text,uuid,jsonb,jsonb,jsonb,uuid,uuid,text,text,text)' ) is null"
    ) in sql

    for block in (edit, create):
        assert "language plpgsql security invoker set search_path = ''" in block
        assert "security definer" not in block
        assert "pg_get_functiondef" not in block
        assert "execute format" not in block

    edit_signature = (
        "public.server_update_public_tournament_registration_edit( "
        "text, text, timestamptz, jsonb, jsonb, jsonb )"
    )
    create_signature = (
        "public.server_create_public_tournament_registration_with_commerce( "
        "text, uuid, jsonb, jsonb, jsonb, uuid, uuid, text, text, text )"
    )
    for signature in (edit_signature, create_signature):
        assert (
            f"revoke all on function {signature} "
            "from public, anon, authenticated"
        ) in sql
        assert f"grant execute on function {signature} to service_role" in sql


def test_public_edit_rpc_preserves_cas_lock_and_relationship_guards() -> None:
    edit = _normalized(_function_block(MIGRATION, EDIT_DECLARATION))

    for marker in (
        "private.lock_tournament_registration_selection_scope",
        "pg_catalog.pg_advisory_xact_lock",
        "registration_imported_to_draw",
        "stale_registration_version",
        "stale_selection_version",
        "registration_relationship_locked",
        "and registration.updated_at = p_expected_updated_at",
        "pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true)",
    ):
        assert marker in edit


def test_public_create_rpc_preserves_commerce_idempotency_and_audit() -> None:
    create = _normalized(
        _function_block(MIGRATION, CREATE_WITH_COMMERCE_DECLARATION)
    )

    for marker in (
        "private.tournament_commerce_operation_begin",
        "if v_operation.status = 'completed'",
        "jsonb_build_object('idempotent_replay', true)",
        "public.server_apply_tournament_commerce_order",
        "update public.tournament_commerce_operations",
        "insert into public.tournament_commerce_audit_log",
        "'registration_create_with_commerce'",
    ):
        assert marker in create
