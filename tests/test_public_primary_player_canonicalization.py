from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase/migrations/20261108010000_public_primary_player_canonicalization.sql"
).read_text(encoding="utf-8")
SQL = MIGRATION.lower()
PROJECTION_REPAIR_SQL = (
    ROOT
    / "supabase/migrations/"
    "20261108014000_tournament_registration_player_projection_backfill.sql"
).read_text(encoding="utf-8").lower()
FINALIZED_PROJECTION_REPAIR_SQL = (
    ROOT
    / "supabase/migrations/"
    "20261108015000_tournament_player_projection_finalized_review_repair.sql"
).read_text(encoding="utf-8").lower()


def _split_top_level_sql_list(source: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quoted = False
    for index, character in enumerate(source):
        if character == "'":
            quoted = not quoted
        elif not quoted and character in "([":
            depth += 1
        elif not quoted and character in ")]":
            depth -= 1
        elif not quoted and character == "," and depth == 0:
            parts.append(source[start:index].strip())
            start = index + 1
    parts.append(source[start:].strip())
    return parts


def test_public_primary_player_is_resolved_inside_both_atomic_create_paths() -> None:
    assert "private.resolve_public_tournament_primary_player_v1" in SQL
    assert "p_registration - 'player_id'" in SQL
    assert "create_tournament_registration_canonical_unlinked_v1" in SQL
    assert (
        "server_create_public_tournament_registration_with_commerce_unlinked_v1"
        in SQL
    )
    assert "insert into public.players" in SQL
    assert "insert into public.tournament_primary_player_reconciliation" in SQL
    assert "pg_advisory_xact_lock" in SQL


def test_historical_unlinked_players_are_backfilled_only_when_unambiguous() -> None:
    assert "do $historical_primary_player_backfill$" in SQL
    assert "where registration.player_id is null" in SQL
    assert "lock table public.tournament_registrations in share row exclusive mode" in SQL
    assert "lock table public.players in share row exclusive mode" in SQL
    assert "existing_player_name_collision" in SQL
    assert "set player_id = (v_resolution ->> 'player_id')::integer" in SQL
    assert "and registration.player_id is null" in SQL
    assert "select player.id" not in SQL


def test_historical_backfill_queues_finalized_identity_before_resolving() -> None:
    backfill = SQL.split("do $historical_primary_player_backfill$", 1)[1].split(
        "$historical_primary_player_backfill$;", 1
    )[0]
    finalized_guard = backfill.index(
        "review.review_phase = 'registration_close'"
    )
    resolver = backfill.index(
        "v_resolution := private.resolve_public_tournament_primary_player_v1"
    )

    assert finalized_guard < resolver
    assert "review.finalized_at is not null" in backfill
    assert "'state', 'staff_reconciliation_required'" in backfill
    assert (
        "'reason_code', 'finalized_registration_close_review'" in backfill
    )
    assert "'player_id', null" in backfill


def test_primary_player_backfill_repairs_only_null_live_team_projections() -> None:
    for source in (
        SQL,
        PROJECTION_REPAIR_SQL,
        FINALIZED_PROJECTION_REPAIR_SQL,
    ):
        assert (
            "update public.tournament_registration_team_members as member"
            in source
        )
        assert "set player_id = registration.player_id" in source
        assert "where member.player_id is null" in source
        assert "member.status = 'active'" in source
        assert "registration.id = member.registration_id" in source
        assert (
            "registration.tournament_id = member.tournament_id::text"
            in source
        )

        assert source.count(
            "update public.tournament_registration_team_links as link"
        ) == 2
        assert "set player1_id = registration.player_id" in source
        assert "where link.player1_id is null" in source
        assert "registration.id = link.registration1_id" in source
        assert "set player2_id = registration.player_id" in source
        assert "where link.player2_id is null" in source
        assert "registration.id = link.registration2_id" in source
        assert (
            "link.status in ('confirmed', 'admin_confirmed')" in source
        )
        assert (
            "registration.tournament_id = link.tournament_id::text"
            in source
        )
        assert source.count("registration.player_id is not null") >= 3


def test_projection_backfill_skips_immutable_or_cross_club_sources_and_queues() -> None:
    for source in (
        SQL,
        PROJECTION_REPAIR_SQL,
        FINALIZED_PROJECTION_REPAIR_SQL,
    ):
        assert (
            "create table if not exists public."
            "tournament_registration_player_projection_reconciliation"
            in source
        )
        assert "finalized_registration_close_review" in source
        assert "registration_player_club_mismatch" in source
        assert "with projection_candidates" in source
        assert "on conflict (projection_kind, relationship_id) do update" in source
        assert "enable row level security" in source
        assert "force row level security" in source
        assert (
            "revoke all on table public."
            "tournament_registration_player_projection_reconciliation"
            in source
        )
        assert "from public, anon, authenticated, service_role" in source
        assert "to service_role" in source
        assert source.count("review.review_phase = 'registration_close'") >= 4
        assert source.count("review.finalized_at is not null") >= 4
        assert "review.selection_id::text = member.selection_id::text" in source
        assert source.count("review.selection_id::text in (") >= 2
        assert source.count("player.club_id = tournament.club_id") >= 3
        assert source.count("on player.id = registration.player_id") >= 3


def test_projection_candidate_cte_has_matching_alias_and_select_arity() -> None:
    expected_aliases = (
        "projection_kind",
        "relationship_id",
        "tournament_id",
        "registration_id",
        "selection_id",
        "trigger_selection_ids",
        "registration_player_id",
    )
    for source in (
        SQL,
        PROJECTION_REPAIR_SQL,
        FINALIZED_PROJECTION_REPAIR_SQL,
    ):
        match = re.search(
            r"with projection_candidates\s*\((.*?)\)\s*as\s*\((.*?)\)\s*"
            r"insert into public\."
            r"tournament_registration_player_projection_reconciliation",
            source,
            re.DOTALL,
        )
        assert match is not None
        aliases = tuple(
            item.strip() for item in match.group(1).split(",")
        )
        assert aliases == expected_aliases

        select_arms = re.split(r"\s+union all\s+", match.group(2))
        assert len(select_arms) == 3
        for arm in select_arms:
            select_list = arm.split("select", 1)[1].split(
                "\n  from public.", 1
            )[0]
            assert len(_split_top_level_sql_list(select_list)) == len(
                expected_aliases
            )


def test_unverified_email_collision_never_auto_links_an_existing_player() -> None:
    assert "email supplied during initial intake is not verified" in SQL
    assert "staff_reconciliation_required" in SQL
    assert "v_prior_identity_count > 0" in SQL
    collision_block = SQL.split(
        "if v_prior_identity_count > 0 then", 1
    )[1].split("end if;", 1)[0]
    assert "'player_id', null" in collision_block
    assert "select player.id" not in collision_block


def test_live_resolver_serializes_and_queues_normalized_name_collisions() -> None:
    start = SQL.index(
        "create or replace function private.resolve_public_tournament_primary_player_v1"
    )
    end = SQL.index(
        "revoke all on function private.resolve_public_tournament_primary_player_v1",
        start,
    )
    resolver = SQL[start:end]

    assert "v_normalized_name := pg_catalog.lower(pg_catalog.btrim(v_display_name))" in resolver
    assert "public-primary-player-email:" in resolver
    assert "public-primary-player-name:" in resolver
    assert "when v_email_lock <= v_name_lock then v_email_lock else v_name_lock" in resolver
    assert "pg_catalog.lower(pg_catalog.btrim(player.name)) = v_normalized_name" in resolver
    assert "exception when unique_violation then" in resolver
    assert resolver.count("'reason_code', 'existing_player_name_collision'") == 2


def test_idempotent_retry_preserves_created_player_provenance() -> None:
    start = SQL.index(
        "create or replace function private.resolve_public_tournament_primary_player_v1"
    )
    end = SQL.index(
        "revoke all on function private.resolve_public_tournament_primary_player_v1",
        start,
    )
    resolver = SQL[start:end]

    assert "from public.tournament_primary_player_reconciliation as reconciliation" in resolver
    assert "v_existing_reconciliation.created_player_id = v_existing_registration.player_id" in resolver
    assert "v_existing_reconciliation.state in ('created', 'created_unrated')" in resolver
    assert "'state', v_existing_reconciliation.state" in resolver


def test_reconciliation_queue_is_server_only_and_does_not_store_email() -> None:
    table_block = SQL.split(
        "create table if not exists public.tournament_primary_player_reconciliation",
        1,
    )[1].split(");", 1)[0]
    assert "email" not in table_block
    assert "identity_hash text not null" in table_block
    assert "enable row level security" in SQL
    assert "force row level security" in SQL
    assert "from public, anon, authenticated, service_role" in SQL
    assert "to service_role" in SQL
    assert "extensions.digest" in SQL


def test_reconciliation_foreign_keys_have_leading_indexes() -> None:
    assert "tournament_primary_player_reconciliation_tournament_idx" in SQL
    assert "(tournament_id);" in SQL
    assert "tournament_primary_player_reconciliation_created_player_idx" in SQL
    assert "(created_player_id)" in SQL


def test_public_wrappers_remain_service_role_only() -> None:
    assert (
        "grant execute on function private.resolve_public_tournament_primary_player_v1"
        in SQL
    )
    assert SQL.count(
        "revoke all on function public.create_tournament_registration_canonical_v1"
    ) == 1
    assert SQL.count(
        "revoke all on function public.server_create_public_tournament_registration_with_commerce"
    ) >= 2
    assert "grant execute on function public.create_tournament_registration_canonical_v1" in SQL
    assert (
        "grant execute on function public.server_create_public_tournament_registration_with_commerce"
        in SQL
    )


def test_admin_registration_has_a_separate_validated_player_link_path() -> None:
    start = SQL.index(
        "create or replace function public.create_admin_tournament_registration_canonical_v1"
    )
    end = SQL.index(
        "create or replace function public.server_create_public_tournament_registration_with_commerce",
        start,
    )
    admin = SQL[start:end]
    assert "p_registration ->> 'player_id'" in admin
    assert "from public.players as player" in admin
    assert "player.club_id = v_club_id" in admin
    assert "create_tournament_registration_canonical_unlinked_v1(" in admin
    assert "p_registration - 'player_id'" not in admin
    assert "jupr_admin_registration_player_mismatch" in admin
    assert (
        "grant execute on function public.create_admin_tournament_registration_canonical_v1"
        in SQL
    )
