-- Enforce canonical uniqueness constraints (phase A) and runtime invariant RPC.
--
-- This migration can be replayed in environments where legacy core tables might
-- be provisioned outside this repo's migration chain. Guard index creation to
-- avoid hard-failing on missing relations; runtime preflight still blocks writes
-- until required invariants exist.

do $$
begin
  if to_regclass('public.matches') is not null then
    execute $sql$
      create unique index if not exists matches_club_id_idempotency_key_uq
        on public.matches (club_id, idempotency_key)
        where idempotency_key is not null
    $sql$;
  end if;

  if to_regclass('public.players') is not null then
    execute $sql$
      create unique index if not exists players_club_id_normalized_name_uq
        on public.players (club_id, normalized_name)
        where normalized_name is not null
    $sql$;
  end if;
end;
$$;

create or replace function public.assert_app_invariants()
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  if not exists (
    select 1
    from pg_indexes
    where schemaname = 'public'
      and indexname = 'matches_club_id_idempotency_key_uq'
  ) then
    raise exception using
      message = 'Missing required index: matches_club_id_idempotency_key_uq',
      hint = 'Apply migration 202602200001_enforce_uniques_and_preflight.sql before enabling writes.';
  end if;

  if not exists (
    select 1
    from pg_indexes
    where schemaname = 'public'
      and indexname = 'players_club_id_normalized_name_uq'
  ) then
    raise exception using
      message = 'Missing required index: players_club_id_normalized_name_uq',
      hint = 'Apply migration 202602200001_enforce_uniques_and_preflight.sql before enabling writes.';
  end if;
end;
$$;

grant execute on function public.assert_app_invariants() to anon, authenticated, service_role;
