-- Badge queue tenant isolation + atomic dequeue + event-key idempotency.
--
-- Guard badge-queue DDL for environments where badge tables are provisioned
-- outside this repository's migration chain.

do $$
begin
  if to_regclass('public.badge_eval_queue') is not null then
    execute $sql$
      alter table public.badge_eval_queue
        add column if not exists event_key text
    $sql$;

    execute $sql$
      update public.badge_eval_queue
      set event_key = coalesce(nullif(match_id, ''), 'legacy:' || id::text)
      where event_key is null or event_key = ''
    $sql$;

    execute $sql$
      alter table public.badge_eval_queue
        alter column event_key set not null
    $sql$;

    execute $sql$
      create unique index if not exists badge_eval_queue_club_event_eventkey_uidx
        on public.badge_eval_queue (club_id, event_type, event_key)
    $sql$;

    execute $sql$
      create or replace function public.dequeue_badge_eval_jobs(
        p_club_id text,
        p_limit integer default 1
      )
      returns setof public.badge_eval_queue
      language sql
      security definer
      set search_path = public
      as $fn$
        with candidates as (
          select q.id
          from public.badge_eval_queue q
          where q.club_id = p_club_id
            and q.status = 'pending'
          order by q.created_at asc, q.id asc
          for update skip locked
          limit greatest(coalesce(p_limit, 1), 1)
        )
        update public.badge_eval_queue q
        set status = 'processing',
            attempts = coalesce(q.attempts, 0) + 1
        from candidates c
        where q.id = c.id
        returning q.*
      $fn$
    $sql$;

    execute $sql$
      grant execute on function public.dequeue_badge_eval_jobs(text, integer)
      to anon, authenticated, service_role
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
    select 1 from pg_indexes where schemaname = 'public' and indexname = 'matches_club_id_idempotency_key_uq'
  ) then
    raise exception using
      message = 'Missing required index: matches_club_id_idempotency_key_uq',
      hint = 'Apply migration 202602200001_enforce_uniques_and_preflight.sql before enabling writes.';
  end if;

  if not exists (
    select 1 from pg_indexes where schemaname = 'public' and indexname = 'players_club_id_normalized_name_uq'
  ) then
    raise exception using
      message = 'Missing required index: players_club_id_normalized_name_uq',
      hint = 'Apply migration 202602200001_enforce_uniques_and_preflight.sql before enabling writes.';
  end if;

  if not exists (
    select 1 from pg_indexes where schemaname = 'public' and indexname = 'badge_eval_queue_club_event_eventkey_uidx'
  ) then
    raise exception using
      message = 'Missing required index: badge_eval_queue_club_event_eventkey_uidx',
      hint = 'Apply migration 202602200002_badge_queue_tenant_atomic.sql before enabling writes.';
  end if;
end;
$$;

grant execute on function public.assert_app_invariants() to anon, authenticated, service_role;
