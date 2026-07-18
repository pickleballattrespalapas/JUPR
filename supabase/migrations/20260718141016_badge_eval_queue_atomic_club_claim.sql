-- Make badge evaluation queue claims atomic and strictly club-scoped.
--
-- The queue used to deduplicate on (event_type, match_id), which prevented the
-- same source match identifier from being queued independently for two clubs.
-- Workers also selected and updated jobs in separate PostgREST requests, so two
-- concurrent workers could claim the same pending row. Keep queue access
-- service-role-only and expose one atomic claim RPC for server workers.

create table if not exists public.badge_eval_queue (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  club_id text not null,
  context_id text,
  event_type text not null,
  player_ids bigint[] not null default '{}'::bigint[],
  match_id text,
  payload_json jsonb not null default '{}'::jsonb,
  status text not null default 'pending',
  attempts integer not null default 0,
  last_error text,
  processed_at timestamptz
);

do $migration_preflight$
begin
  if exists (
    select 1
    from public.badge_eval_queue as queue
    where queue.match_id is not null
    group by queue.club_id, queue.event_type, queue.match_id
    having count(*) > 1
  ) then
    raise exception using
      errcode = '23505',
      message = 'JUPR_BADGE_QUEUE_DUPLICATE: remove duplicate club/event/match queue rows before applying atomic queue claims.';
  end if;
end
$migration_preflight$;

-- Remove both legacy names seen in repository and staging history. Drop a
-- same-named constraint first in case an environment created one instead of a
-- standalone index.
alter table public.badge_eval_queue
  drop constraint if exists badge_eval_queue_event_match_key;
alter table public.badge_eval_queue
  drop constraint if exists badge_eval_queue_event_match_uidx;
alter table public.badge_eval_queue
  drop constraint if exists badge_eval_queue_club_event_match_uidx;
alter table public.badge_eval_queue
  drop constraint if exists uq_badge_eval_queue_club_event_match;

drop index if exists public.badge_eval_queue_event_match_key;
drop index if exists public.badge_eval_queue_event_match_uidx;
drop index if exists public.badge_eval_queue_club_event_match_uidx;
drop index if exists public.uq_badge_eval_queue_club_event_match;

-- PostgreSQL unique indexes allow multiple NULL values by default, so this
-- supports non-match queue events while giving PostgREST a non-partial unique
-- target for on_conflict=club_id,event_type,match_id.
create unique index if not exists badge_eval_queue_club_event_match_key
  on public.badge_eval_queue (club_id, event_type, match_id);

create index if not exists badge_eval_queue_club_status_created_idx
  on public.badge_eval_queue (club_id, status, created_at, id);

create or replace function public.claim_badge_eval_queue_job(
  p_club_id text
)
returns setof public.badge_eval_queue
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_job_id uuid;
begin
  if nullif(pg_catalog.btrim(p_club_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_BADGE_QUEUE_CLUB_REQUIRED: club_id is required to claim a badge queue job.';
  end if;

  select queue.id
  into v_job_id
  from public.badge_eval_queue as queue
  where queue.club_id = pg_catalog.btrim(p_club_id)
    and queue.status = 'pending'
  order by queue.created_at asc, queue.id asc
  for update skip locked
  limit 1;

  if v_job_id is null then
    return;
  end if;

  return query
  update public.badge_eval_queue as queue
  set
    status = 'processing',
    attempts = queue.attempts + 1
  where queue.id = v_job_id
    and queue.club_id = pg_catalog.btrim(p_club_id)
    and queue.status = 'pending'
  returning queue.*;
end
$function$;

alter table public.badge_eval_queue enable row level security;

revoke all on table public.badge_eval_queue from public, anon, authenticated;
grant select, insert, update on table public.badge_eval_queue to service_role;

revoke all on function public.claim_badge_eval_queue_job(text)
  from public, anon, authenticated;
grant execute on function public.claim_badge_eval_queue_job(text)
  to service_role;

notify pgrst, 'reload schema';
