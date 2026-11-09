-- Opt-in ownership for new claims. Do not alter or replay existing jobs.
-- Expired processing jobs stay processing for operator reconciliation.
alter table public.badge_eval_queue
  add column if not exists claim_token uuid,
  add column if not exists worker_id text,
  add column if not exists started_at timestamptz,
  add column if not exists heartbeat_at timestamptz,
  add column if not exists lease_expires_at timestamptz;

create index if not exists badge_eval_queue_active_lease_idx
  on public.badge_eval_queue (club_id, lease_expires_at)
  where status = 'processing' and claim_token is not null;

create or replace function public.claim_badge_eval_queue_job_leased(
  p_club_id text, p_worker_id text, p_lease_seconds integer default 120
) returns setof public.badge_eval_queue
language plpgsql security invoker set search_path = ''
as $function$
declare
  v_job_id uuid;
  v_now timestamptz := clock_timestamp();
begin
  if nullif(btrim(p_club_id), '') is null or nullif(btrim(p_worker_id), '') is null
     or length(p_worker_id) > 200 or p_lease_seconds is null or p_lease_seconds not between 60 and 900 then
    raise exception 'Invalid badge claim parameters';
  end if;
  -- Serialize the short claim transaction, not the whole calculation. At most
  -- one lease-aware calculation per club; legacy claims are left untouched.
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('badge-lease:' || btrim(p_club_id), 0));
  v_now := clock_timestamp();
  if exists (select 1 from public.badge_eval_queue q
             where q.club_id = btrim(p_club_id) and q.status = 'processing'
               and q.claim_token is not null and q.lease_expires_at > v_now) then
    return;
  end if;
  select q.id into v_job_id from public.badge_eval_queue q
    where q.club_id = btrim(p_club_id) and q.status = 'pending'
    order by q.created_at, q.id for update skip locked limit 1;
  if v_job_id is null then return; end if;
  return query update public.badge_eval_queue q set
    status = 'processing', attempts = q.attempts + 1,
    claim_token = gen_random_uuid(), worker_id = p_worker_id,
    started_at = v_now, heartbeat_at = v_now,
    lease_expires_at = v_now + make_interval(secs => p_lease_seconds),
    processed_at = null, last_error = null
    where q.id = v_job_id and q.status = 'pending' returning q.*;
end
$function$;

create or replace function public.heartbeat_badge_eval_queue_job(
  p_job_id uuid, p_claim_token uuid, p_lease_seconds integer default 120
) returns boolean
language plpgsql security invoker set search_path = ''
as $function$
declare v_count integer; v_now timestamptz := clock_timestamp();
begin
  if p_lease_seconds is null or p_lease_seconds not between 60 and 900 then raise exception 'Invalid badge lease duration'; end if;
  update public.badge_eval_queue q set heartbeat_at = v_now,
    lease_expires_at = v_now + make_interval(secs => p_lease_seconds)
    where q.id = p_job_id and q.claim_token = p_claim_token
      and q.status = 'processing' and q.lease_expires_at > v_now;
  get diagnostics v_count = row_count;
  return v_count = 1;
end
$function$;

create or replace function public.finish_badge_eval_queue_job(
  p_job_id uuid, p_claim_token uuid, p_status text, p_error text default null
) returns boolean
language plpgsql security invoker set search_path = ''
as $function$
declare v_count integer; v_now timestamptz := clock_timestamp();
begin
  if p_status is null or p_status not in ('done', 'error') then raise exception 'Invalid badge completion status'; end if;
  update public.badge_eval_queue q set status = p_status, processed_at = v_now,
    heartbeat_at = v_now, lease_expires_at = null, last_error = left(p_error, 2000)
    where q.id = p_job_id and q.claim_token = p_claim_token
      and q.status = 'processing' and q.lease_expires_at > v_now;
  get diagnostics v_count = row_count;
  return v_count = 1;
end
$function$;

revoke all on function public.claim_badge_eval_queue_job_leased(text,text,integer) from public, anon, authenticated;
revoke all on function public.heartbeat_badge_eval_queue_job(uuid,uuid,integer) from public, anon, authenticated;
revoke all on function public.finish_badge_eval_queue_job(uuid,uuid,text,text) from public, anon, authenticated;
grant execute on function public.claim_badge_eval_queue_job_leased(text,text,integer) to service_role;
grant execute on function public.heartbeat_badge_eval_queue_job(uuid,uuid,integer) to service_role;
grant execute on function public.finish_badge_eval_queue_job(uuid,uuid,text,text) to service_role;

notify pgrst, 'reload schema';
