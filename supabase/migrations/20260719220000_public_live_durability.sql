-- Order 25: durable, rate-limited public JUPR Live mutations.
--
-- The browser receives a high-entropy edit token once. Only its SHA-256 digest
-- is retained. FastAPI owns every read/write through the service role; neither
-- anonymous nor authenticated browser clients can access recovery state.

alter table public.live_sessions
  add column if not exists version bigint not null default 1,
  add column if not exists edit_token_hash text,
  add column if not exists creation_operation_key text,
  add column if not exists last_operation_key text,
  add column if not exists last_request_fingerprint text,
  add column if not exists pending_operation_key text,
  add column if not exists pending_operation_action text,
  add column if not exists completed_at timestamptz;

-- Migrate legacy web sessions without preserving their plaintext token.
update public.live_sessions
   set edit_token_hash = encode(
         digest(coalesce(state #>> '{private,edit_token}', ''), 'sha256'),
         'hex'
       )
 where nullif(state #>> '{private,edit_token}', '') is not null
   and nullif(edit_token_hash, '') is null;

update public.live_sessions
   set state = state #- '{private,edit_token}'
 where state #> '{private,edit_token}' is not null;

create unique index if not exists live_sessions_creation_operation_unique
  on public.live_sessions (club_id, creation_operation_key)
  where creation_operation_key is not null;

create index if not exists live_sessions_last_operation_idx
  on public.live_sessions (club_id, last_operation_key)
  where last_operation_key is not null;

-- Existing Streamlit and admin writers predate optimistic versions. Normalize
-- every update at the table boundary so their writes invalidate public CAS
-- snapshots instead of being silently overwritten. While completion is
-- reserved, only the final operation that clears that reservation may update or
-- delete the row.
create or replace function public.guard_live_session_version_and_completion()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
begin
  if tg_op = 'DELETE' then
    if old.pending_operation_key is not null then
      raise exception using errcode = '55000', message = 'live session completion is reserved';
    end if;
    return old;
  end if;
  if old.pending_operation_key is not null then
    if new.pending_operation_key is not null then
      raise exception using errcode = '55000', message = 'live session completion is reserved';
    end if;
    if new.last_operation_key is distinct from old.pending_operation_key then
      raise exception using errcode = '55000', message = 'live session completion reservation does not match';
    end if;
  end if;
  if new.version is null or new.version <= old.version then
    new.version := old.version + 1;
  end if;
  return new;
end;
$$;

revoke all on function public.guard_live_session_version_and_completion()
  from public, anon, authenticated;
grant execute on function public.guard_live_session_version_and_completion()
  to service_role;

drop trigger if exists guard_live_session_version_and_completion_before_write
  on public.live_sessions;
create trigger guard_live_session_version_and_completion_before_write
before update or delete on public.live_sessions
for each row execute function public.guard_live_session_version_and_completion();

create table if not exists public.public_live_operations (
  operation_key text primary key,
  club_id text not null,
  session_key text,
  action text not null,
  idempotency_key text not null,
  request_fingerprint text not null,
  requester_hash text not null,
  expected_version bigint,
  status text not null default 'intent',
  request_json jsonb not null default '{}'::jsonb,
  result_json jsonb not null default '{}'::jsonb,
  error_text text,
  executor_token text,
  lease_expires_at timestamptz,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  completed_at timestamptz,
  constraint public_live_operations_status_check check (
    status in ('intent', 'running', 'applied', 'completed', 'rejected', 'recovery_required')
  ),
  constraint public_live_operations_key_check check (char_length(operation_key) = 64),
  constraint public_live_operations_fingerprint_check check (char_length(request_fingerprint) = 64),
  constraint public_live_operations_requester_check check (char_length(requester_hash) = 64),
  constraint public_live_operations_idempotency_unique unique (club_id, action, idempotency_key)
);

create index if not exists public_live_operations_rate_idx
  on public.public_live_operations (club_id, requester_hash, action, created_at desc);

create index if not exists public_live_operations_club_rate_idx
  on public.public_live_operations (club_id, action, created_at desc);

create index if not exists public_live_operations_requester_mutation_rate_idx
  on public.public_live_operations (club_id, requester_hash, created_at desc)
  where action <> 'create';

create index if not exists public_live_operations_club_mutation_rate_idx
  on public.public_live_operations (club_id, created_at desc)
  where action <> 'create';

create index if not exists public_live_operations_session_idx
  on public.public_live_operations (club_id, session_key, created_at desc)
  where session_key is not null;

-- Application-configured limits fail closed at a lower threshold. These hard
-- database ceilings provide a second, atomic bound even if a caller spoofs a
-- forwarding header or races concurrent inserts.
create or replace function public.enforce_public_live_operation_limits()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
declare
  requester_count integer;
  club_count integer;
  requester_limit integer;
  club_limit integer;
begin
  perform pg_advisory_xact_lock(hashtextextended('public-live:club:' || new.club_id, 0));
  perform pg_advisory_xact_lock(
    hashtextextended('public-live:requester:' || new.club_id || ':' || new.requester_hash, 0)
  );

  if new.action = 'create' then
    requester_limit := 30;
    club_limit := 100;
    select count(*) into requester_count
      from public.public_live_operations
     where club_id = new.club_id
       and requester_hash = new.requester_hash
       and action = 'create'
       and created_at >= timezone('utc', now()) - interval '1 hour';
    select count(*) into club_count
      from public.public_live_operations
     where club_id = new.club_id
       and action = 'create'
       and created_at >= timezone('utc', now()) - interval '1 hour';
  else
    requester_limit := 500;
    club_limit := 5000;
    select count(*) into requester_count
      from public.public_live_operations
     where club_id = new.club_id
       and requester_hash = new.requester_hash
       and action <> 'create'
       and created_at >= timezone('utc', now()) - interval '1 hour';
    select count(*) into club_count
      from public.public_live_operations
     where club_id = new.club_id
       and action <> 'create'
       and created_at >= timezone('utc', now()) - interval '1 hour';
  end if;

  if requester_count >= requester_limit or club_count >= club_limit then
    raise exception using
      errcode = 'P0001',
      message = 'public live rate limit exceeded';
  end if;
  return new;
end;
$$;

revoke all on function public.enforce_public_live_operation_limits() from public, anon, authenticated;
grant execute on function public.enforce_public_live_operation_limits() to service_role;

drop trigger if exists public_live_operation_limits_before_insert on public.public_live_operations;
create trigger public_live_operation_limits_before_insert
before insert on public.public_live_operations
for each row execute function public.enforce_public_live_operation_limits();

alter table public.live_sessions enable row level security;
alter table public.live_sessions force row level security;
revoke all on table public.live_sessions from public, anon, authenticated;
grant select, insert, update, delete on table public.live_sessions to service_role;

alter table public.public_live_operations enable row level security;
alter table public.public_live_operations force row level security;
revoke all on table public.public_live_operations from public, anon, authenticated;
grant select, insert, update on table public.public_live_operations to service_role;

-- Club Social completion spans the private session row plus the existing
-- moderation tables. Claim exactly one executor before entering that multi-row
-- submit so concurrent same-key retries cannot both create/link participants.
create or replace function public.claim_public_live_completion_executor(
  p_club_id text,
  p_operation_key text,
  p_executor_token text,
  p_lease_seconds integer default 300
)
returns setof public.public_live_operations
language plpgsql
set search_path = public, pg_temp
as $$
begin
  if nullif(trim(p_executor_token), '') is null then
    raise exception using errcode = '22023', message = 'executor token is required';
  end if;
  return query
  update public.public_live_operations as operation
     set executor_token = p_executor_token,
         lease_expires_at = timezone('utc', now())
           + make_interval(secs => greatest(30, least(coalesce(p_lease_seconds, 300), 600))),
         status = 'running',
         error_text = null,
         updated_at = timezone('utc', now())
   where club_id = p_club_id
     and operation_key = p_operation_key
     and action = 'complete'
     and status in ('intent', 'running', 'applied', 'recovery_required')
     and (
       executor_token is null
       or lease_expires_at is null
       or lease_expires_at <= timezone('utc', now())
       or executor_token = p_executor_token
     )
  returning operation.*;
end;
$$;

revoke all on function public.claim_public_live_completion_executor(text, text, text, integer)
  from public, anon, authenticated;
grant execute on function public.claim_public_live_completion_executor(text, text, text, integer)
  to service_role;

comment on column public.live_sessions.edit_token_hash is
  'SHA-256 digest of the one-time public edit token. The plaintext token must never be stored.';
comment on column public.live_sessions.pending_operation_key is
  'Server-only reservation for a recoverable multi-table completion; ordinary writes fail closed while present.';
comment on table public.public_live_operations is
  'FastAPI-private idempotency, recovery, and durable anti-abuse ledger for public JUPR Live writes.';

notify pgrst, 'reload schema';
