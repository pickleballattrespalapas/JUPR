create extension if not exists pgcrypto;

-- Idempotent schema contract for durable JUPR Live sessions.
-- The public API reads this table with the backend service-role key and returns
-- a sanitized public projection. Do not grant anonymous clients direct SELECT
-- on this table because state contains recoverable admin/Streamlit session data.

create table if not exists public.live_sessions (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    session_key text not null,
    created_by uuid null,
    created_by_email text null,
    status text not null default 'active',
    title text null,
    state jsonb not null default '{}'::jsonb,
    source text not null default 'jupr_live_admin',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    expires_at timestamptz null,
    constraint live_sessions_status_check
        check (status in ('active', 'completed', 'abandoned', 'archived'))
);

alter table public.live_sessions
    add column if not exists created_by uuid null,
    add column if not exists created_by_email text null,
    add column if not exists status text not null default 'active',
    add column if not exists title text null,
    add column if not exists state jsonb not null default '{}'::jsonb,
    add column if not exists source text not null default 'jupr_live_admin',
    add column if not exists created_at timestamptz not null default now(),
    add column if not exists updated_at timestamptz not null default now(),
    add column if not exists last_seen_at timestamptz not null default now(),
    add column if not exists expires_at timestamptz null;

alter table public.live_sessions
    alter column status set default 'active',
    alter column state set default '{}'::jsonb,
    alter column source set default 'jupr_live_admin',
    alter column created_at set default now(),
    alter column updated_at set default now(),
    alter column last_seen_at set default now();

create unique index if not exists live_sessions_club_session_key_idx
    on public.live_sessions (club_id, session_key);

create index if not exists live_sessions_club_status_updated_idx
    on public.live_sessions (club_id, status, updated_at desc);

create index if not exists live_sessions_expires_at_idx
    on public.live_sessions (expires_at)
    where expires_at is not null;

create index if not exists live_sessions_state_event_type_idx
    on public.live_sessions ((state->>'event_type'));

create or replace function public.set_updated_at_timestamp()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists live_sessions_set_updated_at on public.live_sessions;
create trigger live_sessions_set_updated_at
before update on public.live_sessions
for each row
execute function public.set_updated_at_timestamp();

grant usage on schema public to service_role;
grant select, insert, update, delete on public.live_sessions to service_role;

comment on table public.live_sessions is
    'Durable recoverable state for JUPR Live quick sessions. Read by the FastAPI backend with service_role only, then sanitized before public exposure.';

comment on column public.live_sessions.state is
    'JSONB recoverable JUPR Live page state. Contains internal recovery data and must not be granted directly to anonymous clients.';
