create extension if not exists pgcrypto;

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

comment on table public.live_sessions is
    'Durable recoverable state for JUPR Live quick sessions. Used to restore admin scoring sessions after browser refresh, phone close, or temporary disconnect.';

comment on column public.live_sessions.session_key is
    'Opaque browser URL/session key scoped by club_id. Do not expose Supabase row ids as recovery keys.';

comment on column public.live_sessions.state is
    'JSONB recoverable JUPR Live page state. The app writes a curated payload rather than the full Streamlit session_state.';
