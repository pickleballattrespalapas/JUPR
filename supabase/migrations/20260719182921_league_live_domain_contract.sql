-- Canonical, server-only League Manager Live persistence contract.
--
-- The legacy root migration remains the Streamlit/Next bootstrap source. This
-- migration makes the same schema available through the canonical Supabase
-- chain, adds durable idempotency, and opts each private table out of browser
-- Data API access. FastAPI is the only supported data-plane client and must use
-- SUPABASE_SERVICE_ROLE_KEY.

create extension if not exists pgcrypto;

create table if not exists public.league_live_sessions (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    league_name text not null,
    week_tag text not null default '',
    status text not null default 'setup',
    total_rounds integer not null default 1,
    current_round integer not null default 1,
    roster_json jsonb not null default '[]'::jsonb,
    current_court_state_json jsonb not null default '[]'::jsonb,
    notes text,
    created_by text,
    updated_by text,
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint league_live_sessions_status_check
        check (status in ('setup', 'active', 'paused', 'complete', 'archived')),
    constraint league_live_sessions_total_rounds_check
        check (total_rounds between 1 and 50),
    constraint league_live_sessions_current_round_check
        check (current_round >= 1)
);

create table if not exists public.league_live_rounds (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    session_id uuid not null references public.league_live_sessions(id) on delete cascade,
    round_number integer not null,
    round_label text,
    status text not null default 'draft',
    match_date date,
    preview_json jsonb not null default '{}'::jsonb,
    matches_json jsonb not null default '[]'::jsonb,
    movement_json jsonb not null default '{}'::jsonb,
    operation_key text,
    submitted_match_count integer not null default 0,
    submitted_match_ids jsonb not null default '[]'::jsonb,
    submitted_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint league_live_rounds_status_check
        check (status in ('draft', 'generated', 'submitted', 'voided')),
    constraint league_live_rounds_round_number_check check (round_number >= 1),
    constraint league_live_rounds_submitted_count_check check (submitted_match_count >= 0),
    constraint league_live_rounds_unique_round unique (session_id, round_number)
);

alter table public.league_live_rounds
    add column if not exists operation_key text;

create table if not exists public.league_live_courts (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    session_id uuid not null references public.league_live_sessions(id) on delete cascade,
    round_number integer not null,
    court_number integer not null,
    format_type text not null default '4-Player',
    player_names jsonb not null default '[]'::jsonb,
    players_json jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint league_live_courts_round_check check (round_number >= 1),
    constraint league_live_courts_court_check check (court_number >= 1),
    constraint league_live_courts_unique_court unique (session_id, round_number, court_number)
);

create index if not exists idx_league_live_sessions_club_status_updated
    on public.league_live_sessions (club_id, status, updated_at desc);

create index if not exists idx_league_live_sessions_club_league_updated
    on public.league_live_sessions (club_id, league_name, updated_at desc);

create index if not exists idx_league_live_rounds_club_session_round
    on public.league_live_rounds (club_id, session_id, round_number);

create unique index if not exists idx_league_live_rounds_session_operation
    on public.league_live_rounds (session_id, operation_key)
    where operation_key is not null and operation_key <> '';

create index if not exists idx_league_live_courts_club_session_round
    on public.league_live_courts (club_id, session_id, round_number, court_number);

alter table public.league_live_sessions enable row level security;
alter table public.league_live_rounds enable row level security;
alter table public.league_live_courts enable row level security;

revoke all on table public.league_live_sessions from public, anon, authenticated;
revoke all on table public.league_live_rounds from public, anon, authenticated;
revoke all on table public.league_live_courts from public, anon, authenticated;

grant usage on schema public to service_role;
grant select, insert, update, delete on table public.league_live_sessions to service_role;
grant select, insert, update, delete on table public.league_live_rounds to service_role;
grant select, insert, update, delete on table public.league_live_courts to service_role;

comment on table public.league_live_sessions is
    'Private, resumable League Manager Live state. FastAPI service-role access only; never project roster_json directly to browsers.';
comment on table public.league_live_rounds is
    'Private League Live round state, Python-authoritative movement plans, and idempotent operation keys.';
comment on table public.league_live_courts is
    'Private per-round court snapshots used for League Live recovery and audit context.';

notify pgrst, 'reload schema';
