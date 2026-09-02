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
    constraint league_live_sessions_status_check check (status in ('setup', 'active', 'paused', 'complete', 'archived')),
    constraint league_live_sessions_total_rounds_check check (total_rounds between 1 and 50),
    constraint league_live_sessions_current_round_check check (current_round >= 1)
);
create index if not exists idx_league_live_sessions_club_status_updated
    on public.league_live_sessions (club_id, status, updated_at desc);
create index if not exists idx_league_live_sessions_club_league_updated
    on public.league_live_sessions (club_id, league_name, updated_at desc);

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
    submitted_match_count integer not null default 0,
    submitted_match_ids jsonb not null default '[]'::jsonb,
    submitted_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint league_live_rounds_status_check check (status in ('draft', 'generated', 'submitted', 'voided')),
    constraint league_live_rounds_round_number_check check (round_number >= 1),
    constraint league_live_rounds_submitted_count_check check (submitted_match_count >= 0),
    constraint league_live_rounds_unique_round unique (session_id, round_number)
);
create index if not exists idx_league_live_rounds_club_session_round
    on public.league_live_rounds (club_id, session_id, round_number);

create table if not exists public.league_live_courts (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    session_id uuid not null references public.league_live_sessions(id) on delete cascade,
    round_number integer not null,
    court_number integer not null,
    format_type text not null default '4-player',
    player_names jsonb not null default '[]'::jsonb,
    players_json jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint league_live_courts_round_check check (round_number >= 1),
    constraint league_live_courts_court_check check (court_number >= 1),
    constraint league_live_courts_unique_court unique (session_id, round_number, court_number)
);
create index if not exists idx_league_live_courts_club_session_round
    on public.league_live_courts (club_id, session_id, round_number, court_number);
