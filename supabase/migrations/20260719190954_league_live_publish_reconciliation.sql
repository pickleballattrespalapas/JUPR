-- Durable order-21 League Live publish/reconciliation contract.
--
-- These tables are FastAPI-private coordination records. Supabase JWTs authorize
-- the operator at FastAPI, while only the server-side service role reaches this
-- data plane. Matches use deterministic UUID contexts so an interrupted request
-- can reconcile a completed publish without publishing the round twice.

create table if not exists public.league_live_publish_operations (
    id uuid primary key,
    club_id text not null,
    session_id uuid not null references public.league_live_sessions(id) on delete cascade,
    round_number integer not null,
    idempotency_key text not null,
    request_fingerprint text not null,
    plan_operation_key text not null,
    expected_session_updated_at timestamptz not null,
    status text not null default 'intent',
    request_json jsonb not null default '{}'::jsonb,
    match_context_ids jsonb not null default '[]'::jsonb,
    published_match_ids jsonb not null default '[]'::jsonb,
    rating_before_json jsonb not null default '[]'::jsonb,
    rating_after_json jsonb not null default '[]'::jsonb,
    result_json jsonb not null default '{}'::jsonb,
    error_text text,
    attempt_count integer not null default 0,
    created_by text not null,
    updated_by text not null,
    completed_at timestamptz,
    completion_audited_at timestamptz,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint league_live_publish_operations_round_check check (round_number >= 1),
    constraint league_live_publish_operations_attempt_check check (attempt_count >= 0),
    constraint league_live_publish_operations_status_check check (
        status in ('intent', 'publishing', 'published', 'reconciling', 'retryable', 'recovery_required', 'completed', 'compensated')
    ),
    constraint league_live_publish_operations_round_unique unique (session_id, round_number),
    constraint league_live_publish_operations_idempotency_unique unique (club_id, idempotency_key)
);

create index if not exists idx_league_live_publish_operations_club_status_updated
    on public.league_live_publish_operations (club_id, status, updated_at desc);

create index if not exists idx_league_live_publish_operations_session_round
    on public.league_live_publish_operations (club_id, session_id, round_number);

create table if not exists public.league_live_guest_players (
    id uuid primary key,
    club_id text not null,
    session_id uuid not null references public.league_live_sessions(id) on delete cascade,
    player_id bigint,
    idempotency_key text not null,
    request_fingerprint text not null,
    guest_name text not null,
    starting_jupr numeric(4, 2) not null,
    reason text not null,
    status text not null default 'intent',
    error_text text,
    created_by text not null,
    updated_by text not null,
    completed_at timestamptz,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint league_live_guest_players_starting_jupr_check check (starting_jupr between 1.0 and 7.0),
    constraint league_live_guest_players_status_check check (status in ('intent', 'completed', 'recovery_required', 'compensated')),
    constraint league_live_guest_players_idempotency_unique unique (club_id, idempotency_key),
    constraint league_live_guest_players_session_player_unique unique (session_id, player_id)
);

create index if not exists idx_league_live_guest_players_session_created
    on public.league_live_guest_players (club_id, session_id, created_at desc);

-- The application derives one UUIDv5 text value per session/round/match slot and
-- verifies every row after the Python match service returns.
create unique index if not exists idx_matches_league_live_publish_context
    on public.matches (club_id, context_type, context_id)
    where context_type = 'league_live_session' and context_id is not null;

alter table public.league_live_publish_operations enable row level security;
alter table public.league_live_guest_players enable row level security;

revoke all on table public.league_live_publish_operations from public, anon, authenticated;
revoke all on table public.league_live_guest_players from public, anon, authenticated;

grant select, insert, update, delete on table public.league_live_publish_operations to service_role;
grant select, insert, update, delete on table public.league_live_guest_players to service_role;

comment on table public.league_live_publish_operations is
    'FastAPI-private League Live all-match publish intents and reconciliation state.';
comment on table public.league_live_guest_players is
    'FastAPI-private idempotency and recovery records for League Live guest creation.';

notify pgrst, 'reload schema';
