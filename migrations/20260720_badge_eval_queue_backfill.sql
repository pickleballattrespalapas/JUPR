create table if not exists public.badge_eval_queue (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    club_id text not null,
    context_id text,
    event_type text not null,
    player_ids bigint[] not null default '{}',
    match_id text,
    payload_json jsonb not null default '{}'::jsonb,
    status text not null default 'pending',
    attempts integer not null default 0,
    last_error text,
    processed_at timestamptz
);

create index if not exists badge_eval_queue_status_created_idx
    on public.badge_eval_queue (status, created_at);

create index if not exists badge_eval_queue_club_status_idx
    on public.badge_eval_queue (club_id, status);

create unique index if not exists badge_eval_queue_event_match_uidx
    on public.badge_eval_queue (event_type, match_id)
    where match_id is not null;
