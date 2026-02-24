begin;

create table if not exists public.live_ladder_sessions (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_name text not null,
  week text not null,
  total_rounds integer not null,
  ladder_round_num integer not null default 1,
  state text not null default 'SETUP',
  players_per_court integer not null default 4,
  court_sizes jsonb not null default '[]'::jsonb,
  ordered_player_ids jsonb not null default '[]'::jsonb,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists live_ladder_sessions_active_uniq
on public.live_ladder_sessions (club_id, league_name, week)
where active = true;

create table if not exists public.live_ladder_rounds (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.live_ladder_sessions(id) on delete cascade,
  round_num integer not null,
  status text not null default 'SUBMITTED',
  schedule_sig text,
  schedule_json jsonb,
  results_json jsonb,
  round_stats_json jsonb,
  movement_preview_json jsonb,
  ordered_ids_before jsonb,
  ordered_ids_after jsonb,
  court_sizes jsonb,
  roster_ids jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (session_id, round_num)
);

create index if not exists idx_live_ladder_rounds_session_round
  on public.live_ladder_rounds (session_id, round_num desc);

drop trigger if exists live_ladder_sessions_set_updated_at on public.live_ladder_sessions;
create trigger live_ladder_sessions_set_updated_at
before update on public.live_ladder_sessions
for each row execute function public.set_updated_at_timestamp();

drop trigger if exists live_ladder_rounds_set_updated_at on public.live_ladder_rounds;
create trigger live_ladder_rounds_set_updated_at
before update on public.live_ladder_rounds
for each row execute function public.set_updated_at_timestamp();

commit;
