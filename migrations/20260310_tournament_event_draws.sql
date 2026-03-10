-- Division-scoped tournament operations layer

create table if not exists public.tournament_event_draws (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  registration_day_id text null,
  event_option_id text null,
  name text not null,
  status text not null default 'draft',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_tournament_event_draws_tournament on public.tournament_event_draws (tournament_id);
create unique index if not exists uq_tournament_event_draw_unique on public.tournament_event_draws (tournament_id, registration_day_id, event_option_id, name);

alter table public.tournament_teams
  add column if not exists draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  add column if not exists registration_day_id text null,
  add column if not exists event_option_id text null,
  add column if not exists source text null,
  add column if not exists notes text null;

alter table public.tournament_games
  add column if not exists draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  add column if not exists registration_day_id text null,
  add column if not exists event_option_id text null;

create index if not exists idx_tournament_teams_draw_id on public.tournament_teams (draw_id);
create index if not exists idx_tournament_games_draw_id on public.tournament_games (draw_id);

create unique index if not exists uq_tournament_teams_draw_team
  on public.tournament_teams (tournament_id, draw_id, team_number);
