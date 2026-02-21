begin;

-- Tournaments
create table if not exists public.tournaments (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  name text not null,
  format text not null,
  status text not null default 'draft',
  created_at timestamptz not null default now()
);

create index if not exists idx_tournaments_club on public.tournaments (club_id);

-- Tournament Teams
create table if not exists public.tournament_teams (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  team_number integer not null,
  player1_id bigint,
  player2_id bigint,
  created_at timestamptz not null default now(),
  constraint uq_tournament_team_number unique (club_id, tournament_id, team_number)
);

create index if not exists idx_tournament_teams_club_tournament
  on public.tournament_teams (club_id, tournament_id);

-- Tournament Games
create table if not exists public.tournament_games (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  round integer,
  team1_id uuid references public.tournament_teams(id) on delete cascade,
  team2_id uuid references public.tournament_teams(id) on delete cascade,
  score1 integer,
  score2 integer,
  created_at timestamptz not null default now()
);

create index if not exists idx_tournament_games_club_tournament
  on public.tournament_games (club_id, tournament_id);

-- Tournament Podium
create table if not exists public.tournament_podium (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  team_id uuid not null references public.tournament_teams(id) on delete cascade,
  placement integer not null,
  created_at timestamptz not null default now(),
  constraint uq_podium_placement unique (club_id, tournament_id, placement),
  constraint uq_podium_team unique (club_id, tournament_id, team_id)
);

create index if not exists idx_tournament_podium_club_tournament
  on public.tournament_podium (club_id, tournament_id);

commit;
