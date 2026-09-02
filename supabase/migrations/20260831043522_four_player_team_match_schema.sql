-- Canonical source for the four-player team-match schema first applied to
-- staging as migration 20260831043522. The staging ledger retained the
-- migration, but the SQL source was not committed with the original change.
--
-- The JUPR data plane is server-only: browsers authenticate with Supabase and
-- call FastAPI, while FastAPI uses service_role after its own authorization.
-- These tables therefore use forced, deny-by-default RLS with no client policy.

create table if not exists public.team_match_competitions (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  scope_type text not null,
  scope_key text not null,
  name text not null,
  roster_mode text not null default 'MIXED_2M2W',
  scoring_mode text not null default 'RALLY',
  target_score smallint not null default 21,
  win_by smallint not null default 2,
  tiebreak_mode text not null default 'DREAMBREAKER',
  round_robin_cycles smallint not null default 1,
  status text not null default 'DRAFT',
  tournament_id uuid null
    references public.tournaments(id) on delete cascade,
  event_option_id text null
    references public.tournament_event_options(id) on delete cascade,
  league_name text null,
  version integer not null default 1,
  created_by text null,
  updated_by text null,
  created_at timestamptz not null default clock_timestamp(),
  updated_at timestamptz not null default clock_timestamp(),
  completed_at timestamptz null,
  constraint team_match_competitions_scope_chk
    check (scope_type in ('GENERATOR', 'LEAGUE', 'TOURNAMENT')),
  constraint team_match_competitions_roster_mode_chk
    check (roster_mode in ('MIXED_2M2W', 'SAME_GENDER', 'OPEN')),
  constraint team_match_competitions_scoring_mode_chk
    check (scoring_mode in ('RALLY', 'SIDEOUT')),
  constraint team_match_competitions_score_chk
    check (target_score between 1 and 99 and win_by between 1 and 10),
  constraint team_match_competitions_tiebreak_chk
    check (tiebreak_mode in ('DREAMBREAKER', 'NONE')),
  constraint team_match_competitions_cycles_chk
    check (round_robin_cycles between 1 and 6),
  constraint team_match_competitions_status_chk
    check (status in ('DRAFT', 'ACTIVE', 'COMPLETE', 'ARCHIVED')),
  constraint team_match_competitions_scope_link_chk
    check (
      (
        scope_type = 'GENERATOR'
        and tournament_id is null
        and event_option_id is null
        and league_name is null
      )
      or (
        scope_type = 'LEAGUE'
        and tournament_id is null
        and event_option_id is null
        and nullif(btrim(league_name), '') is not null
      )
      or (
        scope_type = 'TOURNAMENT'
        and tournament_id is not null
        and event_option_id is not null
        and league_name is null
      )
    )
);

create unique index if not exists team_match_competitions_open_scope_idx
  on public.team_match_competitions (club_id, scope_type, scope_key)
  where status in ('DRAFT', 'ACTIVE');
create index if not exists team_match_competitions_scope_idx
  on public.team_match_competitions (
    club_id,
    scope_type,
    status,
    updated_at desc
  );

create table if not exists public.team_match_teams (
  id uuid primary key default gen_random_uuid(),
  competition_id uuid not null
    references public.team_match_competitions(id) on delete cascade,
  club_id text not null,
  name text not null,
  pairing_code text not null,
  seed integer null,
  status text not null default 'ACTIVE',
  version integer not null default 1,
  created_by text null,
  updated_by text null,
  created_at timestamptz not null default clock_timestamp(),
  updated_at timestamptz not null default clock_timestamp(),
  constraint team_match_teams_competition_id_id_key
    unique (competition_id, id),
  constraint team_match_teams_pairing_chk
    check (
      pairing_code in (
        'STRAIGHT',
        'CROSS',
        'CYCLE_12_34__13_24',
        'CYCLE_12_34__14_23',
        'CYCLE_13_24__14_23'
      )
    ),
  constraint team_match_teams_status_chk
    check (status in ('ACTIVE', 'WITHDRAWN', 'ARCHIVED'))
);

create unique index if not exists team_match_teams_name_idx
  on public.team_match_teams (competition_id, lower(btrim(name)))
  where status = 'ACTIVE';
create index if not exists team_match_teams_scope_idx
  on public.team_match_teams (competition_id, status, seed, created_at);

create table if not exists public.team_match_team_members (
  id uuid primary key default gen_random_uuid(),
  competition_id uuid not null
    references public.team_match_competitions(id) on delete cascade,
  team_id uuid not null,
  club_id text not null,
  slot_key text not null,
  player_id bigint not null references public.players(id) on delete restrict,
  display_name_snapshot text not null,
  gender_snapshot text null,
  status text not null default 'ACTIVE',
  created_at timestamptz not null default clock_timestamp(),
  updated_at timestamptz not null default clock_timestamp(),
  constraint team_match_team_members_competition_id_team_id_fkey
    foreign key (competition_id, team_id)
    references public.team_match_teams(competition_id, id)
    on delete cascade,
  constraint team_match_team_members_slot_chk
    check (slot_key in ('M1', 'M2', 'W1', 'W2', 'P1', 'P2', 'P3', 'P4')),
  constraint team_match_team_members_status_chk
    check (status in ('ACTIVE', 'SUBSTITUTE', 'REMOVED'))
);

create unique index if not exists team_match_team_members_comp_player_idx
  on public.team_match_team_members (competition_id, player_id)
  where status = 'ACTIVE';
create unique index if not exists team_match_team_members_slot_idx
  on public.team_match_team_members (team_id, slot_key)
  where status = 'ACTIVE';
create unique index if not exists team_match_team_members_team_player_idx
  on public.team_match_team_members (team_id, player_id)
  where status = 'ACTIVE';
create index if not exists team_match_team_members_scope_idx
  on public.team_match_team_members (
    competition_id,
    team_id,
    status,
    slot_key
  );

create table if not exists public.team_match_matchups (
  id uuid primary key default gen_random_uuid(),
  competition_id uuid not null
    references public.team_match_competitions(id) on delete cascade,
  club_id text not null,
  stage text not null default 'ROUND_ROBIN',
  round_number integer not null,
  slot_number integer not null,
  team_a_id uuid not null,
  team_b_id uuid not null,
  status text not null default 'SCHEDULED',
  team_a_game_wins smallint not null default 0,
  team_b_game_wins smallint not null default 0,
  team_a_points integer not null default 0,
  team_b_points integer not null default 0,
  winner_team_id uuid null,
  version integer not null default 1,
  created_at timestamptz not null default clock_timestamp(),
  updated_at timestamptz not null default clock_timestamp(),
  constraint team_match_matchups_competition_id_id_key
    unique (competition_id, id),
  constraint team_match_matchups_competition_id_stage_round_number_slot__key
    unique (competition_id, stage, round_number, slot_number),
  constraint team_match_matchups_competition_id_team_a_id_fkey
    foreign key (competition_id, team_a_id)
    references public.team_match_teams(competition_id, id)
    on delete restrict,
  constraint team_match_matchups_competition_id_team_b_id_fkey
    foreign key (competition_id, team_b_id)
    references public.team_match_teams(competition_id, id)
    on delete restrict,
  constraint team_match_matchups_competition_id_winner_team_id_fkey
    foreign key (competition_id, winner_team_id)
    references public.team_match_teams(competition_id, id)
    on delete restrict,
  constraint team_match_matchups_stage_chk
    check (stage in ('ROUND_ROBIN', 'PLAYOFF', 'EXHIBITION')),
  constraint team_match_matchups_status_chk
    check (
      status in (
        'SCHEDULED',
        'IN_PROGRESS',
        'TIEBREAK_REQUIRED',
        'FINAL',
        'VOID'
      )
    ),
  constraint team_match_matchups_distinct_teams_chk
    check (team_a_id <> team_b_id)
);

create index if not exists team_match_matchups_scope_idx
  on public.team_match_matchups (
    competition_id,
    stage,
    round_number,
    slot_number
  );
create index if not exists team_match_matchups_status_idx
  on public.team_match_matchups (competition_id, status);

create table if not exists public.team_match_games (
  id uuid primary key default gen_random_uuid(),
  competition_id uuid not null
    references public.team_match_competitions(id) on delete cascade,
  matchup_id uuid not null,
  club_id text not null,
  game_code text not null,
  game_order smallint not null,
  label text not null,
  team_a_player_ids bigint[] not null,
  team_b_player_ids bigint[] not null,
  counts_for_rating boolean not null default true,
  score_a integer null,
  score_b integer null,
  status text not null default 'SCHEDULED',
  publish_status text not null default 'NOT_STARTED',
  publish_operation_key text null,
  score_request_fingerprint text null,
  official_match_id bigint null references public.matches(id) on delete set null,
  version integer not null default 1,
  created_at timestamptz not null default clock_timestamp(),
  updated_at timestamptz not null default clock_timestamp(),
  finalized_at timestamptz null,
  constraint team_match_games_competition_id_matchup_id_fkey
    foreign key (competition_id, matchup_id)
    references public.team_match_matchups(competition_id, id)
    on delete cascade,
  constraint team_match_games_matchup_id_game_code_key
    unique (matchup_id, game_code),
  constraint team_match_games_code_chk
    check (game_code in ('G1', 'G2', 'G3', 'G4', 'TIEBREAK')),
  constraint team_match_games_order_chk check (game_order between 1 and 5),
  constraint team_match_games_lineup_chk
    check (
      (
        counts_for_rating
        and cardinality(team_a_player_ids) = 2
        and cardinality(team_b_player_ids) = 2
      )
      or (
        not counts_for_rating
        and cardinality(team_a_player_ids) = 4
        and cardinality(team_b_player_ids) = 4
      )
    ),
  constraint team_match_games_score_chk
    check (
      (score_a is null and score_b is null)
      or (
        score_a is not null
        and score_b is not null
        and score_a >= 0
        and score_b >= 0
        and score_a <> score_b
      )
    ),
  constraint team_match_games_status_chk
    check (status in ('BLOCKED', 'SCHEDULED', 'PUBLISHING', 'FINAL', 'VOID')),
  constraint team_match_games_publish_status_chk
    check (
      publish_status in (
        'NOT_STARTED',
        'PUBLISHING',
        'PUBLISHED',
        'NOT_REQUIRED',
        'ERROR'
      )
    )
);

create unique index if not exists team_match_games_official_match_idx
  on public.team_match_games (official_match_id)
  where official_match_id is not null;
create index if not exists team_match_games_recovery_idx
  on public.team_match_games (club_id, publish_status)
  where publish_status in ('PUBLISHING', 'ERROR');
create index if not exists team_match_games_scope_idx
  on public.team_match_games (competition_id, matchup_id, game_order);

alter table public.team_match_competitions enable row level security;
alter table public.team_match_competitions force row level security;
alter table public.team_match_teams enable row level security;
alter table public.team_match_teams force row level security;
alter table public.team_match_team_members enable row level security;
alter table public.team_match_team_members force row level security;
alter table public.team_match_matchups enable row level security;
alter table public.team_match_matchups force row level security;
alter table public.team_match_games enable row level security;
alter table public.team_match_games force row level security;

revoke all on table public.team_match_competitions
  from public, anon, authenticated, service_role;
revoke all on table public.team_match_teams
  from public, anon, authenticated, service_role;
revoke all on table public.team_match_team_members
  from public, anon, authenticated, service_role;
revoke all on table public.team_match_matchups
  from public, anon, authenticated, service_role;
revoke all on table public.team_match_games
  from public, anon, authenticated, service_role;

grant select, insert, update, delete
  on table public.team_match_competitions to service_role;
grant select, insert, update, delete
  on table public.team_match_teams to service_role;
grant select, insert, update, delete
  on table public.team_match_team_members to service_role;
grant select, insert, update, delete
  on table public.team_match_matchups to service_role;
grant select, insert, update, delete
  on table public.team_match_games to service_role;

notify pgrst, 'reload schema';
