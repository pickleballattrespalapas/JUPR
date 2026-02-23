begin;

-- Session Ladder Engine MVP schema

create table if not exists public.session_ladder_sessions (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_id text not null,
  season_id text null,
  session_starts_at timestamptz not null,
  session_ends_at timestamptz null,
  courts_available integer not null check (courts_available > 0),
  players_per_court integer not null check (players_per_court in (4, 5)),
  state text not null default 'draft' check (state in ('draft', 'setup', 'active', 'completed', 'cancelled', 'archived')),
  created_by text not null,
  updated_by text null,
  notes text null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint session_ladder_sessions_club_fk
    foreign key (club_id) references public.clubs(id) on update cascade on delete restrict
);

create index if not exists idx_session_ladder_sessions_club_starts
  on public.session_ladder_sessions (club_id, session_starts_at desc);
create index if not exists idx_session_ladder_sessions_club_state
  on public.session_ladder_sessions (club_id, state, session_starts_at desc);

create table if not exists public.session_ladder_roster_entries (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.session_ladder_sessions(id) on delete cascade,
  player_id bigint not null references public.players(id) on delete restrict,
  status text not null default 'EXPECTED' check (status in ('EXPECTED', 'CHECKED_IN', 'NO_SHOW', 'WALK_IN')),
  rating_snapshot numeric(10,3) not null,
  seed_order integer null,
  created_by text null,
  updated_by text null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint session_ladder_roster_entries_unique unique (session_id, player_id)
);

create index if not exists idx_session_ladder_roster_entries_session_status
  on public.session_ladder_roster_entries (session_id, status, seed_order);
create index if not exists idx_session_ladder_roster_entries_club_player
  on public.session_ladder_roster_entries (club_id, player_id);

create table if not exists public.session_ladder_court_pods (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.session_ladder_sessions(id) on delete cascade,
  round_number integer not null check (round_number > 0),
  court_number integer not null check (court_number > 0),
  state text not null default 'planned' check (state in ('planned', 'in_progress', 'complete', 'void')),
  created_by text null,
  updated_by text null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint session_ladder_court_pods_unique unique (session_id, round_number, court_number)
);

create index if not exists idx_session_ladder_court_pods_session_round
  on public.session_ladder_court_pods (session_id, round_number, court_number);
create index if not exists idx_session_ladder_court_pods_club_round
  on public.session_ladder_court_pods (club_id, round_number, court_number);

create table if not exists public.session_ladder_court_pod_players (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.session_ladder_sessions(id) on delete cascade,
  court_pod_id uuid not null references public.session_ladder_court_pods(id) on delete cascade,
  player_id bigint not null references public.players(id) on delete restrict,
  player_label text null,
  player_order smallint not null check (player_order > 0),
  created_at timestamptz not null default now(),
  constraint session_ladder_court_pod_players_unique_order unique (court_pod_id, player_order),
  constraint session_ladder_court_pod_players_unique_player unique (court_pod_id, player_id)
);

create index if not exists idx_session_ladder_court_pod_players_session
  on public.session_ladder_court_pod_players (session_id, court_pod_id);
create index if not exists idx_session_ladder_court_pod_players_player
  on public.session_ladder_court_pod_players (club_id, player_id);

create table if not exists public.session_ladder_games (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.session_ladder_sessions(id) on delete cascade,
  court_pod_id uuid not null references public.session_ladder_court_pods(id) on delete cascade,
  game_number integer not null check (game_number > 0),
  team_a_player_ids bigint[] not null check (cardinality(team_a_player_ids) = 2),
  team_b_player_ids bigint[] not null check (cardinality(team_b_player_ids) = 2),
  score_a integer null check (score_a is null or score_a >= 0),
  score_b integer null check (score_b is null or score_b >= 0),
  edited_by text null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint session_ladder_games_unique unique (court_pod_id, game_number)
);

create index if not exists idx_session_ladder_games_session_round
  on public.session_ladder_games (session_id, court_pod_id, game_number);
create index if not exists idx_session_ladder_games_club_created
  on public.session_ladder_games (club_id, created_at desc);

create table if not exists public.session_ladder_round_stats_cache (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.session_ladder_sessions(id) on delete cascade,
  round_number integer not null check (round_number > 0),
  player_id bigint not null references public.players(id) on delete restrict,
  wins integer not null default 0,
  losses integer not null default 0,
  points_for integer not null default 0,
  points_against integer not null default 0,
  point_differential integer not null default 0,
  source_hash text null,
  computed_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint session_ladder_round_stats_cache_unique unique (session_id, round_number, player_id)
);

create index if not exists idx_session_ladder_round_stats_cache_session_round
  on public.session_ladder_round_stats_cache (session_id, round_number);

-- shared updated_at trigger helper (already present in repo, recreated idempotently)
create or replace function public.set_updated_at_timestamp()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create or replace function public.session_ladder_set_club_from_session()
returns trigger
language plpgsql
as $$
declare
  parent_club text;
begin
  select club_id into parent_club
  from public.session_ladder_sessions
  where id = new.session_id;

  if parent_club is null then
    raise exception 'Invalid session_id %', new.session_id;
  end if;

  if new.club_id is null or new.club_id = '' then
    new.club_id := parent_club;
  end if;

  if new.club_id <> parent_club then
    raise exception 'club_id mismatch for session ladder child row';
  end if;

  return new;
end;
$$;

create or replace function public.session_ladder_set_from_court_pod()
returns trigger
language plpgsql
as $$
declare
  parent_club text;
  parent_session uuid;
begin
  select club_id, session_id into parent_club, parent_session
  from public.session_ladder_court_pods
  where id = new.court_pod_id;

  if parent_club is null or parent_session is null then
    raise exception 'Invalid court_pod_id %', new.court_pod_id;
  end if;

  if new.club_id is null or new.club_id = '' then
    new.club_id := parent_club;
  end if;

  if new.session_id is null then
    new.session_id := parent_session;
  end if;

  if new.club_id <> parent_club then
    raise exception 'club_id mismatch for court pod child row';
  end if;

  if new.session_id <> parent_session then
    raise exception 'session_id mismatch for court pod child row';
  end if;

  return new;
end;
$$;

drop trigger if exists trg_session_ladder_roster_club_guard on public.session_ladder_roster_entries;
create trigger trg_session_ladder_roster_club_guard
before insert or update of session_id, club_id on public.session_ladder_roster_entries
for each row execute function public.session_ladder_set_club_from_session();

drop trigger if exists trg_session_ladder_court_pods_club_guard on public.session_ladder_court_pods;
create trigger trg_session_ladder_court_pods_club_guard
before insert or update of session_id, club_id on public.session_ladder_court_pods
for each row execute function public.session_ladder_set_club_from_session();

drop trigger if exists trg_session_ladder_court_pod_players_guard on public.session_ladder_court_pod_players;
create trigger trg_session_ladder_court_pod_players_guard
before insert or update of court_pod_id, session_id, club_id on public.session_ladder_court_pod_players
for each row execute function public.session_ladder_set_from_court_pod();

drop trigger if exists trg_session_ladder_games_guard on public.session_ladder_games;
create trigger trg_session_ladder_games_guard
before insert or update of court_pod_id, session_id, club_id on public.session_ladder_games
for each row execute function public.session_ladder_set_from_court_pod();

drop trigger if exists session_ladder_sessions_set_updated_at on public.session_ladder_sessions;
create trigger session_ladder_sessions_set_updated_at
before update on public.session_ladder_sessions
for each row execute function public.set_updated_at_timestamp();

drop trigger if exists session_ladder_roster_entries_set_updated_at on public.session_ladder_roster_entries;
create trigger session_ladder_roster_entries_set_updated_at
before update on public.session_ladder_roster_entries
for each row execute function public.set_updated_at_timestamp();

drop trigger if exists session_ladder_court_pods_set_updated_at on public.session_ladder_court_pods;
create trigger session_ladder_court_pods_set_updated_at
before update on public.session_ladder_court_pods
for each row execute function public.set_updated_at_timestamp();

drop trigger if exists session_ladder_games_set_updated_at on public.session_ladder_games;
create trigger session_ladder_games_set_updated_at
before update on public.session_ladder_games
for each row execute function public.set_updated_at_timestamp();

drop trigger if exists session_ladder_round_stats_cache_set_updated_at on public.session_ladder_round_stats_cache;
create trigger session_ladder_round_stats_cache_set_updated_at
before update on public.session_ladder_round_stats_cache
for each row execute function public.set_updated_at_timestamp();

-- RLS helpers + policies
create or replace function public.jwt_club_id()
returns text
language sql
stable
as $$
  select coalesce(
    public.jwt_claims() ->> 'club_id',
    public.jwt_claims() -> 'user_metadata' ->> 'club_id',
    ''
  );
$$;

create or replace function public.session_ladder_can_write(target_club_id text)
returns boolean
language plpgsql
stable
security definer
set search_path = public
as $$
declare
  claims jsonb;
  user_club text;
  jwt_role text;
  uid uuid;
begin
  claims := public.jwt_claims();
  user_club := coalesce(claims ->> 'club_id', claims -> 'user_metadata' ->> 'club_id', '');
  jwt_role := lower(coalesce(claims ->> 'role', claims -> 'user_metadata' ->> 'role', claims -> 'app_metadata' ->> 'role', ''));

  if user_club = '' or target_club_id is null or target_club_id = '' or user_club <> target_club_id then
    return false;
  end if;

  if jwt_role in ('admin', 'manager') then
    return true;
  end if;

  begin
    uid := nullif(claims ->> 'sub', '')::uuid;
  exception
    when others then
      uid := null;
  end;

  if uid is null then
    return false;
  end if;

  return exists (
    select 1
    from public.club_user_roles cur
    where cur.club_id = target_club_id
      and cur.user_id = uid
      and cur.role in ('admin', 'coordinator', 'score_entry')
  );
end;
$$;

alter table public.session_ladder_sessions enable row level security;
alter table public.session_ladder_roster_entries enable row level security;
alter table public.session_ladder_court_pods enable row level security;
alter table public.session_ladder_court_pod_players enable row level security;
alter table public.session_ladder_games enable row level security;
alter table public.session_ladder_round_stats_cache enable row level security;

drop policy if exists session_ladder_sessions_select_by_club on public.session_ladder_sessions;
create policy session_ladder_sessions_select_by_club
on public.session_ladder_sessions
for select
using (club_id = public.jwt_club_id());

drop policy if exists session_ladder_sessions_write_by_role on public.session_ladder_sessions;
create policy session_ladder_sessions_write_by_role
on public.session_ladder_sessions
for all
using (public.session_ladder_can_write(club_id))
with check (public.session_ladder_can_write(club_id));

drop policy if exists session_ladder_roster_entries_select_by_club on public.session_ladder_roster_entries;
create policy session_ladder_roster_entries_select_by_club
on public.session_ladder_roster_entries
for select
using (club_id = public.jwt_club_id());

drop policy if exists session_ladder_roster_entries_write_by_role on public.session_ladder_roster_entries;
create policy session_ladder_roster_entries_write_by_role
on public.session_ladder_roster_entries
for all
using (public.session_ladder_can_write(club_id))
with check (public.session_ladder_can_write(club_id));

drop policy if exists session_ladder_court_pods_select_by_club on public.session_ladder_court_pods;
create policy session_ladder_court_pods_select_by_club
on public.session_ladder_court_pods
for select
using (club_id = public.jwt_club_id());

drop policy if exists session_ladder_court_pods_write_by_role on public.session_ladder_court_pods;
create policy session_ladder_court_pods_write_by_role
on public.session_ladder_court_pods
for all
using (public.session_ladder_can_write(club_id))
with check (public.session_ladder_can_write(club_id));

drop policy if exists session_ladder_court_pod_players_select_by_club on public.session_ladder_court_pod_players;
create policy session_ladder_court_pod_players_select_by_club
on public.session_ladder_court_pod_players
for select
using (club_id = public.jwt_club_id());

drop policy if exists session_ladder_court_pod_players_write_by_role on public.session_ladder_court_pod_players;
create policy session_ladder_court_pod_players_write_by_role
on public.session_ladder_court_pod_players
for all
using (public.session_ladder_can_write(club_id))
with check (public.session_ladder_can_write(club_id));

drop policy if exists session_ladder_games_select_by_club on public.session_ladder_games;
create policy session_ladder_games_select_by_club
on public.session_ladder_games
for select
using (club_id = public.jwt_club_id());

drop policy if exists session_ladder_games_write_by_role on public.session_ladder_games;
create policy session_ladder_games_write_by_role
on public.session_ladder_games
for all
using (public.session_ladder_can_write(club_id))
with check (public.session_ladder_can_write(club_id));

drop policy if exists session_ladder_round_stats_cache_select_by_club on public.session_ladder_round_stats_cache;
create policy session_ladder_round_stats_cache_select_by_club
on public.session_ladder_round_stats_cache
for select
using (club_id = public.jwt_club_id());

drop policy if exists session_ladder_round_stats_cache_write_by_role on public.session_ladder_round_stats_cache;
create policy session_ladder_round_stats_cache_write_by_role
on public.session_ladder_round_stats_cache
for all
using (public.session_ladder_can_write(club_id))
with check (public.session_ladder_can_write(club_id));

commit;
