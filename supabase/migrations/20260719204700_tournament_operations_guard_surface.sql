-- Order-27 canonical Tournament Ops schema, security, and atomic write seam.
-- The existing order-26 operation ledger remains the only durable retry ledger.

create table if not exists public.tournament_event_draws (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  registration_day_id text null,
  event_option_id text null,
  name text not null,
  status text not null default 'draft',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.tournament_teams (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  registration_day_id text null,
  event_option_id text null,
  team_number integer not null,
  player1_id integer null,
  player2_id integer null,
  seed integer null,
  source text null,
  notes text null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.tournament_games (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  registration_day_id text null,
  event_option_id text null,
  stage text not null,
  rr_round_number integer null,
  rr_slot_number integer null,
  playoff_game_code text null,
  playoff_round text null,
  team_a_id uuid null references public.tournament_teams(id),
  team_b_id uuid null references public.tournament_teams(id),
  team_a_source jsonb null,
  team_b_source jsonb null,
  score_a integer null,
  score_b integer null,
  winner_team_id uuid null references public.tournament_teams(id),
  loser_team_id uuid null references public.tournament_teams(id),
  finalized_at timestamptz null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.tournament_podium (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  placement integer not null check (placement between 1 and 3),
  team_id uuid not null references public.tournament_teams(id) on delete restrict,
  source text not null,
  created_at timestamptz not null default timezone('utc', now())
);

alter table public.matches
  add column if not exists context_type text null,
  add column if not exists context_id uuid null,
  add column if not exists tournament_id uuid null,
  add column if not exists tournament_game_id uuid null;

do $$
begin
  if to_regclass('public.tournaments') is null
     or to_regclass('public.tournament_event_draws') is null
     or to_regclass('public.tournament_teams') is null
     or to_regclass('public.tournament_games') is null
     or to_regclass('public.tournament_podium') is null
     or to_regclass('public.matches') is null
     or to_regclass('public.player_badges') is null then
    raise exception 'Tournament Ops base tables must exist before order-27 migration';
  end if;
end
$$;

alter table public.tournament_admin_operations
  drop constraint if exists tournament_admin_surface_check;

alter table public.tournament_admin_operations
  add constraint tournament_admin_surface_check
  check (surface in ('tournament', 'setup', 'registration', 'import_handoff', 'tournament_live', 'operations'));

alter table public.tournament_teams
  add column if not exists draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  add column if not exists registration_day_id text null,
  add column if not exists event_option_id text null,
  add column if not exists source text null,
  add column if not exists notes text null,
  add column if not exists created_at timestamptz not null default timezone('utc', now()),
  add column if not exists updated_at timestamptz not null default timezone('utc', now());

alter table public.tournament_games
  add column if not exists draw_id uuid null references public.tournament_event_draws(id) on delete cascade,
  add column if not exists registration_day_id text null,
  add column if not exists event_option_id text null;

do $$
begin
  if not exists (
    select 1 from pg_constraint
     where conrelid = 'public.tournament_games'::regclass
       and conname = 'tournament_games_nonnegative_scores'
  ) then
    alter table public.tournament_games
      add constraint tournament_games_nonnegative_scores
      check ((score_a is null or score_a >= 0) and (score_b is null or score_b >= 0)) not valid;
  end if;
end
$$;

alter table public.tournament_podium
  add column if not exists draw_id uuid null references public.tournament_event_draws(id) on delete cascade;

alter table public.tournament_teams drop constraint if exists uq_tournament_team_number;
alter table public.tournament_games drop constraint if exists tournament_games_rr_unique;
alter table public.tournament_games drop constraint if exists tournament_games_playoff_unique;
alter table public.tournament_podium
  drop constraint if exists tournament_podium_unique_placement,
  drop constraint if exists tournament_podium_unique_team;

drop index if exists public.uq_tournament_event_draw_unique;
create unique index uq_tournament_event_draw_unique
  on public.tournament_event_draws (
    tournament_id,
    coalesce(registration_day_id, ''),
    coalesce(event_option_id, ''),
    lower(btrim(name))
  );
create index if not exists idx_tournament_event_draws_tournament
  on public.tournament_event_draws (tournament_id);
create index if not exists idx_tournament_teams_draw_id
  on public.tournament_teams (draw_id);
create index if not exists idx_tournament_games_draw_id
  on public.tournament_games (draw_id);

drop index if exists public.uq_tournament_teams_draw_team;
create unique index uq_tournament_teams_draw_team
  on public.tournament_teams (tournament_id, draw_id, team_number)
  where draw_id is not null;
create unique index if not exists uq_tournament_teams_legacy_team
  on public.tournament_teams (tournament_id, team_number)
  where draw_id is null;

create unique index if not exists uq_tournament_games_draw_rr
  on public.tournament_games (tournament_id, draw_id, rr_round_number, rr_slot_number)
  where draw_id is not null and stage = 'ROUND_ROBIN';
create unique index if not exists uq_tournament_games_draw_playoff
  on public.tournament_games (tournament_id, draw_id, playoff_game_code)
  where draw_id is not null and stage = 'PLAYOFF' and playoff_game_code is not null;
create unique index if not exists uq_tournament_games_legacy_rr
  on public.tournament_games (tournament_id, rr_round_number, rr_slot_number)
  where draw_id is null and stage = 'ROUND_ROBIN';
create unique index if not exists uq_tournament_games_legacy_playoff
  on public.tournament_games (tournament_id, playoff_game_code)
  where draw_id is null and stage = 'PLAYOFF' and playoff_game_code is not null;

create unique index if not exists uq_tournament_podium_draw_placement
  on public.tournament_podium (tournament_id, draw_id, placement)
  where draw_id is not null;
create unique index if not exists uq_tournament_podium_draw_team
  on public.tournament_podium (tournament_id, draw_id, team_id)
  where draw_id is not null;
create unique index if not exists uq_tournament_podium_legacy_placement
  on public.tournament_podium (tournament_id, placement)
  where draw_id is null;
create unique index if not exists uq_tournament_podium_legacy_team
  on public.tournament_podium (tournament_id, team_id)
  where draw_id is null;
create unique index if not exists idx_matches_unique_tournament_game_id
  on public.matches (tournament_game_id)
  where tournament_game_id is not null;

alter table public.tournament_event_draws enable row level security;
alter table public.tournament_teams enable row level security;
alter table public.tournament_games enable row level security;
alter table public.tournament_podium enable row level security;

revoke all on table public.tournament_event_draws from public, anon, authenticated;
revoke all on table public.tournament_teams from public, anon, authenticated;
revoke all on table public.tournament_games from public, anon, authenticated;
revoke all on table public.tournament_podium from public, anon, authenticated;
grant select, insert, update, delete on table public.tournament_event_draws to service_role;
grant select, insert, update, delete on table public.tournament_teams to service_role;
grant select, insert, update, delete on table public.tournament_games to service_role;
grant select, insert, update, delete on table public.tournament_podium to service_role;

-- Any child write, including a direct service-role/Streamlit write, advances
-- the draw version. Guarded computed writes can therefore compare one reviewed
-- draw version under row lock and cannot commit over an interleaving change.
create or replace function public.touch_tournament_draw_version_from_child()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_old_draw_id uuid;
  v_new_draw_id uuid;
  v_team_structural_change boolean := false;
  v_game_derivation_change boolean := false;
begin
  if tg_op in ('UPDATE', 'DELETE') then
    v_old_draw_id := old.draw_id;
  end if;
  if tg_op in ('INSERT', 'UPDATE') then
    v_new_draw_id := new.draw_id;
  end if;
  update public.tournament_event_draws
     set updated_at = clock_timestamp()
   where id in (v_old_draw_id, v_new_draw_id);

  if tg_table_name = 'tournament_teams' then
    v_team_structural_change := tg_op <> 'UPDATE';
    if tg_op = 'UPDATE' then
      v_team_structural_change := old.id is distinct from new.id
        or old.tournament_id is distinct from new.tournament_id
        or old.draw_id is distinct from new.draw_id
        or old.registration_day_id is distinct from new.registration_day_id
        or old.event_option_id is distinct from new.event_option_id
        or old.team_number is distinct from new.team_number
        or old.player1_id is distinct from new.player1_id
        or old.player2_id is distinct from new.player2_id;
    end if;
  end if;
  if v_team_structural_change
     and coalesce(current_setting('jupr.tournament_results_import_structural_write', true), 'off') <> 'on'
     and exists (
      select 1 from public.tournament_games g
       where g.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_GAMES';
  end if;

  if tg_table_name = 'tournament_games' then
    v_game_derivation_change := tg_op <> 'UPDATE';
    if tg_op = 'UPDATE' then
      v_game_derivation_change := old.stage is distinct from new.stage
        or old.team_a_id is distinct from new.team_a_id
        or old.team_b_id is distinct from new.team_b_id
        or old.team_a_source is distinct from new.team_a_source
        or old.team_b_source is distinct from new.team_b_source
        or old.score_a is distinct from new.score_a
        or old.score_b is distinct from new.score_b
        or old.winner_team_id is distinct from new.winner_team_id
        or old.loser_team_id is distinct from new.loser_team_id
        or old.finalized_at is distinct from new.finalized_at;
    end if;
  end if;
  if v_game_derivation_change and exists (
      select 1 from public.tournament_podium p
       where p.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PODIUM_LOCK';
  end if;
  if v_game_derivation_change and exists (
      select 1
        from public.matches m
        join public.tournament_games published_game on published_game.id = m.tournament_game_id
       where published_game.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK';
  end if;
  if v_game_derivation_change
     and (
       (tg_op in ('UPDATE', 'DELETE') and upper(coalesce(old.stage, '')) = 'ROUND_ROBIN')
       or (tg_op in ('INSERT', 'UPDATE') and upper(coalesce(new.stage, '')) = 'ROUND_ROBIN')
     )
     and exists (
       select 1 from public.tournament_games playoff
        where playoff.draw_id in (v_old_draw_id, v_new_draw_id)
          and upper(coalesce(playoff.stage, '')) = 'PLAYOFF'
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DOWNSTREAM_SCORE_LOCK';
  end if;
  if tg_table_name = 'tournament_podium' and exists (
    select 1
      from public.tournament_event_draws d
      join public.tournaments t on t.id = d.tournament_id
      join public.player_badges badge
        on badge.club_id = t.club_id
       and badge.context_type = 'tournament'
       and badge.context_id::text like d.tournament_id::text || ':draw:' || d.id::text || ':podium:%'
     where d.id in (v_old_draw_id, v_new_draw_id)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_ALREADY_AWARDED';
  end if;
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_tournament_teams_touch_draw_version on public.tournament_teams;
create trigger trg_tournament_teams_touch_draw_version
before insert or update or delete on public.tournament_teams
for each row execute function public.touch_tournament_draw_version_from_child();

drop trigger if exists trg_tournament_games_touch_draw_version on public.tournament_games;
create trigger trg_tournament_games_touch_draw_version
before insert or update or delete on public.tournament_games
for each row execute function public.touch_tournament_draw_version_from_child();

drop trigger if exists trg_tournament_podium_touch_draw_version on public.tournament_podium;
create trigger trg_tournament_podium_touch_draw_version
before insert or update or delete on public.tournament_podium
for each row execute function public.touch_tournament_draw_version_from_child();

revoke all on function public.touch_tournament_draw_version_from_child() from public, anon, authenticated;
grant execute on function public.touch_tournament_draw_version_from_child() to service_role;

-- Draw-scoped podium badge writes participate in the same child -> draw
-- serialization contract. Inserts validate their placement against the current
-- podium after acquiring the draw lock, so an award computed from a podium that
-- was concurrently replaced cannot commit stale recipients.
create or replace function public.touch_tournament_draw_version_from_badge()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_context_id text;
  v_context_type text;
  v_club_id text;
  v_player_id integer;
  v_tournament_id text;
  v_draw_id text;
  v_placement integer;
begin
  if tg_op = 'UPDATE'
     and (
       (lower(coalesce(old.context_type::text, '')) = 'tournament'
        and old.context_id::text ~ '^[^:]+:draw:[^:]+:podium:[123]$')
       or (lower(coalesce(new.context_type::text, '')) = 'tournament'
        and new.context_id::text ~ '^[^:]+:draw:[^:]+:podium:[123]$')
     )
     and (
       old.context_id is distinct from new.context_id
       or old.context_type is distinct from new.context_type
       or old.club_id is distinct from new.club_id
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_BADGE_CONTEXT_IMMUTABLE';
  end if;
  if tg_op = 'DELETE' then
    v_context_id := old.context_id::text;
    v_context_type := old.context_type::text;
    v_club_id := old.club_id::text;
  else
    v_context_id := new.context_id::text;
    v_context_type := new.context_type::text;
    v_club_id := new.club_id::text;
    v_player_id := new.player_id;
  end if;
  if lower(coalesce(v_context_type, '')) <> 'tournament'
     or v_context_id !~ '^[^:]+:draw:[^:]+:podium:[123]$' then
    if tg_op = 'DELETE' then return old; end if;
    return new;
  end if;

  v_tournament_id := split_part(v_context_id, ':', 1);
  v_draw_id := split_part(v_context_id, ':', 3);
  v_placement := split_part(v_context_id, ':', 5)::integer;
  update public.tournament_event_draws d
     set updated_at = clock_timestamp()
    from public.tournaments t
   where d.id::text = v_draw_id
     and d.tournament_id::text = v_tournament_id
     and t.id = d.tournament_id
     and t.club_id = v_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_BADGE_DRAW_STALE';
  end if;
  if tg_op <> 'DELETE' and not exists (
    select 1
      from public.tournament_podium p
      join public.tournament_teams team on team.id = p.team_id
     where p.tournament_id::text = v_tournament_id
       and p.draw_id::text = v_draw_id
       and p.placement = v_placement
       and v_player_id in (team.player1_id, team.player2_id)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_BADGE_PODIUM_STALE';
  end if;
  if tg_op = 'DELETE' then return old; end if;
  return new;
end;
$$;

drop trigger if exists trg_player_badges_touch_tournament_draw_version on public.player_badges;
create trigger trg_player_badges_touch_tournament_draw_version
before insert or update or delete on public.player_badges
for each row execute function public.touch_tournament_draw_version_from_badge();

revoke all on function public.touch_tournament_draw_version_from_badge() from public, anon, authenticated;
grant execute on function public.touch_tournament_draw_version_from_badge() to service_role;

drop function if exists public.admin_write_tournament_draw_teams_cas(text, text, text, boolean, jsonb);
create or replace function public.admin_write_tournament_draw_teams_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_replace boolean,
  p_teams jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_saved jsonb;
begin
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if jsonb_typeof(coalesce(p_teams, '[]'::jsonb)) <> 'array' or jsonb_array_length(coalesce(p_teams, '[]'::jsonb)) = 0 then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAMS_REQUIRED';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(p_teams) as x(player1_id integer, player2_id integer)
      left join public.players p1 on p1.id = x.player1_id and p1.club_id = p_club_id
      left join public.players p2 on p2.id = x.player2_id and p2.club_id = p_club_id
     where p1.id is null or (x.player2_id is not null and p2.id is null)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_PLAYER_OUTSIDE_CLUB';
  end if;

  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
  perform game.id
    from public.tournament_games game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
   order by game.id
   for update;
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and d.updated_at = p_expected_draw_updated_at
   for update of d;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if exists (
    select 1 from public.tournament_games
     where tournament_id::text = p_tournament_id and draw_id::text = p_draw_id
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_GAMES';
  end if;

  if p_replace then
    delete from public.tournament_teams
     where tournament_id::text = p_tournament_id and draw_id::text = p_draw_id;
  end if;

  with inserted as (
    insert into public.tournament_teams (
      id, tournament_id, draw_id, registration_day_id, event_option_id,
      team_number, player1_id, player2_id, seed, source, notes, created_at, updated_at
    )
    select
      coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
      v_draw.tournament_id,
      v_draw.id,
      coalesce(nullif(x.registration_day_id, ''), v_draw.registration_day_id),
      coalesce(nullif(x.event_option_id, ''), v_draw.event_option_id),
      x.team_number,
      x.player1_id,
      x.player2_id,
      x.seed,
      coalesce(nullif(x.source, ''), 'MANUAL'),
      nullif(x.notes, ''),
      coalesce(x.created_at, timezone('utc', now())),
      timezone('utc', now())
    from jsonb_to_recordset(p_teams) as x(
      id text, registration_day_id text, event_option_id text, team_number integer,
      player1_id integer, player2_id integer, seed integer, source text, notes text,
      created_at timestamptz
    )
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.team_number), '[]'::jsonb)
    into v_saved from inserted;

  update public.tournament_event_draws
     set updated_at = clock_timestamp()
   where id = v_draw.id;
  return jsonb_build_object('ok', true, 'teams', v_saved);
end;
$$;

create or replace function public.admin_insert_tournament_draw_games_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_mode text,
  p_expected_teams jsonb,
  p_expected_source_games jsonb,
  p_games jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_saved jsonb;
  v_mode text := upper(coalesce(p_mode, ''));
begin
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if v_mode not in ('ROUND_ROBIN', 'PLAYOFF') then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_MODE_INVALID';
  end if;
  if jsonb_typeof(coalesce(p_games, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_games, '[]'::jsonb)) = 0 then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAMES_REQUIRED';
  end if;
  if jsonb_typeof(coalesce(p_expected_teams, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_teams, '[]'::jsonb)) = 0
     or exists (
       select 1 from jsonb_to_recordset(coalesce(p_expected_teams, '[]'::jsonb)) as x(id text, updated_at timestamptz)
        where nullif(x.id, '') is null or x.updated_at is null
     )
     or (
       select count(distinct x.id)
         from jsonb_to_recordset(coalesce(p_expected_teams, '[]'::jsonb)) as x(id text)
     ) <> jsonb_array_length(coalesce(p_expected_teams, '[]'::jsonb)) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;

  -- Direct UPDATE/DELETE statements already own their child row before the
  -- BEFORE trigger advances the draw version. Lock every existing dependency
  -- in deterministic order before taking the draw CAS lock, so both paths use
  -- child -> draw ordering. A direct INSERT reaches the BEFORE trigger before
  -- acquiring a unique-index entry, so it waits on the draw without creating
  -- the inverse edge.
  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
  perform source_game.id
    from public.tournament_games source_game
   where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
   order by source_game.id
   for update;
  perform badge.id
    from public.player_badges badge
   where badge.club_id = p_club_id
     and badge.context_type = 'tournament'
     and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
   order by badge.id
   for update;

  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and d.updated_at = p_expected_draw_updated_at
   for update of d;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  if (
       select count(*) from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_teams)
     or exists (
       select 1 from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
          and not exists (
            select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
             where x.id = team.id::text and x.updated_at = team.updated_at
          )
     )
     or exists (
       select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
        where not exists (
          select 1 from public.tournament_teams team
           where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
             and team.id::text = x.id and team.updated_at = x.updated_at
        )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(p_games) as x(stage text)
     where upper(coalesce(x.stage, '')) <> v_mode
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_STAGE_INVALID';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(p_games) as x(team_a_id text, team_b_id text)
      left join public.tournament_teams a
        on a.id = nullif(x.team_a_id, '')::uuid
       and a.tournament_id = v_draw.tournament_id and a.draw_id = v_draw.id
      left join public.tournament_teams b
        on b.id = nullif(x.team_b_id, '')::uuid
       and b.tournament_id = v_draw.tournament_id and b.draw_id = v_draw.id
     where (nullif(x.team_a_id, '') is not null and a.id is null)
        or (nullif(x.team_b_id, '') is not null and b.id is null)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_TEAM_OUTSIDE_DRAW';
  end if;

  if v_mode = 'ROUND_ROBIN' then
    if exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_GAMES';
    end if;
  else
    if jsonb_typeof(coalesce(p_expected_source_games, '[]'::jsonb)) <> 'array'
       or jsonb_array_length(coalesce(p_expected_source_games, '[]'::jsonb)) = 0
       or exists (
         select 1 from jsonb_to_recordset(coalesce(p_expected_source_games, '[]'::jsonb)) as x(id text, updated_at timestamptz)
          where nullif(x.id, '') is null or x.updated_at is null
       )
       or (
         select count(distinct x.id)
           from jsonb_to_recordset(coalesce(p_expected_source_games, '[]'::jsonb)) as x(id text)
       ) <> jsonb_array_length(coalesce(p_expected_source_games, '[]'::jsonb)) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE';
    end if;
    if (
         select count(*) from public.tournament_games source_game
          where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
       ) <> jsonb_array_length(p_expected_source_games)
       or exists (
         select 1 from public.tournament_games source_game
          where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
            and not exists (
              select 1 from jsonb_to_recordset(p_expected_source_games) as x(id text, updated_at timestamptz)
               where x.id = source_game.id::text and x.updated_at = source_game.updated_at
            )
       )
       or exists (
         select 1 from jsonb_to_recordset(p_expected_source_games) as x(id text, updated_at timestamptz)
          where not exists (
            select 1 from public.tournament_games source_game
             where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
               and source_game.id::text = x.id and source_game.updated_at = x.updated_at
          )
       ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE';
    end if;
    if exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id and g.stage = 'PLAYOFF'
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_PLAYOFFS';
    end if;
    if exists (
      select 1 from public.tournament_podium p
       where p.tournament_id = v_draw.tournament_id and p.draw_id = v_draw.id
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_PODIUM';
    end if;
    if not exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id and g.stage = 'ROUND_ROBIN'
    ) or exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id and g.stage = 'ROUND_ROBIN'
         and (g.score_a is null or g.score_b is null or g.winner_team_id is null)
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_INCOMPLETE';
    end if;
  end if;

  with inserted as (
    insert into public.tournament_games (
      id, tournament_id, draw_id, registration_day_id, event_option_id, stage,
      rr_round_number, rr_slot_number, playoff_game_code, playoff_round,
      team_a_id, team_b_id, team_a_source, team_b_source,
      score_a, score_b, winner_team_id, loser_team_id, finalized_at,
      created_at, updated_at
    )
    select
      coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
      v_draw.tournament_id, v_draw.id,
      coalesce(nullif(x.registration_day_id, ''), v_draw.registration_day_id),
      coalesce(nullif(x.event_option_id, ''), v_draw.event_option_id),
      v_mode, x.rr_round_number, x.rr_slot_number,
      nullif(x.playoff_game_code, ''), nullif(x.playoff_round, ''),
      nullif(x.team_a_id, '')::uuid, nullif(x.team_b_id, '')::uuid,
      x.team_a_source, x.team_b_source,
      x.score_a, x.score_b,
      nullif(x.winner_team_id, '')::uuid, nullif(x.loser_team_id, '')::uuid,
      x.finalized_at,
      coalesce(x.created_at, clock_timestamp()), clock_timestamp()
    from jsonb_to_recordset(p_games) as x(
      id text, registration_day_id text, event_option_id text, stage text,
      rr_round_number integer, rr_slot_number integer,
      playoff_game_code text, playoff_round text,
      team_a_id text, team_b_id text, team_a_source jsonb, team_b_source jsonb,
      score_a integer, score_b integer, winner_team_id text, loser_team_id text,
      finalized_at timestamptz, created_at timestamptz
    )
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.stage, inserted.rr_round_number, inserted.rr_slot_number, inserted.playoff_game_code), '[]'::jsonb)
    into v_saved from inserted;
  update public.tournament_event_draws set updated_at = clock_timestamp() where id = v_draw.id;
  return jsonb_build_object('ok', true, 'games', v_saved);
end;
$$;

drop function if exists public.admin_replace_tournament_draw_podium_cas(text, text, text, jsonb);
create or replace function public.admin_replace_tournament_draw_podium_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_expected_source_games jsonb,
  p_podium jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_saved jsonb;
begin
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if jsonb_typeof(coalesce(p_expected_teams, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_teams, '[]'::jsonb)) = 0
     or jsonb_typeof(coalesce(p_expected_source_games, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_source_games, '[]'::jsonb)) = 0 then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE';
  end if;

  -- Existing replace targets are child rows too. Lock them before the draw,
  -- followed by the reviewed dependencies, matching direct child-write order.
  perform existing_podium.id
    from public.tournament_podium existing_podium
   where existing_podium.tournament_id = v_draw.tournament_id
     and existing_podium.draw_id = v_draw.id
   order by existing_podium.id
   for update;
  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
  perform source_game.id
    from public.tournament_games source_game
   where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
   order by source_game.id
   for update;
  perform badge.id
    from public.player_badges badge
   where badge.club_id = p_club_id
     and badge.context_type = 'tournament'
     and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
   order by badge.id
   for update;

  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and d.updated_at = p_expected_draw_updated_at
   for update of d;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  if (
       select count(*) from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_teams)
     or (
       select count(distinct x.id) from jsonb_to_recordset(p_expected_teams) as x(id text)
     ) <> jsonb_array_length(p_expected_teams)
     or exists (
       select 1 from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
          and not exists (
            select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
             where x.id = team.id::text and x.updated_at = team.updated_at
          )
     )
     or exists (
       select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
        where nullif(x.id, '') is null or x.updated_at is null or not exists (
          select 1 from public.tournament_teams team
           where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
             and team.id::text = x.id and team.updated_at = x.updated_at
        )
     )
     or (
       select count(*) from public.tournament_games source_game
        where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_source_games)
     or (
       select count(distinct x.id) from jsonb_to_recordset(p_expected_source_games) as x(id text)
     ) <> jsonb_array_length(p_expected_source_games)
     or exists (
       select 1 from public.tournament_games source_game
        where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
          and not exists (
            select 1 from jsonb_to_recordset(p_expected_source_games) as x(id text, updated_at timestamptz)
             where x.id = source_game.id::text and x.updated_at = source_game.updated_at
          )
     )
     or exists (
       select 1 from jsonb_to_recordset(p_expected_source_games) as x(id text, updated_at timestamptz)
        where nullif(x.id, '') is null or x.updated_at is null or not exists (
          select 1 from public.tournament_games source_game
           where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
             and source_game.id::text = x.id and source_game.updated_at = x.updated_at
        )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(coalesce(p_podium, '[]'::jsonb)) as x(team_id text)
      left join public.tournament_teams team
        on team.id = nullif(x.team_id, '')::uuid
       and team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
     where team.id is null
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_TEAM_OUTSIDE_DRAW';
  end if;
  if exists (
    select 1 from public.player_badges
     where club_id = p_club_id
       and context_type = 'tournament'
       and context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_ALREADY_AWARDED';
  end if;

  delete from public.tournament_podium
   where tournament_id::text = p_tournament_id and draw_id::text = p_draw_id;
  with inserted as (
    insert into public.tournament_podium (id, tournament_id, draw_id, placement, team_id, source, created_at)
    select
      coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
      v_draw.tournament_id,
      v_draw.id,
      x.placement,
      nullif(x.team_id, '')::uuid,
      coalesce(nullif(x.source, ''), 'ROUND_ROBIN'),
      coalesce(x.created_at, timezone('utc', now()))
    from jsonb_to_recordset(p_podium) as x(
      id text, placement integer, team_id text, source text, created_at timestamptz
    )
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.placement), '[]'::jsonb)
    into v_saved from inserted;
  update public.tournament_event_draws set updated_at = clock_timestamp() where id = v_draw.id;
  return jsonb_build_object('ok', true, 'podium', v_saved);
end;
$$;

drop function if exists public.admin_score_tournament_game_cas(text, text, text, timestamptz, jsonb, jsonb);
create or replace function public.admin_score_tournament_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_game_id text,
  p_expected_updated_at timestamptz,
  p_expected_draw_updated_at timestamptz,
  p_game_patch jsonb,
  p_dependency_updates jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_game public.tournament_games%rowtype;
  v_dependency jsonb;
  v_dependency_ids uuid[] := '{}'::uuid[];
  v_locked_game_ids uuid[] := '{}'::uuid[];
  v_dependencies jsonb := '[]'::jsonb;
  v_locked_game_count integer := 0;
begin
  select coalesce(array_agg((value->>'id')::uuid order by (value->>'id')::uuid), '{}'::uuid[])
    into v_dependency_ids
    from jsonb_array_elements(coalesce(p_dependency_updates, '[]'::jsonb));
  if cardinality(v_dependency_ids) <> cardinality(array(select distinct unnest(v_dependency_ids))) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DEPENDENCY_DUPLICATE';
  end if;

  select g.* into v_game
    from public.tournament_games g
    join public.tournaments t on t.id = g.tournament_id
   where g.id::text = p_game_id
     and g.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_STALE';
  end if;
  if v_game.id = any(v_dependency_ids) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DEPENDENCY_DUPLICATE';
  end if;
  select coalesce(array_agg(game_id order by game_id), '{}'::uuid[])
    into v_locked_game_ids
    from unnest(array_append(v_dependency_ids, v_game.id)) as locked(game_id);

  perform g.id
    from public.tournament_games g
   where g.id = any(v_locked_game_ids)
     and g.tournament_id = v_game.tournament_id
     and g.draw_id is not distinct from v_game.draw_id
   order by g.id
   for update;
  get diagnostics v_locked_game_count = row_count;
  if v_locked_game_count <> cardinality(v_locked_game_ids) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DEPENDENCY_STALE';
  end if;
  if v_game.draw_id is not null then
    perform d.id
      from public.tournament_event_draws d
     where d.id = v_game.draw_id
       and d.updated_at = p_expected_draw_updated_at
     for update;
    if not found then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
    end if;
  end if;

  select g.* into v_game
    from public.tournament_games g
    join public.tournaments t on t.id = g.tournament_id
   where g.id::text = p_game_id
     and g.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and g.updated_at = p_expected_updated_at
   for update of g;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_STALE';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(coalesce(p_dependency_updates, '[]'::jsonb))
        as expected(id text, expected_updated_at timestamptz)
      left join public.tournament_games dependency on dependency.id = nullif(expected.id, '')::uuid
     where expected.expected_updated_at is null
        or dependency.id is null
        or dependency.tournament_id <> v_game.tournament_id
        or dependency.draw_id is distinct from v_game.draw_id
        or dependency.updated_at is distinct from expected.expected_updated_at
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DEPENDENCY_STALE';
  end if;
  if exists (
    select 1 from public.tournament_podium p
     where p.tournament_id = v_game.tournament_id
       and p.draw_id is not distinct from v_game.draw_id
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PODIUM_LOCK';
  end if;
  if exists (
    select 1
      from public.matches m
      join public.tournament_games g on g.id = m.tournament_game_id
     where g.tournament_id = v_game.tournament_id
       and g.draw_id is not distinct from v_game.draw_id
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK';
  end if;
  if (p_game_patch->>'score_a')::integer < 0 or (p_game_patch->>'score_b')::integer < 0 then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_NEGATIVE';
  end if;

  update public.tournament_games
     set score_a = (p_game_patch->>'score_a')::integer,
         score_b = (p_game_patch->>'score_b')::integer,
         winner_team_id = nullif(p_game_patch->>'winner_team_id', '')::uuid,
         loser_team_id = nullif(p_game_patch->>'loser_team_id', '')::uuid,
         finalized_at = nullif(p_game_patch->>'finalized_at', '')::timestamptz,
         updated_at = coalesce(nullif(p_game_patch->>'updated_at', '')::timestamptz, timezone('utc', now()))
   where id = v_game.id
   returning * into v_game;

  for v_dependency in select value from jsonb_array_elements(coalesce(p_dependency_updates, '[]'::jsonb))
  loop
    if exists (
      select 1 from public.tournament_games g
       where g.id = nullif(v_dependency->>'id', '')::uuid
         and g.tournament_id = v_game.tournament_id
         and g.draw_id is not distinct from v_game.draw_id
         and (g.score_a is not null or g.score_b is not null or g.winner_team_id is not null or g.finalized_at is not null)
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DOWNSTREAM_SCORE_LOCK';
    end if;
    update public.tournament_games g
       set team_a_id = case when v_dependency ? 'team_a_id' then nullif(v_dependency->>'team_a_id', '')::uuid else g.team_a_id end,
           team_b_id = case when v_dependency ? 'team_b_id' then nullif(v_dependency->>'team_b_id', '')::uuid else g.team_b_id end,
           score_a = case when v_dependency ? 'score_a' then nullif(v_dependency->>'score_a', '')::integer else g.score_a end,
           score_b = case when v_dependency ? 'score_b' then nullif(v_dependency->>'score_b', '')::integer else g.score_b end,
           winner_team_id = case when v_dependency ? 'winner_team_id' then nullif(v_dependency->>'winner_team_id', '')::uuid else g.winner_team_id end,
           loser_team_id = case when v_dependency ? 'loser_team_id' then nullif(v_dependency->>'loser_team_id', '')::uuid else g.loser_team_id end,
           finalized_at = case when v_dependency ? 'finalized_at' then nullif(v_dependency->>'finalized_at', '')::timestamptz else g.finalized_at end,
           updated_at = timezone('utc', now())
     where g.id = nullif(v_dependency->>'id', '')::uuid
       and g.tournament_id = v_game.tournament_id
       and g.draw_id is not distinct from v_game.draw_id;
    if not found then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DEPENDENCY_STALE';
    end if;
  end loop;

  if cardinality(v_dependency_ids) > 0 then
    select coalesce(jsonb_agg(to_jsonb(g) order by g.id), '[]'::jsonb)
      into v_dependencies from public.tournament_games g where g.id = any(v_dependency_ids);
  end if;
  if v_game.draw_id is not null then
    update public.tournament_event_draws set updated_at = clock_timestamp() where id = v_game.draw_id;
  end if;
  return jsonb_build_object('ok', true, 'game', to_jsonb(v_game), 'dependency_updates', v_dependencies);
end;
$$;

drop function if exists public.admin_import_tournament_draw_results_cas(text, text, text, text, jsonb, jsonb, jsonb, jsonb);
create or replace function public.admin_import_tournament_draw_results_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_import_mode text,
  p_new_players jsonb,
  p_teams jsonb,
  p_games jsonb,
  p_podium jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_team_count integer := 0;
  v_game_count integer := 0;
  v_podium_count integer := 0;
  v_created_player_count integer := 0;
  v_player jsonb;
  v_player_id integer;
  v_player_map jsonb := '{}'::jsonb;
  v_team jsonb;
  v_player1_id integer;
  v_player2_id integer;
  v_assigned_player_ids integer[] := '{}'::integer[];
  v_resolved_teams jsonb := '[]'::jsonb;
begin
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if upper(coalesce(p_import_mode, '')) not in ('REPLACE', 'APPEND') then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULTS_MODE_INVALID';
  end if;

  perform existing_podium.id
    from public.tournament_podium existing_podium
   where existing_podium.tournament_id = v_draw.tournament_id
     and existing_podium.draw_id = v_draw.id
   order by existing_podium.id
   for update;
  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
  perform game.id
    from public.tournament_games game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
   order by game.id
   for update;
  perform badge.id
    from public.player_badges badge
   where badge.club_id = p_club_id
     and badge.context_type = 'tournament'
     and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
   order by badge.id
   for update;

  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and d.updated_at = p_expected_draw_updated_at
   for update of d;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if exists (
      select 1 from public.matches m join public.tournament_games g on g.id = m.tournament_game_id
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULTS_ALREADY_PUBLISHED';
  end if;
  if exists (
      select 1 from public.player_badges
       where club_id = p_club_id and context_type = 'tournament'
         and context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULTS_ALREADY_AWARDED';
  end if;

  -- The atomic reviewed-results function may append teams and their games to a
  -- scheduled draw. This transaction-local marker narrowly bypasses the direct
  -- structural-team guard while the draw and all existing children are locked.
  perform set_config('jupr.tournament_results_import_structural_write', 'on', true);

  -- Player creation is part of this transaction. A duplicate reviewed
  -- create_new name is a conflict, never an implicit reuse of an existing row.
  for v_player in select value from jsonb_array_elements(coalesce(p_new_players, '[]'::jsonb))
  loop
    if nullif(btrim(v_player->>'ref'), '') is null
       or nullif(btrim(v_player->>'name'), '') is null
       or v_player->>'ref' not like 'create:%' then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULT_NEW_PLAYER_INVALID';
    end if;
    if exists (
      select 1 from public.players p
       where p.club_id = p_club_id
         and lower(regexp_replace(btrim(p.name), '\s+', ' ', 'g')) =
             lower(regexp_replace(btrim(v_player->>'name'), '\s+', ' ', 'g'))
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULT_NEW_PLAYER_CONFLICT';
    end if;
    insert into public.players (
      club_id, name, rating, starting_rating, wins, losses, matches_played,
      active, last_game_at, inactive_at
    ) values (
      p_club_id, btrim(v_player->>'name'), 4000.0, 4000.0, 0, 0, 0,
      true, null, null
    ) returning id into v_player_id;
    v_player_map := v_player_map || jsonb_build_object(v_player->>'ref', v_player_id);
    v_created_player_count := v_created_player_count + 1;
  end loop;

  -- Resolve and validate the complete roster before any destructive REPLACE.
  -- The same player cannot occupy two teams, and APPEND cannot reuse a player
  -- already assigned in this draw.
  for v_team in select value from jsonb_array_elements(coalesce(p_teams, '[]'::jsonb))
  loop
    if coalesce(v_team->>'player1_ref', '') like 'existing:%' then
      begin
        v_player1_id := substring(v_team->>'player1_ref' from 10)::integer;
      exception when others then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULT_PLAYER_REF_INVALID';
      end;
    else
      v_player1_id := nullif(v_player_map->>(v_team->>'player1_ref'), '')::integer;
    end if;
    if nullif(v_team->>'player2_ref', '') is null then
      v_player2_id := null;
    elsif v_team->>'player2_ref' like 'existing:%' then
      begin
        v_player2_id := substring(v_team->>'player2_ref' from 10)::integer;
      exception when others then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULT_PLAYER_REF_INVALID';
      end;
    else
      v_player2_id := nullif(v_player_map->>(v_team->>'player2_ref'), '')::integer;
    end if;
    if v_player1_id is null
       or not exists (select 1 from public.players p where p.id = v_player1_id and p.club_id = p_club_id)
       or (v_player2_id is not null and not exists (
            select 1 from public.players p where p.id = v_player2_id and p.club_id = p_club_id
          )) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULT_PLAYER_OUTSIDE_CLUB';
    end if;
    if v_player1_id = v_player2_id
       or v_player1_id = any(v_assigned_player_ids)
       or (v_player2_id is not null and v_player2_id = any(v_assigned_player_ids)) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULT_PLAYER_MULTIPLE_TEAMS';
    end if;
    if upper(p_import_mode) = 'APPEND' and exists (
      select 1 from public.tournament_teams current_team
       where current_team.tournament_id = v_draw.tournament_id
         and current_team.draw_id = v_draw.id
         and (
           current_team.player1_id in (v_player1_id, v_player2_id)
           or current_team.player2_id in (v_player1_id, v_player2_id)
         )
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_RESULT_APPEND_PLAYER_ASSIGNED';
    end if;
    v_assigned_player_ids := array_append(v_assigned_player_ids, v_player1_id);
    if v_player2_id is not null then
      v_assigned_player_ids := array_append(v_assigned_player_ids, v_player2_id);
    end if;
    v_resolved_teams := v_resolved_teams || jsonb_build_array(
      v_team || jsonb_build_object('player1_id', v_player1_id, 'player2_id', v_player2_id)
    );
  end loop;

  if upper(p_import_mode) = 'REPLACE' then
    delete from public.tournament_podium where tournament_id = v_draw.tournament_id and draw_id = v_draw.id;
    delete from public.tournament_games
     where tournament_id = v_draw.tournament_id and draw_id = v_draw.id and upper(stage) = 'PLAYOFF';
    delete from public.tournament_games
     where tournament_id = v_draw.tournament_id and draw_id = v_draw.id and upper(stage) <> 'PLAYOFF';
    delete from public.tournament_teams where tournament_id = v_draw.tournament_id and draw_id = v_draw.id;
  elsif jsonb_array_length(coalesce(p_podium, '[]'::jsonb)) > 0 then
    delete from public.tournament_podium where tournament_id = v_draw.tournament_id and draw_id = v_draw.id;
  end if;

  insert into public.tournament_teams (
    id, tournament_id, draw_id, registration_day_id, event_option_id, team_number,
    player1_id, player2_id, seed, source, notes, created_at, updated_at
  )
  select nullif(x.id, '')::uuid, v_draw.tournament_id, v_draw.id,
         v_draw.registration_day_id, v_draw.event_option_id, x.team_number,
         x.player1_id, x.player2_id, x.seed, 'RESULTS_IMPORT', nullif(x.notes, ''),
         timezone('utc', now()), timezone('utc', now())
    from jsonb_to_recordset(v_resolved_teams) as x(
      id text, team_number integer, player1_id integer, player2_id integer, seed integer, notes text
    );
  get diagnostics v_team_count = row_count;

  insert into public.tournament_games (
    id, tournament_id, draw_id, registration_day_id, event_option_id, stage,
    rr_round_number, rr_slot_number, playoff_game_code, playoff_round,
    team_a_id, team_b_id, score_a, score_b, winner_team_id, loser_team_id,
    finalized_at, created_at, updated_at
  )
  select nullif(x.id, '')::uuid, v_draw.tournament_id, v_draw.id,
         v_draw.registration_day_id, v_draw.event_option_id, x.stage,
         x.rr_round_number, x.rr_slot_number, nullif(x.playoff_game_code, ''), nullif(x.playoff_round, ''),
         nullif(x.team_a_id, '')::uuid, nullif(x.team_b_id, '')::uuid,
         x.score_a, x.score_b, nullif(x.winner_team_id, '')::uuid, nullif(x.loser_team_id, '')::uuid,
         x.finalized_at, timezone('utc', now()), timezone('utc', now())
    from jsonb_to_recordset(coalesce(p_games, '[]'::jsonb)) as x(
      id text, stage text, rr_round_number integer, rr_slot_number integer,
      playoff_game_code text, playoff_round text, team_a_id text, team_b_id text,
      score_a integer, score_b integer, winner_team_id text, loser_team_id text,
      finalized_at timestamptz
    )
   order by case when upper(coalesce(x.stage, '')) = 'ROUND_ROBIN' then 0 else 1 end,
            x.rr_round_number, x.rr_slot_number, x.playoff_game_code, x.id;
  get diagnostics v_game_count = row_count;

  if jsonb_array_length(coalesce(p_podium, '[]'::jsonb)) > 0 then
    insert into public.tournament_podium (id, tournament_id, draw_id, placement, team_id, source, created_at)
    select nullif(x.id, '')::uuid, v_draw.tournament_id, v_draw.id,
           x.placement, nullif(x.team_id, '')::uuid, 'DRAW_RESULTS_IMPORT', timezone('utc', now())
      from jsonb_to_recordset(p_podium) as x(id text, placement integer, team_id text);
    get diagnostics v_podium_count = row_count;
  end if;

  update public.tournament_event_draws set updated_at = clock_timestamp() where id = v_draw.id;
  return jsonb_build_object(
    'ok', true,
    'team_count', v_team_count,
    'game_count', v_game_count,
    'podium_count', v_podium_count,
    'created_player_count', v_created_player_count
  );
end;
$$;

revoke all on function public.admin_write_tournament_draw_teams_cas(text, text, text, timestamptz, boolean, jsonb) from public, anon, authenticated;
revoke all on function public.admin_insert_tournament_draw_games_cas(text, text, text, timestamptz, text, jsonb, jsonb, jsonb) from public, anon, authenticated;
revoke all on function public.admin_replace_tournament_draw_podium_cas(text, text, text, timestamptz, jsonb, jsonb, jsonb) from public, anon, authenticated;
revoke all on function public.admin_score_tournament_game_cas(text, text, text, timestamptz, timestamptz, jsonb, jsonb) from public, anon, authenticated;
revoke all on function public.admin_import_tournament_draw_results_cas(text, text, text, timestamptz, text, jsonb, jsonb, jsonb, jsonb) from public, anon, authenticated;
grant execute on function public.admin_write_tournament_draw_teams_cas(text, text, text, timestamptz, boolean, jsonb) to service_role;
grant execute on function public.admin_insert_tournament_draw_games_cas(text, text, text, timestamptz, text, jsonb, jsonb, jsonb) to service_role;
grant execute on function public.admin_replace_tournament_draw_podium_cas(text, text, text, timestamptz, jsonb, jsonb, jsonb) to service_role;
grant execute on function public.admin_score_tournament_game_cas(text, text, text, timestamptz, timestamptz, jsonb, jsonb) to service_role;
grant execute on function public.admin_import_tournament_draw_results_cas(text, text, text, timestamptz, text, jsonb, jsonb, jsonb, jsonb) to service_role;

comment on table public.tournament_admin_operations is
  'FastAPI-private Tournament Admin and Tournament Ops mutation intents, results, and response-loss recovery state.';

notify pgrst, 'reload schema';
