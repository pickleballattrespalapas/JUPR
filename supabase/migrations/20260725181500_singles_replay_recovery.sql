-- Preserve pre-migration singles aggregates as a stable replay baseline.
--
-- Legacy singles rows were not guaranteed to carry deterministic rating
-- snapshots, so they remain represented by this baseline. New singles writes
-- opt into replay management and can then be rebuilt or excluded without
-- changing the legacy aggregate.

alter table public.players
  add column if not exists singles_replay_baseline jsonb;

update public.players
set singles_replay_baseline = jsonb_build_object(
  'rating', coalesce(singles_rating, rating, 1200.0),
  'wins', coalesce(singles_wins, 0),
  'losses', coalesce(singles_losses, 0),
  'matches_played', coalesce(singles_matches_played, 0),
  'last_game_at', singles_last_game_at
)
where singles_replay_baseline is null;

alter table public.players
  alter column singles_replay_baseline drop default,
  alter column singles_replay_baseline set not null;

alter table public.matches
  add column if not exists singles_replay_managed boolean not null default false;

do $singles_replay_constraints$
begin
  if not exists (
    select 1
      from pg_catalog.pg_constraint
     where conname = 'players_singles_replay_baseline_shape_check'
       and conrelid = 'public.players'::regclass
  ) then
    alter table public.players
      add constraint players_singles_replay_baseline_shape_check
      check (
        jsonb_typeof(singles_replay_baseline) = 'object'
        and singles_replay_baseline ?& array[
          'rating',
          'wins',
          'losses',
          'matches_played'
        ]
        and jsonb_typeof(singles_replay_baseline->'rating') = 'number'
        and jsonb_typeof(singles_replay_baseline->'wins') = 'number'
        and jsonb_typeof(singles_replay_baseline->'losses') = 'number'
        and jsonb_typeof(singles_replay_baseline->'matches_played') = 'number'
        and (singles_replay_baseline->>'wins')::numeric >= 0
        and (singles_replay_baseline->>'losses')::numeric >= 0
        and (singles_replay_baseline->>'matches_played')::numeric >= 0
        and (singles_replay_baseline->>'wins')::numeric
          = trunc((singles_replay_baseline->>'wins')::numeric)
        and (singles_replay_baseline->>'losses')::numeric
          = trunc((singles_replay_baseline->>'losses')::numeric)
        and (singles_replay_baseline->>'matches_played')::numeric
          = trunc((singles_replay_baseline->>'matches_played')::numeric)
        and (
          not (singles_replay_baseline ? 'last_game_at')
          or jsonb_typeof(singles_replay_baseline->'last_game_at')
            in ('null', 'string')
        )
      );
  end if;

  if not exists (
    select 1
      from pg_catalog.pg_constraint
     where conname = 'matches_singles_replay_managed_shape_check'
       and conrelid = 'public.matches'::regclass
  ) then
    alter table public.matches
      add constraint matches_singles_replay_managed_shape_check
      check (
        not singles_replay_managed
        or (
          match_format = 'singles'
          and t1_p1 is not null
          and t1_p2 is null
          and t2_p1 is not null
          and t2_p2 is null
          and t1_p1 <> t2_p1
          and score_t1 is not null
          and score_t2 is not null
          and score_t1 >= 0
          and score_t2 >= 0
          and score_t1 <> score_t2
          and score_t1 + score_t2 > 0
          and coalesce(rating_bonus_elo, 0) >= 0
        )
      );
  end if;
end
$singles_replay_constraints$;

create index if not exists idx_matches_club_singles_replay_order
  on public.matches (club_id, date, id)
  where singles_replay_managed;

comment on column public.players.singles_replay_baseline is
  'Immutable aggregate baseline preceding singles_replay_managed match rows.';

comment on column public.matches.singles_replay_managed is
  'True when the singles row is covered by deterministic Replay History.';

-- An omitted singles rating seeds from the player's current doubles rating.
-- Initialize only missing baselines so imports that explicitly carry a
-- baseline, along with every row backfilled above, remain unchanged.
create or replace function public.initialize_player_singles_replay_baseline()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
begin
  if new.singles_replay_baseline is null then
    new.singles_replay_baseline := jsonb_build_object(
      'rating', coalesce(new.singles_rating, new.rating, 1200.0),
      'wins', coalesce(new.singles_wins, 0),
      'losses', coalesce(new.singles_losses, 0),
      'matches_played', coalesce(new.singles_matches_played, 0),
      'last_game_at', new.singles_last_game_at
    );
  end if;
  return new;
end;
$$;

drop trigger if exists trg_01_players_initialize_singles_replay_baseline
  on public.players;
create trigger trg_01_players_initialize_singles_replay_baseline
before insert on public.players
for each row
execute function public.initialize_player_singles_replay_baseline();

revoke all on function public.initialize_player_singles_replay_baseline()
  from public, anon, authenticated;
grant execute on function public.initialize_player_singles_replay_baseline()
  to service_role;

create or replace function public.protect_player_singles_replay_baseline()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
begin
  if new.singles_replay_baseline is distinct from old.singles_replay_baseline then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_SINGLES_REPLAY_BASELINE_IMMUTABLE';
  end if;
  return new;
end;
$$;

drop trigger if exists trg_02_players_protect_singles_replay_baseline
  on public.players;
create trigger trg_02_players_protect_singles_replay_baseline
before update on public.players
for each row
execute function public.protect_player_singles_replay_baseline();

revoke all on function public.protect_player_singles_replay_baseline()
  from public, anon, authenticated;
grant execute on function public.protect_player_singles_replay_baseline()
  to service_role;

-- The tournament official-publish CAS predates singles replay management and
-- inserts an explicit matches column list. Its transaction-local operation
-- identity is the narrow, authoritative seam for preserving the marker
-- without rewriting the already-applied CAS migration.
create or replace function public.mark_atomic_tournament_singles_replay_managed()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_operation_key text;
begin
  v_operation_key := nullif(
    btrim(pg_catalog.current_setting('jupr.official_publish_operation_key', true)),
    ''
  );

  if v_operation_key is null
     or new.match_format is distinct from 'singles' then
    return new;
  end if;

  if new.context_type is distinct from 'tournament_game'
     or new.tournament_id is null
     or new.tournament_game_id is null
     or new.context_id is distinct from new.tournament_game_id::text
     or not exists (
       select 1
         from public.tournament_admin_operations operation
         join public.tournament_games game
           on game.id = new.tournament_game_id
          and game.tournament_id = new.tournament_id
        where operation.club_id = new.club_id
          and operation.operation_key = v_operation_key
          and operation.entity_type = 'tournament_event_draw'
          and operation.entity_id = game.draw_id::text
          and operation.action in (
            'ops_official_publish',
            'tournament_live_official_publish'
          )
          and operation.status = 'intent'
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_SINGLES_REPLAY_IDENTITY_INVALID';
  end if;

  new.singles_replay_managed := true;
  return new;
end;
$$;

drop trigger if exists trg_01_matches_atomic_tournament_singles_replay_managed
  on public.matches;
create trigger trg_01_matches_atomic_tournament_singles_replay_managed
before insert on public.matches
for each row
execute function public.mark_atomic_tournament_singles_replay_managed();

revoke all on function public.mark_atomic_tournament_singles_replay_managed()
  from public, anon, authenticated;
grant execute on function public.mark_atomic_tournament_singles_replay_managed()
  to service_role;

create or replace function public.protect_match_singles_replay_contract()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_legacy_singles boolean;
begin
  v_legacy_singles := old.singles_replay_managed is not true
    and (
      pg_catalog.lower(pg_catalog.btrim(coalesce(old.match_format, '')))
        = 'singles'
      or (
        old.t1_p2 is null
        and old.t2_p2 is null
        and pg_catalog.strpos(
          pg_catalog.lower(coalesce(old.match_type, '')),
          'singles'
        ) > 0
      )
  );

  if tg_op = 'DELETE' then
    if old.singles_replay_managed is true then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_MANAGED_SINGLES_HARD_DELETE_UNSUPPORTED';
    end if;
    if v_legacy_singles then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_LEGACY_SINGLES_HARD_DELETE_UNSUPPORTED';
    end if;
    return old;
  end if;

  if new.singles_replay_managed is distinct from old.singles_replay_managed then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_SINGLES_REPLAY_MARKER_IMMUTABLE';
  end if;

  if old.singles_replay_managed is true
     and (
       new.id is distinct from old.id
       or new.club_id is distinct from old.club_id
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_SINGLES_REPLAY_IDENTITY_IMMUTABLE';
  end if;

  if old.match_format is distinct from new.match_format
     and (
       pg_catalog.lower(pg_catalog.btrim(coalesce(old.match_format, '')))
         = 'singles'
       or pg_catalog.lower(pg_catalog.btrim(coalesce(new.match_format, '')))
         = 'singles'
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_SINGLES_MATCH_FORMAT_IMMUTABLE';
  end if;

  if v_legacy_singles
     and (
       new.id is distinct from old.id
       or new.club_id is distinct from old.club_id
       or new.date is distinct from old.date
       or new.match_type is distinct from old.match_type
       or new.match_format is distinct from old.match_format
       or new.t1_p1 is distinct from old.t1_p1
       or new.t1_p2 is distinct from old.t1_p2
       or new.t2_p1 is distinct from old.t2_p1
       or new.t2_p2 is distinct from old.t2_p2
       or new.score_t1 is distinct from old.score_t1
       or new.score_t2 is distinct from old.score_t2
       or new.rating_scope is distinct from old.rating_scope
       or new.rating_bonus_elo is distinct from old.rating_bonus_elo
       or new.rating_bonus_reason is distinct from old.rating_bonus_reason
       or new.deleted_at is distinct from old.deleted_at
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_LEGACY_SINGLES_RATING_MUTATION_UNSUPPORTED';
  end if;

  return new;
end;
$$;

drop trigger if exists trg_02_matches_protect_singles_replay_contract
  on public.matches;
create trigger trg_02_matches_protect_singles_replay_contract
before update or delete on public.matches
for each row
execute function public.protect_match_singles_replay_contract();

revoke all on function public.protect_match_singles_replay_contract()
  from public, anon, authenticated;
grant execute on function public.protect_match_singles_replay_contract()
  to service_role;

create or replace function public.bulk_update_player_singles_stats(rows jsonb)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  updated_count integer;
begin
  with data as (
    select *
    from jsonb_to_recordset(rows) as x(
      id bigint,
      club_id text,
      singles_rating double precision,
      singles_wins integer,
      singles_losses integer,
      singles_matches_played integer,
      singles_last_game_at timestamptz
    )
  ),
  upd as (
    update public.players p
    set
      singles_rating = d.singles_rating,
      singles_wins = d.singles_wins,
      singles_losses = d.singles_losses,
      singles_matches_played = d.singles_matches_played,
      singles_last_game_at = d.singles_last_game_at
    from data d
    where p.club_id = d.club_id
      and p.id = d.id
    returning 1
  )
  select count(*) into updated_count from upd;

  return updated_count;
end;
$$;

revoke all on function public.bulk_update_player_singles_stats(jsonb)
  from public, anon, authenticated;
grant execute on function public.bulk_update_player_singles_stats(jsonb)
  to service_role;

notify pgrst, 'reload schema';
