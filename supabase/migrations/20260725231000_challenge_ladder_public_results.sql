-- Durable public result relations for newly published Challenge Ladder matches.
--
-- Legacy/imported challenges are intentionally not backfilled. A public result
-- is available only when the guarded FastAPI publish path can prove both exact
-- official match IDs. Scores, teams, and rating snapshots remain authoritative
-- in matches and are projected by FastAPI through those IDs.

do $challenge_ladder_public_result_source_guard$
declare
  missing_columns text[];
begin
  if to_regclass('public.ladder_challenges') is null
     or to_regclass('public.ladder_roster') is null
     or to_regclass('public.matches') is null
     or to_regclass('public.players') is null
     or to_regclass('public.league_ratings') is null
     or to_regclass('public.leagues_metadata') is null
     or to_regclass('public.live_ladder_admin_operations') is null then
    raise exception using
      errcode = '42P01',
      message = 'challenge ladder public result migration requires ladder_challenges, ladder_roster, matches, players, league_ratings, leagues_metadata, and live_ladder_admin_operations';
  end if;

  select array_agg(
           required.table_name || '.' || required.column_name
           order by required.table_name, required.column_name
         )
    into missing_columns
    from (
      values
        ('ladder_challenges', 'id'),
        ('ladder_challenges', 'club_id'),
        ('ladder_challenges', 'challenger_id'),
        ('ladder_challenges', 'defender_id'),
        ('ladder_challenges', 'tier_id'),
        ('ladder_challenges', 'status'),
        ('ladder_challenges', 'winner_id'),
        ('ladder_challenges', 'completed_at'),
        ('ladder_challenges', 'resolution_notes'),
        ('ladder_challenges', 'forfeit_by'),
        ('ladder_challenges', 'forfeit_reason'),
        ('ladder_roster', 'id'),
        ('ladder_roster', 'club_id'),
        ('ladder_roster', 'player_id'),
        ('ladder_roster', 'tier_id'),
        ('ladder_roster', 'rank'),
        ('ladder_roster', 'is_active'),
        ('ladder_roster', 'updated_at'),
        ('matches', 'id'),
        ('matches', 'club_id'),
        ('matches', 'context_type'),
        ('matches', 'context_id'),
        ('matches', 't1_p1'),
        ('matches', 't1_p2'),
        ('matches', 't2_p1'),
        ('matches', 't2_p2'),
        ('matches', 'score_t1'),
        ('matches', 'score_t2'),
        ('matches', 'date'),
        ('matches', 'league'),
        ('matches', 'elo_delta'),
        ('matches', 'match_type'),
        ('matches', 'week_tag'),
        ('matches', 't1_p1_r'),
        ('matches', 't1_p2_r'),
        ('matches', 't2_p1_r'),
        ('matches', 't2_p2_r'),
        ('matches', 't1_p1_r_end'),
        ('matches', 't1_p2_r_end'),
        ('matches', 't2_p1_r_end'),
        ('matches', 't2_p2_r_end'),
        ('matches', 'rating_scope'),
        ('matches', 'match_format'),
        ('matches', 'rating_bonus_elo'),
        ('matches', 'rating_bonus_reason'),
        ('matches', 'deleted_at'),
        ('players', 'id'),
        ('players', 'club_id'),
        ('players', 'rating'),
        ('players', 'wins'),
        ('players', 'losses'),
        ('players', 'matches_played'),
        ('players', 'last_game_at'),
        ('players', 'inactive_at'),
        ('players', 'active'),
        ('league_ratings', 'id'),
        ('league_ratings', 'club_id'),
        ('league_ratings', 'player_id'),
        ('league_ratings', 'league_name'),
        ('league_ratings', 'rating'),
        ('league_ratings', 'wins'),
        ('league_ratings', 'losses'),
        ('league_ratings', 'matches_played'),
        ('league_ratings', 'starting_rating'),
        ('league_ratings', 'is_active'),
        ('league_ratings', 'inactive_at'),
        ('leagues_metadata', 'id'),
        ('leagues_metadata', 'club_id'),
        ('leagues_metadata', 'league_name'),
        ('leagues_metadata', 'k_factor'),
        ('live_ladder_admin_operations', 'operation_key'),
        ('live_ladder_admin_operations', 'club_id'),
        ('live_ladder_admin_operations', 'surface'),
        ('live_ladder_admin_operations', 'operation_type'),
        ('live_ladder_admin_operations', 'entity_id'),
        ('live_ladder_admin_operations', 'status'),
        ('live_ladder_admin_operations', 'request_json'),
        ('live_ladder_admin_operations', 'result_json'),
        ('live_ladder_admin_operations', 'recovery_json'),
        ('live_ladder_admin_operations', 'error_text'),
        ('live_ladder_admin_operations', 'completed_at'),
        ('live_ladder_admin_operations', 'updated_at')
    ) as required(table_name, column_name)
   where not exists (
     select 1
       from information_schema.columns columns
      where columns.table_schema = 'public'
        and columns.table_name = required.table_name
        and columns.column_name = required.column_name
   );

  if missing_columns is not null then
    raise exception using
      errcode = '42703',
      message = 'challenge ladder public result migration is missing required columns: '
        || array_to_string(missing_columns, ', ');
  end if;
end
$challenge_ladder_public_result_source_guard$;

alter table public.ladder_challenges
  add column if not exists updated_at timestamptz not null
    default timezone('utc', now()),
  add column if not exists public_result_json jsonb;

create or replace function public.validate_challenge_ladder_public_result(
  result_json jsonb
)
returns boolean
language plpgsql
immutable
set search_path = pg_catalog
as $$
declare
  match_ids jsonb;
  rank_change jsonb;
  challenger jsonb;
  defender jsonb;
begin
  if result_json is null then
    return true;
  end if;
  if jsonb_typeof(result_json) is distinct from 'object'
     or not (result_json ?& array['version', 'match_ids', 'rank_change'])
     or result_json - array['version', 'match_ids', 'rank_change'] <> '{}'::jsonb
     or result_json->'version' is distinct from '1'::jsonb then
    return false;
  end if;

  match_ids := result_json->'match_ids';
  if jsonb_typeof(match_ids) is distinct from 'object'
     or not (match_ids ?& array['a', 'b'])
     or match_ids - array['a', 'b'] <> '{}'::jsonb
     or jsonb_typeof(match_ids->'a') is distinct from 'number'
     or jsonb_typeof(match_ids->'b') is distinct from 'number'
     or (match_ids->>'a')::numeric <= 0
     or (match_ids->>'b')::numeric <= 0
     or (match_ids->>'a')::numeric <> trunc((match_ids->>'a')::numeric)
     or (match_ids->>'b')::numeric <> trunc((match_ids->>'b')::numeric)
     or match_ids->>'a' = match_ids->>'b' then
    return false;
  end if;

  rank_change := result_json->'rank_change';
  if jsonb_typeof(rank_change) is distinct from 'object'
     or not (rank_change ?& array['swapped', 'challenger', 'defender'])
     or rank_change - array['swapped', 'challenger', 'defender'] <> '{}'::jsonb
     or jsonb_typeof(rank_change->'swapped') is distinct from 'boolean' then
    return false;
  end if;

  challenger := rank_change->'challenger';
  defender := rank_change->'defender';
  if jsonb_typeof(challenger) is distinct from 'object'
     or jsonb_typeof(defender) is distinct from 'object'
     or not (challenger ?& array['player_id', 'before', 'after'])
     or not (defender ?& array['player_id', 'before', 'after'])
     or challenger - array['player_id', 'before', 'after'] <> '{}'::jsonb
     or defender - array['player_id', 'before', 'after'] <> '{}'::jsonb then
    return false;
  end if;

  if jsonb_typeof(challenger->'player_id') is distinct from 'number'
     or jsonb_typeof(challenger->'before') is distinct from 'number'
     or jsonb_typeof(challenger->'after') is distinct from 'number'
     or jsonb_typeof(defender->'player_id') is distinct from 'number'
     or jsonb_typeof(defender->'before') is distinct from 'number'
     or jsonb_typeof(defender->'after') is distinct from 'number' then
    return false;
  end if;

  if (challenger->>'player_id')::numeric <= 0
     or (defender->>'player_id')::numeric <= 0
     or (challenger->>'before')::numeric <= 0
     or (challenger->>'after')::numeric <= 0
     or (defender->>'before')::numeric <= 0
     or (defender->>'after')::numeric <= 0
     or (challenger->>'player_id')::numeric
        <> trunc((challenger->>'player_id')::numeric)
     or (defender->>'player_id')::numeric
        <> trunc((defender->>'player_id')::numeric)
     or (challenger->>'before')::numeric
        <> trunc((challenger->>'before')::numeric)
     or (challenger->>'after')::numeric
        <> trunc((challenger->>'after')::numeric)
     or (defender->>'before')::numeric
        <> trunc((defender->>'before')::numeric)
     or (defender->>'after')::numeric
        <> trunc((defender->>'after')::numeric) then
    return false;
  end if;

  return true;
exception
  when others then
    return false;
end;
$$;

revoke all on function public.validate_challenge_ladder_public_result(jsonb)
  from public, anon, authenticated;
grant execute on function public.validate_challenge_ladder_public_result(jsonb)
  to service_role;

do $challenge_ladder_public_result_constraint$
begin
  if not exists (
    select 1
      from pg_constraint
     where conrelid = 'public.ladder_challenges'::regclass
       and conname = 'ladder_challenges_public_result_json_check'
  ) then
    alter table public.ladder_challenges
      add constraint ladder_challenges_public_result_json_check
      check (public.validate_challenge_ladder_public_result(public_result_json));
  end if;
end
$challenge_ladder_public_result_constraint$;

comment on column public.ladder_challenges.public_result_json is
  'Public-safe v1 relation to two exact official match IDs plus the transactional ladder rank change. Null for legacy, imported, forfeit, or unlinked results.';

create or replace function public.set_ladder_challenge_updated_at()
returns trigger
language plpgsql
set search_path = pg_catalog
as $$
begin
  new.updated_at := timezone('utc', now());
  return new;
end;
$$;

revoke all on function public.set_ladder_challenge_updated_at()
  from public, anon, authenticated;
grant execute on function public.set_ladder_challenge_updated_at()
  to service_role;

drop trigger if exists ladder_challenges_set_updated_at
  on public.ladder_challenges;
create trigger ladder_challenges_set_updated_at
before update on public.ladder_challenges
for each row execute function public.set_ladder_challenge_updated_at();

-- A Challenge Ladder official publish owns the club-wide doubles-rating domain
-- until its durable receipt is completed or explicitly reconciled. The partial
-- unique index prevents two ladder publishes from claiming the same club, while
-- the row triggers also make tournament and legacy match writers fail closed.
create unique index if not exists idx_live_ladder_admin_operations_active_challenge_rating
  on public.live_ladder_admin_operations (club_id)
  where surface = 'challenge_ladder'
    and operation_type = 'publish_result'
    and status in ('intent', 'running', 'mutated', 'recovery_required');

create or replace function public.block_rating_change_during_challenge_ladder_publish()
returns trigger
language plpgsql
security invoker
set search_path = pg_catalog
as $$
declare
  transaction_operation_key text;
  old_club_id text;
  new_club_id text;
begin
  old_club_id := case when tg_op in ('UPDATE', 'DELETE') then old.club_id else null end;
  new_club_id := case when tg_op in ('INSERT', 'UPDATE') then new.club_id else null end;
  transaction_operation_key := current_setting(
    'jupr.challenge_ladder_atomic_core',
    true
  );

  if exists (
    select 1
      from public.live_ladder_admin_operations operation
     where operation.club_id in (
             coalesce(old_club_id, new_club_id),
             coalesce(new_club_id, old_club_id)
           )
       and operation.surface = 'challenge_ladder'
       and operation.operation_type = 'publish_result'
       and operation.status in ('intent', 'running', 'mutated', 'recovery_required')
       and operation.operation_key is distinct from transaction_operation_key
  ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_RATING_CORE_LOCK';
  end if;

  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_01_matches_challenge_ladder_rating_lock
  on public.matches;
create trigger trg_01_matches_challenge_ladder_rating_lock
before insert or update or delete on public.matches
for each row execute function public.block_rating_change_during_challenge_ladder_publish();

drop trigger if exists trg_01_players_challenge_ladder_rating_lock
  on public.players;
create trigger trg_01_players_challenge_ladder_rating_lock
before insert or update or delete on public.players
for each row execute function public.block_rating_change_during_challenge_ladder_publish();

drop trigger if exists trg_01_league_ratings_challenge_ladder_rating_lock
  on public.league_ratings;
create trigger trg_01_league_ratings_challenge_ladder_rating_lock
before insert or update or delete on public.league_ratings
for each row execute function public.block_rating_change_during_challenge_ladder_publish();

drop trigger if exists trg_01_leagues_metadata_challenge_ladder_rating_lock
  on public.leagues_metadata;
create trigger trg_01_leagues_metadata_challenge_ladder_rating_lock
before insert or update or delete on public.leagues_metadata
for each row execute function public.block_rating_change_during_challenge_ladder_publish();

revoke all on function public.block_rating_change_during_challenge_ladder_publish()
  from public, anon, authenticated;
grant execute on function public.block_rating_change_during_challenge_ladder_publish()
  to service_role;

create or replace function public.admin_finalize_challenge_ladder_result_v1(
  p_club_id text,
  p_challenge_id bigint,
  p_operation_key text,
  p_winner_id bigint,
  p_completed_at timestamptz,
  p_resolution_notes text,
  p_public_result_json jsonb,
  p_match_context_a text default null,
  p_match_context_b text default null
)
returns jsonb
language plpgsql
security invoker
set search_path = pg_catalog
as $$
declare
  operation_row public.live_ladder_admin_operations%rowtype;
  challenge_row public.ladder_challenges%rowtype;
  challenger_roster public.ladder_roster%rowtype;
  defender_roster public.ladder_roster%rowtype;
  match_a public.matches%rowtype;
  match_b public.matches%rowtype;
  updated_challenge public.ladder_challenges%rowtype;
  final_public_result jsonb;
  rank_change jsonb;
  receipt jsonb;
  atomic_core jsonb;
  challenge_expected jsonb;
  tier_roster_expected jsonb;
  temp_rank public.ladder_roster.rank%type;
  active_ranked_count integer;
  write_count integer;
  swapped boolean;
begin
  select *
    into operation_row
    from public.live_ladder_admin_operations
   where operation_key = p_operation_key
     and club_id = p_club_id
   for update;
  if not found
     or operation_row.surface is distinct from 'challenge_ladder'
     or operation_row.operation_type is distinct from 'publish_result'
     or operation_row.entity_id is distinct from p_challenge_id::text
     or operation_row.status is distinct from 'running' then
    raise exception using errcode = '55000', message = 'running durable challenge result operation not found';
  end if;

  atomic_core := operation_row.request_json->'atomic_core';
  challenge_expected := atomic_core->'challenge_expected';
  tier_roster_expected := atomic_core->'tier_roster_expected';
  if jsonb_typeof(atomic_core) is distinct from 'object'
     or atomic_core->'version' is distinct from '1'::jsonb
     or jsonb_typeof(challenge_expected) is distinct from 'object'
     or jsonb_typeof(tier_roster_expected) is distinct from 'array'
     or jsonb_typeof(atomic_core->'write_plan') is distinct from 'object'
     or jsonb_typeof(atomic_core #> '{write_plan,match_rows}') is distinct from 'array'
     or jsonb_typeof(atomic_core #> '{write_plan,player_updates}') is distinct from 'array'
     or jsonb_typeof(atomic_core #> '{write_plan,league_rating_updates}') is distinct from 'array'
     or jsonb_typeof(atomic_core #> '{write_plan,league_metadata_expectations}') is distinct from 'array'
     or jsonb_typeof(atomic_core->'match_context_ids') is distinct from 'array'
     or jsonb_typeof(atomic_core->'publish_official_matches') is distinct from 'boolean'
     or jsonb_typeof(atomic_core->'winner_id') is distinct from 'number'
     or jsonb_typeof(atomic_core->'completed_at') is distinct from 'string'
     or jsonb_typeof(atomic_core->'resolution_notes') is distinct from 'string'
     or nullif(atomic_core->>'plan_fingerprint', '') is null
     or challenge_expected->>'club_id' is distinct from p_club_id
     or (challenge_expected->>'id')::bigint is distinct from p_challenge_id
     or (atomic_core->>'winner_id')::bigint is distinct from p_winner_id
     or (atomic_core->>'completed_at')::timestamptz is distinct from p_completed_at
     or atomic_core->>'resolution_notes' is distinct from p_resolution_notes
     or (atomic_core->>'publish_official_matches')::boolean
        is distinct from (p_public_result_json is not null)
     or atomic_core->'match_context_ids' is distinct from (
          case
            when p_public_result_json is null then '[]'::jsonb
            else jsonb_build_array(p_match_context_a, p_match_context_b)
          end
        ) then
    raise exception using
      errcode = '55000',
      message = 'durable challenge result operation plan does not match finalization parameters';
  end if;

  if p_public_result_json is null
     and (
       jsonb_array_length(coalesce(atomic_core #> '{write_plan,match_rows}', '[]'::jsonb)) <> 0
       or jsonb_array_length(coalesce(atomic_core #> '{write_plan,player_updates}', '[]'::jsonb)) <> 0
       or jsonb_array_length(coalesce(atomic_core #> '{write_plan,league_rating_updates}', '[]'::jsonb)) <> 0
       or jsonb_array_length(coalesce(atomic_core #> '{write_plan,league_metadata_expectations}', '[]'::jsonb)) <> 0
     ) then
    raise exception using
      errcode = '55000',
      message = 'non-official challenge result plan contains rating writes';
  end if;

  select *
    into challenge_row
    from public.ladder_challenges
   where club_id = p_club_id
     and id = p_challenge_id
   for update;
  if not found then
    raise exception using errcode = 'P0002', message = 'challenge not found';
  end if;
  if challenge_row.id is distinct from (challenge_expected->>'id')::bigint
     or challenge_row.club_id is distinct from challenge_expected->>'club_id'
     or challenge_row.challenger_id is distinct from (challenge_expected->>'challenger_id')::bigint
     or challenge_row.defender_id is distinct from (challenge_expected->>'defender_id')::bigint
     or challenge_row.tier_id is distinct from challenge_expected->>'tier_id'
     or challenge_row.status::text is distinct from challenge_expected->>'status'
     or challenge_row.updated_at is distinct from nullif(challenge_expected->>'updated_at', '')::timestamptz
     or challenge_row.winner_id is distinct from nullif(challenge_expected->>'winner_id', '')::bigint
     or challenge_row.completed_at is distinct from nullif(challenge_expected->>'completed_at', '')::timestamptz
     or challenge_row.public_result_json is distinct from (
          case
            when challenge_expected->'public_result_json' is null
              or challenge_expected->'public_result_json' = 'null'::jsonb
              then null
            else challenge_expected->'public_result_json'
          end
        ) then
    raise exception using
      errcode = '55000',
      message = 'challenge changed after the atomic result plan was prepared';
  end if;
  if challenge_row.status::text not in (
       'ACCEPTED_SCHEDULING',
       'ACCEPTED',
       'IN_PROGRESS',
       'AWAITING_VERIFICATION',
       'OVERDUE_PLAY'
     )
     or challenge_row.completed_at is not null
     or challenge_row.winner_id is not null then
    raise exception using errcode = '55000', message = 'challenge cannot accept a played result';
  end if;
  if p_winner_id not in (
       challenge_row.challenger_id::bigint,
       challenge_row.defender_id::bigint
     ) then
    raise exception using errcode = '22023', message = 'winner must be a ranked challenge participant';
  end if;

  -- Lock the complete tier before choosing a temporary rank. This avoids the
  -- immediate unique-index collision caused by swapping two ranks directly.
  perform 1
    from public.ladder_roster
   where club_id = p_club_id
     and tier_id = challenge_row.tier_id
   order by id
   for update;

  if (
       select count(*)
         from public.ladder_roster roster
        where roster.club_id = p_club_id
          and roster.tier_id = challenge_row.tier_id
     ) <> jsonb_array_length(tier_roster_expected)
     or (
       select count(distinct expected.id)
         from jsonb_to_recordset(tier_roster_expected)
           as expected(
             id bigint,
             player_id bigint,
             tier_id text,
             rank integer,
             is_active boolean,
             updated_at timestamptz
           )
     ) <> jsonb_array_length(tier_roster_expected)
     or exists (
       select 1
         from jsonb_to_recordset(tier_roster_expected)
           as expected(
             id bigint,
             player_id bigint,
             tier_id text,
             rank integer,
             is_active boolean,
             updated_at timestamptz
           )
        where expected.id is null
           or expected.player_id is null
           or expected.tier_id is distinct from challenge_row.tier_id
           or expected.rank is null
           or expected.is_active is null
           or expected.updated_at is null
           or not exists (
             select 1
               from public.ladder_roster roster
              where roster.club_id = p_club_id
                and roster.tier_id = challenge_row.tier_id
                and roster.id = expected.id
                and roster.player_id is not distinct from expected.player_id
                and roster.rank is not distinct from expected.rank
                and roster.is_active is not distinct from expected.is_active
                and roster.updated_at is not distinct from expected.updated_at
           )
     )
     or exists (
       select 1
         from public.ladder_roster roster
        where roster.club_id = p_club_id
          and roster.tier_id = challenge_row.tier_id
          and not exists (
            select 1
              from jsonb_to_recordset(tier_roster_expected)
                as expected(
                  id bigint,
                  player_id bigint,
                  tier_id text,
                  rank integer,
                  is_active boolean,
                  updated_at timestamptz
                )
             where expected.id = roster.id
               and expected.player_id is not distinct from roster.player_id
               and expected.tier_id is not distinct from roster.tier_id
               and expected.rank is not distinct from roster.rank
               and expected.is_active is not distinct from roster.is_active
               and expected.updated_at is not distinct from roster.updated_at
          )
     ) then
    raise exception using
      errcode = '55000',
      message = 'challenge tier roster changed after the atomic result plan was prepared';
  end if;

  select count(*)
    into active_ranked_count
    from public.ladder_roster
   where club_id = p_club_id
     and tier_id = challenge_row.tier_id
     and player_id in (
       challenge_row.challenger_id,
       challenge_row.defender_id
     )
     and is_active is not false;
  if active_ranked_count <> 2 then
    raise exception using errcode = '55000', message = 'exact active ladder rows are required for both challenge participants';
  end if;

  select *
    into challenger_roster
    from public.ladder_roster
   where club_id = p_club_id
     and player_id = challenge_row.challenger_id
     and tier_id = challenge_row.tier_id
     and is_active is not false
   limit 1;
  select *
    into defender_roster
    from public.ladder_roster
   where club_id = p_club_id
     and player_id = challenge_row.defender_id
     and tier_id = challenge_row.tier_id
     and is_active is not false
   limit 1;

  if challenger_roster.tier_id is distinct from defender_roster.tier_id
     or challenger_roster.tier_id is distinct from challenge_row.tier_id
     or challenger_roster.rank is null
     or defender_roster.rank is null then
    raise exception using errcode = '55000', message = 'challenge participants must have ranked rows in the challenge tier';
  end if;

  if p_public_result_json is not null then
    if current_setting('jupr.challenge_ladder_atomic_core', true)
         is distinct from p_operation_key then
      raise exception using
        errcode = '55000',
        message = 'official challenge matches may only be finalized by the bound atomic core';
    end if;
    if jsonb_typeof(p_public_result_json) is distinct from 'object'
       or not (p_public_result_json ?& array['version', 'match_ids'])
       or p_public_result_json - array['version', 'match_ids'] <> '{}'::jsonb
       or p_public_result_json->'version' is distinct from '1'::jsonb
       or jsonb_typeof(p_public_result_json->'match_ids') is distinct from 'object'
       or not ((p_public_result_json->'match_ids') ?& array['a', 'b'])
       or (p_public_result_json->'match_ids') - array['a', 'b'] <> '{}'::jsonb
       or jsonb_typeof(p_public_result_json #> '{match_ids,a}') is distinct from 'number'
       or jsonb_typeof(p_public_result_json #> '{match_ids,b}') is distinct from 'number'
       or (p_public_result_json #>> '{match_ids,a}')::bigint <= 0
       or (p_public_result_json #>> '{match_ids,b}')::bigint <= 0
       or (p_public_result_json #>> '{match_ids,a}')
          = (p_public_result_json #>> '{match_ids,b}') then
      raise exception using errcode = '22023', message = 'invalid challenge public result relation';
    end if;
    if nullif(btrim(p_match_context_a), '') is null
       or nullif(btrim(p_match_context_b), '') is null
       or p_match_context_a = p_match_context_b then
      raise exception using errcode = '22023', message = 'two exact official match contexts are required';
    end if;
    if jsonb_typeof(operation_row.recovery_json->'match_context_ids') is distinct from 'array'
       or jsonb_array_length(operation_row.recovery_json->'match_context_ids') <> 2
       or operation_row.recovery_json->'match_context_ids'
          is distinct from jsonb_build_array(p_match_context_a, p_match_context_b) then
      raise exception using errcode = '55000', message = 'official match contexts do not match the durable operation';
    end if;

    select *
      into match_a
      from public.matches
     where club_id = p_club_id
       and id = (p_public_result_json #>> '{match_ids,a}')::bigint
       and context_type = 'challenge_ladder'
       and context_id = p_match_context_a
       and deleted_at is null
     for share;
    if not found then
      raise exception using errcode = 'P0002', message = 'official challenge match A not found';
    end if;
    select *
      into match_b
      from public.matches
     where club_id = p_club_id
       and id = (p_public_result_json #>> '{match_ids,b}')::bigint
       and context_type = 'challenge_ladder'
       and context_id = p_match_context_b
       and deleted_at is null
     for share;
    if not found then
      raise exception using errcode = 'P0002', message = 'official challenge match B not found';
    end if;

    if match_a.t1_p1::bigint is distinct from challenge_row.challenger_id::bigint
       or match_a.t2_p1::bigint is distinct from challenge_row.defender_id::bigint
       or match_b.t1_p1::bigint is distinct from challenge_row.challenger_id::bigint
       or match_b.t2_p1::bigint is distinct from challenge_row.defender_id::bigint
       or match_a.t1_p2 is null
       or match_a.t2_p2 is null
       or match_b.t1_p2 is null
       or match_b.t2_p2 is null
       or match_a.t1_p2::bigint is not distinct from challenge_row.challenger_id::bigint
       or match_a.t1_p2::bigint is not distinct from challenge_row.defender_id::bigint
       or match_a.t2_p2::bigint is not distinct from challenge_row.challenger_id::bigint
       or match_a.t2_p2::bigint is not distinct from challenge_row.defender_id::bigint
       or match_b.t1_p2::bigint is not distinct from challenge_row.challenger_id::bigint
       or match_b.t1_p2::bigint is not distinct from challenge_row.defender_id::bigint
       or match_b.t2_p2::bigint is not distinct from challenge_row.challenger_id::bigint
       or match_b.t2_p2::bigint is not distinct from challenge_row.defender_id::bigint
       or match_a.t1_p2 is not distinct from match_a.t2_p2
       or match_a.t1_p2 is distinct from match_b.t2_p2
       or match_a.t2_p2 is distinct from match_b.t1_p2
       or match_a.score_t1 is null
       or match_a.score_t2 is null
       or match_b.score_t1 is null
       or match_b.score_t2 is null then
      raise exception using errcode = '22023', message = 'official challenge matches do not match the ranked participants and swing-partner format';
    end if;
  elsif p_match_context_a is not null
        or p_match_context_b is not null
        or operation_row.recovery_json->'match_context_ids' is distinct from '[]'::jsonb then
    raise exception using errcode = '22023', message = 'match contexts require a public result relation';
  end if;

  swapped := p_winner_id = challenge_row.challenger_id::bigint;
  rank_change := jsonb_build_object(
    'swapped', swapped,
    'challenger', jsonb_build_object(
      'player_id', challenge_row.challenger_id,
      'before', challenger_roster.rank,
      'after', case when swapped then defender_roster.rank else challenger_roster.rank end
    ),
    'defender', jsonb_build_object(
      'player_id', challenge_row.defender_id,
      'before', defender_roster.rank,
      'after', case when swapped then challenger_roster.rank else defender_roster.rank end
    )
  );

  if swapped then
    select coalesce(max(rank), 0) + 1
      into temp_rank
      from public.ladder_roster
     where club_id = p_club_id
       and tier_id = challenge_row.tier_id;

    update public.ladder_roster
       set rank = temp_rank,
           updated_at = p_completed_at
     where id = challenger_roster.id;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using errcode = '55000', message = 'challenger temporary rank write was incomplete';
    end if;
    update public.ladder_roster
       set rank = challenger_roster.rank,
           updated_at = p_completed_at
     where id = defender_roster.id;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using errcode = '55000', message = 'defender rank write was incomplete';
    end if;
    update public.ladder_roster
       set rank = defender_roster.rank,
           updated_at = p_completed_at
     where id = challenger_roster.id;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using errcode = '55000', message = 'challenger final rank write was incomplete';
    end if;
  end if;

  final_public_result := case
    when p_public_result_json is null then null
    else jsonb_set(p_public_result_json, '{rank_change}', rank_change, true)
  end;

  update public.ladder_challenges
     set status = 'COMPLETED',
         winner_id = p_winner_id,
         completed_at = p_completed_at,
         resolution_notes = p_resolution_notes,
         public_result_json = final_public_result,
         updated_at = p_completed_at
   where club_id = p_club_id
     and id = p_challenge_id
   returning * into updated_challenge;
  get diagnostics write_count = row_count;
  if write_count <> 1 then
    raise exception using errcode = '55000', message = 'challenge result write was incomplete';
  end if;

  receipt := jsonb_build_object(
    'ok', true,
    'core_committed', true,
    'mode', 'challenge_ladder_result_core',
    'challenge', to_jsonb(updated_challenge),
    'rank_result', rank_change,
    'public_result_json', final_public_result,
    'official_matches', case
      when final_public_result is null then jsonb_build_object(
        'inserted', 0,
        'skipped', true,
        'atomic', true,
        'match_ids', '{}'::jsonb,
        'match_context_ids', '[]'::jsonb
      )
      else jsonb_build_object(
        'inserted', 2,
        'skipped', false,
        'atomic', true,
        'match_ids', final_public_result->'match_ids',
        'match_context_ids', jsonb_build_array(
          p_match_context_a,
          p_match_context_b
        )
      )
    end,
    'post_processors', jsonb_build_object(
      'status',
      case when final_public_result is null then 'complete' else 'pending' end
    ),
    'side_effect_context', coalesce(
      atomic_core->'side_effect_context',
      '{}'::jsonb
    ),
    'plan_fingerprint', atomic_core->>'plan_fingerprint'
  );

  update public.live_ladder_admin_operations
     set status = 'mutated',
         result_json = receipt,
         error_text = null,
         completed_at = p_completed_at,
         updated_at = timezone('utc', now())
   where operation_key = p_operation_key
     and club_id = p_club_id
     and status = 'running';
  get diagnostics write_count = row_count;
  if write_count <> 1 then
    raise exception using
      errcode = '55000',
      message = 'durable challenge result receipt write was incomplete';
  end if;

  return receipt;
end;
$$;

revoke all on function public.admin_finalize_challenge_ladder_result_v1(
  text, bigint, text, bigint, timestamptz, text, jsonb, text, text
) from public, anon, authenticated;
grant execute on function public.admin_finalize_challenge_ladder_result_v1(
  text, bigint, text, bigint, timestamptz, text, jsonb, text, text
) to service_role;

create or replace function public.admin_apply_challenge_ladder_result_atomic_v1(
  p_club_id text,
  p_challenge_id bigint,
  p_operation_key text,
  p_atomic_core jsonb,
  p_plan_fingerprint text,
  p_winner_id bigint,
  p_completed_at timestamptz,
  p_resolution_notes text,
  p_match_rows jsonb,
  p_player_updates jsonb,
  p_league_rating_updates jsonb,
  p_league_metadata_expectations jsonb,
  p_match_context_a text,
  p_match_context_b text
)
returns jsonb
language plpgsql
security invoker
set search_path = pg_catalog
as $$
declare
  operation_row public.live_ladder_admin_operations%rowtype;
  challenge_row public.ladder_challenges%rowtype;
  player_row public.players%rowtype;
  league_rating_row public.league_ratings%rowtype;
  metadata_row public.leagues_metadata%rowtype;
  match_plan_a record;
  match_plan_b record;
  plan_item record;
  expected_json jsonb;
  after_json jsonb;
  challenge_expected jsonb;
  tier_roster_expected jsonb;
  participant_ids bigint[];
  inserted_count integer := 0;
  write_count integer := 0;
  match_a_id bigint;
  match_b_id bigint;
  public_relation jsonb;
  receipt jsonb;
begin
  if nullif(btrim(p_club_id), '') is null
     or p_challenge_id is null
     or p_challenge_id <= 0
     or nullif(btrim(p_operation_key), '') is null
     or nullif(btrim(p_plan_fingerprint), '') is null
     or p_winner_id is null
     or p_completed_at is null
     or nullif(btrim(p_resolution_notes), '') is null
     or nullif(btrim(p_match_context_a), '') is null
     or nullif(btrim(p_match_context_b), '') is null
     or p_match_context_a = p_match_context_b
     or jsonb_typeof(p_atomic_core) is distinct from 'object'
     or p_atomic_core->'version' is distinct from '1'::jsonb
     or p_atomic_core->>'plan_fingerprint' is distinct from p_plan_fingerprint
     or p_atomic_core->'publish_official_matches' is distinct from 'true'::jsonb
     or jsonb_typeof(p_atomic_core->'challenge_expected') is distinct from 'object'
     or jsonb_typeof(p_atomic_core->'tier_roster_expected') is distinct from 'array'
     or jsonb_typeof(p_atomic_core->'match_payloads') is distinct from 'array'
     or jsonb_typeof(p_atomic_core->'match_context_ids') is distinct from 'array'
     or jsonb_typeof(p_atomic_core->'write_plan') is distinct from 'object'
     or jsonb_typeof(p_atomic_core #> '{write_plan,match_rows}') is distinct from 'array'
     or jsonb_typeof(p_atomic_core #> '{write_plan,player_updates}') is distinct from 'array'
     or jsonb_typeof(p_atomic_core #> '{write_plan,league_rating_updates}') is distinct from 'array'
     or jsonb_typeof(p_atomic_core #> '{write_plan,league_metadata_expectations}') is distinct from 'array'
     or jsonb_typeof(p_match_rows) is distinct from 'array'
     or jsonb_array_length(p_match_rows) <> 2
     or jsonb_typeof(p_player_updates) is distinct from 'array'
     or jsonb_array_length(p_player_updates) <> 4
     or jsonb_typeof(p_league_rating_updates) is distinct from 'array'
     or jsonb_array_length(p_league_rating_updates) <> 4
     or jsonb_typeof(p_league_metadata_expectations) is distinct from 'array'
     or jsonb_array_length(p_league_metadata_expectations) <> 1
     or p_atomic_core #> '{write_plan,match_rows}' is distinct from p_match_rows
     or p_atomic_core #> '{write_plan,player_updates}' is distinct from p_player_updates
     or p_atomic_core #> '{write_plan,league_rating_updates}' is distinct from p_league_rating_updates
     or p_atomic_core #> '{write_plan,league_metadata_expectations}'
        is distinct from p_league_metadata_expectations
     or p_atomic_core->'match_context_ids'
        is distinct from jsonb_build_array(p_match_context_a, p_match_context_b)
     or (p_atomic_core->>'winner_id')::bigint is distinct from p_winner_id
     or (p_atomic_core->>'completed_at')::timestamptz is distinct from p_completed_at
     or p_atomic_core->>'resolution_notes' is distinct from p_resolution_notes then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_PLAN_INVALID';
  end if;

  select operation.*
    into operation_row
    from public.live_ladder_admin_operations operation
   where operation.operation_key = p_operation_key
     and operation.club_id = p_club_id
   for update;
  if not found
     or operation_row.surface is distinct from 'challenge_ladder'
     or operation_row.operation_type is distinct from 'publish_result'
     or operation_row.entity_id is distinct from p_challenge_id::text
     or operation_row.status is distinct from 'running'
     or operation_row.request_json->'atomic_core' is distinct from p_atomic_core
     or operation_row.recovery_json->'match_context_ids'
        is distinct from jsonb_build_array(p_match_context_a, p_match_context_b) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_OPERATION_STALE';
  end if;

  perform set_config(
    'jupr.challenge_ladder_atomic_core',
    p_operation_key,
    true
  );

  challenge_expected := p_atomic_core->'challenge_expected';
  tier_roster_expected := p_atomic_core->'tier_roster_expected';
  select challenge.*
    into challenge_row
    from public.ladder_challenges challenge
   where challenge.club_id = p_club_id
     and challenge.id = p_challenge_id
   for update;
  if not found
     or challenge_row.id is distinct from (challenge_expected->>'id')::bigint
     or challenge_row.club_id is distinct from challenge_expected->>'club_id'
     or challenge_row.challenger_id is distinct from (challenge_expected->>'challenger_id')::bigint
     or challenge_row.defender_id is distinct from (challenge_expected->>'defender_id')::bigint
     or challenge_row.tier_id is distinct from challenge_expected->>'tier_id'
     or challenge_row.status::text is distinct from challenge_expected->>'status'
     or challenge_row.updated_at is distinct from nullif(challenge_expected->>'updated_at', '')::timestamptz
     or challenge_row.winner_id is distinct from nullif(challenge_expected->>'winner_id', '')::bigint
     or challenge_row.completed_at is distinct from nullif(challenge_expected->>'completed_at', '')::timestamptz
     or challenge_row.public_result_json is distinct from (
          case
            when challenge_expected->'public_result_json' is null
              or challenge_expected->'public_result_json' = 'null'::jsonb
              then null
            else challenge_expected->'public_result_json'
          end
        ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_CHALLENGE_STALE';
  end if;
  if challenge_row.status::text not in (
       'ACCEPTED_SCHEDULING',
       'ACCEPTED',
       'IN_PROGRESS',
       'AWAITING_VERIFICATION',
       'OVERDUE_PLAY'
     )
     or challenge_row.completed_at is not null
     or challenge_row.winner_id is not null
     or p_winner_id not in (
       challenge_row.challenger_id::bigint,
       challenge_row.defender_id::bigint
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_CHALLENGE_STALE';
  end if;

  perform roster.id
    from public.ladder_roster roster
   where roster.club_id = p_club_id
     and roster.tier_id = challenge_row.tier_id
   order by roster.id
   for update;
  if (
       select count(*)
         from public.ladder_roster roster
        where roster.club_id = p_club_id
          and roster.tier_id = challenge_row.tier_id
     ) <> jsonb_array_length(tier_roster_expected)
     or (
       select count(distinct expected.id)
         from jsonb_to_recordset(tier_roster_expected)
           as expected(
             id bigint,
             player_id bigint,
             tier_id text,
             rank integer,
             is_active boolean,
             updated_at timestamptz
           )
     ) <> jsonb_array_length(tier_roster_expected)
     or exists (
       select 1
         from jsonb_to_recordset(tier_roster_expected)
           as expected(
             id bigint,
             player_id bigint,
             tier_id text,
             rank integer,
             is_active boolean,
             updated_at timestamptz
           )
        where expected.id is null
           or expected.player_id is null
           or expected.tier_id is distinct from challenge_row.tier_id
           or expected.rank is null
           or expected.is_active is null
           or expected.updated_at is null
           or not exists (
             select 1
               from public.ladder_roster roster
              where roster.club_id = p_club_id
                and roster.tier_id = challenge_row.tier_id
                and roster.id = expected.id
                and roster.player_id is not distinct from expected.player_id
                and roster.rank is not distinct from expected.rank
                and roster.is_active is not distinct from expected.is_active
                and roster.updated_at is not distinct from expected.updated_at
           )
     )
     or exists (
       select 1
         from public.ladder_roster roster
        where roster.club_id = p_club_id
          and roster.tier_id = challenge_row.tier_id
          and not exists (
            select 1
              from jsonb_to_recordset(tier_roster_expected)
                as expected(
                  id bigint,
                  player_id bigint,
                  tier_id text,
                  rank integer,
                  is_active boolean,
                  updated_at timestamptz
                )
             where expected.id = roster.id
               and expected.player_id is not distinct from roster.player_id
               and expected.tier_id is not distinct from roster.tier_id
               and expected.rank is not distinct from roster.rank
               and expected.is_active is not distinct from roster.is_active
               and expected.updated_at is not distinct from roster.updated_at
          )
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_ROSTER_STALE';
  end if;

  if (
       select count(*)
         from public.ladder_roster roster
        where roster.club_id = p_club_id
          and roster.tier_id = challenge_row.tier_id
          and roster.player_id in (
            challenge_row.challenger_id,
            challenge_row.defender_id
          )
          and roster.is_active is not false
     ) <> 2 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_ROSTER_STALE';
  end if;

  if jsonb_array_length(p_atomic_core->'match_payloads') <> 2
     or (
       select count(distinct row.value->>'context_id')
         from jsonb_array_elements(p_match_rows) row(value)
     ) <> 2
     or exists (
       select 1
         from jsonb_array_elements(p_match_rows) row(value)
        where row.value->>'club_id' is distinct from p_club_id
           or row.value->>'context_type' is distinct from 'challenge_ladder'
           or row.value->>'context_id' not in (
             p_match_context_a,
             p_match_context_b
           )
           or row.value->>'league' is distinct from 'OVERALL'
           or row.value->>'match_type' is distinct from 'ChallengeLadder'
           or coalesce(row.value->>'match_format', 'doubles') is distinct from 'doubles'
           or coalesce(row.value->>'rating_scope', '') is distinct from ''
           or coalesce((row.value->>'rating_bonus_elo')::double precision, 0) <> 0
           or nullif(row.value->>'rating_bonus_reason', '') is not null
           or nullif(row.value->>'date', '') is null
           or (row.value->>'score_t1')::integer < 0
           or (row.value->>'score_t2')::integer < 0
           or (row.value->>'score_t1')::integer
              = (row.value->>'score_t2')::integer
           or (row.value->>'score_t1')::integer
              + (row.value->>'score_t2')::integer <= 0
     )
     or exists (
       select 1
         from jsonb_array_elements(p_match_rows)
           with ordinality planned(value, ordinal_position)
         join jsonb_array_elements(p_atomic_core->'match_payloads')
           with ordinality payload(value, ordinal_position)
           using (ordinal_position)
        where (planned.value->>'date')::timestamptz
                is distinct from (payload.value->>'date')::timestamptz
           or planned.value->>'league' is distinct from payload.value->>'league'
           or planned.value->>'match_type' is distinct from payload.value->>'match_type'
           or planned.value->>'context_type' is distinct from payload.value->>'context_type'
           or planned.value->>'context_id' is distinct from payload.value->>'context_id'
           or (planned.value->>'t1_p1')::bigint
                is distinct from (payload.value->>'t1_p1')::bigint
           or (planned.value->>'t1_p2')::bigint
                is distinct from (payload.value->>'t1_p2')::bigint
           or (planned.value->>'t2_p1')::bigint
                is distinct from (payload.value->>'t2_p1')::bigint
           or (planned.value->>'t2_p2')::bigint
                is distinct from (payload.value->>'t2_p2')::bigint
           or (planned.value->>'score_t1')::integer
                is distinct from (payload.value->>'s1')::integer
           or (planned.value->>'score_t2')::integer
                is distinct from (payload.value->>'s2')::integer
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_MATCH_PLAN_INVALID';
  end if;

  select item.*
    into match_plan_a
    from jsonb_to_recordset(p_match_rows)
      as item(
        club_id text,
        date timestamptz,
        league text,
        t1_p1 bigint,
        t1_p2 bigint,
        t2_p1 bigint,
        t2_p2 bigint,
        score_t1 integer,
        score_t2 integer,
        elo_delta double precision,
        match_type text,
        week_tag text,
        t1_p1_r double precision,
        t1_p2_r double precision,
        t2_p1_r double precision,
        t2_p2_r double precision,
        t1_p1_r_end double precision,
        t1_p2_r_end double precision,
        t2_p1_r_end double precision,
        t2_p2_r_end double precision,
        context_type text,
        context_id text,
        rating_scope text,
        match_format text,
        rating_bonus_elo double precision,
        rating_bonus_reason text
      )
   where item.context_id = p_match_context_a;
  if not found then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_MATCH_PLAN_INVALID';
  end if;
  select item.*
    into match_plan_b
    from jsonb_to_recordset(p_match_rows)
      as item(
        club_id text,
        date timestamptz,
        league text,
        t1_p1 bigint,
        t1_p2 bigint,
        t2_p1 bigint,
        t2_p2 bigint,
        score_t1 integer,
        score_t2 integer,
        elo_delta double precision,
        match_type text,
        week_tag text,
        t1_p1_r double precision,
        t1_p2_r double precision,
        t2_p1_r double precision,
        t2_p2_r double precision,
        t1_p1_r_end double precision,
        t1_p2_r_end double precision,
        t2_p1_r_end double precision,
        t2_p2_r_end double precision,
        context_type text,
        context_id text,
        rating_scope text,
        match_format text,
        rating_bonus_elo double precision,
        rating_bonus_reason text
      )
   where item.context_id = p_match_context_b;
  if not found
     or match_plan_a.t1_p1 is distinct from challenge_row.challenger_id
     or match_plan_a.t2_p1 is distinct from challenge_row.defender_id
     or match_plan_b.t1_p1 is distinct from challenge_row.challenger_id
     or match_plan_b.t2_p1 is distinct from challenge_row.defender_id
     or match_plan_a.t1_p2 is null
     or match_plan_a.t2_p2 is null
     or match_plan_b.t1_p2 is null
     or match_plan_b.t2_p2 is null
     or match_plan_a.t1_p2 in (
       challenge_row.challenger_id,
       challenge_row.defender_id
     )
     or match_plan_a.t2_p2 in (
       challenge_row.challenger_id,
       challenge_row.defender_id
     )
     or match_plan_b.t1_p2 in (
       challenge_row.challenger_id,
       challenge_row.defender_id
     )
     or match_plan_b.t2_p2 in (
       challenge_row.challenger_id,
       challenge_row.defender_id
     )
     or match_plan_a.t1_p2 = match_plan_a.t2_p2
     or match_plan_a.t1_p2 is distinct from match_plan_b.t2_p2
     or match_plan_a.t2_p2 is distinct from match_plan_b.t1_p2 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_MATCH_PLAN_INVALID';
  end if;

  participant_ids := array[
    challenge_row.challenger_id::bigint,
    match_plan_a.t1_p2::bigint,
    challenge_row.defender_id::bigint,
    match_plan_a.t2_p2::bigint
  ];
  if (
       select count(distinct participant_id)
         from unnest(participant_ids) participant_id
     ) <> 4
     or (
       select count(distinct item.player_id)
         from jsonb_to_recordset(p_player_updates)
           as item(player_id bigint)
     ) <> 4
     or exists (
       select 1
         from jsonb_to_recordset(p_player_updates)
           as item(
             player_id bigint,
             rating_mode text,
             expected jsonb,
             after jsonb
           )
        where item.player_id <> all(participant_ids)
           or item.rating_mode is distinct from 'doubles'
           or jsonb_typeof(item.expected) is distinct from 'object'
           or jsonb_typeof(item.after) is distinct from 'object'
     )
     or exists (
       select 1
         from unnest(participant_ids) participant_id
        where not exists (
          select 1
            from jsonb_to_recordset(p_player_updates)
              as item(player_id bigint)
           where item.player_id = participant_id
        )
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_PLAYER_PLAN_INVALID';
  end if;

  if (
       select count(distinct (item.player_id, item.league_name))
         from jsonb_to_recordset(p_league_rating_updates)
           as item(player_id bigint, league_name text)
     ) <> 4
     or exists (
       select 1
         from jsonb_to_recordset(p_league_rating_updates)
           as item(
             player_id bigint,
             league_name text,
             expected jsonb,
             after jsonb
           )
        where item.player_id <> all(participant_ids)
           or item.league_name is distinct from 'OVERALL'
           or (
             item.expected is not null
             and item.expected <> 'null'::jsonb
             and jsonb_typeof(item.expected) is distinct from 'object'
           )
           or jsonb_typeof(item.after) is distinct from 'object'
     )
     or (
       select count(distinct item.league_name)
         from jsonb_to_recordset(p_league_metadata_expectations)
           as item(league_name text)
     ) <> 1
     or exists (
       select 1
         from jsonb_to_recordset(p_league_metadata_expectations)
           as item(league_name text, expected jsonb)
        where item.league_name is distinct from 'OVERALL'
           or (
             item.expected is not null
             and item.expected <> 'null'::jsonb
             and jsonb_typeof(item.expected) is distinct from 'object'
           )
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_LEAGUE_PLAN_INVALID';
  end if;

  perform player.id
    from public.players player
    join jsonb_to_recordset(p_player_updates)
      as item(player_id bigint)
      on item.player_id = player.id
   where player.club_id = p_club_id
   order by player.id
   for update;
  if (
       select count(*)
         from public.players player
        where player.club_id = p_club_id
          and player.id = any(participant_ids)
     ) <> 4 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_PLAYER_STALE';
  end if;

  for plan_item in
    select *
      from jsonb_to_recordset(p_player_updates)
        as item(
          player_id bigint,
          rating_mode text,
          expected jsonb,
          after jsonb
        )
  loop
    select player.*
      into player_row
      from public.players player
     where player.club_id = p_club_id
       and player.id = plan_item.player_id
     for update;
    expected_json := plan_item.expected;
    if not found
       or player_row.rating is distinct from (expected_json->>'rating')::double precision
       or player_row.wins is distinct from (expected_json->>'wins')::integer
       or player_row.losses is distinct from (expected_json->>'losses')::integer
       or player_row.matches_played is distinct from (expected_json->>'matches_played')::integer
       or player_row.last_game_at is distinct from nullif(expected_json->>'last_game_at', '')::timestamptz
       or player_row.inactive_at is distinct from nullif(expected_json->>'inactive_at', '')::timestamptz
       or player_row.active is distinct from (expected_json->>'active')::boolean then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_CHALLENGE_LADDER_ATOMIC_PLAYER_STALE';
    end if;
  end loop;

  for plan_item in
    select *
      from jsonb_to_recordset(p_league_metadata_expectations)
        as item(league_name text, expected jsonb)
  loop
    select metadata.*
      into metadata_row
      from public.leagues_metadata metadata
     where metadata.club_id = p_club_id
       and lower(metadata.league_name) = lower(plan_item.league_name)
     for update;
    expected_json := plan_item.expected;
    if expected_json is null or expected_json = 'null'::jsonb then
      if found then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_CHALLENGE_LADDER_ATOMIC_LEAGUE_METADATA_STALE';
      end if;
    elsif not found
       or metadata_row.id::text is distinct from expected_json->>'id'
       or metadata_row.club_id is distinct from expected_json->>'club_id'
       or metadata_row.league_name is distinct from expected_json->>'league_name'
       or metadata_row.k_factor is distinct from (expected_json->>'k_factor')::integer then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_CHALLENGE_LADDER_ATOMIC_LEAGUE_METADATA_STALE';
    end if;
  end loop;

  perform rating.id
    from public.league_ratings rating
    join jsonb_to_recordset(p_league_rating_updates)
      as item(player_id bigint, league_name text)
      on item.player_id = rating.player_id
     and item.league_name = rating.league_name
   where rating.club_id = p_club_id
   order by rating.id
   for update;
  for plan_item in
    select *
      from jsonb_to_recordset(p_league_rating_updates)
        as item(
          player_id bigint,
          league_name text,
          expected jsonb,
          after jsonb
        )
  loop
    select rating.*
      into league_rating_row
      from public.league_ratings rating
     where rating.club_id = p_club_id
       and rating.player_id = plan_item.player_id
       and rating.league_name = plan_item.league_name
     for update;
    expected_json := plan_item.expected;
    if expected_json is null or expected_json = 'null'::jsonb then
      if found then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_CHALLENGE_LADDER_ATOMIC_LEAGUE_RATING_STALE';
      end if;
    elsif not found
       or league_rating_row.id is distinct from (expected_json->>'id')::bigint
       or league_rating_row.rating is distinct from (expected_json->>'rating')::double precision
       or league_rating_row.wins is distinct from (expected_json->>'wins')::integer
       or league_rating_row.losses is distinct from (expected_json->>'losses')::integer
       or league_rating_row.matches_played is distinct from (expected_json->>'matches_played')::integer
       or league_rating_row.starting_rating is distinct from (expected_json->>'starting_rating')::double precision
       or league_rating_row.is_active is distinct from (expected_json->>'is_active')::boolean
       or league_rating_row.inactive_at is distinct from nullif(expected_json->>'inactive_at', '')::timestamptz then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_CHALLENGE_LADDER_ATOMIC_LEAGUE_RATING_STALE';
    end if;
  end loop;

  if exists (
    select 1
      from public.matches match
     where match.club_id = p_club_id
       and match.context_type = 'challenge_ladder'
       and match.context_id in (p_match_context_a, p_match_context_b)
  ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_MATCH_EXISTS';
  end if;

  insert into public.matches (
    club_id,
    date,
    league,
    t1_p1,
    t1_p2,
    t2_p1,
    t2_p2,
    score_t1,
    score_t2,
    elo_delta,
    match_type,
    week_tag,
    t1_p1_r,
    t1_p2_r,
    t2_p1_r,
    t2_p2_r,
    t1_p1_r_end,
    t1_p2_r_end,
    t2_p1_r_end,
    t2_p2_r_end,
    context_type,
    context_id,
    rating_scope,
    match_format,
    rating_bonus_elo,
    rating_bonus_reason
  )
  select
    p_club_id,
    item.date,
    item.league,
    item.t1_p1,
    item.t1_p2,
    item.t2_p1,
    item.t2_p2,
    item.score_t1,
    item.score_t2,
    item.elo_delta,
    item.match_type,
    item.week_tag,
    item.t1_p1_r,
    item.t1_p2_r,
    item.t2_p1_r,
    item.t2_p2_r,
    item.t1_p1_r_end,
    item.t1_p2_r_end,
    item.t2_p1_r_end,
    item.t2_p2_r_end,
    item.context_type,
    item.context_id,
    item.rating_scope,
    coalesce(nullif(item.match_format, ''), 'doubles'),
    coalesce(item.rating_bonus_elo, 0),
    item.rating_bonus_reason
  from jsonb_to_recordset(p_match_rows)
    as item(
      club_id text,
      date timestamptz,
      league text,
      t1_p1 bigint,
      t1_p2 bigint,
      t2_p1 bigint,
      t2_p2 bigint,
      score_t1 integer,
      score_t2 integer,
      elo_delta double precision,
      match_type text,
      week_tag text,
      t1_p1_r double precision,
      t1_p2_r double precision,
      t2_p1_r double precision,
      t2_p2_r double precision,
      t1_p1_r_end double precision,
      t1_p2_r_end double precision,
      t2_p1_r_end double precision,
      t2_p2_r_end double precision,
      context_type text,
      context_id text,
      rating_scope text,
      match_format text,
      rating_bonus_elo double precision,
      rating_bonus_reason text
    );
  get diagnostics inserted_count = row_count;
  if inserted_count <> 2 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_MATCH_INSERT_INCOMPLETE';
  end if;

  for plan_item in
    select *
      from jsonb_to_recordset(p_player_updates)
        as item(
          player_id bigint,
          rating_mode text,
          expected jsonb,
          after jsonb
        )
  loop
    after_json := plan_item.after;
    update public.players
       set rating = (after_json->>'rating')::double precision,
           wins = (after_json->>'wins')::integer,
           losses = (after_json->>'losses')::integer,
           matches_played = (after_json->>'matches_played')::integer,
           last_game_at = nullif(after_json->>'last_game_at', '')::timestamptz,
           inactive_at = nullif(after_json->>'inactive_at', '')::timestamptz,
           active = (after_json->>'active')::boolean
     where club_id = p_club_id
       and id = plan_item.player_id;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_CHALLENGE_LADDER_ATOMIC_PLAYER_WRITE_INCOMPLETE';
    end if;
  end loop;

  for plan_item in
    select *
      from jsonb_to_recordset(p_league_rating_updates)
        as item(
          player_id bigint,
          league_name text,
          expected jsonb,
          after jsonb
        )
  loop
    expected_json := plan_item.expected;
    after_json := plan_item.after;
    if expected_json is null or expected_json = 'null'::jsonb then
      insert into public.league_ratings (
        club_id,
        player_id,
        league_name,
        rating,
        wins,
        losses,
        matches_played,
        starting_rating,
        is_active,
        inactive_at
      ) values (
        p_club_id,
        plan_item.player_id,
        plan_item.league_name,
        (after_json->>'rating')::double precision,
        (after_json->>'wins')::integer,
        (after_json->>'losses')::integer,
        (after_json->>'matches_played')::integer,
        (after_json->>'starting_rating')::double precision,
        (after_json->>'is_active')::boolean,
        nullif(after_json->>'inactive_at', '')::timestamptz
      );
    else
      update public.league_ratings
         set rating = (after_json->>'rating')::double precision,
             wins = (after_json->>'wins')::integer,
             losses = (after_json->>'losses')::integer,
             matches_played = (after_json->>'matches_played')::integer,
             starting_rating = (after_json->>'starting_rating')::double precision,
             is_active = (after_json->>'is_active')::boolean,
             inactive_at = nullif(after_json->>'inactive_at', '')::timestamptz
       where club_id = p_club_id
         and id = (expected_json->>'id')::bigint;
    end if;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_CHALLENGE_LADDER_ATOMIC_LEAGUE_RATING_WRITE_INCOMPLETE';
    end if;
  end loop;

  select match.id
    into match_a_id
    from public.matches match
   where match.club_id = p_club_id
     and match.context_type = 'challenge_ladder'
     and match.context_id = p_match_context_a
     and match.deleted_at is null
   for share;
  select match.id
    into match_b_id
    from public.matches match
   where match.club_id = p_club_id
     and match.context_type = 'challenge_ladder'
     and match.context_id = p_match_context_b
     and match.deleted_at is null
   for share;
  if match_a_id is null
     or match_b_id is null
     or match_a_id = match_b_id then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_MATCH_INSERT_INCOMPLETE';
  end if;

  public_relation := jsonb_build_object(
    'version',
    1,
    'match_ids',
    jsonb_build_object('a', match_a_id, 'b', match_b_id)
  );
  receipt := public.admin_finalize_challenge_ladder_result_v1(
    p_club_id,
    p_challenge_id,
    p_operation_key,
    p_winner_id,
    p_completed_at,
    p_resolution_notes,
    public_relation,
    p_match_context_a,
    p_match_context_b
  );
  receipt := receipt || jsonb_build_object(
    'official_matches',
    jsonb_build_object(
      'inserted',
      2,
      'skipped',
      false,
      'atomic',
      true,
      'match_ids',
      jsonb_build_object('a', match_a_id, 'b', match_b_id),
      'match_context_ids',
      jsonb_build_array(p_match_context_a, p_match_context_b)
    ),
    'post_processors',
    jsonb_build_object('status', 'pending'),
    'side_effect_context',
    p_atomic_core->'side_effect_context',
    'plan_fingerprint',
    p_plan_fingerprint
  );

  update public.live_ladder_admin_operations
     set status = 'mutated',
         result_json = receipt,
         error_text = null,
         completed_at = p_completed_at,
         updated_at = timezone('utc', now())
   where operation_key = p_operation_key
     and club_id = p_club_id
     and status = 'mutated';
  get diagnostics write_count = row_count;
  if write_count <> 1 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_CHALLENGE_LADDER_ATOMIC_RECEIPT_WRITE_INCOMPLETE';
  end if;

  return receipt;
end;
$$;

revoke all on function public.admin_apply_challenge_ladder_result_atomic_v1(
  text,
  bigint,
  text,
  jsonb,
  text,
  bigint,
  timestamptz,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.admin_apply_challenge_ladder_result_atomic_v1(
  text,
  bigint,
  text,
  jsonb,
  text,
  bigint,
  timestamptz,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  text,
  text
) to service_role;

create or replace function public.admin_finalize_challenge_ladder_forfeit_v1(
  p_club_id text,
  p_challenge_id bigint,
  p_operation_key text,
  p_forfeited_by_id bigint,
  p_completed_at timestamptz,
  p_forfeit_reason text
)
returns jsonb
language plpgsql
security invoker
set search_path = pg_catalog
as $$
declare
  operation_row public.live_ladder_admin_operations%rowtype;
  challenge_row public.ladder_challenges%rowtype;
  challenger_roster public.ladder_roster%rowtype;
  defender_roster public.ladder_roster%rowtype;
  updated_challenge public.ladder_challenges%rowtype;
  rank_change jsonb;
  receipt jsonb;
  temp_rank public.ladder_roster.rank%type;
  active_ranked_count integer;
  write_count integer;
  v_winner_id bigint;
  swapped boolean;
begin
  select *
    into operation_row
    from public.live_ladder_admin_operations
   where operation_key = p_operation_key
     and club_id = p_club_id
   for update;
  if not found
     or operation_row.surface is distinct from 'challenge_ladder'
     or operation_row.operation_type is distinct from 'record_forfeit'
     or operation_row.entity_id is distinct from p_challenge_id::text
     or operation_row.status is distinct from 'running' then
    raise exception using errcode = '55000', message = 'running durable challenge forfeit operation not found';
  end if;
  if jsonb_typeof(operation_row.request_json) is distinct from 'object'
     or jsonb_typeof(operation_row.request_json->'forfeited_by_id')
        is distinct from 'number'
     or (operation_row.request_json->>'forfeited_by_id')::bigint
        is distinct from p_forfeited_by_id
     or p_forfeit_reason is distinct from coalesce(
          nullif(
            left(
              btrim(
                replace(
                  replace(
                    coalesce(operation_row.request_json->>'admin_note', ''),
                    '<',
                    ''
                  ),
                  '>',
                  ''
                )
              ),
              500
            ),
            ''
          ),
          'Forfeit'
        ) then
    raise exception using
      errcode = '55000',
      message = 'durable challenge forfeit request does not match finalization parameters';
  end if;

  select *
    into challenge_row
    from public.ladder_challenges
   where club_id = p_club_id
     and id = p_challenge_id
   for update;
  if not found then
    raise exception using errcode = 'P0002', message = 'challenge not found';
  end if;
  if challenge_row.status::text in (
       'CANCELLED',
       'CANCELED',
       'FORFEITED',
       'COMPLETED',
       'EXPIRED_ACCEPTANCE'
     )
     or challenge_row.completed_at is not null
     or challenge_row.winner_id is not null then
    raise exception using errcode = '55000', message = 'challenge cannot accept a forfeit';
  end if;
  if p_forfeited_by_id not in (
       challenge_row.challenger_id::bigint,
       challenge_row.defender_id::bigint
     ) then
    raise exception using errcode = '22023', message = 'forfeited player must be a ranked challenge participant';
  end if;
  v_winner_id := case
    when p_forfeited_by_id = challenge_row.challenger_id::bigint
      then challenge_row.defender_id::bigint
    else challenge_row.challenger_id::bigint
  end;

  perform 1
    from public.ladder_roster
   where club_id = p_club_id
     and tier_id = challenge_row.tier_id
   order by id
   for update;

  select count(*)
    into active_ranked_count
    from public.ladder_roster
   where club_id = p_club_id
     and tier_id = challenge_row.tier_id
     and player_id in (
       challenge_row.challenger_id,
       challenge_row.defender_id
     )
     and is_active is not false;
  if active_ranked_count <> 2 then
    raise exception using errcode = '55000', message = 'exact active ladder rows are required for both challenge participants';
  end if;

  select *
    into challenger_roster
    from public.ladder_roster
   where club_id = p_club_id
     and player_id = challenge_row.challenger_id
     and tier_id = challenge_row.tier_id
     and is_active is not false
   limit 1;
  select *
    into defender_roster
    from public.ladder_roster
   where club_id = p_club_id
     and player_id = challenge_row.defender_id
     and tier_id = challenge_row.tier_id
     and is_active is not false
   limit 1;
  if challenger_roster.tier_id is distinct from defender_roster.tier_id
     or challenger_roster.tier_id is distinct from challenge_row.tier_id
     or challenger_roster.rank is null
     or defender_roster.rank is null then
    raise exception using errcode = '55000', message = 'challenge participants must have ranked rows in the challenge tier';
  end if;

  swapped := v_winner_id = challenge_row.challenger_id::bigint;
  rank_change := jsonb_build_object(
    'swapped', swapped,
    'challenger', jsonb_build_object(
      'player_id', challenge_row.challenger_id,
      'before', challenger_roster.rank,
      'after', case when swapped then defender_roster.rank else challenger_roster.rank end
    ),
    'defender', jsonb_build_object(
      'player_id', challenge_row.defender_id,
      'before', defender_roster.rank,
      'after', case when swapped then challenger_roster.rank else defender_roster.rank end
    )
  );

  if swapped then
    select coalesce(max(rank), 0) + 1
      into temp_rank
      from public.ladder_roster
     where club_id = p_club_id
       and tier_id = challenge_row.tier_id;
    update public.ladder_roster
       set rank = temp_rank,
           updated_at = p_completed_at
     where id = challenger_roster.id;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using errcode = '55000', message = 'forfeit challenger temporary rank write was incomplete';
    end if;
    update public.ladder_roster
       set rank = challenger_roster.rank,
           updated_at = p_completed_at
     where id = defender_roster.id;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using errcode = '55000', message = 'forfeit defender rank write was incomplete';
    end if;
    update public.ladder_roster
       set rank = defender_roster.rank,
           updated_at = p_completed_at
     where id = challenger_roster.id;
    get diagnostics write_count = row_count;
    if write_count <> 1 then
      raise exception using errcode = '55000', message = 'forfeit challenger final rank write was incomplete';
    end if;
  end if;

  update public.ladder_challenges
     set status = 'FORFEITED',
         forfeit_by = p_forfeited_by_id,
         winner_id = v_winner_id,
         completed_at = p_completed_at,
         forfeit_reason = nullif(btrim(p_forfeit_reason), ''),
         public_result_json = null,
         updated_at = p_completed_at
   where club_id = p_club_id
     and id = p_challenge_id
   returning * into updated_challenge;
  get diagnostics write_count = row_count;
  if write_count <> 1 then
    raise exception using errcode = '55000', message = 'challenge forfeit write was incomplete';
  end if;

  receipt := jsonb_build_object(
    'ok', true,
    'core_committed', true,
    'mode', 'challenge_ladder_forfeit_core',
    'challenge', to_jsonb(updated_challenge),
    'rank_result', rank_change,
    'public_result_json', null,
    'post_processors', jsonb_build_object('status', 'complete')
  );

  update public.live_ladder_admin_operations
     set status = 'mutated',
         result_json = receipt,
         error_text = null,
         completed_at = p_completed_at,
         updated_at = timezone('utc', now())
   where operation_key = p_operation_key
     and club_id = p_club_id
     and status = 'running';
  get diagnostics write_count = row_count;
  if write_count <> 1 then
    raise exception using
      errcode = '55000',
      message = 'durable challenge forfeit receipt write was incomplete';
  end if;

  return receipt;
end;
$$;

revoke all on function public.admin_finalize_challenge_ladder_forfeit_v1(
  text, bigint, text, bigint, timestamptz, text
) from public, anon, authenticated;
grant execute on function public.admin_finalize_challenge_ladder_forfeit_v1(
  text, bigint, text, bigint, timestamptz, text
) to service_role;

notify pgrst, 'reload schema';
