-- Make ordinary admin score entry one fail-closed database transaction.
--
-- The application calculates deterministic rating projections, while this
-- SECURITY INVOKER RPC owns the durable compare-and-swap boundary. A match
-- row, player/league aggregates, the operation receipt, and its audit row
-- either all commit or all roll back.

create table if not exists public.admin_direct_match_entry_operations (
  id uuid primary key,
  club_id text not null,
  idempotency_key text not null,
  request_fingerprint text not null,
  match_format text not null,
  source text not null,
  actor_email text not null,
  actor_role text not null,
  request_json jsonb not null,
  result_json jsonb not null,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint admin_direct_match_entry_operations_idempotency_key_check
    check (
      pg_catalog.char_length(idempotency_key) between 8 and 160
      and idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]+$'
    ),
  constraint admin_direct_match_entry_operations_fingerprint_check
    check (request_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint admin_direct_match_entry_operations_format_check
    check (match_format in ('doubles', 'singles')),
  constraint admin_direct_match_entry_operations_request_check
    check (pg_catalog.jsonb_typeof(request_json) = 'object'),
  constraint admin_direct_match_entry_operations_result_check
    check (
      pg_catalog.jsonb_typeof(result_json) = 'object'
      and result_json @> '{"ok": true, "committed": true}'::jsonb
    ),
  unique (club_id, idempotency_key)
);

create index if not exists
  admin_direct_match_entry_operations_club_created_idx
  on public.admin_direct_match_entry_operations (club_id, created_at desc);

alter table public.admin_direct_match_entry_operations enable row level security;

revoke all on table public.admin_direct_match_entry_operations
  from public, anon, authenticated;
grant select, insert on table public.admin_direct_match_entry_operations
  to service_role;

comment on table public.admin_direct_match_entry_operations is
  'Completed atomic direct match-entry receipts. Exact retries return the original result.';

drop function if exists public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
);

create or replace function public.admin_apply_direct_match_entry_atomic_v1(
  p_operation_id uuid,
  p_club_id text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_match_format text,
  p_source text,
  p_actor_email text,
  p_actor_role text,
  p_request_json jsonb,
  p_result_summary jsonb,
  p_match_rows jsonb,
  p_player_updates jsonb,
  p_league_rating_updates jsonb,
  p_league_metadata_expectations jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_idempotency_key text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_request_fingerprint text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_match_format text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_match_format, '')));
  v_source text :=
    coalesce(
      nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
      'admin_direct_match_entry'
    );
  v_actor_email text :=
    coalesce(
      nullif(
        pg_catalog.left(
          pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
          320
        ),
        ''
      ),
      'unknown'
    );
  v_actor_role text :=
    coalesce(
      nullif(
        pg_catalog.left(
          pg_catalog.lower(pg_catalog.btrim(p_actor_role)),
          80
        ),
        ''
      ),
      'unknown'
    );
  v_request_json jsonb := coalesce(p_request_json, '{}'::jsonb);
  v_result_summary jsonb := coalesce(p_result_summary, '{}'::jsonb);
  v_match_rows jsonb := coalesce(p_match_rows, '[]'::jsonb);
  v_player_updates jsonb := coalesce(p_player_updates, '[]'::jsonb);
  v_league_rating_updates jsonb :=
    coalesce(p_league_rating_updates, '[]'::jsonb);
  v_league_metadata_expectations jsonb :=
    coalesce(p_league_metadata_expectations, '[]'::jsonb);
  v_operation public.admin_direct_match_entry_operations%rowtype;
  v_player public.players%rowtype;
  v_league_rating public.league_ratings%rowtype;
  v_metadata public.leagues_metadata%rowtype;
  v_item record;
  v_expected jsonb;
  v_after jsonb;
  v_inserted integer := 0;
  v_row_count integer := 0;
  v_match_ids jsonb := '[]'::jsonb;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_idempotency_key is null
     or pg_catalog.char_length(v_idempotency_key) not between 8 and 160
     or v_idempotency_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]+$'
     or v_request_fingerprint !~ '^[0-9a-f]{64}$'
     or v_match_format not in ('doubles', 'singles')
     or pg_catalog.jsonb_typeof(v_request_json) <> 'object'
     or pg_catalog.jsonb_typeof(v_result_summary) <> 'object'
     or pg_catalog.jsonb_typeof(v_match_rows) <> 'array'
     or pg_catalog.jsonb_array_length(v_match_rows) not between 1 and 200
     or pg_catalog.jsonb_typeof(v_player_updates) <> 'array'
     or pg_catalog.jsonb_typeof(v_league_rating_updates) <> 'array'
     or pg_catalog.jsonb_typeof(v_league_metadata_expectations) <> 'array'
     or coalesce((v_result_summary->>'inserted')::integer, -1)
          <> pg_catalog.jsonb_array_length(v_match_rows) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_PLAN_INVALID: exact operation, request, summary, and write-plan values are required.';
  end if;

  -- One short club-scoped transaction lock serializes direct rating plans and
  -- makes the idempotency-key receipt deterministic.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:direct-match-entry:' || v_club_id,
      0
    )
  );

  select operation.*
    into v_operation
    from public.admin_direct_match_entry_operations as operation
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_idempotency_key;

  if found then
    if v_operation.request_fingerprint is distinct from v_request_fingerprint
       or v_operation.match_format is distinct from v_match_format then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_IDEMPOTENCY_CONFLICT: this key already belongs to a different request.';
    end if;
    return v_operation.result_json || pg_catalog.jsonb_build_object(
      'idempotent', true,
      'duplicate_request', false
    );
  end if;

  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(v_match_rows) as match_row(
        club_id text,
        date timestamptz,
        league text,
        t1_p1 bigint,
        t1_p2 bigint,
        t2_p1 bigint,
        t2_p2 bigint,
        score_t1 integer,
        score_t2 integer,
        match_type text,
        context_type text,
        tournament_id text,
        tournament_game_id text,
        rating_scope text,
        match_format text,
        singles_replay_managed boolean
      )
     where match_row.club_id is distinct from v_club_id
        or match_row.date is null
        or nullif(pg_catalog.btrim(match_row.league), '') is null
        or match_row.t1_p1 is null
        or match_row.t2_p1 is null
        or match_row.t1_p1 = match_row.t2_p1
        or match_row.score_t1 is null
        or match_row.score_t2 is null
        or match_row.score_t1 < 0
        or match_row.score_t2 < 0
        or match_row.score_t1 + match_row.score_t2 <= 0
        or match_row.score_t1 = match_row.score_t2
        or pg_catalog.lower(
             pg_catalog.btrim(coalesce(match_row.match_format, 'doubles'))
           ) is distinct from v_match_format
        or coalesce(match_row.rating_scope, '') not in (
             '',
             'overall_only',
             'unrated'
           )
        or nullif(pg_catalog.btrim(match_row.tournament_id), '') is not null
        or nullif(
             pg_catalog.btrim(match_row.tournament_game_id),
             ''
           ) is not null
        or coalesce(pg_catalog.btrim(match_row.context_type), '') not in (
             '',
             'event'
           )
        or (
          v_match_format = 'doubles'
          and (
            match_row.t1_p2 is null
            or match_row.t2_p2 is null
            or pg_catalog.cardinality(
              array[
                match_row.t1_p1,
                match_row.t1_p2,
                match_row.t2_p1,
                match_row.t2_p2
              ]
            ) <> (
              select pg_catalog.count(distinct player_id)
              from pg_catalog.unnest(
                array[
                  match_row.t1_p1,
                  match_row.t1_p2,
                  match_row.t2_p1,
                  match_row.t2_p2
                ]
              ) as player(player_id)
            )
            or coalesce(match_row.singles_replay_managed, false)
          )
        )
        or (
          v_match_format = 'singles'
          and (
            match_row.t1_p2 is not null
            or match_row.t2_p2 is not null
            or match_row.singles_replay_managed is not true
          )
        )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_ROWS_INVALID: match rows do not satisfy the direct doubles/singles contract.';
  end if;

  if (
    select pg_catalog.count(distinct update_row.player_id)
      from pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
        player_id bigint
      )
  ) <> pg_catalog.jsonb_array_length(v_player_updates)
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
           player_id bigint,
           rating_mode text,
           expected jsonb,
           after jsonb
         )
        where update_row.player_id is null
           or update_row.player_id <= 0
           or update_row.rating_mode is distinct from v_match_format
           or pg_catalog.jsonb_typeof(update_row.expected) <> 'object'
           or pg_catalog.jsonb_typeof(update_row.after) <> 'object'
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_PLAYER_PLAN_INVALID: unique exact player projections are required.';
  end if;

  if exists (
    with rated_match_players as (
      select match_player.player_id
        from pg_catalog.jsonb_to_recordset(v_match_rows) as match_row(
          t1_p1 bigint,
          t1_p2 bigint,
          t2_p1 bigint,
          t2_p2 bigint,
          rating_scope text
        )
        cross join lateral pg_catalog.unnest(
          case
            when v_match_format = 'singles'
              then array[match_row.t1_p1, match_row.t2_p1]
            else array[
              match_row.t1_p1,
              match_row.t1_p2,
              match_row.t2_p1,
              match_row.t2_p2
            ]
          end
        ) as match_player(player_id)
       where coalesce(match_row.rating_scope, '') <> 'unrated'
    ),
    planned_players as (
      select update_row.player_id
        from pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
          player_id bigint
        )
    )
    (
      select player_id from rated_match_players
      except
      select player_id from planned_players
    )
    union all
    (
      select player_id from planned_players
      except
      select player_id from rated_match_players
    )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_PLAYER_PLAN_INVALID: rated match players and player projections differ.';
  end if;

  if v_match_format = 'singles'
     and (
       pg_catalog.jsonb_array_length(v_league_rating_updates) <> 0
       or pg_catalog.jsonb_array_length(v_league_metadata_expectations) <> 0
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_LEAGUE_PLAN_INVALID: singles may not mutate doubles league ratings.';
  end if;

  if (
    select pg_catalog.count(
      distinct (
        league_update.player_id,
        pg_catalog.lower(league_update.league_name)
      )
    )
      from pg_catalog.jsonb_to_recordset(
        v_league_rating_updates
      ) as league_update(
        player_id bigint,
        league_name text
      )
  ) <> pg_catalog.jsonb_array_length(v_league_rating_updates)
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(
           v_league_rating_updates
         ) as league_update(
           player_id bigint,
           league_name text,
           expected jsonb,
           after jsonb
         )
        where league_update.player_id is null
           or league_update.player_id <= 0
           or nullif(pg_catalog.btrim(league_update.league_name), '') is null
           or (
             league_update.expected is not null
             and pg_catalog.jsonb_typeof(league_update.expected) <> 'object'
           )
           or pg_catalog.jsonb_typeof(league_update.after) <> 'object'
     )
     or (
       select pg_catalog.count(
         distinct pg_catalog.lower(metadata_expectation.league_name)
       )
         from pg_catalog.jsonb_to_recordset(
           v_league_metadata_expectations
         ) as metadata_expectation(league_name text)
     ) <> pg_catalog.jsonb_array_length(v_league_metadata_expectations)
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(
           v_league_metadata_expectations
         ) as metadata_expectation(
           league_name text,
           expected jsonb
         )
        where nullif(
                pg_catalog.btrim(metadata_expectation.league_name),
                ''
              ) is null
           or (
             metadata_expectation.expected is not null
             and pg_catalog.jsonb_typeof(metadata_expectation.expected)
                   <> 'object'
           )
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_LEAGUE_PLAN_INVALID: unique exact league projections are required.';
  end if;

  -- Lock affected rows in one deterministic order before comparing snapshots.
  perform player.id
    from public.players as player
    join pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
      player_id bigint
    ) on update_row.player_id = player.id
   where player.club_id = v_club_id
   order by player.id
   for update;

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
        player_id bigint,
        rating_mode text,
        expected jsonb,
        after jsonb
      )
     order by update_row.player_id
  loop
    select player.*
      into v_player
      from public.players as player
     where player.club_id = v_club_id
       and player.id = v_item.player_id
     for update;

    if not found then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_PLAYER_STALE: a planned player no longer exists.';
    end if;

    v_expected := v_item.expected;
    if v_match_format = 'doubles' then
      if v_player.rating is distinct from
           (v_expected->>'rating')::numeric(10,4)
         or v_player.wins is distinct from
           (v_expected->>'wins')::integer
         or v_player.losses is distinct from
           (v_expected->>'losses')::integer
         or v_player.matches_played is distinct from
           (v_expected->>'matches_played')::integer
         or v_player.last_game_at is distinct from
           nullif(v_expected->>'last_game_at', '')::timestamptz
         or v_player.inactive_at is distinct from
           nullif(v_expected->>'inactive_at', '')::timestamptz
         or v_player.active is distinct from
           (v_expected->>'active')::boolean then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_DIRECT_MATCH_PLAYER_STALE: doubles player state changed before commit.';
      end if;
    else
      if v_player.singles_rating is distinct from
           (v_expected->>'singles_rating')::double precision
         or v_player.singles_wins is distinct from
           (v_expected->>'singles_wins')::integer
         or v_player.singles_losses is distinct from
           (v_expected->>'singles_losses')::integer
         or v_player.singles_matches_played is distinct from
           (v_expected->>'singles_matches_played')::integer
         or v_player.singles_last_game_at is distinct from
           nullif(v_expected->>'singles_last_game_at', '')::timestamptz
         or v_player.last_game_at is distinct from
           nullif(v_expected->>'last_game_at', '')::timestamptz
         or v_player.inactive_at is distinct from
           nullif(v_expected->>'inactive_at', '')::timestamptz
         or v_player.active is distinct from
           (v_expected->>'active')::boolean then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_DIRECT_MATCH_PLAYER_STALE: singles player state changed before commit.';
      end if;
    end if;
  end loop;

  perform metadata.id
    from public.leagues_metadata as metadata
    join pg_catalog.jsonb_to_recordset(
      v_league_metadata_expectations
    ) as metadata_expectation(league_name text)
      on pg_catalog.lower(metadata_expectation.league_name)
       = pg_catalog.lower(metadata.league_name)
   where metadata.club_id = v_club_id
   order by metadata.id
   for update;

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(
        v_league_metadata_expectations
      ) as metadata_expectation(
        league_name text,
        expected jsonb
      )
     order by pg_catalog.lower(metadata_expectation.league_name)
  loop
    select metadata.*
      into v_metadata
      from public.leagues_metadata as metadata
     where metadata.club_id = v_club_id
       and pg_catalog.lower(metadata.league_name)
           = pg_catalog.lower(v_item.league_name)
     for update;

    if v_item.expected is null then
      if found then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: league metadata appeared before commit.';
      end if;
    elsif not found
       or v_metadata.id::text is distinct from v_item.expected->>'id'
       or v_metadata.club_id is distinct from v_item.expected->>'club_id'
       or v_metadata.league_name is distinct from
            v_item.expected->>'league_name'
       or v_metadata.k_factor is distinct from
            (v_item.expected->>'k_factor')::integer then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: league metadata changed before commit.';
    end if;
  end loop;

  perform league_rating.id
    from public.league_ratings as league_rating
    join pg_catalog.jsonb_to_recordset(
      v_league_rating_updates
    ) as league_update(
      player_id bigint,
      league_name text
    ) on league_update.player_id = league_rating.player_id
      and league_update.league_name = league_rating.league_name
   where league_rating.club_id = v_club_id
   order by league_rating.id
   for update;

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(
        v_league_rating_updates
      ) as league_update(
        player_id bigint,
        league_name text,
        expected jsonb,
        after jsonb
      )
     order by league_update.player_id, league_update.league_name
  loop
    select league_rating.*
      into v_league_rating
      from public.league_ratings as league_rating
     where league_rating.club_id = v_club_id
       and league_rating.player_id = v_item.player_id
       and league_rating.league_name = v_item.league_name
     for update;

    if v_item.expected is null then
      if found then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_DIRECT_MATCH_LEAGUE_RATING_STALE: a league rating appeared before commit.';
      end if;
    elsif not found
       or v_league_rating.id is distinct from
            (v_item.expected->>'id')::bigint
       or v_league_rating.rating is distinct from
            (v_item.expected->>'rating')::numeric(10,4)
       or v_league_rating.wins is distinct from
            (v_item.expected->>'wins')::integer
       or v_league_rating.losses is distinct from
            (v_item.expected->>'losses')::integer
       or v_league_rating.matches_played is distinct from
            (v_item.expected->>'matches_played')::integer
       or v_league_rating.starting_rating is distinct from
            (v_item.expected->>'starting_rating')::numeric(10,4)
       or v_league_rating.is_active is distinct from
            (v_item.expected->>'is_active')::boolean
       or v_league_rating.inactive_at is distinct from
            nullif(v_item.expected->>'inactive_at', '')::timestamptz then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_RATING_STALE: a league rating changed before commit.';
    end if;
  end loop;

  with inserted as (
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
      tournament_id,
      tournament_game_id,
      rating_scope,
      match_format,
      rating_bonus_elo,
      rating_bonus_reason,
      singles_replay_managed
    )
    select
      v_club_id,
      match_row.date,
      match_row.league,
      match_row.t1_p1,
      match_row.t1_p2,
      match_row.t2_p1,
      match_row.t2_p2,
      match_row.score_t1,
      match_row.score_t2,
      match_row.elo_delta,
      match_row.match_type,
      match_row.week_tag,
      match_row.t1_p1_r,
      match_row.t1_p2_r,
      match_row.t2_p1_r,
      match_row.t2_p2_r,
      match_row.t1_p1_r_end,
      match_row.t1_p2_r_end,
      match_row.t2_p1_r_end,
      match_row.t2_p2_r_end,
      nullif(match_row.context_type, ''),
      nullif(match_row.context_id, ''),
      nullif(match_row.tournament_id, '')::uuid,
      nullif(match_row.tournament_game_id, '')::uuid,
      coalesce(match_row.rating_scope, ''),
      v_match_format,
      coalesce(match_row.rating_bonus_elo, 0),
      nullif(match_row.rating_bonus_reason, ''),
      v_match_format = 'singles'
    from pg_catalog.jsonb_to_recordset(v_match_rows) as match_row(
      club_id text,
      date timestamptz,
      league text,
      t1_p1 bigint,
      t1_p2 bigint,
      t2_p1 bigint,
      t2_p2 bigint,
      score_t1 integer,
      score_t2 integer,
      elo_delta numeric(10,4),
      match_type text,
      week_tag text,
      t1_p1_r numeric(10,4),
      t1_p2_r numeric(10,4),
      t2_p1_r numeric(10,4),
      t2_p2_r numeric(10,4),
      t1_p1_r_end numeric(10,4),
      t1_p2_r_end numeric(10,4),
      t2_p1_r_end numeric(10,4),
      t2_p2_r_end numeric(10,4),
      context_type text,
      context_id text,
      tournament_id text,
      tournament_game_id text,
      rating_scope text,
      match_format text,
      rating_bonus_elo numeric(10,4),
      rating_bonus_reason text,
      singles_replay_managed boolean
    )
    returning id
  )
  select
    pg_catalog.count(*)::integer,
    coalesce(
      pg_catalog.jsonb_agg(inserted.id::text order by inserted.id),
      '[]'::jsonb
    )
    into v_inserted, v_match_ids
    from inserted;

  if v_inserted <> pg_catalog.jsonb_array_length(v_match_rows) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_DIRECT_MATCH_INSERT_INCOMPLETE: not every planned match row was inserted.';
  end if;

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
        player_id bigint,
        rating_mode text,
        expected jsonb,
        after jsonb
      )
     order by update_row.player_id
  loop
    v_after := v_item.after;
    if v_match_format = 'doubles' then
      update public.players
         set rating = (v_after->>'rating')::numeric(10,4),
             wins = (v_after->>'wins')::integer,
             losses = (v_after->>'losses')::integer,
             matches_played = (v_after->>'matches_played')::integer,
             last_game_at =
               nullif(v_after->>'last_game_at', '')::timestamptz,
             inactive_at =
               nullif(v_after->>'inactive_at', '')::timestamptz,
             active = (v_after->>'active')::boolean
       where club_id = v_club_id
         and id = v_item.player_id;
    else
      update public.players
         set singles_rating =
               (v_after->>'singles_rating')::double precision,
             singles_wins = (v_after->>'singles_wins')::integer,
             singles_losses = (v_after->>'singles_losses')::integer,
             singles_matches_played =
               (v_after->>'singles_matches_played')::integer,
             singles_last_game_at =
               nullif(
                 v_after->>'singles_last_game_at',
                 ''
               )::timestamptz,
             last_game_at =
               nullif(v_after->>'last_game_at', '')::timestamptz,
             inactive_at =
               nullif(v_after->>'inactive_at', '')::timestamptz,
             active = (v_after->>'active')::boolean
       where club_id = v_club_id
         and id = v_item.player_id;
    end if;

    get diagnostics v_row_count = row_count;
    if v_row_count <> 1 then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_PLAYER_WRITE_INCOMPLETE: one exact player row was not updated.';
    end if;
  end loop;

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(
        v_league_rating_updates
      ) as league_update(
        player_id bigint,
        league_name text,
        expected jsonb,
        after jsonb
      )
     order by league_update.player_id, league_update.league_name
  loop
    v_after := v_item.after;
    if v_item.expected is null then
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
        v_club_id,
        v_item.player_id,
        v_item.league_name,
        (v_after->>'rating')::numeric(10,4),
        (v_after->>'wins')::integer,
        (v_after->>'losses')::integer,
        (v_after->>'matches_played')::integer,
        (v_after->>'starting_rating')::numeric(10,4),
        (v_after->>'is_active')::boolean,
        nullif(v_after->>'inactive_at', '')::timestamptz
      );
    else
      update public.league_ratings
         set rating = (v_after->>'rating')::numeric(10,4),
             wins = (v_after->>'wins')::integer,
             losses = (v_after->>'losses')::integer,
             matches_played =
               (v_after->>'matches_played')::integer,
             starting_rating =
               (v_after->>'starting_rating')::numeric(10,4),
             is_active = (v_after->>'is_active')::boolean,
             inactive_at =
               nullif(v_after->>'inactive_at', '')::timestamptz
       where club_id = v_club_id
         and id = (v_item.expected->>'id')::bigint;
    end if;

    get diagnostics v_row_count = row_count;
    if v_row_count <> 1 then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_RATING_WRITE_INCOMPLETE: one exact league-rating row was not written.';
    end if;
  end loop;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'idempotent', false,
    'duplicate_request', false,
    'operation_id', p_operation_id,
    'club_id', v_club_id,
    'idempotency_key', v_idempotency_key,
    'request_fingerprint', v_request_fingerprint,
    'match_format', v_match_format,
    'inserted', v_inserted,
    'match_ids', v_match_ids,
    'player_update_count', pg_catalog.jsonb_array_length(v_player_updates),
    'league_rating_update_count',
      pg_catalog.jsonb_array_length(v_league_rating_updates),
    'result_summary', v_result_summary,
    'player_updates', v_player_updates
  );

  insert into public.admin_direct_match_entry_operations (
    id,
    club_id,
    idempotency_key,
    request_fingerprint,
    match_format,
    source,
    actor_email,
    actor_role,
    request_json,
    result_json
  ) values (
    p_operation_id,
    v_club_id,
    v_idempotency_key,
    v_request_fingerprint,
    v_match_format,
    v_source,
    v_actor_email,
    v_actor_role,
    v_request_json,
    v_result
  );

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_club_id,
    v_actor_email,
    v_actor_role,
    case
      when v_match_format = 'singles'
        then 'submit_singles_match_uploader_atomic'
      else 'submit_direct_match_batch_atomic'
    end,
    'direct_match_entry_operation',
    p_operation_id::text,
    null,
    pg_catalog.jsonb_build_object(
      'operation',
      v_result - 'player_updates',
      'request',
      v_request_json
    ),
    null,
    v_source,
    v_match_format = 'singles'
  );

  return v_result;
exception
  when unique_violation then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_DIRECT_MATCH_CONCURRENT_CONFLICT: another write changed a unique dependency; no part of this plan committed.';
end
$function$;

revoke all on function public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) from public, anon, authenticated;

grant execute on function public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) to service_role;

comment on function public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) is
  'Atomically applies one idempotent direct doubles/singles match rating plan with CAS guards and audit evidence.';

notify pgrst, 'reload schema';
