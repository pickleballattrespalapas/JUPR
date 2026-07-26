-- Replay projection payloads originate as JSON numbers, while the durable
-- rating and match snapshot columns store numeric(10,4). Decode those fields
-- at the durable column precision before both mutation and exact verification.
-- The terminal verification failure is an application invariant, not a
-- serialization failure, so it must not trigger PostgREST transaction retries.

create or replace function public.apply_replay_write_batch_atomic(
  p_job_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_target_reset text,
  p_write_kind text,
  p_rows jsonb default '[]'::jsonb,
  p_delete_all boolean default false,
  p_league_names text[] default '{}'::text[]
)
returns integer
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_target_reset text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_target_reset), 160), '');
  v_write_kind text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_write_kind, '')));
  v_rows jsonb := coalesce(p_rows, '[]'::jsonb);
  v_league_names text[] := coalesce(p_league_names, '{}'::text[]);
  v_expected integer;
  v_verified integer;
  v_changed integer;
begin
  if v_club_id is null
     or v_target_reset is null
     or v_write_kind not in (
       'players_stats',
       'player_singles_stats',
       'delete_league_ratings',
       'insert_league_ratings',
       'match_snapshots'
     )
     or pg_catalog.jsonb_typeof(v_rows) <> 'array' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_WRITE_BATCH_INVALID: exact club, target, write kind, and rows array are required.';
  end if;

  v_expected := pg_catalog.jsonb_array_length(v_rows);

  if v_write_kind = 'delete_league_ratings' then
    if v_expected <> 0
       or (
         coalesce(p_delete_all, false)
         and v_target_reset <> 'ALL (Full System Reset)'
       )
       or (
         not coalesce(p_delete_all, false)
         and (
           v_target_reset = 'ALL (Full System Reset)'
           or pg_catalog.cardinality(v_league_names) not between 1 and 2
           or not (v_target_reset = any(v_league_names))
           or exists (
             select 1
             from pg_catalog.unnest(v_league_names)
               as league_name(value)
             where nullif(pg_catalog.btrim(league_name.value), '') is null
           )
         )
       ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_RATING_DELETE_INVALID: full reset must delete all ratings; league reset must name its one exact scope.';
    end if;
  elsif v_expected not between 1 and 500
        or coalesce(p_delete_all, false)
        or pg_catalog.cardinality(v_league_names) <> 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_WRITE_ROWS_INVALID: mutation batches require 1 to 500 rows and no delete-only arguments.';
  end if;

  perform public.assert_replay_write_fence_atomic(
    p_job_id,
    v_club_id,
    p_lease_token,
    p_worker_id,
    v_target_reset
  );

  if v_write_kind = 'players_stats' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text,
        rating numeric(10,4),
        wins integer,
        losses integer,
        matches_played integer,
        last_game_at timestamptz
      )
      where data.id is null
         or data.id <= 0
         or data.club_id is distinct from v_club_id
         or data.rating is null
         or data.wins is null
         or data.losses is null
         or data.matches_played is null
         or data.wins < 0
         or data.losses < 0
         or data.matches_played < 0
    ) or exists (
      select 1
      from pg_catalog.jsonb_array_elements(v_rows) as row_item(item)
      where not (row_item.item ? 'last_game_at')
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      group by data.id
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_PLAYER_BATCH_INVALID: exact unique club player projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        id bigint,
        club_id text,
        rating numeric(10,4),
        wins integer,
        losses integer,
        matches_played integer,
        last_game_at timestamptz
      )
    )
    update public.players as player
    set
      rating = data.rating,
      wins = data.wins,
      losses = data.losses,
      matches_played = data.matches_played,
      last_game_at = data.last_game_at
    from data
    where player.club_id::text = v_club_id
      and player.id = data.id
      and row(
        player.rating,
        player.wins,
        player.losses,
        player.matches_played,
        player.last_game_at
      ) is distinct from row(
        data.rating,
        data.wins,
        data.losses,
        data.matches_played,
        data.last_game_at
      );

    select pg_catalog.count(*)
    into v_verified
    from public.players as player
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      id bigint,
      club_id text,
      rating numeric(10,4),
      wins integer,
      losses integer,
      matches_played integer,
      last_game_at timestamptz
    ) on data.id = player.id
      and data.club_id = player.club_id::text
    where player.club_id::text = v_club_id
      and row(
        player.rating,
        player.wins,
        player.losses,
        player.matches_played,
        player.last_game_at
      ) is not distinct from row(
        data.rating,
        data.wins,
        data.losses,
        data.matches_played,
        data.last_game_at
      );

  elsif v_write_kind = 'player_singles_stats' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text,
        singles_rating double precision,
        singles_wins integer,
        singles_losses integer,
        singles_matches_played integer,
        singles_last_game_at timestamptz
      )
      where data.id is null
         or data.id <= 0
         or data.club_id is distinct from v_club_id
         or data.singles_rating is null
         or data.singles_wins is null
         or data.singles_losses is null
         or data.singles_matches_played is null
         or data.singles_wins < 0
         or data.singles_losses < 0
         or data.singles_matches_played < 0
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      group by data.id
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_SINGLES_PLAYER_BATCH_INVALID: exact unique club singles projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        id bigint,
        club_id text,
        singles_rating double precision,
        singles_wins integer,
        singles_losses integer,
        singles_matches_played integer,
        singles_last_game_at timestamptz
      )
    )
    update public.players as player
    set
      singles_rating = data.singles_rating,
      singles_wins = data.singles_wins,
      singles_losses = data.singles_losses,
      singles_matches_played = data.singles_matches_played,
      singles_last_game_at = data.singles_last_game_at
    from data
    where player.club_id::text = v_club_id
      and player.id = data.id
      and row(
        player.singles_rating,
        player.singles_wins,
        player.singles_losses,
        player.singles_matches_played,
        player.singles_last_game_at
      ) is distinct from row(
        data.singles_rating,
        data.singles_wins,
        data.singles_losses,
        data.singles_matches_played,
        data.singles_last_game_at
      );

    select pg_catalog.count(*)
    into v_verified
    from public.players as player
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      id bigint,
      club_id text,
      singles_rating double precision,
      singles_wins integer,
      singles_losses integer,
      singles_matches_played integer,
      singles_last_game_at timestamptz
    ) on data.id = player.id
      and data.club_id = player.club_id::text
    where player.club_id::text = v_club_id
      and row(
        player.singles_rating,
        player.singles_wins,
        player.singles_losses,
        player.singles_matches_played,
        player.singles_last_game_at
      ) is not distinct from row(
        data.singles_rating,
        data.singles_wins,
        data.singles_losses,
        data.singles_matches_played,
        data.singles_last_game_at
      );

  elsif v_write_kind = 'delete_league_ratings' then
    if coalesce(p_delete_all, false) then
      delete from public.league_ratings as league_rating
      where league_rating.club_id::text = v_club_id;
    else
      delete from public.league_ratings as league_rating
      where league_rating.club_id::text = v_club_id
        and league_rating.league_name = any(v_league_names);
    end if;
    get diagnostics v_changed = row_count;
    return v_changed;

  elsif v_write_kind = 'insert_league_ratings' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        club_id text,
        player_id bigint,
        league_name text,
        rating numeric(10,4),
        wins integer,
        losses integer,
        matches_played integer,
        starting_rating numeric(10,4)
      )
      where data.club_id is distinct from v_club_id
         or data.player_id is null
         or data.player_id <= 0
         or nullif(pg_catalog.btrim(data.league_name), '') is null
         or data.rating is null
         or data.starting_rating is null
         or data.wins is null
         or data.losses is null
         or data.matches_played is null
         or data.wins < 0
         or data.losses < 0
         or data.matches_played < 0
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        club_id text,
        player_id bigint,
        league_name text
      )
      group by data.club_id, data.player_id, data.league_name
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_LEAGUE_RATING_BATCH_INVALID: exact unique club/player/league projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        club_id text,
        player_id bigint,
        league_name text,
        rating numeric(10,4),
        wins integer,
        losses integer,
        matches_played integer,
        starting_rating numeric(10,4)
      )
    )
    update public.league_ratings as league_rating
    set
      rating = data.rating,
      wins = data.wins,
      losses = data.losses,
      matches_played = data.matches_played,
      starting_rating = data.starting_rating,
      is_active = true,
      inactive_at = null
    from data
    where league_rating.club_id::text = data.club_id
      and league_rating.player_id = data.player_id
      and league_rating.league_name = data.league_name
      and row(
        league_rating.rating,
        league_rating.wins,
        league_rating.losses,
        league_rating.matches_played,
        league_rating.starting_rating,
        league_rating.is_active,
        league_rating.inactive_at
      ) is distinct from row(
        data.rating,
        data.wins,
        data.losses,
        data.matches_played,
        data.starting_rating,
        true,
        null::timestamptz
      );

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
    )
    select
      data.club_id,
      data.player_id,
      data.league_name,
      data.rating,
      data.wins,
      data.losses,
      data.matches_played,
      data.starting_rating,
      true,
      null
    from pg_catalog.jsonb_to_recordset(v_rows) as data(
      club_id text,
      player_id bigint,
      league_name text,
      rating numeric(10,4),
      wins integer,
      losses integer,
      matches_played integer,
      starting_rating numeric(10,4)
    )
    where not exists (
      select 1
      from public.league_ratings as existing
      where existing.club_id::text = data.club_id
        and existing.player_id = data.player_id
        and existing.league_name = data.league_name
    )
    on conflict (club_id, player_id, league_name) do update
    set
      rating = excluded.rating,
      wins = excluded.wins,
      losses = excluded.losses,
      matches_played = excluded.matches_played,
      starting_rating = excluded.starting_rating,
      is_active = true,
      inactive_at = null
    where row(
      league_ratings.rating,
      league_ratings.wins,
      league_ratings.losses,
      league_ratings.matches_played,
      league_ratings.starting_rating,
      league_ratings.is_active,
      league_ratings.inactive_at
    ) is distinct from row(
      excluded.rating,
      excluded.wins,
      excluded.losses,
      excluded.matches_played,
      excluded.starting_rating,
      true,
      null::timestamptz
    );

    select pg_catalog.count(*)
    into v_verified
    from public.league_ratings as league_rating
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      club_id text,
      player_id bigint,
      league_name text,
      rating numeric(10,4),
      wins integer,
      losses integer,
      matches_played integer,
      starting_rating numeric(10,4)
    ) on data.club_id = league_rating.club_id::text
      and data.player_id = league_rating.player_id
      and data.league_name = league_rating.league_name
    where row(
      league_rating.rating,
      league_rating.wins,
      league_rating.losses,
      league_rating.matches_played,
      league_rating.starting_rating,
      league_rating.is_active,
      league_rating.inactive_at
    ) is not distinct from row(
      data.rating,
      data.wins,
      data.losses,
      data.matches_played,
      data.starting_rating,
      true,
      null::timestamptz
    );

  elsif v_write_kind = 'match_snapshots' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      where data.id is null
         or data.id <= 0
         or data.club_id is distinct from v_club_id
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      group by data.id
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_MATCH_SNAPSHOT_BATCH_INVALID: exact unique club match projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        id bigint,
        club_id text,
        elo_delta numeric(10,4),
        t1_p1_r numeric(10,4),
        t1_p2_r numeric(10,4),
        t2_p1_r numeric(10,4),
        t2_p2_r numeric(10,4),
        t1_p1_r_end numeric(10,4),
        t1_p2_r_end numeric(10,4),
        t2_p1_r_end numeric(10,4),
        t2_p2_r_end numeric(10,4)
      )
    )
    update public.matches as match_row
    set
      elo_delta = data.elo_delta,
      t1_p1_r = data.t1_p1_r,
      t1_p2_r = data.t1_p2_r,
      t2_p1_r = data.t2_p1_r,
      t2_p2_r = data.t2_p2_r,
      t1_p1_r_end = data.t1_p1_r_end,
      t1_p2_r_end = data.t1_p2_r_end,
      t2_p1_r_end = data.t2_p1_r_end,
      t2_p2_r_end = data.t2_p2_r_end
    from data
    where match_row.club_id::text = v_club_id
      and match_row.id = data.id
      and row(
        match_row.elo_delta,
        match_row.t1_p1_r,
        match_row.t1_p2_r,
        match_row.t2_p1_r,
        match_row.t2_p2_r,
        match_row.t1_p1_r_end,
        match_row.t1_p2_r_end,
        match_row.t2_p1_r_end,
        match_row.t2_p2_r_end
      ) is distinct from row(
        data.elo_delta,
        data.t1_p1_r,
        data.t1_p2_r,
        data.t2_p1_r,
        data.t2_p2_r,
        data.t1_p1_r_end,
        data.t1_p2_r_end,
        data.t2_p1_r_end,
        data.t2_p2_r_end
      );

    select pg_catalog.count(*)
    into v_verified
    from public.matches as match_row
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      id bigint,
      club_id text,
      elo_delta numeric(10,4),
      t1_p1_r numeric(10,4),
      t1_p2_r numeric(10,4),
      t2_p1_r numeric(10,4),
      t2_p2_r numeric(10,4),
      t1_p1_r_end numeric(10,4),
      t1_p2_r_end numeric(10,4),
      t2_p1_r_end numeric(10,4),
      t2_p2_r_end numeric(10,4)
    ) on data.id = match_row.id
      and data.club_id = match_row.club_id::text
    where row(
      match_row.elo_delta,
      match_row.t1_p1_r,
      match_row.t1_p2_r,
      match_row.t2_p1_r,
      match_row.t2_p2_r,
      match_row.t1_p1_r_end,
      match_row.t1_p2_r_end,
      match_row.t2_p1_r_end,
      match_row.t2_p2_r_end
    ) is not distinct from row(
      data.elo_delta,
      data.t1_p1_r,
      data.t1_p2_r,
      data.t2_p1_r,
      data.t2_p2_r,
      data.t1_p1_r_end,
      data.t1_p2_r_end,
      data.t2_p1_r_end,
      data.t2_p2_r_end
    );
  end if;

  if v_verified is distinct from v_expected then
    raise exception using
      errcode = 'P0001',
      message = pg_catalog.format(
        'JUPR_REPLAY_WRITE_BATCH_INCOMPLETE: %s verified %s of %s exact rows.',
        v_write_kind,
        coalesce(v_verified, 0),
        v_expected
      );
  end if;

  return v_expected;
end
$function$;

revoke all on function public.apply_replay_write_batch_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  text,
  jsonb,
  boolean,
  text[]
) from public, anon, authenticated;

grant execute on function public.apply_replay_write_batch_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  text,
  jsonb,
  boolean,
  text[]
) to service_role;

comment on function public.apply_replay_write_batch_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  text,
  jsonb,
  boolean,
  text[]
) is
  'Service-only idempotent replay projection batch fenced by the exact active job lease, normalized to durable numeric precision, in the same transaction.';

notify pgrst, 'reload schema';
