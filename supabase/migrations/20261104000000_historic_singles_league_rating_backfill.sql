-- Preserve league-roster lifecycle during Replay History and repair the two
-- deterministic Acceptance Singles rows omitted before official singles
-- league_ratings were written atomically.

do $migration_preflight$
declare
  v_missing text[];
begin
  select pg_catalog.array_agg(
           required.table_name || '.' || required.column_name
           order by required.table_name, required.column_name
         )
    into v_missing
    from (
      values
        ('admin_activity_log', 'after_json'),
        ('admin_league_roster_batch_operations', 'idempotency_key'),
        ('league_ratings', 'inactive_at'),
        ('league_ratings', 'is_active'),
        ('league_ratings', 'starting_rating'),
        ('leagues_metadata', 'ended_at'),
        ('leagues_metadata', 'is_active'),
        ('leagues_metadata', 'match_format'),
        ('leagues_metadata', 'status'),
        ('matches', 'singles_replay_managed'),
        ('players', 'singles_rating'),
        ('replay_jobs', 'status')
    ) as required(table_name, column_name)
   where not exists (
     select 1
       from information_schema.columns as actual
      where actual.table_schema = 'public'
        and actual.table_name = required.table_name
        and actual.column_name = required.column_name
   );

  if v_missing is not null
     or pg_catalog.to_regprocedure(
          'public.assert_replay_write_fence_atomic(uuid,text,uuid,text,text)'
        ) is null
     or pg_catalog.to_regprocedure(
          'public.admin_apply_league_roster_batch_atomic_v1(uuid,text,text,text,text,text,jsonb,numeric,text,text,text)'
        ) is null then
    raise exception using
      errcode = '42703',
      message =
        'historic singles league-rating backfill requires the replay fence, roster v1, and columns: '
        || coalesce(pg_catalog.array_to_string(v_missing, ', '), '(none)');
  end if;
end
$migration_preflight$;

-- Unlike the legacy replay batch writer, this exact-row writer carries
-- is_active/inactive_at through the lease-fenced transaction so replayed and
-- stale-stat-reset rows both preserve an existing roster lifecycle exactly.
create or replace function public.apply_replay_league_rating_rows_atomic(
  p_job_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_target_reset text,
  p_rows jsonb default '[]'::jsonb
)
returns integer
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_rows jsonb := coalesce(p_rows, '[]'::jsonb);
  v_expected integer;
  v_verified integer;
begin
  if v_club_id is null
     or pg_catalog.jsonb_typeof(v_rows) <> 'array' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_LEAGUE_RATING_ROWS_INVALID: exact club and rows array are required.';
  end if;

  v_expected := pg_catalog.jsonb_array_length(v_rows);
  if v_expected not between 1 and 500
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(v_rows) as data(
           club_id text,
           player_id bigint,
           league_name text,
           rating numeric,
           wins integer,
           losses integer,
           matches_played integer,
           starting_rating numeric,
           is_active boolean,
           inactive_at timestamptz
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
           or data.is_active is null
           or data.wins < 0
           or data.losses < 0
           or data.matches_played < 0
     )
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(v_rows) as data(
           player_id bigint,
           league_name text
         )
        group by
          data.player_id,
          pg_catalog.lower(pg_catalog.btrim(data.league_name))
       having pg_catalog.count(*) > 1
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_LEAGUE_RATING_ROWS_INVALID: complete normalized-unique rows are required.';
  end if;

  perform public.assert_replay_write_fence_atomic(
    p_job_id,
    v_club_id,
    p_lease_token,
    p_worker_id,
    p_target_reset
  );

  if (
    select pg_catalog.count(*)
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        player_id bigint
      )
      join public.players as player
        on player.id = data.player_id
       and player.club_id::text = v_club_id
  ) <> v_expected then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_LEAGUE_RATING_PLAYER_INVALID: every player must belong to this club.';
  end if;

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
    data.is_active,
    data.inactive_at
  from pg_catalog.jsonb_to_recordset(v_rows) as data(
    club_id text,
    player_id bigint,
    league_name text,
    rating numeric,
    wins integer,
    losses integer,
    matches_played integer,
    starting_rating numeric,
    is_active boolean,
    inactive_at timestamptz
  )
  on conflict (club_id, player_id, league_name) do update
  set
    rating = excluded.rating,
    wins = excluded.wins,
    losses = excluded.losses,
    matches_played = excluded.matches_played,
    starting_rating = excluded.starting_rating,
    is_active = excluded.is_active,
    inactive_at = excluded.inactive_at
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
    excluded.is_active,
    excluded.inactive_at
  );

  select pg_catalog.count(*)
    into v_verified
    from public.league_ratings as league_rating
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      club_id text,
      player_id bigint,
      league_name text,
      rating numeric,
      wins integer,
      losses integer,
      matches_played integer,
      starting_rating numeric,
      is_active boolean,
      inactive_at timestamptz
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
     data.is_active,
     data.inactive_at
   );

  if v_verified <> v_expected then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_REPLAY_LEAGUE_RATING_VERIFY_FAILED: exact rows were not persisted.';
  end if;
  return v_verified;
end
$function$;

revoke all on function public.apply_replay_league_rating_rows_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  jsonb
) from public, anon, authenticated;

grant execute on function public.apply_replay_league_rating_rows_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  jsonb
) to service_role;

-- Keep v1 as the single batch implementation, but put an authoritative
-- lifecycle gate in front of every new operation.  A committed receipt is
-- deliberately delegated to v1 before metadata is consulted, so an exact
-- idempotent retry remains readable after the league has been paused or
-- otherwise closed.  The row lock keeps an allowed new operation and a
-- concurrent lifecycle transition in one serial order.
create or replace function public.admin_apply_league_roster_batch_atomic_v2(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_action text,
  p_player_ids jsonb,
  p_starting_rating numeric,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_league_name), 120), '');
  v_idempotency_key text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_has_operation boolean := false;
  v_league public.leagues_metadata%rowtype;
  v_raw_league_status text;
  v_league_status text;
  v_result jsonb;
  v_receipt_rows jsonb;
begin
  -- Let v1 retain its exact validation/error contract when the identifiers
  -- required to acquire this wrapper's locks are themselves malformed.
  if v_club_id is null
     or v_league_name is null
     or v_idempotency_key is null then
    return public.admin_apply_league_roster_batch_atomic_v1(
      p_operation_id,
      p_club_id,
      p_league_name,
      p_idempotency_key,
      p_request_fingerprint,
      p_action,
      p_player_ids,
      p_starting_rating,
      p_actor_email,
      p_actor_role,
      p_source
    );
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:league-roster-batch:' || v_club_id || ':' || v_league_name,
      0
    )
  );
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select exists (
    select 1
      from public.admin_league_roster_batch_operations as operation
     where operation.club_id = v_club_id
       and operation.idempotency_key = v_idempotency_key
  )
    into v_has_operation;

  if v_has_operation then
    v_result := public.admin_apply_league_roster_batch_atomic_v1(
      p_operation_id,
      p_club_id,
      p_league_name,
      p_idempotency_key,
      p_request_fingerprint,
      p_action,
      p_player_ids,
      p_starting_rating,
      p_actor_email,
      p_actor_role,
      p_source
    );
    select activity.after_json -> 'league_ratings'
      into v_receipt_rows
      from public.admin_activity_log as activity
     where activity.club_id = v_club_id
       and activity.entity_type = 'league_roster_batch_operation'
       and activity.entity_id = v_result ->> 'operation_id'
     limit 1;
    if pg_catalog.jsonb_typeof(v_receipt_rows) = 'array' then
      v_result := v_result || pg_catalog.jsonb_build_object(
        'league_ratings',
        v_receipt_rows
      );
    end if;
    return v_result;
  end if;

  select metadata.*
    into v_league
    from public.leagues_metadata as metadata
   where metadata.club_id = v_club_id
     and metadata.league_name = v_league_name
   for update;

  if not found then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_LEAGUE_NOT_FOUND: league not found.';
  end if;

  -- Match jupr_app.domain.leagues.normalize_league_status and the v1
  -- consistency check exactly before narrowing mutable states to draft/active.
  v_raw_league_status :=
    nullif(pg_catalog.lower(pg_catalog.btrim(v_league.status)), '');
  v_league_status := case
    when v_raw_league_status = 'archived' then 'archived'
    when v_raw_league_status in ('ended', 'completed', 'complete', 'done')
      then 'ended'
    when v_raw_league_status in ('active', 'running', 'live')
      then 'active'
    when v_raw_league_status = 'paused' then 'paused'
    when v_raw_league_status in ('draft', 'planned') then 'draft'
    when v_league.ended_at is not null then 'ended'
    when v_league.is_active is null then 'draft'
    when v_league.is_active then 'active'
    when v_raw_league_status is null then 'ended'
    else 'draft'
  end;

  if coalesce(v_league.is_active, false)
       is distinct from (v_league_status = 'active') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_LIFECYCLE_INVALID: league status and active state are inconsistent.';
  end if;

  if v_league_status not in ('draft', 'active') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_READ_ONLY: only draft and active league rosters are mutable.';
  end if;

  v_result := public.admin_apply_league_roster_batch_atomic_v1(
    p_operation_id,
    p_club_id,
    p_league_name,
    p_idempotency_key,
    p_request_fingerprint,
    p_action,
    p_player_ids,
    p_starting_rating,
    p_actor_email,
    p_actor_role,
    p_source
  );
  select activity.after_json -> 'league_ratings'
    into v_receipt_rows
    from public.admin_activity_log as activity
   where activity.club_id = v_club_id
     and activity.entity_type = 'league_roster_batch_operation'
     and activity.entity_id = v_result ->> 'operation_id'
   limit 1;
  if pg_catalog.jsonb_typeof(v_receipt_rows) = 'array' then
    v_result := v_result || pg_catalog.jsonb_build_object(
      'league_ratings',
      v_receipt_rows
    );
  end if;
  return v_result;
end
$function$;

revoke all on function public.admin_apply_league_roster_batch_atomic_v2(
  uuid,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  numeric,
  text,
  text,
  text
) from public, anon, authenticated;

grant execute on function public.admin_apply_league_roster_batch_atomic_v2(
  uuid,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  numeric,
  text,
  text,
  text
) to service_role;

comment on function public.admin_apply_league_roster_batch_atomic_v2(
  uuid,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  numeric,
  text,
  text,
  text
) is
  'Lifecycle-gates new roster batches while preserving v1 idempotent receipt semantics under the same transaction locks.';

do $historic_backfill$
declare
  v_club_id constant text := 'tres_palapas';
  v_league_name constant text := 'Acceptance Singles League 0731';
  v_match_id constant bigint := 49;
  v_entity_id constant text :=
    'tres_palapas:Acceptance Singles League 0731:49';
  v_source_page constant text :=
    'migration:20261104000000_historic_singles_league_rating_backfill';
  v_league public.leagues_metadata%rowtype;
  v_match public.matches%rowtype;
  v_player_count integer;
  v_existing_audit_count integer;
  v_after jsonb;
begin
  -- Match the lock order used by roster operations, then exclude direct match
  -- entry before inspecting or repairing any projection row.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:league-roster-batch:' || v_club_id || ':' || v_league_name,
      0
    )
  );
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:direct-match-entry:' || v_club_id, 0)
  );

  -- Production normally has no league with this staging acceptance-fixture
  -- identity. In that case the data repair is an intentional no-op while the
  -- replay writer above remains installed. Check only after the same locks as
  -- every competing writer so absence is a stable decision.
  if not exists (
    select 1
      from public.leagues_metadata as metadata
     where metadata.club_id = v_club_id
       and metadata.league_name = v_league_name
  ) then
    return;
  end if;

  if exists (
    select 1
      from public.replay_jobs as replay_job
     where replay_job.club_id = v_club_id
       and pg_catalog.lower(coalesce(replay_job.status, ''))
         in ('pending', 'running')
  ) then
    raise exception using
      errcode = '55006',
      message = 'JUPR_HISTORIC_SINGLES_BACKFILL_REPLAY_IN_PROGRESS: Replay History owns this club.';
  end if;

  select metadata.*
    into strict v_league
    from public.leagues_metadata as metadata
   where metadata.club_id = v_club_id
     and metadata.league_name = v_league_name
   for update;

  if pg_catalog.lower(
       pg_catalog.btrim(coalesce(v_league.match_format, ''))
     ) <> 'singles'
     or coalesce(v_league.is_active, false) is not true
     or pg_catalog.lower(pg_catalog.btrim(coalesce(v_league.status, '')))
          not in ('active', 'running', 'live')
     or v_league.ended_at is not null
     or v_league.k_factor is distinct from 32 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_HISTORIC_SINGLES_BACKFILL_LEAGUE_MISMATCH: exact singles metadata is required.';
  end if;

  select match_row.*
    into strict v_match
    from public.matches as match_row
   where match_row.club_id = v_club_id
     and match_row.id = v_match_id
   for update;

  if v_match.league is distinct from v_league_name
     or pg_catalog.lower(pg_catalog.btrim(coalesce(v_match.match_format, '')))
          <> 'singles'
     or pg_catalog.lower(pg_catalog.btrim(coalesce(v_match.match_type, '')))
          not in ('league', 'singles')
     or v_match.t1_p1 is distinct from 22
     or v_match.t1_p2 is not null
     or v_match.t2_p1 is distinct from 23
     or v_match.t2_p2 is not null
     or v_match.date::date is distinct from date '2026-07-31'
     or v_match.score_t1 is distinct from 11
     or v_match.score_t2 is distinct from 8
     or coalesce(pg_catalog.btrim(v_match.rating_scope), '') <> ''
     or coalesce(v_match.rating_bonus_elo, 0) <> 0
     or v_match.singles_replay_managed is not true
     or v_match.deleted_at is not null
     or v_match.t1_p2_r is not null
     or v_match.t2_p2_r is not null
     or v_match.t1_p2_r_end is not null
     or v_match.t2_p2_r_end is not null
     or pg_catalog.round(v_match.elo_delta::numeric, 4)
          is distinct from 5.0526::numeric
     or pg_catalog.round(v_match.t1_p1_r::numeric, 4)
          is distinct from 1200.0000::numeric
     or pg_catalog.round(v_match.t2_p1_r::numeric, 4)
          is distinct from 1200.0000::numeric
     or pg_catalog.round(v_match.t1_p1_r_end::numeric, 4)
          is distinct from 1205.0526::numeric
     or pg_catalog.round(v_match.t2_p1_r_end::numeric, 4)
          is distinct from 1194.9474::numeric then
    raise exception using
      errcode = '22023',
      message = 'JUPR_HISTORIC_SINGLES_BACKFILL_MATCH_MISMATCH: match 49 no longer has the reviewed final snapshots.';
  end if;

  perform 1
    from public.players as player
   where player.club_id = v_club_id
     and player.id in (22, 23)
   order by player.id
   for update;

  select pg_catalog.count(*)
    into v_player_count
    from public.players as player
   where player.club_id = v_club_id
     and (
       (player.id = 22 and player.name = 'Test B')
       or (player.id = 23 and player.name = 'Test C')
     );
  if v_player_count <> 2 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_HISTORIC_SINGLES_BACKFILL_PLAYER_MISMATCH: exact Test B/Test C identities are required.';
  end if;

  perform 1
    from public.league_ratings as league_rating
   where league_rating.club_id = v_club_id
     and league_rating.player_id in (22, 23)
     and pg_catalog.lower(pg_catalog.btrim(league_rating.league_name)) =
           pg_catalog.lower(pg_catalog.btrim(v_league_name))
   order by league_rating.player_id, league_rating.league_name
   for update;

  if exists (
    with expected as (
      select
        22::bigint as player_id,
        pg_catalog.round(v_match.t1_p1_r::numeric, 4) as starting_rating,
        pg_catalog.round(v_match.t1_p1_r_end::numeric, 4) as rating,
        1::integer as wins,
        0::integer as losses
      union all
      select
        23::bigint,
        pg_catalog.round(v_match.t2_p1_r::numeric, 4),
        pg_catalog.round(v_match.t2_p1_r_end::numeric, 4),
        0::integer,
        1::integer
    )
    select 1
      from public.league_ratings as league_rating
      join expected
        on expected.player_id = league_rating.player_id
     where league_rating.club_id = v_club_id
       and pg_catalog.lower(pg_catalog.btrim(league_rating.league_name)) =
             pg_catalog.lower(pg_catalog.btrim(v_league_name))
       and (
         league_rating.league_name is distinct from v_league_name
         or row(
           league_rating.starting_rating,
           league_rating.rating,
           league_rating.wins,
           league_rating.losses,
           league_rating.matches_played,
           league_rating.is_active,
           league_rating.inactive_at
         ) is distinct from row(
           expected.starting_rating,
           expected.rating,
           expected.wins,
           expected.losses,
           1,
           true,
           null::timestamptz
         )
       )
  ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_HISTORIC_SINGLES_BACKFILL_CONFLICT: an existing normalized row has different state.';
  end if;

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
    v_club_id,
    expected.player_id,
    v_league_name,
    expected.rating,
    expected.wins,
    expected.losses,
    1,
    expected.starting_rating,
    true,
    null
  from (
    select
      22::bigint as player_id,
      pg_catalog.round(v_match.t1_p1_r::numeric, 4) as starting_rating,
      pg_catalog.round(v_match.t1_p1_r_end::numeric, 4) as rating,
      1::integer as wins,
      0::integer as losses
    union all
    select
      23::bigint,
      pg_catalog.round(v_match.t2_p1_r::numeric, 4),
      pg_catalog.round(v_match.t2_p1_r_end::numeric, 4),
      0::integer,
      1::integer
  ) as expected
  on conflict (club_id, player_id, league_name) do nothing;

  if (
    select pg_catalog.count(*)
      from public.league_ratings as league_rating
     where league_rating.club_id = v_club_id
       and league_rating.league_name = v_league_name
       and (
         (
           league_rating.player_id = 22
           and row(
             league_rating.starting_rating,
             league_rating.rating,
             league_rating.wins,
             league_rating.losses,
             league_rating.matches_played,
             league_rating.is_active,
             league_rating.inactive_at
           ) is not distinct from row(
             1200.0000::numeric,
             1205.0526::numeric,
             1,
             0,
             1,
             true,
             null::timestamptz
           )
         )
         or (
           league_rating.player_id = 23
           and row(
             league_rating.starting_rating,
             league_rating.rating,
             league_rating.wins,
             league_rating.losses,
             league_rating.matches_played,
             league_rating.is_active,
             league_rating.inactive_at
           ) is not distinct from row(
             1200.0000::numeric,
             1194.9474::numeric,
             0,
             1,
             1,
             true,
             null::timestamptz
           )
         )
       )
  ) <> 2 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_HISTORIC_SINGLES_BACKFILL_VERIFY_FAILED: exact repaired rows were not found.';
  end if;

  v_after := pg_catalog.jsonb_build_object(
    'source', 'authoritative_final_match_snapshots',
    'match_id', v_match_id,
    'league_name', v_league_name,
    'league_ratings', pg_catalog.jsonb_build_array(
      pg_catalog.jsonb_build_object(
        'player_id', 22,
        'starting_rating', 1200.0000,
        'rating', 1205.0526,
        'wins', 1,
        'losses', 0,
        'matches_played', 1,
        'is_active', true,
        'inactive_at', null
      ),
      pg_catalog.jsonb_build_object(
        'player_id', 23,
        'starting_rating', 1200.0000,
        'rating', 1194.9474,
        'wins', 0,
        'losses', 1,
        'matches_played', 1,
        'is_active', true,
        'inactive_at', null
      )
    )
  );

  select pg_catalog.count(*)
    into v_existing_audit_count
    from public.admin_activity_log as activity
   where activity.club_id = v_club_id
     and activity.entity_id = v_entity_id
     and activity.source_page = v_source_page;
  if v_existing_audit_count > 1
     or exists (
       select 1
         from public.admin_activity_log as activity
        where activity.club_id = v_club_id
          and activity.entity_id = v_entity_id
          and activity.source_page = v_source_page
          and (
            activity.action_type is distinct from
              'historic_singles_league_rating_backfill'
            or activity.entity_type is distinct from 'league_rating_backfill'
            or activity.after_json is distinct from v_after
          )
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_HISTORIC_SINGLES_BACKFILL_AUDIT_CONFLICT: the stable audit identity has different content.';
  end if;

  if v_existing_audit_count = 0 then
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
      'migration@juprleagues.com',
      'system',
      'historic_singles_league_rating_backfill',
      'league_rating_backfill',
      v_entity_id,
      pg_catalog.jsonb_build_object(
        'known_missing_player_ids',
        pg_catalog.jsonb_build_array(22, 23)
      ),
      v_after,
      'Repaired the reviewed match-49 singles league projection from final snapshots.',
      v_source_page,
      true
    );
  end if;
end
$historic_backfill$;

notify pgrst, 'reload schema';
