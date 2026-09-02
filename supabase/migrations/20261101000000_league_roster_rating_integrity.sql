-- Keep official league membership and league-scoped ratings in the same
-- transaction as match entry. Singles and doubles now share the same
-- participant-membership invariant while retaining their independent rating
-- baselines.

do $migration$
declare
  missing_columns text[];
begin
  select pg_catalog.array_agg(
           required.table_name || '.' || required.column_name
           order by required.table_name, required.column_name
         )
    into missing_columns
    from (
      values
        ('players', 'singles_rating'),
        ('leagues_metadata', 'match_format')
    ) as required(table_name, column_name)
    where not exists (
      select 1
        from information_schema.columns as actual
       where actual.table_schema = 'public'
         and actual.table_name = required.table_name
         and actual.column_name = required.column_name
    );

  if missing_columns is not null then
    raise exception using
      errcode = '42703',
      message = 'league roster/rating integrity requires columns: '
        || pg_catalog.array_to_string(missing_columns, ', ');
  end if;
end
$migration$;

do $migration$
begin
  if exists (
    select 1
      from public.league_ratings as league_rating
     group by
       league_rating.club_id,
       league_rating.player_id,
       pg_catalog.lower(pg_catalog.btrim(league_rating.league_name))
    having pg_catalog.count(*) > 1
  ) then
    raise exception using
      errcode = '23505',
      message = 'league roster/rating integrity requires case-insensitive unique league-rating rows';
  end if;
end
$migration$;

-- Keep every writer on the same canonical membership key. The preflight above
-- makes a partial/legacy install fail safely; this index prevents a later
-- case- or whitespace-variant row from making normalized lookups ambiguous.
create unique index if not exists league_ratings_club_player_normalized_league_uidx
  on public.league_ratings (
    club_id,
    player_id,
    (pg_catalog.lower(pg_catalog.btrim(league_name)))
  );

-- Preserve the full lifecycle and match-format wrapper chain as the
-- compatibility implementation. This outer layer owns only the additional
-- official-league participant coverage and singles league_ratings CAS work.
do $migration$
declare
  v_public_signature constant text :=
    'public.admin_apply_direct_match_entry_atomic_v1(uuid,text,text,text,text,text,text,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb)';
  v_base_signature constant text :=
    'public.admin_apply_direct_match_entry_base_20261101(uuid,text,text,text,text,text,text,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb)';
begin
  if pg_catalog.to_regprocedure(v_base_signature) is null then
    if pg_catalog.to_regprocedure(v_public_signature) is null then
      raise exception using
        errcode = '42883',
        message = 'league roster/rating integrity requires the direct-match atomic RPC';
    end if;

    execute 'alter function ' || v_public_signature
      || ' rename to admin_apply_direct_match_entry_base_20261101';
  end if;
end
$migration$;

revoke all on function public.admin_apply_direct_match_entry_base_20261101(
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

grant execute on function public.admin_apply_direct_match_entry_base_20261101(
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
  v_match_format text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_match_format, '')));
  v_match_rows jsonb := coalesce(p_match_rows, '[]'::jsonb);
  v_league_rating_updates jsonb :=
    coalesce(p_league_rating_updates, '[]'::jsonb);
  v_league_metadata_expectations jsonb :=
    coalesce(p_league_metadata_expectations, '[]'::jsonb);
  v_operation public.admin_direct_match_entry_operations%rowtype;
  v_league_rating public.league_ratings%rowtype;
  v_item record;
  v_after jsonb;
  v_result jsonb;
  v_row_count integer := 0;
begin
  if v_club_id is not null and v_idempotency_key is not null then
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
      v_result := public.admin_apply_direct_match_entry_base_20261101(
        p_operation_id,
        p_club_id,
        p_idempotency_key,
        p_request_fingerprint,
        p_match_format,
        p_source,
        p_actor_email,
        p_actor_role,
        p_request_json,
        p_result_summary,
        p_match_rows,
        p_player_updates,
        p_league_rating_updates,
        p_league_metadata_expectations
      );

      if v_match_format = 'singles'
         and pg_catalog.jsonb_typeof(v_result->'result_summary') = 'object'
         and (
           v_result->'result_summary'
         ) ? 'official_league_rating_update_count' then
        v_result := v_result || pg_catalog.jsonb_build_object(
          'league_rating_update_count',
          (
            v_result->'result_summary'
              ->>'official_league_rating_update_count'
          )::integer
        );
      end if;
      return v_result;
    end if;
  end if;

  -- Let the established implementation retain its exact validation errors for
  -- malformed plans. Coverage queries below only run over well-shaped arrays.
  if v_match_format not in ('doubles', 'singles')
     or pg_catalog.jsonb_typeof(v_match_rows) <> 'array'
     or pg_catalog.jsonb_typeof(v_league_rating_updates) <> 'array'
     or pg_catalog.jsonb_typeof(v_league_metadata_expectations) <> 'array' then
    return public.admin_apply_direct_match_entry_base_20261101(
      p_operation_id,
      p_club_id,
      p_idempotency_key,
      p_request_fingerprint,
      p_match_format,
      p_source,
      p_actor_email,
      p_actor_role,
      p_request_json,
      p_result_summary,
      p_match_rows,
      p_player_updates,
      p_league_rating_updates,
      p_league_metadata_expectations
    );
  end if;

  -- Every official-league participant must have exactly one planned
  -- league_ratings projection. A metadata expectation identifies which match
  -- labels are managed leagues; tournament and event labels may also occupy
  -- matches.league and must remain outside this invariant. Reserved aggregate
  -- leagues remain outside it as well. The projection must leave the
  -- participant active in the league.
  if exists (
    with managed_leagues as (
      select distinct
        pg_catalog.lower(
          pg_catalog.btrim(metadata_expectation.league_name)
        ) as league_key
      from pg_catalog.jsonb_to_recordset(v_league_metadata_expectations)
        as metadata_expectation(
          league_name text,
          expected jsonb
        )
      where nullif(
              pg_catalog.btrim(metadata_expectation.league_name),
              ''
            ) is not null
        and pg_catalog.lower(
              pg_catalog.btrim(metadata_expectation.league_name)
            ) not in ('overall', 'popup', 'singles')
    ),
    required_memberships as (
      select distinct
        pg_catalog.lower(pg_catalog.btrim(match_row.league)) as league_key,
        participant.player_id
      from pg_catalog.jsonb_to_recordset(v_match_rows) as match_row(
        league text,
        t1_p1 bigint,
        t1_p2 bigint,
        t2_p1 bigint,
        t2_p2 bigint
      )
      cross join lateral (
        values
          (match_row.t1_p1, true),
          (match_row.t1_p2, v_match_format = 'doubles'),
          (match_row.t2_p1, true),
          (match_row.t2_p2, v_match_format = 'doubles')
      ) as participant(player_id, included)
      join managed_leagues as managed_league
        on managed_league.league_key =
             pg_catalog.lower(pg_catalog.btrim(match_row.league))
      where participant.included
        and participant.player_id is not null
        and nullif(pg_catalog.btrim(match_row.league), '') is not null
        and pg_catalog.lower(pg_catalog.btrim(match_row.league))
              not in ('overall', 'popup', 'singles')
    ),
    planned_memberships as (
      select distinct
        pg_catalog.lower(pg_catalog.btrim(update_row.league_name)) as league_key,
        update_row.player_id
      from pg_catalog.jsonb_to_recordset(v_league_rating_updates)
        as update_row(
          player_id bigint,
          league_name text
        )
      where update_row.player_id is not null
        and nullif(pg_catalog.btrim(update_row.league_name), '') is not null
    ),
    symmetric_difference as (
      (
        select league_key, player_id from required_memberships
        except
        select league_key, player_id from planned_memberships
      )
      union all
      (
        select league_key, player_id from planned_memberships
        except
        select league_key, player_id from required_memberships
      )
    )
    select 1 from symmetric_difference
  )
  or exists (
    select 1
      from pg_catalog.jsonb_to_recordset(v_league_rating_updates)
        as update_row(
          player_id bigint,
          league_name text,
          after jsonb
        )
     where update_row.after->>'is_active' is distinct from 'true'
        or nullif(update_row.after->>'inactive_at', '') is not null
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_LEAGUE_ROSTER_PLAN_INVALID: official league participants require exact active league-rating projections.';
  end if;

  -- Doubles already owns league_ratings CAS work in the compatibility base.
  if v_match_format = 'doubles' then
    return public.admin_apply_direct_match_entry_base_20261101(
      p_operation_id,
      p_club_id,
      p_idempotency_key,
      p_request_fingerprint,
      p_match_format,
      p_source,
      p_actor_email,
      p_actor_role,
      p_request_json,
      p_result_summary,
      p_match_rows,
      p_player_updates,
      p_league_rating_updates,
      p_league_metadata_expectations
    );
  end if;

  -- Singles historically rejected league_ratings plans. Validate and lock the
  -- new projections here, call the compatibility base with an empty projection,
  -- then apply the locked rows before this outer transaction can commit.
  if (
    select pg_catalog.count(
      distinct (
        update_row.player_id,
        pg_catalog.lower(pg_catalog.btrim(update_row.league_name))
      )
    )
      from pg_catalog.jsonb_to_recordset(v_league_rating_updates)
        as update_row(
          player_id bigint,
          league_name text
        )
  ) <> pg_catalog.jsonb_array_length(v_league_rating_updates)
  or exists (
    select 1
      from pg_catalog.jsonb_to_recordset(v_league_rating_updates)
        as update_row(
          player_id bigint,
          league_name text,
          expected jsonb,
          after jsonb
        )
     where update_row.player_id is null
        or update_row.player_id <= 0
        or nullif(pg_catalog.btrim(update_row.league_name), '') is null
        or (
          update_row.expected is not null
          and pg_catalog.jsonb_typeof(update_row.expected) <> 'object'
        )
        or pg_catalog.jsonb_typeof(update_row.after) <> 'object'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_LEAGUE_PLAN_INVALID: unique exact league projections are required.';
  end if;

  perform league_rating.id
    from public.league_ratings as league_rating
    join pg_catalog.jsonb_to_recordset(v_league_rating_updates)
      as update_row(
        player_id bigint,
        league_name text
      ) on update_row.player_id = league_rating.player_id
        and pg_catalog.lower(pg_catalog.btrim(update_row.league_name)) =
              pg_catalog.lower(pg_catalog.btrim(league_rating.league_name))
   where league_rating.club_id = v_club_id
   order by league_rating.id
   for update;

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(v_league_rating_updates)
        as update_row(
          player_id bigint,
          league_name text,
          expected jsonb,
          after jsonb
        )
     order by update_row.player_id, update_row.league_name
  loop
    select league_rating.*
      into v_league_rating
      from public.league_ratings as league_rating
     where league_rating.club_id = v_club_id
       and league_rating.player_id = v_item.player_id
       and pg_catalog.lower(pg_catalog.btrim(league_rating.league_name)) =
             pg_catalog.lower(pg_catalog.btrim(v_item.league_name))
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

  v_result := public.admin_apply_direct_match_entry_base_20261101(
    p_operation_id,
    p_club_id,
    p_idempotency_key,
    p_request_fingerprint,
    p_match_format,
    p_source,
    p_actor_email,
    p_actor_role,
    p_request_json,
    coalesce(p_result_summary, '{}'::jsonb)
      || pg_catalog.jsonb_build_object(
           'official_league_rating_update_count',
           pg_catalog.jsonb_array_length(v_league_rating_updates)
         ),
    p_match_rows,
    p_player_updates,
    '[]'::jsonb,
    p_league_metadata_expectations
  );

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(v_league_rating_updates)
        as update_row(
          player_id bigint,
          league_name text,
          expected jsonb,
          after jsonb
        )
     order by update_row.player_id, update_row.league_name
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
             matches_played = (v_after->>'matches_played')::integer,
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

  v_result := v_result || pg_catalog.jsonb_build_object(
    'league_rating_update_count',
    pg_catalog.jsonb_array_length(v_league_rating_updates)
  );

  -- The compatibility base had to receive an empty singles projection, so
  -- replace its zero count in both durable records after the outer CAS writes
  -- succeed. Exact retries then return the same receipt without recomputing or
  -- replaying a league_ratings mutation.
  update public.admin_direct_match_entry_operations as operation
     set result_json = v_result
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_idempotency_key
     and operation.id = p_operation_id;

  get diagnostics v_row_count = row_count;
  if v_row_count <> 1 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_DIRECT_MATCH_RECEIPT_WRITE_INCOMPLETE: the durable singles receipt was not updated.';
  end if;

  update public.admin_activity_log as activity
     set after_json = pg_catalog.jsonb_set(
           activity.after_json,
           '{operation}',
           v_result - 'player_updates',
           false
         )
   where activity.club_id = v_club_id
     and activity.entity_type = 'direct_match_entry_operation'
     and activity.entity_id = p_operation_id::text;

  get diagnostics v_row_count = row_count;
  if v_row_count <> 1 then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_DIRECT_MATCH_AUDIT_WRITE_INCOMPLETE: the singles audit receipt was not updated.';
  end if;

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
  'Applies idempotent direct match plans and atomically CAS-writes active league ratings for every official-league participant.';

-- Forward-replace the roster batch implementation so a missing explicit
-- override selects the rating baseline that belongs to the league's format.
-- Existing league_ratings rows retain their current value when reactivated.
create or replace function public.admin_apply_league_roster_batch_atomic_v1(
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
  v_request_fingerprint text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_action text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_action, '')));
  v_player_ids_json jsonb := coalesce(p_player_ids, '[]'::jsonb);
  v_player_ids bigint[];
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
  v_source text :=
    coalesce(
      nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
      'next_league_manager_bulk_roster_editor'
    );
  v_operation public.admin_league_roster_batch_operations%rowtype;
  v_league public.leagues_metadata%rowtype;
  v_raw_league_status text;
  v_league_status text;
  v_league_match_format text;
  v_before jsonb := '[]'::jsonb;
  v_after jsonb := '[]'::jsonb;
  v_request jsonb;
  v_result jsonb;
  v_target_count integer;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_idempotency_key is null
     or pg_catalog.char_length(v_idempotency_key) not between 8 and 160
     or v_idempotency_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]+$'
     or v_request_fingerprint !~ '^[0-9a-f]{64}$'
     or v_action not in ('activate', 'deactivate')
     or pg_catalog.jsonb_typeof(v_player_ids_json) <> 'array'
     or pg_catalog.jsonb_array_length(v_player_ids_json) not between 1 and 500
     or exists (
       select 1
         from pg_catalog.jsonb_array_elements(v_player_ids_json) as item(value)
        where pg_catalog.jsonb_typeof(item.value) <> 'number'
           or item.value::text !~ '^[1-9][0-9]*$'
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_INVALID: exact operation, league, action, and player IDs are required.';
  end if;

  select pg_catalog.array_agg(
           item.value::text::bigint
           order by item.value::text::bigint
         )
    into v_player_ids
    from pg_catalog.jsonb_array_elements(v_player_ids_json) as item(value);

  v_target_count := pg_catalog.cardinality(v_player_ids);
  if (
    select pg_catalog.count(distinct player_id)
      from pg_catalog.unnest(v_player_ids) as selected(player_id)
  ) <> v_target_count then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_INVALID: player IDs must be unique.';
  end if;

  if p_starting_rating is not null
     and (
       case
         when p_starting_rating <= 20 then p_starting_rating * 400
         else p_starting_rating
       end
     ) not between 400 and 2800 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_RATING_INVALID: use JUPR 1.0-7.0 or Elo 400-2800.';
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

  select operation.*
    into v_operation
    from public.admin_league_roster_batch_operations as operation
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_idempotency_key;

  if found then
    if v_operation.league_name is distinct from v_league_name
       or v_operation.request_fingerprint is distinct from v_request_fingerprint
       or v_operation.action is distinct from v_action then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_LEAGUE_ROSTER_BATCH_IDEMPOTENCY_CONFLICT: this key already belongs to a different request.';
    end if;
    return v_operation.result_json || pg_catalog.jsonb_build_object(
      'idempotent', true
    );
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
      message = 'JUPR_LEAGUE_ROSTER_BATCH_REPLAY_IN_PROGRESS: Replay History is rebuilding this club.';
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

  v_league_match_format := pg_catalog.lower(
    pg_catalog.btrim(coalesce(v_league.match_format, ''))
  );
  if v_league_match_format not in ('doubles', 'singles') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_FORMAT_INVALID: league match format must be singles or doubles.';
  end if;

  -- Keep database lifecycle interpretation identical to
  -- jupr_app.domain.leagues.normalize_league_status.
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

  if v_league_status in ('ended', 'archived') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_READ_ONLY: ended and archived league rosters are read-only.';
  end if;

  perform 1
    from public.players as player
   where player.club_id = v_club_id
     and player.id = any(v_player_ids)
   order by player.id
   for update;

  if (
    select pg_catalog.count(*)
      from public.players as player
     where player.club_id = v_club_id
       and player.id = any(v_player_ids)
  ) <> v_target_count then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_PLAYER_NOT_FOUND: every player must belong to this club.';
  end if;

  perform 1
    from public.league_ratings as rating
   where rating.club_id = v_club_id
     and rating.league_name = v_league_name
     and rating.player_id = any(v_player_ids)
   order by rating.player_id
   for update;

  select coalesce(
           pg_catalog.jsonb_agg(
             pg_catalog.to_jsonb(rating)
             order by rating.player_id
           ),
           '[]'::jsonb
         )
    into v_before
    from public.league_ratings as rating
   where rating.club_id = v_club_id
     and rating.league_name = v_league_name
     and rating.player_id = any(v_player_ids);

  if v_action = 'activate' then
    if exists (
      select 1
        from public.players as player
       where player.club_id = v_club_id
         and player.id = any(v_player_ids)
         and (
           coalesce(player.active, true) is not true
           or player.inactive_at is not null
         )
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_LEAGUE_ROSTER_BATCH_PLAYER_INACTIVE: inactive club players cannot be added to a league.';
    end if;

    if exists (
      select 1
        from public.league_ratings as rating
       where rating.club_id = v_club_id
         and rating.league_name = v_league_name
         and rating.player_id = any(v_player_ids)
         and coalesce(rating.is_active, true)
         and rating.inactive_at is null
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_LEAGUE_ROSTER_BATCH_ALREADY_ACTIVE: at least one player is already active in this league.';
    end if;

    if exists (
      select 1
        from public.players as player
        left join public.league_ratings as rating
          on rating.club_id = v_club_id
         and rating.league_name = v_league_name
         and rating.player_id = player.id
        cross join lateral (
          select case
            when v_league_match_format = 'singles'
              then coalesce(player.singles_rating, 1200)::numeric
            else coalesce(player.rating, 1200)::numeric
          end as raw_rating
        ) as baseline
       where player.club_id = v_club_id
         and player.id = any(v_player_ids)
         and rating.id is null
         and (
           case
             when p_starting_rating is not null then
               case
                 when p_starting_rating <= 20 then p_starting_rating * 400
                 else p_starting_rating
               end
             when baseline.raw_rating <= 20 then baseline.raw_rating * 400
             else baseline.raw_rating
           end
         ) not between 400 and 2800
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_LEAGUE_ROSTER_BATCH_RATING_INVALID: every new league member needs a valid starting rating.';
    end if;

    update public.league_ratings as rating
       set is_active = true,
           inactive_at = null
     where rating.club_id = v_club_id
       and rating.league_name = v_league_name
       and rating.player_id = any(v_player_ids);

    insert into public.league_ratings (
      club_id,
      player_id,
      league_name,
      rating,
      starting_rating,
      wins,
      losses,
      matches_played,
      is_active,
      inactive_at
    )
    select
      v_club_id,
      player.id,
      v_league_name,
      case
        when p_starting_rating is not null then
          case
            when p_starting_rating <= 20 then p_starting_rating * 400
            else p_starting_rating
          end
        when baseline.raw_rating <= 20 then baseline.raw_rating * 400
        else baseline.raw_rating
      end,
      case
        when p_starting_rating is not null then
          case
            when p_starting_rating <= 20 then p_starting_rating * 400
            else p_starting_rating
          end
        when baseline.raw_rating <= 20 then baseline.raw_rating * 400
        else baseline.raw_rating
      end,
      0,
      0,
      0,
      true,
      null
    from public.players as player
    cross join lateral (
      select case
        when v_league_match_format = 'singles'
          then coalesce(player.singles_rating, 1200)::numeric
        else coalesce(player.rating, 1200)::numeric
      end as raw_rating
    ) as baseline
   where player.club_id = v_club_id
     and player.id = any(v_player_ids)
     and not exists (
       select 1
         from public.league_ratings as rating
        where rating.club_id = v_club_id
          and rating.league_name = v_league_name
          and rating.player_id = player.id
     );
  else
    if (
      select pg_catalog.count(*)
        from public.league_ratings as rating
       where rating.club_id = v_club_id
         and rating.league_name = v_league_name
         and rating.player_id = any(v_player_ids)
         and coalesce(rating.is_active, true)
         and rating.inactive_at is null
    ) <> v_target_count then
      raise exception using
        errcode = '22023',
        message = 'JUPR_LEAGUE_ROSTER_BATCH_NOT_ACTIVE: every selected player must currently be active in this league.';
    end if;

    update public.league_ratings as rating
       set is_active = false,
           inactive_at = pg_catalog.clock_timestamp()
     where rating.club_id = v_club_id
       and rating.league_name = v_league_name
       and rating.player_id = any(v_player_ids)
       and coalesce(rating.is_active, true)
       and rating.inactive_at is null;
  end if;

  select coalesce(
           pg_catalog.jsonb_agg(
             pg_catalog.to_jsonb(rating)
             order by rating.player_id
           ),
           '[]'::jsonb
         )
    into v_after
    from public.league_ratings as rating
   where rating.club_id = v_club_id
     and rating.league_name = v_league_name
     and rating.player_id = any(v_player_ids);

  if pg_catalog.jsonb_array_length(v_after) <> v_target_count then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_WRITE_CONFLICT: the final roster rows do not match the requested players.';
  end if;

  v_request := pg_catalog.jsonb_build_object(
    'league_name', v_league_name,
    'action', v_action,
    'player_ids', pg_catalog.to_jsonb(v_player_ids),
    'starting_rating', p_starting_rating
  );
  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'mode', 'league_manager_roster_batch_update',
    'operation_id', p_operation_id,
    'league_name', v_league_name,
    'action', v_action,
    'player_ids', pg_catalog.to_jsonb(v_player_ids),
    'updated_count', v_target_count,
    'idempotent', false
  );

  insert into public.admin_league_roster_batch_operations (
    id,
    club_id,
    league_name,
    idempotency_key,
    request_fingerprint,
    action,
    source,
    actor_email,
    actor_role,
    request_json,
    result_json
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_idempotency_key,
    v_request_fingerprint,
    v_action,
    v_source,
    v_actor_email,
    v_actor_role,
    v_request,
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
    'update_league_manager_roster_membership_batch_admin',
    'league_roster_batch_operation',
    p_operation_id::text,
    pg_catalog.jsonb_build_object(
      'league_name', v_league_name,
      'league_ratings', v_before
    ),
    pg_catalog.jsonb_build_object(
      'league_name', v_league_name,
      'action', v_action,
      'player_ids', pg_catalog.to_jsonb(v_player_ids),
      'league_ratings', v_after
    ),
    null,
    v_source,
    true
  );

  return v_result;
exception
  when unique_violation then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_LEAGUE_ROSTER_BATCH_CONCURRENT_CONFLICT: another roster write won the race; no part of this batch committed.';
end
$function$;

revoke all on function public.admin_apply_league_roster_batch_atomic_v1(
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

grant execute on function public.admin_apply_league_roster_batch_atomic_v1(
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

comment on function public.admin_apply_league_roster_batch_atomic_v1(
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
  'Atomically applies an idempotent roster batch and seeds new members from the league-format rating unless an explicit override is supplied.';

notify pgrst, 'reload schema';
