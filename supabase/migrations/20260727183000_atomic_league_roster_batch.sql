-- Apply multi-player League Manager roster changes as one idempotent,
-- fail-closed transaction. The operation receipt, every league_ratings
-- mutation, and the required audit row commit together or all roll back.

do $migration$
declare
  missing_columns text[];
begin
  if to_regclass('public.players') is null
     or to_regclass('public.league_ratings') is null
     or to_regclass('public.leagues_metadata') is null
     or to_regclass('public.admin_activity_log') is null
     or to_regclass('public.replay_jobs') is null then
    raise exception using
      errcode = '42P01',
      message = 'atomic League Manager roster batches require players, league_ratings, leagues_metadata, admin_activity_log, and replay_jobs';
  end if;

  select pg_catalog.array_agg(
           required.table_name || '.' || required.column_name
           order by required.table_name, required.column_name
         )
    into missing_columns
    from (
      values
        ('players', 'id'),
        ('players', 'club_id'),
        ('players', 'rating'),
        ('players', 'active'),
        ('players', 'inactive_at'),
        ('league_ratings', 'id'),
        ('league_ratings', 'club_id'),
        ('league_ratings', 'league_name'),
        ('league_ratings', 'player_id'),
        ('league_ratings', 'rating'),
        ('league_ratings', 'starting_rating'),
        ('league_ratings', 'wins'),
        ('league_ratings', 'losses'),
        ('league_ratings', 'matches_played'),
        ('league_ratings', 'is_active'),
        ('league_ratings', 'inactive_at'),
        ('leagues_metadata', 'club_id'),
        ('leagues_metadata', 'league_name'),
        ('leagues_metadata', 'status'),
        ('leagues_metadata', 'ended_at'),
        ('leagues_metadata', 'is_active'),
        ('admin_activity_log', 'club_id'),
        ('admin_activity_log', 'actor_email'),
        ('admin_activity_log', 'actor_role'),
        ('admin_activity_log', 'action_type'),
        ('admin_activity_log', 'entity_type'),
        ('admin_activity_log', 'entity_id'),
        ('admin_activity_log', 'before_json'),
        ('admin_activity_log', 'after_json'),
        ('admin_activity_log', 'note'),
        ('admin_activity_log', 'source_page'),
        ('admin_activity_log', 'flagged_for_review'),
        ('replay_jobs', 'club_id'),
        ('replay_jobs', 'status')
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
      message = 'atomic League Manager roster batch schema is missing required columns: '
        || pg_catalog.array_to_string(missing_columns, ', ');
  end if;
end
$migration$;

create table if not exists public.admin_league_roster_batch_operations (
  id uuid primary key,
  club_id text not null,
  league_name text not null,
  idempotency_key text not null,
  request_fingerprint text not null,
  action text not null,
  source text not null,
  actor_email text not null,
  actor_role text not null,
  request_json jsonb not null,
  result_json jsonb not null,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint admin_league_roster_batch_idempotency_key_check
    check (
      pg_catalog.char_length(idempotency_key) between 8 and 160
      and idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]+$'
    ),
  constraint admin_league_roster_batch_fingerprint_check
    check (request_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint admin_league_roster_batch_action_check
    check (action in ('activate', 'deactivate')),
  constraint admin_league_roster_batch_request_check
    check (pg_catalog.jsonb_typeof(request_json) = 'object'),
  constraint admin_league_roster_batch_result_check
    check (
      pg_catalog.jsonb_typeof(result_json) = 'object'
      and result_json @> '{"ok": true, "committed": true}'::jsonb
    ),
  unique (club_id, idempotency_key)
);

create index if not exists admin_league_roster_batch_club_created_idx
  on public.admin_league_roster_batch_operations (club_id, created_at desc);

alter table public.admin_league_roster_batch_operations enable row level security;
alter table public.admin_league_roster_batch_operations force row level security;

revoke all on table public.admin_league_roster_batch_operations
  from public, anon, authenticated;
grant select, insert on table public.admin_league_roster_batch_operations
  to service_role;

comment on table public.admin_league_roster_batch_operations is
  'Completed atomic League Manager roster-batch receipts. Exact retries return the original result.';

drop function if exists public.admin_apply_league_roster_batch_atomic_v1(
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
);

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

  select pg_catalog.array_agg(item.value::text::bigint order by item.value::text::bigint)
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
             when coalesce(player.rating, 1200) <= 20
               then coalesce(player.rating, 1200) * 400
             else coalesce(player.rating, 1200)
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
        when coalesce(player.rating, 1200) <= 20
          then coalesce(player.rating, 1200) * 400
        else coalesce(player.rating, 1200)
      end,
      case
        when p_starting_rating is not null then
          case
            when p_starting_rating <= 20 then p_starting_rating * 400
            else p_starting_rating
          end
        when coalesce(player.rating, 1200) <= 20
          then coalesce(player.rating, 1200) * 400
        else coalesce(player.rating, 1200)
      end,
      0,
      0,
      0,
      true,
      null
    from public.players as player
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
  'Atomically applies one idempotent multi-player League Manager roster change with audit evidence.';

notify pgrst, 'reload schema';
