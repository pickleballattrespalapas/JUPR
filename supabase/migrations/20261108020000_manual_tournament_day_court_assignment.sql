-- Tournament Day Live manual court assignment.
--
-- Draw activation and result progression now stop at the durable WAITING
-- queue. Operators explicitly assign one eligible game to the next open court
-- or to a reviewed court, and can atomically move or requeue a mistaken
-- assignment. The legacy bulk allocator remains callable only through its
-- explicit guarded auto-fill command for backwards compatibility.

do $migration$
begin
  if pg_catalog.to_regprocedure(
       'public.fill_tournament_day_live_courts_explicit(uuid,text,text)'
     ) is null then
    if pg_catalog.to_regprocedure(
         'public.fill_tournament_day_live_courts(uuid,text,text)'
       ) is null then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_MIGRATION: court allocator is missing.';
    end if;
    execute 'alter function public.fill_tournament_day_live_courts(uuid, text, text) rename to fill_tournament_day_live_courts_explicit';
  end if;
end;
$migration$;

create or replace function public.fill_tournament_day_live_courts(
  p_run_id uuid,
  p_operation_key text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_action text;
begin
  select operation.action into v_action
    from public.tournament_admin_operations as operation
    join public.tournament_day_live_runs as run
      on run.id = p_run_id
     and run.club_id = operation.club_id
     and operation.entity_type = 'tournament_registration_day'
     and operation.entity_id = pg_catalog.concat_ws(
       ':', run.tournament_id::text, run.registration_day_id
     )
     and operation.lock_scope = pg_catalog.concat_ws(
       ':', 'tournament', run.tournament_id::text,
       'day', run.registration_day_id
     )
   where operation.operation_key = p_operation_key
     and operation.surface = 'tournament_live'
     and operation.status = 'intent';

  if v_action is distinct from 'tournament_day_live_auto_fill_courts' then
    return '[]'::jsonb;
  end if;
  return public.fill_tournament_day_live_courts_explicit(
    p_run_id, p_operation_key, p_actor
  );
end;
$function$;

create or replace function public.admin_assign_tournament_day_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_action text,
  p_game_id text,
  p_court_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_queue_entry_version bigint,
  p_expected_game_updated_at timestamptz,
  p_expected_court_version bigint,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_action text := pg_catalog.upper(pg_catalog.btrim(coalesce(p_action, '')));
  v_command_action text;
  v_operation_action text;
  v_run public.tournament_day_live_runs%rowtype;
  v_court public.tournament_day_live_courts%rowtype;
  v_queue public.tournament_day_live_queue%rowtype;
  v_day_draw public.tournament_day_live_draws%rowtype;
  v_game public.tournament_games%rowtype;
  v_intent jsonb;
  v_player_ids integer[];
  v_locked_team_count integer;
  v_now timestamptz := pg_catalog.clock_timestamp();
begin
  if v_action = 'NEXT_OPEN' then
    v_command_action := 'assign_next_court';
  elsif v_action = 'SELECTED' then
    v_command_action := 'assign_game_to_court';
  else
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_ACTION: NEXT_OPEN or SELECTED is required.';
  end if;
  v_operation_action := 'tournament_day_live_' || v_command_action;
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint, v_operation_action
  );

  if v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,action}' is distinct from v_command_action
     or v_intent #> '{payload,payload}' is distinct from (
       case when v_action = 'NEXT_OPEN'
         then pg_catalog.jsonb_build_object('game_id', p_game_id)
         else pg_catalog.jsonb_build_object(
           'game_id', p_game_id, 'court_id', p_court_id
         )
       end
     )
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,queue_entry_version}')::bigint
          is distinct from p_expected_queue_entry_version
     or nullif(v_intent #>> '{payload,expected,game_version}', '')::timestamptz
          is distinct from p_expected_game_updated_at
     or nullif(v_intent #>> '{payload,expected,court_version}', '')::bigint
          is distinct from p_expected_court_version then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: court-assignment arguments do not match durable intent.';
  end if;
  if (v_action = 'NEXT_OPEN' and (
        nullif(pg_catalog.btrim(coalesce(p_court_id, '')), '') is not null
        or p_expected_court_version is not null
      ))
     or (v_action = 'SELECTED' and (
       nullif(pg_catalog.btrim(coalesce(p_court_id, '')), '') is null
       or p_expected_court_version is null
     )) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_COURT: reviewed court arguments are invalid.';
  end if;

  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state = 'ACTIVE'
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: day run changed or is not active.';
  end if;

  if v_action = 'NEXT_OPEN' then
    select court.* into v_court
      from public.tournament_day_live_courts as court
     where court.run_id = v_run.id
       and court.state = 'OPEN'
       and not exists (
         select 1
           from public.tournament_day_live_queue as occupied
          where occupied.run_id = v_run.id
            and occupied.court_id = court.id
            and occupied.released_at is null
       )
     order by court.position, court.id
     for update
     limit 1;
  else
    select court.* into v_court
      from public.tournament_day_live_courts as court
     where court.run_id = v_run.id
       and court.id::text = p_court_id
       and court.version = p_expected_court_version
       and court.state = 'OPEN'
       and not exists (
         select 1
           from public.tournament_day_live_queue as occupied
          where occupied.run_id = v_run.id
            and occupied.court_id = court.id
            and occupied.released_at is null
       )
     for update;
  end if;
  if v_court.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: reviewed court is no longer open.';
  end if;

  select queue.* into v_queue
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
     and queue.tournament_id = v_run.tournament_id
     and queue.registration_day_id = v_run.registration_day_id
     and queue.game_id::text = p_game_id
     and queue.version = p_expected_queue_entry_version
     and queue.state = 'WAITING'
     and queue.court_id is null
     and queue.released_at is null
   for update;
  if v_queue.id is null or v_queue.team_a_id is null or v_queue.team_b_id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: queued matchup changed or is no longer waiting.';
  end if;

  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.id = v_queue.day_draw_id
     and day_draw.run_id = v_run.id
     and day_draw.draw_id = v_queue.draw_id
     and day_draw.state = 'ACTIVE'
   for update;
  if v_day_draw.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: queued draw is no longer active.';
  end if;

  select game.* into v_game
    from public.tournament_games as game
    join public.tournament_registration_days as source_day
      on source_day.id = v_run.registration_day_id
     and source_day.tournament_id = v_run.tournament_id::text
     and source_day.enabled is true
    join public.tournament_event_draws as source_draw
      on source_draw.id = v_queue.draw_id
     and source_draw.tournament_id = v_run.tournament_id
     and source_draw.updated_at = v_day_draw.source_draw_updated_at
     and pg_catalog.upper(coalesce(source_draw.status, 'DRAFT')) not in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     and coalesce(source_draw.hidden_from_primary_ops, false) is false
     and pg_catalog.upper(coalesce(source_draw.draw_kind, 'STANDARD')) = 'STANDARD'
    join public.tournament_event_options as source_event
      on source_event.id = source_draw.event_option_id
     and source_event.id = game.event_option_id
     and source_event.tournament_id = v_run.tournament_id::text
     and source_event.enabled is true
     and pg_catalog.upper(coalesce(source_event.status, 'DRAFT')) not in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
   where game.id = v_queue.game_id
     and game.tournament_id = v_run.tournament_id
     and game.draw_id = v_queue.draw_id
     and game.registration_day_id = v_run.registration_day_id
     and game.team_a_id = v_queue.team_a_id
     and game.team_b_id = v_queue.team_b_id
     and game.finalized_at is null
     and game.updated_at = p_expected_game_updated_at
     and (
       case
         when source_draw.registration_day_id is not null then
           source_draw.registration_day_id = v_run.registration_day_id
           and case
             when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
              and pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) > 0
               then source_event.scheduled_day_ids ? source_draw.registration_day_id
             when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
               then source_event.registration_day_id is null
                 or source_event.registration_day_id = source_draw.registration_day_id
             else false
           end
         else case
           when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
            and pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 1
             then source_event.scheduled_day_ids ? v_run.registration_day_id
           when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
            and pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 0
             then source_event.registration_day_id = v_run.registration_day_id
           else false
         end
       end
     ) is true
   for share of game, source_day, source_draw, source_event;
  if v_game.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: game or scheduled draw setup changed.';
  end if;

  perform team.id
    from public.tournament_teams as team
   where team.tournament_id = v_run.tournament_id
     and team.draw_id = v_queue.draw_id
     and team.registration_day_id = v_run.registration_day_id
     and team.event_option_id = v_game.event_option_id
     and team.id in (v_queue.team_a_id, v_queue.team_b_id)
   order by team.id
   for share;
  get diagnostics v_locked_team_count = row_count;
  if v_locked_team_count <> 2 then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: exact game teams changed.';
  end if;

  if exists (
    select 1
      from public.tournament_day_live_queue as earlier
     where earlier.run_id = v_run.id
       and earlier.draw_id = v_queue.draw_id
       and earlier.priority < v_queue.priority
       and earlier.state not in ('COMPLETED', 'WITHDRAWN')
       and (
         earlier.team_a_id in (v_queue.team_a_id, v_queue.team_b_id)
         or earlier.team_b_id in (v_queue.team_a_id, v_queue.team_b_id)
       )
  ) or (
    v_game.stage = 'PLAYOFF' and exists (
      select 1
        from public.tournament_day_live_queue as rr_queue
        join public.tournament_games as rr_game
          on rr_game.id = rr_queue.game_id
         and rr_game.tournament_id = rr_queue.tournament_id
         and rr_game.draw_id = rr_queue.draw_id
       where rr_queue.run_id = v_run.id
         and rr_queue.draw_id = v_queue.draw_id
         and rr_game.stage = 'ROUND_ROBIN'
         and rr_queue.state not in ('COMPLETED', 'WITHDRAWN')
    )
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: an earlier required game is unfinished.';
  end if;

  v_player_ids := public.tournament_day_live_game_player_ids(
    v_run.tournament_id::text, v_queue.draw_id, v_queue.game_id
  );
  if pg_catalog.cardinality(v_player_ids) not in (2, 4)
     or (select pg_catalog.count(*) from pg_catalog.unnest(v_player_ids) as player(id))
          <> (select pg_catalog.count(distinct id) from pg_catalog.unnest(v_player_ids) as player(id))
     or not public.tournament_day_live_players_ready(
       v_run.club_id, v_run.tournament_id::text,
       v_run.registration_day_id, v_player_ids
     ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: participants are no longer ready.';
  end if;

  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.queue_id = v_queue.id
   order by claim.player_id, claim.id
   for update;
  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.tournament_id = v_run.tournament_id
     and claim.released_at is null
     and claim.player_id = any(v_player_ids)
   order by claim.player_id, claim.id
   for update;
  if exists (
    select 1
      from public.tournament_day_live_participant_claims as claim
     where claim.tournament_id = v_run.tournament_id
       and claim.released_at is null
       and claim.player_id = any(v_player_ids)
       and claim.queue_id <> v_queue.id
  ) or exists (
    select 1
      from public.tournament_day_live_participant_claims as claim
     where claim.queue_id = v_queue.id
       and not (claim.player_id = any(v_player_ids))
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_CLAIM_STALE: participant claim set changed.';
  end if;

  update public.tournament_day_live_queue as queue
     set state = 'ON_COURT',
         court_id = v_court.id,
         blocker_code = null,
         blocker_detail = null,
         held_at = null,
         called_at = null,
         started_at = v_now,
         released_at = null,
         completed_at = null,
         version = queue.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where queue.id = v_queue.id
     and queue.version = p_expected_queue_entry_version
     and queue.state = 'WAITING';
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_STALE: queue changed while assigning.';
  end if;

  update public.tournament_day_live_participant_claims as claim
     set state = 'ON_COURT',
         version = claim.version + 1,
         last_operation_key = p_operation_key,
         claimed_at = v_now,
         released_at = null,
         released_by = null,
         updated_at = v_now
   where claim.queue_id = v_queue.id
     and claim.player_id = any(v_player_ids)
     and claim.released_at is not null;
  insert into public.tournament_day_live_participant_claims (
    run_id, tournament_id, registration_day_id, queue_id, game_id,
    player_id, state, version, last_operation_key, claimed_at, updated_at
  )
  select
    v_run.id, v_run.tournament_id, v_run.registration_day_id,
    v_queue.id, v_queue.game_id, player.id,
    'ON_COURT', 1, p_operation_key, v_now, v_now
  from pg_catalog.unnest(v_player_ids) as player(id)
  on conflict (queue_id, player_id) do nothing;

  if (
    select pg_catalog.count(*)
      from public.tournament_day_live_participant_claims as claim
     where claim.queue_id = v_queue.id
       and claim.game_id = v_queue.game_id
       and claim.state = 'ON_COURT'
       and claim.released_at is null
       and claim.player_id = any(v_player_ids)
  ) <> pg_catalog.cardinality(v_player_ids) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ASSIGNMENT_CLAIM_STALE: exact participant claims could not be acquired.';
  end if;

  update public.tournament_day_live_courts as court
     set version = court.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where court.id = v_court.id;
  update public.tournament_day_live_draws as day_draw
     set last_assigned_at = v_now,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where day_draw.id = v_day_draw.id;
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where run.id = v_run.id
   returning * into v_run;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'run', pg_catalog.to_jsonb(v_run),
    'action', v_action,
    'assignments', pg_catalog.jsonb_build_array(
      pg_catalog.jsonb_build_object(
        'queue_id', v_queue.id,
        'game_id', v_queue.game_id,
        'draw_id', v_queue.draw_id,
        'court_id', v_court.id,
        'court_key', v_court.court_key
      )
    )
  );
end;
$function$;

create or replace function public.admin_reassign_tournament_day_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_action text,
  p_game_id text,
  p_target_court_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_queue_entry_version bigint,
  p_expected_game_updated_at timestamptz,
  p_expected_source_court_version bigint,
  p_expected_target_court_version bigint,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_action text := pg_catalog.upper(pg_catalog.btrim(coalesce(p_action, '')));
  v_command_action text;
  v_operation_action text;
  v_run public.tournament_day_live_runs%rowtype;
  v_queue public.tournament_day_live_queue%rowtype;
  v_game public.tournament_games%rowtype;
  v_source_court public.tournament_day_live_courts%rowtype;
  v_target_court public.tournament_day_live_courts%rowtype;
  v_intent jsonb;
  v_player_ids integer[];
  v_now timestamptz := pg_catalog.clock_timestamp();
begin
  if v_action = 'REQUEUE' then
    v_command_action := 'requeue_game';
  elsif v_action = 'MOVE' then
    v_command_action := 'move_game_to_court';
  else
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_ACTION: REQUEUE or MOVE is required.';
  end if;
  v_operation_action := 'tournament_day_live_' || v_command_action;
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint, v_operation_action
  );

  if v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,action}' is distinct from v_command_action
     or v_intent #> '{payload,payload}' is distinct from (
       case when v_action = 'REQUEUE'
         then pg_catalog.jsonb_build_object('game_id', p_game_id)
         else pg_catalog.jsonb_build_object(
           'game_id', p_game_id, 'court_id', p_target_court_id
         )
       end
     )
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,queue_entry_version}')::bigint
          is distinct from p_expected_queue_entry_version
     or nullif(v_intent #>> '{payload,expected,game_version}', '')::timestamptz
          is distinct from p_expected_game_updated_at
     or nullif(v_intent #>> '{payload,expected,court_version}', '')::bigint
          is distinct from p_expected_source_court_version
     or nullif(v_intent #>> '{payload,expected,target_court_version}', '')::bigint
          is distinct from p_expected_target_court_version then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: court-reassignment arguments do not match durable intent.';
  end if;
  if (v_action = 'REQUEUE' and (
       nullif(pg_catalog.btrim(coalesce(p_target_court_id, '')), '') is not null
       or p_expected_target_court_version is not null
     ))
     or (v_action = 'MOVE' and (
       nullif(pg_catalog.btrim(coalesce(p_target_court_id, '')), '') is null
       or p_expected_target_court_version is null
     )) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_COURT: target court arguments are invalid.';
  end if;

  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state = 'ACTIVE'
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_STALE: day run changed or is not active.';
  end if;

  select queue.* into v_queue
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
     and queue.tournament_id = v_run.tournament_id
     and queue.registration_day_id = v_run.registration_day_id
     and queue.game_id::text = p_game_id
     and queue.version = p_expected_queue_entry_version
     and queue.state = 'ON_COURT'
     and queue.court_id is not null
     and queue.released_at is null
   for update;
  if v_queue.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_STALE: game is no longer on the reviewed court.';
  end if;

  select game.* into v_game
    from public.tournament_games as game
   where game.id = v_queue.game_id
     and game.tournament_id = v_run.tournament_id
     and game.draw_id = v_queue.draw_id
     and game.registration_day_id = v_run.registration_day_id
     and game.team_a_id = v_queue.team_a_id
     and game.team_b_id = v_queue.team_b_id
     and game.finalized_at is null
     and game.updated_at = p_expected_game_updated_at
   for share;
  if v_game.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_STALE: game changed after assignment review.';
  end if;

  select court.* into v_source_court
    from public.tournament_day_live_courts as court
   where court.id = v_queue.court_id
     and court.run_id = v_run.id
     and court.version = p_expected_source_court_version
     and court.state = 'OPEN'
   for update;
  if v_source_court.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_STALE: source court changed after review.';
  end if;

  if v_action = 'MOVE' then
    if p_target_court_id = v_source_court.id::text then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_COURT: destination must differ from the source court.';
    end if;
    select court.* into v_target_court
      from public.tournament_day_live_courts as court
     where court.id::text = p_target_court_id
       and court.run_id = v_run.id
       and court.version = p_expected_target_court_version
       and court.state = 'OPEN'
       and not exists (
         select 1
           from public.tournament_day_live_queue as occupied
          where occupied.run_id = v_run.id
            and occupied.court_id = court.id
            and occupied.released_at is null
       )
     for update;
    if v_target_court.id is null then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_STALE: destination court is no longer open.';
    end if;
  end if;

  v_player_ids := public.tournament_day_live_game_player_ids(
    v_run.tournament_id::text, v_queue.draw_id, v_queue.game_id
  );
  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.queue_id = v_queue.id
   order by claim.player_id, claim.id
   for update;
  if pg_catalog.cardinality(v_player_ids) not in (2, 4)
     or exists (
       select 1
         from public.tournament_day_live_participant_claims as claim
        where claim.queue_id = v_queue.id
          and claim.released_at is null
          and not (claim.player_id = any(v_player_ids))
     )
     or (
       select pg_catalog.count(*)
         from public.tournament_day_live_participant_claims as claim
        where claim.queue_id = v_queue.id
          and claim.game_id = v_queue.game_id
          and claim.state = 'ON_COURT'
          and claim.released_at is null
          and claim.player_id = any(v_player_ids)
     ) <> pg_catalog.cardinality(v_player_ids) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_CLAIM_STALE: exact active participant claims changed.';
  end if;

  if v_action = 'REQUEUE' then
    update public.tournament_day_live_participant_claims as claim
       set state = 'RELEASED',
           released_at = v_now,
           released_by = p_actor,
           version = claim.version + 1,
           last_operation_key = p_operation_key,
           updated_at = v_now
     where claim.queue_id = v_queue.id
       and claim.released_at is null;
    update public.tournament_day_live_queue as queue
       set state = 'WAITING',
           court_id = null,
           blocker_code = null,
           blocker_detail = null,
           eligible_since = coalesce(queue.eligible_since, v_now),
           held_at = null,
           called_at = null,
           started_at = null,
           released_at = null,
           completed_at = null,
           version = queue.version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = v_now
     where queue.id = v_queue.id
       and queue.version = p_expected_queue_entry_version
       and queue.state = 'ON_COURT';
    if not found then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_STALE: queue changed while returning the game.';
    end if;
  else
    update public.tournament_day_live_queue as queue
       set court_id = v_target_court.id,
           version = queue.version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = v_now
     where queue.id = v_queue.id
       and queue.version = p_expected_queue_entry_version
       and queue.state = 'ON_COURT';
    if not found then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_REASSIGNMENT_STALE: queue changed while moving the game.';
    end if;
    update public.tournament_day_live_courts as court
       set version = court.version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = v_now
     where court.id = v_target_court.id;
  end if;

  update public.tournament_day_live_courts as court
     set version = court.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where court.id = v_source_court.id;
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where run.id = v_run.id
   returning * into v_run;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'run', pg_catalog.to_jsonb(v_run),
    'action', v_action,
    'game_id', v_queue.game_id,
    'source_court_id', v_source_court.id,
    'target_court_id', case when v_action = 'MOVE' then v_target_court.id else null end,
    'requeued', v_action = 'REQUEUE'
  );
end;
$function$;

revoke execute on function public.fill_tournament_day_live_courts_explicit(
  uuid, text, text
) from public, anon, authenticated, service_role;
revoke execute on function public.fill_tournament_day_live_courts(
  uuid, text, text
) from public, anon, authenticated, service_role;
revoke execute on function public.admin_assign_tournament_day_game_cas(
  text, text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) from public, anon, authenticated;
revoke execute on function public.admin_reassign_tournament_day_game_cas(
  text, text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, bigint, text, text, text
) from public, anon, authenticated;

grant execute on function public.fill_tournament_day_live_courts_explicit(
  uuid, text, text
) to service_role;
grant execute on function public.fill_tournament_day_live_courts(
  uuid, text, text
) to service_role;
grant execute on function public.admin_assign_tournament_day_game_cas(
  text, text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) to service_role;
grant execute on function public.admin_reassign_tournament_day_game_cas(
  text, text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, bigint, text, text, text
) to service_role;

comment on function public.fill_tournament_day_live_courts(uuid, text, text) is
  'Compatibility gate: only the explicit guarded auto-fill operation may invoke the legacy multi-court allocator; activation and progression leave games queued.';
comment on function public.fill_tournament_day_live_courts_explicit(uuid, text, text) is
  'Service-role-only legacy multi-court allocator retained behind the explicit auto-fill operation gate.';
comment on function public.admin_assign_tournament_day_game_cas(
  text, text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) is
  'Service-role-only one-game court assignment with durable intent, run/queue/game/court CAS, eligibility recheck, and exact participant claims.';
comment on function public.admin_reassign_tournament_day_game_cas(
  text, text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, bigint, text, text, text
) is
  'Service-role-only atomic move or requeue of an on-court game with exact source/destination court and participant-claim evidence.';

comment on function public.admin_score_release_tournament_day_game_cas(
  text, text, text, text, bigint, bigint, bigint, timestamptz,
  timestamptz, jsonb, jsonb, text, text, text
) is
  'Atomic Day Live score, progression, claim release, and court release. Subsequent games remain queued until explicitly assigned.';
comment on function public.admin_record_non_played_tournament_day_game_cas(
  text, text, text, text, bigint, bigint, bigint, bigint,
  timestamptz, timestamptz, jsonb, jsonb, text, text, text,
  text, text, text, text
) is
  'Atomic Day Live non-played result, progression, claim release, and court release. Subsequent games remain queued until explicitly assigned.';
