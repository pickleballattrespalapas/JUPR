-- Tournament Day Live occupied-court reservations.
--
-- An operator may reserve one eligible matchup as the next game for an
-- occupied court. The reservation owns the matchup's participant claims so it
-- cannot be assigned elsewhere, and an AFTER UPDATE trigger promotes it in the
-- same transaction that releases or moves the current court assignment.

alter table public.tournament_day_live_queue
  add column if not exists reserved_court_id uuid null
    references public.tournament_day_live_courts(id) on delete restrict,
  add column if not exists reserved_at timestamptz null;

alter table public.tournament_day_live_queue
  drop constraint if exists tournament_day_live_queue_state_chk,
  drop constraint if exists tournament_day_live_queue_court_state_chk,
  drop constraint if exists tournament_day_live_queue_reservation_time_chk;

alter table public.tournament_day_live_queue
  add constraint tournament_day_live_queue_state_chk
    check (state in (
      'WAITING', 'RESERVED', 'HELD', 'CALLED', 'ON_COURT',
      'COMPLETED', 'BLOCKED', 'WITHDRAWN'
    )),
  add constraint tournament_day_live_queue_court_state_chk check (
    (
      state in ('HELD', 'CALLED', 'ON_COURT')
      and court_id is not null
      and reserved_court_id is null
      and released_at is null
    )
    or
    (
      state = 'RESERVED'
      and court_id is null
      and reserved_court_id is not null
      and released_at is null
    )
    or
    (
      state not in ('RESERVED', 'HELD', 'CALLED', 'ON_COURT')
      and court_id is null
      and reserved_court_id is null
    )
  ),
  add constraint tournament_day_live_queue_reservation_time_chk check (
    (state = 'RESERVED' and reserved_at is not null)
    or (state <> 'RESERVED' and reserved_at is null)
  );

alter table public.tournament_day_live_participant_claims
  drop constraint if exists tournament_day_live_claims_state_chk;

alter table public.tournament_day_live_participant_claims
  add constraint tournament_day_live_claims_state_chk
    check (state in ('RESERVED', 'HELD', 'CALLED', 'ON_COURT', 'RELEASED'));

create unique index if not exists uq_tournament_day_live_queue_reserved_court
  on public.tournament_day_live_queue (run_id, reserved_court_id)
  where state = 'RESERVED'
    and reserved_court_id is not null
    and released_at is null;

create or replace function public.admin_reserve_tournament_day_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
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
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_reserve_game_for_court'
  );
  if v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,action}' is distinct from 'reserve_game_for_court'
     or v_intent #> '{payload,payload}' is distinct from
          pg_catalog.jsonb_build_object('game_id', p_game_id, 'court_id', p_court_id)
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: court-reservation arguments do not match durable intent.';
  end if;
  if nullif(pg_catalog.btrim(coalesce(p_court_id, '')), '') is null
     or p_expected_court_version is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_COURT: an occupied reviewed court is required.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: day run changed or is not active.';
  end if;

  select court.* into v_court
    from public.tournament_day_live_courts as court
   where court.run_id = v_run.id
     and court.id::text = p_court_id
     and court.version = p_expected_court_version
     and court.state = 'OPEN'
     and exists (
       select 1
         from public.tournament_day_live_queue as occupied
        where occupied.run_id = v_run.id
          and occupied.court_id = court.id
          and occupied.state in ('HELD', 'CALLED', 'ON_COURT')
          and occupied.released_at is null
     )
     and not exists (
       select 1
         from public.tournament_day_live_queue as reserved
        where reserved.run_id = v_run.id
          and reserved.reserved_court_id = court.id
          and reserved.state = 'RESERVED'
          and reserved.released_at is null
     )
   for update;
  if v_court.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: reviewed court is no longer occupied or already has a next matchup.';
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
     and queue.reserved_court_id is null
     and queue.released_at is null
   for update;
  if v_queue.id is null or v_queue.team_a_id is null or v_queue.team_b_id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: queued matchup changed or is no longer waiting.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: queued draw is no longer active.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: game or scheduled draw setup changed.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: exact game teams changed.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: an earlier required game is unfinished.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: participants are no longer ready.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_CLAIM_STALE: participant claim set changed.';
  end if;

  update public.tournament_day_live_queue as queue
     set state = 'RESERVED',
         reserved_court_id = v_court.id,
         reserved_at = v_now,
         blocker_code = null,
         blocker_detail = null,
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
     and queue.state = 'WAITING';
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: queue changed while reserving the court.';
  end if;

  update public.tournament_day_live_participant_claims as claim
     set state = 'RESERVED',
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
    'RESERVED', 1, p_operation_key, v_now, v_now
  from pg_catalog.unnest(v_player_ids) as player(id)
  on conflict (queue_id, player_id) do nothing;

  if (
    select pg_catalog.count(*)
      from public.tournament_day_live_participant_claims as claim
     where claim.queue_id = v_queue.id
       and claim.game_id = v_queue.game_id
       and claim.state = 'RESERVED'
       and claim.released_at is null
       and claim.player_id = any(v_player_ids)
  ) <> pg_catalog.cardinality(v_player_ids) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_CLAIM_STALE: exact participant claims could not be acquired.';
  end if;

  update public.tournament_day_live_courts as court
     set version = court.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where court.id = v_court.id;
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
    'game_id', v_queue.game_id,
    'reserved_court_id', v_court.id,
    'reserved', true
  );
end;
$function$;

create or replace function public.admin_cancel_tournament_day_court_reservation_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_game_id text,
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
  v_run public.tournament_day_live_runs%rowtype;
  v_queue public.tournament_day_live_queue%rowtype;
  v_game public.tournament_games%rowtype;
  v_court public.tournament_day_live_courts%rowtype;
  v_intent jsonb;
  v_player_ids integer[];
  v_now timestamptz := pg_catalog.clock_timestamp();
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_requeue_game'
  );
  if v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,action}' is distinct from 'requeue_game'
     or v_intent #> '{payload,payload}' is distinct from
          pg_catalog.jsonb_build_object('game_id', p_game_id)
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,queue_entry_version}')::bigint
          is distinct from p_expected_queue_entry_version
     or nullif(v_intent #>> '{payload,expected,game_version}', '')::timestamptz
          is distinct from p_expected_game_updated_at
     or nullif(v_intent #>> '{payload,expected,court_version}', '')::bigint
          is distinct from p_expected_court_version
     or nullif(v_intent #>> '{payload,expected,target_court_version}', '') is not null then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: reservation-cancel arguments do not match durable intent.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: day run changed or is not active.';
  end if;

  select queue.* into v_queue
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
     and queue.game_id::text = p_game_id
     and queue.version = p_expected_queue_entry_version
     and queue.state = 'RESERVED'
     and queue.court_id is null
     and queue.reserved_court_id is not null
     and queue.released_at is null
   for update;
  if v_queue.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: matchup is no longer waiting for the reviewed court.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: reserved game changed after review.';
  end if;

  select court.* into v_court
    from public.tournament_day_live_courts as court
   where court.id = v_queue.reserved_court_id
     and court.run_id = v_run.id
     and court.version = p_expected_court_version
   for update;
  if v_court.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: reserved court changed after review.';
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
          and claim.state = 'RESERVED'
          and claim.released_at is null
          and claim.player_id = any(v_player_ids)
     ) <> pg_catalog.cardinality(v_player_ids) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_CLAIM_STALE: exact reserved participant claims changed.';
  end if;

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
         reserved_court_id = null,
         reserved_at = null,
         blocker_code = null,
         blocker_detail = null,
         eligible_since = coalesce(queue.eligible_since, v_now),
         version = queue.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where queue.id = v_queue.id
     and queue.version = p_expected_queue_entry_version
     and queue.state = 'RESERVED';
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_RESERVATION_STALE: reservation changed while returning it to the queue.';
  end if;
  update public.tournament_day_live_courts as court
     set version = court.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = v_now
   where court.id = v_court.id;
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
    'game_id', v_queue.game_id,
    'source_court_id', v_court.id,
    'requeued', true,
    'reservation_canceled', true
  );
end;
$function$;

create or replace function public.promote_tournament_day_court_reservation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_reservation public.tournament_day_live_queue%rowtype;
  v_run public.tournament_day_live_runs%rowtype;
  v_court public.tournament_day_live_courts%rowtype;
  v_day_draw public.tournament_day_live_draws%rowtype;
  v_game public.tournament_games%rowtype;
  v_player_ids integer[];
  v_ready boolean := true;
  v_now timestamptz := pg_catalog.clock_timestamp();
begin
  if old.court_id is null
     or old.released_at is not null
     or old.state not in ('HELD', 'CALLED', 'ON_COURT') then
    return new;
  end if;
  if new.court_id is not distinct from old.court_id
     and new.released_at is null
     and new.state in ('HELD', 'CALLED', 'ON_COURT') then
    return new;
  end if;

  select queue.* into v_reservation
    from public.tournament_day_live_queue as queue
   where queue.run_id = old.run_id
     and queue.reserved_court_id = old.court_id
     and queue.state = 'RESERVED'
     and queue.court_id is null
     and queue.released_at is null
   for update;
  if v_reservation.id is null then
    return new;
  end if;

  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.id = old.run_id
     and run.state = 'ACTIVE'
   for update;
  select court.* into v_court
    from public.tournament_day_live_courts as court
   where court.id = old.court_id
     and court.run_id = old.run_id
     and court.state = 'OPEN'
   for update;
  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.id = v_reservation.day_draw_id
     and day_draw.run_id = old.run_id
     and day_draw.state in ('ACTIVE', 'PAUSED')
   for update;
  select game.* into v_game
    from public.tournament_games as game
   where game.id = v_reservation.game_id
     and game.tournament_id = v_reservation.tournament_id
     and game.draw_id = v_reservation.draw_id
     and game.registration_day_id = v_reservation.registration_day_id
     and game.team_a_id = v_reservation.team_a_id
     and game.team_b_id = v_reservation.team_b_id
     and game.finalized_at is null
   for share;

  if v_run.id is null
     or v_court.id is null
     or v_day_draw.id is null
     or v_game.id is null
     or exists (
       select 1
         from public.tournament_day_live_queue as occupied
        where occupied.run_id = old.run_id
          and occupied.court_id = old.court_id
          and occupied.released_at is null
          and occupied.state in ('HELD', 'CALLED', 'ON_COURT')
     ) then
    v_ready := false;
  end if;

  v_player_ids := public.tournament_day_live_game_player_ids(
    v_reservation.tournament_id::text,
    v_reservation.draw_id,
    v_reservation.game_id
  );
  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.queue_id = v_reservation.id
   order by claim.player_id, claim.id
   for update;
  if pg_catalog.cardinality(v_player_ids) not in (2, 4)
     or (
       select pg_catalog.count(*)
         from public.tournament_day_live_participant_claims as claim
        where claim.queue_id = v_reservation.id
          and claim.game_id = v_reservation.game_id
          and claim.state = 'RESERVED'
          and claim.released_at is null
          and claim.player_id = any(v_player_ids)
     ) <> pg_catalog.cardinality(v_player_ids)
     or exists (
       select 1
         from public.tournament_day_live_participant_claims as claim
        where claim.queue_id = v_reservation.id
          and claim.released_at is null
          and not (claim.player_id = any(v_player_ids))
     )
     or not public.tournament_day_live_players_ready(
       v_run.club_id,
       v_reservation.tournament_id::text,
       v_reservation.registration_day_id,
       v_player_ids
     ) then
    v_ready := false;
  end if;

  if not v_ready then
    update public.tournament_day_live_participant_claims as claim
       set state = 'RELEASED',
           released_at = v_now,
           released_by = new.updated_by,
           version = claim.version + 1,
           last_operation_key = new.last_operation_key,
           updated_at = v_now
     where claim.queue_id = v_reservation.id
       and claim.released_at is null;
    update public.tournament_day_live_queue as queue
       set state = 'WAITING',
           reserved_court_id = null,
           reserved_at = null,
           blocker_code = null,
           blocker_detail = null,
           eligible_since = coalesce(queue.eligible_since, v_now),
           version = queue.version + 1,
           last_operation_key = new.last_operation_key,
           updated_by = new.updated_by,
           updated_at = v_now
     where queue.id = v_reservation.id
       and queue.state = 'RESERVED';
    if v_court.id is not null then
      update public.tournament_day_live_courts as court
         set version = court.version + 1,
             last_operation_key = new.last_operation_key,
             updated_by = new.updated_by,
             updated_at = v_now
       where court.id = v_court.id;
    end if;
    if v_run.id is not null then
      update public.tournament_day_live_runs as run
         set version = run.version + 1,
             queue_version = run.queue_version + 1,
             last_operation_key = new.last_operation_key,
             updated_by = new.updated_by,
             updated_at = v_now
       where run.id = v_run.id;
    end if;
    return new;
  end if;

  update public.tournament_day_live_queue as queue
     set state = 'ON_COURT',
         court_id = old.court_id,
         reserved_court_id = null,
         reserved_at = null,
         blocker_code = null,
         blocker_detail = null,
         held_at = null,
         called_at = null,
         started_at = v_now,
         released_at = null,
         completed_at = null,
         version = queue.version + 1,
         last_operation_key = new.last_operation_key,
         updated_by = new.updated_by,
         updated_at = v_now
   where queue.id = v_reservation.id
     and queue.state = 'RESERVED'
     and queue.reserved_court_id = old.court_id;
  if not found then
    return new;
  end if;
  update public.tournament_day_live_participant_claims as claim
     set state = 'ON_COURT',
         version = claim.version + 1,
         last_operation_key = new.last_operation_key,
         claimed_at = v_now,
         updated_at = v_now
   where claim.queue_id = v_reservation.id
     and claim.state = 'RESERVED'
     and claim.released_at is null;
  update public.tournament_day_live_courts as court
     set version = court.version + 1,
         last_operation_key = new.last_operation_key,
         updated_by = new.updated_by,
         updated_at = v_now
   where court.id = old.court_id;
  update public.tournament_day_live_draws as day_draw
     set last_assigned_at = v_now,
         version = day_draw.version + 1,
         last_operation_key = new.last_operation_key,
         updated_by = new.updated_by,
         updated_at = v_now
   where day_draw.id = v_day_draw.id;
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = new.last_operation_key,
         updated_by = new.updated_by,
         updated_at = v_now
   where run.id = v_run.id;
  return new;
end;
$function$;

drop trigger if exists trg_tournament_day_live_promote_reserved_court
  on public.tournament_day_live_queue;
create trigger trg_tournament_day_live_promote_reserved_court
after update of state, court_id, released_at
on public.tournament_day_live_queue
for each row
execute function public.promote_tournament_day_court_reservation();

revoke execute on function public.admin_reserve_tournament_day_game_cas(
  text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) from public, anon, authenticated;
revoke execute on function public.admin_cancel_tournament_day_court_reservation_cas(
  text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) from public, anon, authenticated;
revoke all on function public.promote_tournament_day_court_reservation()
  from public, anon, authenticated, service_role;

grant execute on function public.admin_reserve_tournament_day_game_cas(
  text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) to service_role;
grant execute on function public.admin_cancel_tournament_day_court_reservation_cas(
  text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) to service_role;

comment on function public.admin_reserve_tournament_day_game_cas(
  text, text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) is
  'Service-role-only occupied-court reservation with durable intent, exact queue/game/court CAS, readiness recheck, and reserved participant claims.';
comment on function public.admin_cancel_tournament_day_court_reservation_cas(
  text, text, text, text, bigint, bigint, bigint,
  timestamptz, bigint, text, text, text
) is
  'Service-role-only cancellation of an occupied-court reservation back to its original queue priority with exact claim release.';
comment on function public.promote_tournament_day_court_reservation() is
  'Queue trigger that atomically promotes the one reserved next matchup when its selected court is released or vacated.';
