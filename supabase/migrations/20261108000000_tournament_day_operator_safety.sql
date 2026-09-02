-- Tournament Day operator safety: reviewed score policy metadata and atomic
-- non-played outcomes. FastAPI authorizes the JWT; these functions remain
-- service-role-only, invoker-security mutation paths with durable intent/CAS.

alter table public.tournament_games
  add column if not exists result_type text not null default 'PLAYED',
  add column if not exists result_note text null,
  add column if not exists result_recorded_by text null,
  add column if not exists score_review_json jsonb not null default '{}'::jsonb;

alter table public.tournament_games
  drop constraint if exists tournament_games_result_type_chk;
alter table public.tournament_games
  add constraint tournament_games_result_type_chk
  check (result_type in ('PLAYED', 'FORFEIT', 'NO_SHOW', 'RETIREMENT'));
alter table public.tournament_games
  drop constraint if exists tournament_games_result_note_length_chk;
alter table public.tournament_games
  add constraint tournament_games_result_note_length_chk
  check (result_note is null or pg_catalog.char_length(result_note) between 1 and 500);
alter table public.tournament_games
  drop constraint if exists tournament_games_score_review_object_chk;
alter table public.tournament_games
  add constraint tournament_games_score_review_object_chk
  check (pg_catalog.jsonb_typeof(score_review_json) = 'object');

create or replace function public.guard_tournament_game_day_live_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw_id uuid := case when tg_op = 'DELETE' then old.draw_id else new.draw_id end;
  v_game_id uuid := case when tg_op = 'INSERT' then new.id else old.id end;
  v_operation_key text := nullif(
    pg_catalog.current_setting('jupr.day_live_operation_key', true), ''
  );
  v_fenced boolean := false;
  v_authorized boolean := false;
begin
  if v_draw_id is null then
    return case when tg_op = 'DELETE' then old else new end;
  end if;
  if tg_op = 'INSERT' then
    select exists (
      select 1
        from public.tournament_day_live_draws as day_draw
        join public.tournament_day_live_runs as run on run.id = day_draw.run_id
       where day_draw.draw_id = v_draw_id
         and day_draw.state in ('ACTIVE', 'PAUSED')
         and run.state in ('ACTIVE', 'PAUSED')
    ) into v_fenced;
  else
    select exists (
      select 1
        from public.tournament_day_live_queue as queue
        join public.tournament_day_live_runs as run on run.id = queue.run_id
       where queue.game_id = v_game_id
         and run.state in ('ACTIVE', 'PAUSED')
    ) into v_fenced;
  end if;
  if not v_fenced then
    return case when tg_op = 'DELETE' then old else new end;
  end if;
  if v_operation_key is not null then
    select exists (
      select 1
        from public.tournament_admin_operations as operation
        join public.tournament_day_live_runs as run
          on run.club_id = operation.club_id
         and operation.entity_type = 'tournament_registration_day'
         and operation.entity_id = pg_catalog.concat_ws(
           ':', run.tournament_id::text, run.registration_day_id
         )
         and operation.lock_scope = pg_catalog.concat_ws(
           ':', 'tournament', run.tournament_id::text,
           'day', run.registration_day_id
         )
        join public.tournament_day_live_draws as day_draw
          on day_draw.run_id = run.id and day_draw.draw_id = v_draw_id
       where operation.operation_key = v_operation_key
         and operation.surface = 'tournament_live'
         and operation.status = 'intent'
         and operation.action in (
           'tournament_day_live_score_and_release',
           'tournament_day_live_correct_completed_score',
           'tournament_day_live_record_non_played_result',
           'tournament_day_live_generate_playoffs'
         )
    ) into v_authorized;
  end if;
  if not v_authorized then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_SCORE_PATH_REQUIRED: use an authorized day command for a day-fenced game.';
  end if;
  return case when tg_op = 'DELETE' then old else new end;
end;
$function$;

create or replace function public.apply_tournament_day_result_metadata()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation_key text := nullif(
    pg_catalog.current_setting('jupr.day_live_operation_key', true), ''
  );
  v_action text;
  v_payload jsonb;
  v_actor text;
  v_target_game_id text;
  v_outcome jsonb;
begin
  if v_operation_key is null then
    return new;
  end if;
  select operation.action, operation.request_json->'payload', operation.created_by
    into v_action, v_payload, v_actor
    from public.tournament_admin_operations as operation
   where operation.operation_key = v_operation_key
     and operation.surface = 'tournament_live'
     and operation.status = 'intent';
  if not found then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: result metadata requires exact durable intent.';
  end if;
  v_target_game_id := v_payload #>> '{payload,game_id}';
  if new.id::text is distinct from v_target_game_id then
    return new;
  end if;
  if v_action in (
    'tournament_day_live_score_and_release',
    'tournament_day_live_correct_completed_score'
  ) then
    new.result_type := 'PLAYED';
    new.result_note := null;
    new.result_recorded_by := v_actor;
    new.score_review_json := coalesce(
      v_payload #> '{score_evidence,score_review}', '{}'::jsonb
    );
  elsif v_action = 'tournament_day_live_record_non_played_result' then
    v_outcome := v_payload #> '{score_evidence,outcome}';
    if pg_catalog.jsonb_typeof(v_outcome) is distinct from 'object' then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: non-played outcome metadata is missing.';
    end if;
    new.result_type := v_outcome->>'result_type';
    new.result_note := v_outcome->>'result_note';
    new.result_recorded_by := v_outcome->>'result_recorded_by';
    new.score_review_json := coalesce(
      v_payload #> '{score_evidence,score_review}', '{}'::jsonb
    ) || pg_catalog.jsonb_build_object(
      'synthetic_progression_score', true,
      'rating_publish_eligible', false
    );
  end if;
  return new;
end;
$function$;

drop trigger if exists trg_01_tournament_games_day_result_metadata
  on public.tournament_games;
create trigger trg_01_tournament_games_day_result_metadata
before update on public.tournament_games
for each row execute function public.apply_tournament_day_result_metadata();

create or replace function public.admin_record_non_played_tournament_day_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_game_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_queue_entry_version bigint,
  p_expected_court_version bigint,
  p_expected_game_updated_at timestamptz,
  p_expected_draw_updated_at timestamptz,
  p_game_patch jsonb,
  p_dependency_updates jsonb,
  p_result_type text,
  p_winner_team_id text,
  p_result_note text,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text,
  p_actor_role text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_queue public.tournament_day_live_queue%rowtype;
  v_intent jsonb;
  v_score_result jsonb;
  v_assignments jsonb := '[]'::jsonb;
  v_dependency_ids uuid[] := '{}'::uuid[];
  v_court_id uuid;
  v_score_a integer;
  v_score_b integer;
  v_result_type text := pg_catalog.upper(coalesce(p_result_type, ''));
  v_actor_role text := pg_catalog.lower(coalesce(p_actor_role, ''));
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_record_non_played_result'
  );
  if v_result_type not in ('FORFEIT', 'NO_SHOW', 'RETIREMENT')
     or nullif(pg_catalog.btrim(coalesce(p_result_note, '')), '') is null
     or pg_catalog.char_length(p_result_note) > 500 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OUTCOME: a supported type and operator note are required.';
  end if;
  if v_actor_role not in ('super_admin', 'club_owner', 'organizer', 'scorekeeper')
     or v_intent #>> '{payload,operator_authorization,email}'
          is distinct from pg_catalog.lower(pg_catalog.btrim(p_actor))
     or v_intent #>> '{payload,operator_authorization,role}'
          is distinct from v_actor_role
     or v_intent->>'actor' is distinct from p_actor
     or not exists (
       select 1
         from public.admin_role_assignments as assignment
        where assignment.club_id = p_club_id
          and pg_catalog.lower(assignment.email) = pg_catalog.lower(pg_catalog.btrim(p_actor))
          and assignment.role = v_actor_role
     ) then
    raise exception using errcode = '42501',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_AUTHORIZATION: current score-entry staff authorization is required.';
  end if;
  if v_intent #>> '{payload,action}' is distinct from 'record_non_played_result'
     or v_intent #> '{payload,payload}' is distinct from pg_catalog.jsonb_build_object(
       'game_id', p_game_id,
       'result_type', v_result_type,
       'winner_team_id', p_winner_team_id,
       'result_note', p_result_note
     )
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,queue_entry_version}')::bigint
          is distinct from p_expected_queue_entry_version
     or nullif(v_intent #>> '{payload,expected,court_version}', '')::bigint
          is distinct from p_expected_court_version
     or nullif(v_intent #>> '{payload,expected,game_version}', '')::timestamptz
          is distinct from p_expected_game_updated_at
     or nullif(v_intent #>> '{payload,score_evidence,source_draw_updated_at}', '')::timestamptz
          is distinct from p_expected_draw_updated_at
     or v_intent #> '{payload,score_evidence,game_patch}' is distinct from p_game_patch
     or v_intent #> '{payload,score_evidence,dependency_updates}'
          is distinct from p_dependency_updates
     or v_intent #>> '{payload,score_evidence,outcome,result_type}' is distinct from v_result_type
     or v_intent #>> '{payload,score_evidence,outcome,winner_team_id}' is distinct from p_winner_team_id
     or v_intent #>> '{payload,score_evidence,outcome,result_note}' is distinct from p_result_note
     or v_intent #>> '{payload,score_evidence,outcome,result_recorded_by}' is distinct from p_actor
     or (v_intent #>> '{payload,score_evidence,outcome,synthetic_progression_score}')::boolean is distinct from true
     or (v_intent #>> '{payload,score_evidence,outcome,rating_publish_eligible}')::boolean is distinct from false then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: outcome arguments do not match durable intent.';
  end if;
  if pg_catalog.jsonb_typeof(p_game_patch) is distinct from 'object'
     or pg_catalog.jsonb_typeof(p_dependency_updates) is distinct from 'array'
     or nullif(p_winner_team_id, '') is null
     or nullif(p_game_patch->>'winner_team_id', '') is null
     or nullif(p_game_patch->>'winner_team_id', '') is distinct from p_winner_team_id
     or nullif(p_game_patch->>'loser_team_id', '') is null
     or nullif(p_game_patch->>'finalized_at', '') is null
     or nullif(p_game_patch->>'score_a', '') is null
     or nullif(p_game_patch->>'score_b', '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OUTCOME: exact synthetic progression evidence is required.';
  end if;
  begin
    v_score_a := (p_game_patch->>'score_a')::integer;
    v_score_b := (p_game_patch->>'score_b')::integer;
  exception when invalid_text_representation or numeric_value_out_of_range then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OUTCOME: exact synthetic progression evidence is required.';
  end;
  if v_score_a < 0 or v_score_b < 0 or v_score_a = v_score_b then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OUTCOME: exact synthetic progression evidence is required.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: day run changed or is not active.';
  end if;

  select queue.* into v_queue
    from public.tournament_day_live_queue as queue
    join public.tournament_day_live_draws as day_draw
      on day_draw.id = queue.day_draw_id
     and day_draw.state in ('ACTIVE', 'PAUSED')
   where queue.run_id = v_run.id
     and queue.game_id::text = p_game_id
     and queue.version = p_expected_queue_entry_version
     and queue.state in ('WAITING', 'HELD', 'CALLED', 'ON_COURT', 'BLOCKED')
     and queue.released_at is null
   for update of queue;
  if v_queue.id is null
     or v_queue.team_a_id is null
     or v_queue.team_b_id is null
     or p_winner_team_id not in (v_queue.team_a_id::text, v_queue.team_b_id::text)
     or p_game_patch->>'loser_team_id' not in (v_queue.team_a_id::text, v_queue.team_b_id::text) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: queued matchup or reviewed outcome changed.';
  end if;
  -- Keep the two valid outcome shapes explicit. An unparenthesized SQL CASE
  -- inside a PL/pgSQL IF condition is parsed at its inner THEN token and makes
  -- CREATE FUNCTION fail at end-of-input on PostgreSQL 17.
  if not (
    (
      p_winner_team_id = v_queue.team_a_id::text
      and p_game_patch->>'loser_team_id' = v_queue.team_b_id::text
      and v_score_a > v_score_b
    )
    or (
      p_winner_team_id = v_queue.team_b_id::text
      and p_game_patch->>'loser_team_id' = v_queue.team_a_id::text
      and v_score_b > v_score_a
    )
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: queued matchup or reviewed outcome changed.';
  end if;
  v_court_id := v_queue.court_id;
  if v_court_id is null and p_expected_court_version is not null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed court assignment changed.';
  end if;

  select coalesce(
           pg_catalog.array_agg(nullif(dependency.value->>'id', '')::uuid
             order by nullif(dependency.value->>'id', '')::uuid),
           '{}'::uuid[]
         )
    into v_dependency_ids
    from pg_catalog.jsonb_array_elements(p_dependency_updates) as dependency(value);

  perform pg_catalog.set_config('jupr.day_live_operation_key', p_operation_key, true);
  v_score_result := public.admin_score_tournament_game_cas(
    p_club_id,
    p_tournament_id,
    p_game_id,
    p_expected_game_updated_at,
    p_expected_draw_updated_at,
    p_game_patch,
    p_dependency_updates
  );

  -- Match the established game -> court -> participant lock order used by
  -- score-and-release so concurrent score/outcome attempts cannot deadlock.
  if v_court_id is not null then
    perform court.id
      from public.tournament_day_live_courts as court
     where court.id = v_court_id
       and court.run_id = v_run.id
       and court.version = p_expected_court_version
     for update;
    if not found then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: assigned court changed.';
    end if;
  end if;

  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.queue_id = v_queue.id and claim.released_at is null
   order by claim.player_id
   for update;
  update public.tournament_day_live_participant_claims as claim
     set state = 'RELEASED',
         released_at = pg_catalog.clock_timestamp(),
         released_by = p_actor,
         version = claim.version + 1,
         last_operation_key = p_operation_key,
         updated_at = pg_catalog.clock_timestamp()
   where claim.queue_id = v_queue.id and claim.released_at is null;
  update public.tournament_day_live_queue as queue
     set state = 'COMPLETED',
         court_id = null,
         blocker_code = null,
         blocker_detail = null,
         released_at = pg_catalog.clock_timestamp(),
         completed_at = pg_catalog.clock_timestamp(),
         version = queue.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where queue.id = v_queue.id;
  if v_court_id is not null then
    update public.tournament_day_live_courts as court
       set version = court.version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = pg_catalog.clock_timestamp()
     where court.id = v_court_id;
  end if;
  update public.tournament_day_live_queue as queue
     set team_a_id = game.team_a_id,
         team_b_id = game.team_b_id,
         state = 'WAITING',
         blocker_code = null,
         blocker_detail = null,
         eligible_since = pg_catalog.clock_timestamp(),
         version = queue.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
    from public.tournament_games as game
   where queue.run_id = v_run.id
     and queue.draw_id = v_queue.draw_id
     and queue.game_id = game.id
     and queue.game_id = any(v_dependency_ids)
     and queue.state = 'BLOCKED'
     and game.finalized_at is null
     and game.team_a_id is not null
     and game.team_b_id is not null;
  update public.tournament_day_live_draws as day_draw
     set source_draw_updated_at = draw.updated_at,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
    from public.tournament_event_draws as draw
   where day_draw.run_id = v_run.id
     and day_draw.draw_id = v_queue.draw_id
     and draw.id = day_draw.draw_id;
  v_assignments := public.fill_tournament_day_live_courts(
    v_run.id, p_operation_key, p_actor
  );
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return v_score_result || pg_catalog.jsonb_build_object(
    'run', pg_catalog.to_jsonb(v_run),
    'released_court_id', v_court_id,
    'assignments', v_assignments,
    'result_type', v_result_type,
    'rating_publish_eligible', false,
    'non_played_result', true
  );
end;
$function$;

revoke all on function public.apply_tournament_day_result_metadata()
  from public, anon, authenticated, service_role;
revoke all on function public.admin_record_non_played_tournament_day_game_cas(
  text, text, text, text, bigint, bigint, bigint, bigint,
  timestamptz, timestamptz, jsonb, jsonb, text, text, text,
  text, text, text, text
) from public, anon, authenticated;
grant execute on function public.apply_tournament_day_result_metadata()
  to service_role;
grant execute on function public.admin_record_non_played_tournament_day_game_cas(
  text, text, text, text, bigint, bigint, bigint, bigint,
  timestamptz, timestamptz, jsonb, jsonb, text, text, text,
  text, text, text, text
) to service_role;

comment on column public.tournament_games.result_type is
  'PLAYED or a visibly non-played Day Live outcome. Non-played rows carry only synthetic progression scores and are never rating-publish eligible.';
comment on function public.admin_record_non_played_tournament_day_game_cas(
  text, text, text, text, bigint, bigint, bigint, bigint,
  timestamptz, timestamptz, jsonb, jsonb, text, text, text,
  text, text, text, text
) is
  'Service-role-only atomic forfeit/no-show/retirement result: exact durable intent, staff role, run/queue/game/court CAS, claim/court release, dependency promotion, and refill.';

-- Ordinary Ops/Tournament Live score entry needs the same CAS behavior as the
-- established score RPC, plus durable result-policy evidence. Keep Day Live on
-- the established RPC because its exact durable intent trigger is authoritative
-- for score/outcome metadata.
create or replace function public.admin_score_tournament_game_result_cas(
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
set search_path = ''
as $function$
declare
  v_result jsonb;
  v_saved public.tournament_games%rowtype;
  v_scoring_format text;
begin
  if pg_catalog.jsonb_typeof(p_game_patch) is distinct from 'object'
     or p_game_patch ->> 'result_type' is distinct from 'PLAYED'
     or not (p_game_patch ? 'result_note')
     or p_game_patch ->> 'result_note' is not null
     or nullif(
          pg_catalog.btrim(p_game_patch ->> 'result_recorded_by'), ''
        ) is null
     or pg_catalog.jsonb_typeof(p_game_patch -> 'score_review_json')
          is distinct from 'object'
     or p_game_patch #>> '{score_review_json,accepted}' is distinct from 'true'
     or nullif(p_game_patch #>> '{score_review_json,score_a}', '')::integer
          is distinct from (p_game_patch ->> 'score_a')::integer
     or nullif(p_game_patch #>> '{score_review_json,score_b}', '')::integer
          is distinct from (p_game_patch ->> 'score_b')::integer
     or coalesce(
          p_game_patch #>> '{score_review_json,status}', ''
        ) not in ('ordinary', 'unusual')
     or coalesce(
          p_game_patch #>> '{score_review_json,scoring_format}', ''
        ) not in ('GAME_TO_11', 'GAME_TO_15', 'GAME_TO_21', 'BEST_2_OF_3')
     or (
       p_game_patch #>> '{score_review_json,status}' = 'unusual'
       and p_game_patch #>> '{score_review_json,acknowledged}'
             is distinct from 'true'
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_SCORE_METADATA_INVALID';
  end if;

  v_result := public.admin_score_tournament_game_cas(
    p_club_id,
    p_tournament_id,
    p_game_id,
    p_expected_updated_at,
    p_expected_draw_updated_at,
    p_game_patch,
    p_dependency_updates
  );

  select pg_catalog.upper(
           coalesce(
             nullif(pg_catalog.btrim(event.scoring_override), ''),
             nullif(pg_catalog.btrim(event.scoring_default), '')
           )
         ) into v_scoring_format
    from public.tournament_games as game
    left join public.tournament_event_draws as draw on draw.id = game.draw_id
    join public.tournament_event_options as event
      on event.id::text = coalesce(
           nullif(game.event_option_id::text, ''),
           draw.event_option_id::text
         )
     and event.tournament_id::text = game.tournament_id::text
    join public.tournaments as tournament
      on tournament.id = game.tournament_id
     and tournament.club_id::text = p_club_id
   where game.id::text = p_game_id
     and game.tournament_id::text = p_tournament_id
   for share of event;
  if not found
     or v_scoring_format is distinct from
          p_game_patch #>> '{score_review_json,scoring_format}' then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_SCORE_FORMAT_STALE';
  end if;

  -- The base CAS leaves result_type unchanged. Refuse to reinterpret a
  -- forfeit/no-show/retirement through ordinary score entry; the surrounding
  -- exception rolls the base score update back atomically.
  if pg_catalog.upper(
       coalesce(v_result #>> '{game,result_type}', 'PLAYED')
     ) <> 'PLAYED' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_OUTCOME_CONVERSION_REQUIRED';
  end if;

  update public.tournament_games as game
     set result_type = p_game_patch ->> 'result_type',
         result_note = p_game_patch ->> 'result_note',
         result_recorded_by = p_game_patch ->> 'result_recorded_by',
         score_review_json = p_game_patch -> 'score_review_json'
   where game.id::text = p_game_id
     and game.tournament_id::text = p_tournament_id
   returning game.* into v_saved;
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_GAME_STALE';
  end if;

  return pg_catalog.jsonb_set(
    v_result,
    '{game}',
    pg_catalog.to_jsonb(v_saved),
    true
  );
end;
$function$;

-- Results import similarly delegates all draw/roster/podium locking and CAS to
-- the established transaction, then stores the reviewed policy evidence before
-- that transaction commits.
create or replace function public.admin_import_tournament_draw_results_with_metadata_cas(
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
set search_path = ''
as $function$
declare
  v_game_patch jsonb;
  v_result jsonb;
  v_scoring_format text;
begin
  if pg_catalog.jsonb_typeof(p_games) is distinct from 'array' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_RESULTS_METADATA_INVALID';
  end if;
  for v_game_patch in
    select value from pg_catalog.jsonb_array_elements(p_games)
  loop
    if (v_game_patch ->> 'score_a') is not null
       and (v_game_patch ->> 'score_b') is not null
       and (
         v_game_patch ->> 'result_type' is distinct from 'PLAYED'
         or not (v_game_patch ? 'result_note')
         or v_game_patch ->> 'result_note' is not null
         or nullif(
              pg_catalog.btrim(v_game_patch ->> 'result_recorded_by'), ''
            ) is null
         or pg_catalog.jsonb_typeof(v_game_patch -> 'score_review_json')
              is distinct from 'object'
         or v_game_patch #>> '{score_review_json,accepted}' is distinct from 'true'
         or nullif(
              v_game_patch #>> '{score_review_json,score_a}', ''
            )::integer is distinct from (v_game_patch ->> 'score_a')::integer
         or nullif(
              v_game_patch #>> '{score_review_json,score_b}', ''
            )::integer is distinct from (v_game_patch ->> 'score_b')::integer
         or coalesce(
              v_game_patch #>> '{score_review_json,status}', ''
            ) not in ('ordinary', 'unusual')
         or coalesce(
              v_game_patch #>> '{score_review_json,scoring_format}', ''
            ) not in ('GAME_TO_11', 'GAME_TO_15', 'GAME_TO_21', 'BEST_2_OF_3')
         or (
           v_game_patch #>> '{score_review_json,status}' = 'unusual'
           and v_game_patch #>> '{score_review_json,acknowledged}'
                 is distinct from 'true'
         )
       ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_RESULTS_METADATA_INVALID';
    end if;
  end loop;

  v_result := public.admin_import_tournament_draw_results_cas(
    p_club_id,
    p_tournament_id,
    p_draw_id,
    p_expected_draw_updated_at,
    p_import_mode,
    p_new_players,
    p_teams,
    p_games,
    p_podium
  );

  select pg_catalog.upper(
           coalesce(
             nullif(pg_catalog.btrim(event.scoring_override), ''),
             nullif(pg_catalog.btrim(event.scoring_default), '')
           )
         ) into v_scoring_format
    from public.tournament_event_draws as draw
    join public.tournament_event_options as event
      on event.id::text = draw.event_option_id::text
     and event.tournament_id::text = draw.tournament_id::text
    join public.tournaments as tournament
      on tournament.id = draw.tournament_id
     and tournament.club_id::text = p_club_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
   for share of event;
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_RESULTS_FORMAT_STALE';
  end if;

  for v_game_patch in
    select value from pg_catalog.jsonb_array_elements(p_games)
  loop
    if (v_game_patch ->> 'score_a') is not null
       and (v_game_patch ->> 'score_b') is not null then
      if v_scoring_format is distinct from
           v_game_patch #>> '{score_review_json,scoring_format}' then
        raise exception using errcode = '40001',
          message = 'JUPR_TOURNAMENT_RESULTS_FORMAT_STALE';
      end if;
      update public.tournament_games as game
         set result_type = v_game_patch ->> 'result_type',
             result_note = v_game_patch ->> 'result_note',
             result_recorded_by = v_game_patch ->> 'result_recorded_by',
             score_review_json = v_game_patch -> 'score_review_json'
       where game.id::text = v_game_patch ->> 'id'
         and game.tournament_id::text = p_tournament_id
         and game.draw_id::text = p_draw_id;
      if not found then
        raise exception using errcode = '40001',
          message = 'JUPR_TOURNAMENT_RESULTS_METADATA_STALE';
      end if;
    end if;
  end loop;

  return v_result;
end;
$function$;

revoke all on function public.admin_score_tournament_game_result_cas(
  text, text, text, timestamptz, timestamptz, jsonb, jsonb
) from public, anon, authenticated;
grant execute on function public.admin_score_tournament_game_result_cas(
  text, text, text, timestamptz, timestamptz, jsonb, jsonb
) to service_role;
revoke all on function public.admin_import_tournament_draw_results_with_metadata_cas(
  text, text, text, timestamptz, text, jsonb, jsonb, jsonb, jsonb
) from public, anon, authenticated;
grant execute on function public.admin_import_tournament_draw_results_with_metadata_cas(
  text, text, text, timestamptz, text, jsonb, jsonb, jsonb, jsonb
) to service_role;

comment on function public.admin_score_tournament_game_result_cas(
  text, text, text, timestamptz, timestamptz, jsonb, jsonb
) is
  'Atomic ordinary score CAS with PLAYED-only outcome conversion guard and durable score-policy metadata.';
comment on function public.admin_import_tournament_draw_results_with_metadata_cas(
  text, text, text, timestamptz, text, jsonb, jsonb, jsonb, jsonb
) is
  'Atomic reviewed results import with durable PLAYED result and score-policy metadata.';

-- Four-player team child games use the same per-event score policy. Preserve
-- the review on every child game, including non-rating skinny-relay tiebreaks,
-- and mirror it to the canonical rating game when one exists.
alter table public.tournament_team_match_games
  add column if not exists score_review_json jsonb not null default '{}'::jsonb;
alter table public.tournament_team_match_games
  drop constraint if exists tournament_team_match_games_score_review_object_chk;
alter table public.tournament_team_match_games
  add constraint tournament_team_match_games_score_review_object_chk
  check (pg_catalog.jsonb_typeof(score_review_json) = 'object');

create or replace function public.admin_score_tournament_team_match_game_reviewed_cas(
  p_club_id text,
  p_tournament_id text,
  p_match_game_id text,
  p_score_a integer,
  p_score_b integer,
  p_score_review jsonb,
  p_expected_game_version integer,
  p_expected_matchup_version integer,
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
  v_child public.tournament_team_match_games%rowtype;
  v_games jsonb;
  v_result jsonb;
  v_scoring_format text;
begin
  if pg_catalog.jsonb_typeof(p_score_review) is distinct from 'object'
     or p_score_review ->> 'accepted' is distinct from 'true'
     or nullif(p_score_review ->> 'score_a', '')::integer
          is distinct from p_score_a
     or nullif(p_score_review ->> 'score_b', '')::integer
          is distinct from p_score_b
     or coalesce(p_score_review ->> 'status', '')
          not in ('ordinary', 'unusual')
     or coalesce(p_score_review ->> 'scoring_format', '')
          not in ('GAME_TO_11', 'GAME_TO_15', 'GAME_TO_21', 'BEST_2_OF_3')
     or (
       p_score_review ->> 'status' = 'unusual'
       and p_score_review ->> 'acknowledged' is distinct from 'true'
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_TEAM_SCORE_REVIEW_INVALID';
  end if;

  v_result := public.admin_score_tournament_team_match_game_cas(
    p_club_id,
    p_tournament_id,
    p_match_game_id,
    p_score_a,
    p_score_b,
    p_expected_game_version,
    p_expected_matchup_version,
    p_operation_key,
    p_request_fingerprint,
    p_actor
  );

  -- Bind the Python review to the event row in this same transaction. Taking
  -- the row lock after the established score RPC preserves its lock order; an
  -- event-format change that wins the race is observed here and rolls the score
  -- operation back.
  select pg_catalog.upper(
           coalesce(
             nullif(pg_catalog.btrim(event.scoring_override), ''),
             nullif(pg_catalog.btrim(event.scoring_default), '')
           )
         ) into v_scoring_format
    from public.tournament_team_match_games as child
    join public.tournament_team_matchups as matchup
      on matchup.id = child.matchup_id
     and matchup.tournament_id = child.tournament_id
    join public.tournament_event_options as event
      on event.id::text = matchup.event_option_id::text
     and event.tournament_id::text = child.tournament_id::text
    join public.tournaments as tournament
      on tournament.id = child.tournament_id
     and tournament.club_id::text = p_club_id
   where child.id::text = p_match_game_id
     and child.tournament_id::text = p_tournament_id
   for share of event;
  if not found
     or v_scoring_format is distinct from
          p_score_review ->> 'scoring_format' then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_SCORE_FORMAT_STALE';
  end if;

  select child.* into v_child
    from public.tournament_team_match_games as child
   where child.id::text = p_match_game_id
     and child.tournament_id::text = p_tournament_id
   for update;
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_MATCH_GAME_STALE';
  end if;
  if v_child.tournament_game_id is not null and exists (
    select 1
      from public.tournament_games as game
     where game.id = v_child.tournament_game_id
       and pg_catalog.upper(coalesce(game.result_type, 'PLAYED'))
             <> 'PLAYED'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_OUTCOME_CONVERSION_REQUIRED';
  end if;

  update public.tournament_team_match_games as child
     set score_review_json = p_score_review
   where child.id = v_child.id
   returning child.* into v_child;

  if v_child.tournament_game_id is not null then
    update public.tournament_games as game
       set result_type = 'PLAYED',
           result_note = null,
           result_recorded_by = p_actor,
           score_review_json = p_score_review
     where game.id = v_child.tournament_game_id
       and game.tournament_id::text = p_tournament_id;
    if not found then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_TEAM_RATING_GAME_STALE';
    end if;
  end if;

  select coalesce(
           pg_catalog.jsonb_agg(
             pg_catalog.to_jsonb(child) order by child.game_order
           ),
           '[]'::jsonb
         ) into v_games
    from public.tournament_team_match_games as child
   where child.matchup_id = v_child.matchup_id;
  v_result := pg_catalog.jsonb_set(v_result, '{games}', v_games, true)
    || pg_catalog.jsonb_build_object('score_review', p_score_review);
  update public.tournament_team_operations as operation
     set result_json = v_result,
         updated_at = pg_catalog.clock_timestamp()
   where operation.operation_key = p_operation_key
     and operation.request_fingerprint = p_request_fingerprint
     and operation.status = 'COMPLETED';
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_TEAM_OPERATION_COMPLETION_STALE';
  end if;
  return v_result;
end;
$function$;

revoke all on function public.admin_score_tournament_team_match_game_reviewed_cas(
  text, text, text, integer, integer, jsonb, integer, integer, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_score_tournament_team_match_game_reviewed_cas(
  text, text, text, integer, integer, jsonb, integer, integer, text, text, text
) to service_role;

comment on column public.tournament_team_match_games.score_review_json is
  'Durable configured-format review and unusual-score acknowledgement for the finalized team child game.';
comment on function public.admin_score_tournament_team_match_game_reviewed_cas(
  text, text, text, integer, integer, jsonb, integer, integer, text, text, text
) is
  'Atomic four-player team game score CAS with accepted per-event review evidence, durable child metadata, and canonical rating-game metadata.';
