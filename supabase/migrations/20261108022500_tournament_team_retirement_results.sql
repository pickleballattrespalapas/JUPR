-- Durable standard-draw team retirement.
--
-- A retirement has two deliberately separate projections:
--   * played game scores remain unchanged and rating-publish eligible;
--   * tournament standings treat every round-robin meeting involving the
--     retired team as a configured max-score loss.
-- Remaining unplayed games are finalized as non-rated retirement losses. An
-- AFTER UPDATE cascade carries that rule through later playoff dependencies.

alter table public.tournament_teams
  add column if not exists competition_status text not null default 'ACTIVE',
  add column if not exists retired_at timestamptz null,
  add column if not exists retired_by text null,
  add column if not exists retirement_note text null,
  add column if not exists retirement_max_score integer null,
  add column if not exists retirement_game_id uuid null;

alter table public.tournament_teams
  drop constraint if exists tournament_teams_competition_status_chk,
  drop constraint if exists tournament_teams_retirement_note_length_chk,
  drop constraint if exists tournament_teams_retirement_max_score_chk,
  drop constraint if exists tournament_teams_retirement_state_chk;

alter table public.tournament_teams
  add constraint tournament_teams_competition_status_chk
    check (competition_status in ('ACTIVE', 'RETIRED')),
  add constraint tournament_teams_retirement_note_length_chk
    check (
      retirement_note is null
      or pg_catalog.char_length(retirement_note) between 1 and 500
    ),
  add constraint tournament_teams_retirement_max_score_chk
    check (retirement_max_score is null or retirement_max_score > 0),
  add constraint tournament_teams_retirement_state_chk
    check (
      (
        competition_status = 'ACTIVE'
        and retired_at is null
        and retired_by is null
        and retirement_note is null
        and retirement_max_score is null
        and retirement_game_id is null
      )
      or
      (
        competition_status = 'RETIRED'
        and retired_at is not null
        and nullif(pg_catalog.btrim(retired_by), '') is not null
        and retirement_max_score is not null
        and retirement_game_id is not null
      )
    );

comment on column public.tournament_teams.competition_status is
  'ACTIVE or RETIRED for standard-draw progression and standings.';
comment on column public.tournament_teams.retirement_max_score is
  'Configured scoring maximum used for every retirement standings loss and remaining synthetic result.';

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
    new.result_note := nullif(
      pg_catalog.btrim(coalesce(v_outcome->>'result_note', '')), ''
    );
    new.result_recorded_by := v_outcome->>'result_recorded_by';
    new.score_review_json := coalesce(
      v_payload #> '{score_evidence,score_review}', '{}'::jsonb
    ) || pg_catalog.jsonb_build_object(
      'synthetic_progression_score', true,
      'rating_publish_eligible', false,
      'non_playing_team_id', v_outcome->>'non_playing_team_id'
    );
  end if;
  return new;
end;
$function$;

-- Finalize a newly concrete game containing a retired team, release any day
-- resources it held, and carry its winner/loser through playoff dependencies.
-- The same trigger also promotes an ordinary newly concrete dependency from
-- BLOCKED to WAITING. The enclosing guarded day command increments the run's
-- aggregate versions once after all nested updates finish.
create or replace function public.cascade_tournament_team_retirement()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation_key text := nullif(
    pg_catalog.current_setting('jupr.day_live_operation_key', true), ''
  );
  v_operation_actor text;
  v_team_a public.tournament_teams%rowtype;
  v_team_b public.tournament_teams%rowtype;
  v_a_retired boolean := false;
  v_b_retired boolean := false;
  v_target integer;
  v_winner uuid;
  v_loser uuid;
  v_note text;
  v_recorded_by text;
  v_queue public.tournament_day_live_queue%rowtype;
begin
  if v_operation_key is null or new.team_a_id is null or new.team_b_id is null then
    return null;
  end if;
  select operation.created_by
    into v_operation_actor
    from public.tournament_admin_operations as operation
   where operation.operation_key = v_operation_key
     and operation.surface = 'tournament_live'
     and operation.status = 'intent'
     and operation.action in (
       'tournament_day_live_record_non_played_result',
       'tournament_day_live_score_and_release'
     );
  if not found then
    return null;
  end if;

  select team.* into v_team_a
    from public.tournament_teams as team
   where team.id = new.team_a_id
     and team.tournament_id = new.tournament_id
     and team.draw_id is not distinct from new.draw_id;
  select team.* into v_team_b
    from public.tournament_teams as team
   where team.id = new.team_b_id
     and team.tournament_id = new.tournament_id
     and team.draw_id is not distinct from new.draw_id;
  if v_team_a.id is null or v_team_b.id is null then
    return null;
  end if;
  v_a_retired := v_team_a.competition_status = 'RETIRED';
  v_b_retired := v_team_b.competition_status = 'RETIRED';

  if new.finalized_at is null and (v_a_retired or v_b_retired) then
    v_target := greatest(
      case when v_a_retired then v_team_a.retirement_max_score else 0 end,
      case when v_b_retired then v_team_b.retirement_max_score else 0 end
    );
    if v_target is null or v_target <= 0 then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT: configured max score is missing.';
    end if;
    if v_a_retired and not v_b_retired then
      v_winner := v_team_b.id;
      v_loser := v_team_a.id;
      v_note := v_team_a.retirement_note;
      v_recorded_by := v_team_a.retired_by;
    elsif v_b_retired and not v_a_retired then
      v_winner := v_team_a.id;
      v_loser := v_team_b.id;
      v_note := v_team_b.retirement_note;
      v_recorded_by := v_team_b.retired_by;
    elsif v_team_a.retired_at <= v_team_b.retired_at then
      v_winner := v_team_b.id;
      v_loser := v_team_a.id;
      v_note := v_team_a.retirement_note;
      v_recorded_by := v_team_a.retired_by;
    else
      v_winner := v_team_a.id;
      v_loser := v_team_b.id;
      v_note := v_team_b.retirement_note;
      v_recorded_by := v_team_b.retired_by;
    end if;
    update public.tournament_games as game
       set score_a = case when v_winner = new.team_a_id then v_target else 0 end,
           score_b = case when v_winner = new.team_b_id then v_target else 0 end,
           winner_team_id = v_winner,
           loser_team_id = v_loser,
           finalized_at = pg_catalog.clock_timestamp(),
           result_type = 'RETIREMENT',
           result_note = nullif(pg_catalog.btrim(coalesce(v_note, '')), ''),
           result_recorded_by = coalesce(
             nullif(pg_catalog.btrim(v_recorded_by), ''),
             v_operation_actor
           ),
           score_review_json = pg_catalog.jsonb_build_object(
             'synthetic_progression_score', true,
             'rating_publish_eligible', false,
             'retirement_cascade', true,
             'non_playing_team_id', v_loser::text
           ),
           updated_at = pg_catalog.clock_timestamp()
     where game.id = new.id
       and game.finalized_at is null;
    return null;
  end if;

  if old.finalized_at is null
     and new.finalized_at is not null
     and new.result_type = 'RETIREMENT' then
    for v_queue in
      select queue.*
        from public.tournament_day_live_queue as queue
       where queue.game_id = new.id
         and queue.state not in ('COMPLETED', 'WITHDRAWN')
       order by queue.id
       for update
    loop
      update public.tournament_day_live_participant_claims as claim
         set state = 'RELEASED',
             released_at = pg_catalog.clock_timestamp(),
             released_by = v_operation_actor,
             version = claim.version + 1,
             last_operation_key = v_operation_key,
             updated_at = pg_catalog.clock_timestamp()
       where claim.queue_id = v_queue.id
         and claim.released_at is null;
      update public.tournament_day_live_queue as queue
         set team_a_id = new.team_a_id,
             team_b_id = new.team_b_id,
             state = 'COMPLETED',
             court_id = null,
             reserved_court_id = null,
             reserved_at = null,
             blocker_code = null,
             blocker_detail = null,
             released_at = pg_catalog.clock_timestamp(),
             completed_at = pg_catalog.clock_timestamp(),
             version = queue.version + 1,
             last_operation_key = v_operation_key,
             updated_by = v_operation_actor,
             updated_at = pg_catalog.clock_timestamp()
       where queue.id = v_queue.id;
      update public.tournament_day_live_courts as court
         set version = court.version + 1,
             last_operation_key = v_operation_key,
             updated_by = v_operation_actor,
             updated_at = pg_catalog.clock_timestamp()
       where court.id in (v_queue.court_id, v_queue.reserved_court_id);
    end loop;

    if nullif(new.playoff_game_code, '') is not null then
      update public.tournament_games as downstream
         set team_a_id = case
               when downstream.team_a_source->>'winnerOf' = new.playoff_game_code
                 then new.winner_team_id
               when downstream.team_a_source->>'loserOf' = new.playoff_game_code
                 then new.loser_team_id
               else downstream.team_a_id
             end,
             team_b_id = case
               when downstream.team_b_source->>'winnerOf' = new.playoff_game_code
                 then new.winner_team_id
               when downstream.team_b_source->>'loserOf' = new.playoff_game_code
                 then new.loser_team_id
               else downstream.team_b_id
             end,
             updated_at = pg_catalog.clock_timestamp()
       where downstream.tournament_id = new.tournament_id
         and downstream.draw_id is not distinct from new.draw_id
         and downstream.finalized_at is null
         and downstream.id <> new.id
         and (
           downstream.team_a_source->>'winnerOf' = new.playoff_game_code
           or downstream.team_a_source->>'loserOf' = new.playoff_game_code
           or downstream.team_b_source->>'winnerOf' = new.playoff_game_code
           or downstream.team_b_source->>'loserOf' = new.playoff_game_code
         );
    end if;
    return null;
  end if;

  if new.finalized_at is null
     and not v_a_retired
     and not v_b_retired then
    update public.tournament_day_live_queue as queue
       set team_a_id = new.team_a_id,
           team_b_id = new.team_b_id,
           state = 'WAITING',
           blocker_code = null,
           blocker_detail = null,
           eligible_since = pg_catalog.clock_timestamp(),
           version = queue.version + 1,
           last_operation_key = v_operation_key,
           updated_by = v_operation_actor,
           updated_at = pg_catalog.clock_timestamp()
     where queue.game_id = new.id
       and queue.state = 'BLOCKED'
       and queue.court_id is null
       and queue.reserved_court_id is null
       and queue.released_at is null;
  end if;
  return null;
end;
$function$;

drop trigger if exists trg_20_tournament_game_retirement_cascade
  on public.tournament_games;
create trigger trg_20_tournament_game_retirement_cascade
after update on public.tournament_games
for each row execute function public.cascade_tournament_team_retirement();

create or replace function public.admin_record_tournament_day_non_played_result_cas(
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
  p_non_playing_team_id text,
  p_winner_team_id text,
  p_result_note text,
  p_expected_team_updated_at timestamptz,
  p_retirement_max_score integer,
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
  v_game public.tournament_games%rowtype;
  v_team public.tournament_teams%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_intent jsonb;
  v_score_result jsonb;
  v_assignments jsonb := '[]'::jsonb;
  v_dependency_ids uuid[] := '{}'::uuid[];
  v_court_id uuid;
  v_reserved_court_id uuid;
  v_score_a integer;
  v_score_b integer;
  v_result_type text := pg_catalog.upper(coalesce(p_result_type, ''));
  v_actor_role text := pg_catalog.lower(coalesce(p_actor_role, ''));
  v_derived_winner text;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_record_non_played_result'
  );
  if v_result_type not in ('FORFEIT', 'NO_SHOW', 'RETIREMENT')
     or pg_catalog.char_length(coalesce(p_result_note, '')) > 500 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OUTCOME: a supported type is required and the optional note is limited to 500 characters.';
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
       'non_playing_team_id', p_non_playing_team_id,
       'result_note', coalesce(p_result_note, '')
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
     or v_intent #>> '{payload,score_evidence,outcome,non_playing_team_id}' is distinct from p_non_playing_team_id
     or v_intent #>> '{payload,score_evidence,outcome,winner_team_id}' is distinct from p_winner_team_id
     or v_intent #>> '{payload,score_evidence,outcome,result_note}' is distinct from coalesce(p_result_note, '')
     or v_intent #>> '{payload,score_evidence,outcome,result_recorded_by}' is distinct from p_actor
     or (v_intent #>> '{payload,score_evidence,outcome,synthetic_progression_score}')::boolean is distinct from true
     or (v_intent #>> '{payload,score_evidence,outcome,rating_publish_eligible}')::boolean is distinct from false then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: outcome arguments do not match durable intent.';
  end if;
  if pg_catalog.jsonb_typeof(p_game_patch) is distinct from 'object'
     or pg_catalog.jsonb_typeof(p_dependency_updates) is distinct from 'array'
     or nullif(p_non_playing_team_id, '') is null
     or nullif(p_winner_team_id, '') is null
     or nullif(p_game_patch->>'winner_team_id', '') is distinct from p_winner_team_id
     or nullif(p_game_patch->>'loser_team_id', '') is distinct from p_non_playing_team_id
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
     and queue.state in (
       'WAITING', 'RESERVED', 'HELD', 'CALLED', 'ON_COURT', 'BLOCKED'
     )
     and queue.released_at is null
   for update of queue;
  if v_queue.id is null
     or v_queue.team_a_id is null
     or v_queue.team_b_id is null
     or p_non_playing_team_id not in (
       v_queue.team_a_id::text, v_queue.team_b_id::text
     ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: queued matchup or reviewed outcome changed.';
  end if;
  v_derived_winner := case
    when p_non_playing_team_id = v_queue.team_a_id::text
      then v_queue.team_b_id::text
    else v_queue.team_a_id::text
  end;
  if p_winner_team_id is distinct from v_derived_winner
     or (
       p_winner_team_id = v_queue.team_a_id::text
       and not (v_score_a > v_score_b)
     )
     or (
       p_winner_team_id = v_queue.team_b_id::text
       and not (v_score_b > v_score_a)
     ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: non-playing team and synthetic result contradict the reviewed matchup.';
  end if;
  v_court_id := v_queue.court_id;
  v_reserved_court_id := v_queue.reserved_court_id;
  if v_court_id is null and p_expected_court_version is not null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed court assignment changed.';
  end if;
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

  select game.* into v_game
    from public.tournament_games as game
   where game.id = v_queue.game_id
     and game.tournament_id = v_run.tournament_id
     and game.draw_id = v_queue.draw_id
     and game.team_a_id = v_queue.team_a_id
     and game.team_b_id = v_queue.team_b_id
     and game.updated_at = p_expected_game_updated_at
     and game.finalized_at is null
   for update;
  if v_game.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: queued game or team assignment changed.';
  end if;
  select draw.* into v_draw
    from public.tournament_event_draws as draw
   where draw.id = v_game.draw_id
     and draw.tournament_id = v_game.tournament_id
     and draw.updated_at = p_expected_draw_updated_at
   for update;
  if v_draw.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: draw changed after outcome review.';
  end if;
  if exists (
    select 1 from public.tournament_podium as podium
     where podium.tournament_id = v_game.tournament_id
       and podium.draw_id is not distinct from v_game.draw_id
  ) or exists (
    select 1 from public.matches as match
     where match.tournament_id = v_game.tournament_id
       and match.tournament_game_id in (
         select source.id from public.tournament_games as source
          where source.draw_id is not distinct from v_game.draw_id
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OUTCOME_LOCK: published or awarded draws cannot be changed.';
  end if;

  perform pg_catalog.set_config('jupr.day_live_operation_key', p_operation_key, true);
  if v_result_type = 'RETIREMENT' then
    if p_retirement_max_score is null or p_retirement_max_score <= 0
       or nullif(v_intent #>> '{payload,score_evidence,retirement,team_id}', '')
            is distinct from p_non_playing_team_id
       or nullif(v_intent #>> '{payload,score_evidence,retirement,expected_team_updated_at}', '')::timestamptz
            is distinct from p_expected_team_updated_at
       or (v_intent #>> '{payload,score_evidence,retirement,max_score}')::integer
            is distinct from p_retirement_max_score then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT: exact team and scoring evidence is required.';
    end if;
    select team.* into v_team
      from public.tournament_teams as team
     where team.id::text = p_non_playing_team_id
       and team.tournament_id = v_game.tournament_id
       and team.draw_id = v_game.draw_id
       and team.updated_at = p_expected_team_updated_at
       and team.competition_status = 'ACTIVE'
     for update;
    if v_team.id is null then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT_STALE: retiring team changed or is already inactive.';
    end if;
    update public.tournament_teams as team
       set competition_status = 'RETIRED',
           retired_at = pg_catalog.clock_timestamp(),
           retired_by = p_actor,
           retirement_note = nullif(
             pg_catalog.btrim(coalesce(p_result_note, '')), ''
           ),
           retirement_max_score = p_retirement_max_score,
           retirement_game_id = v_game.id,
           updated_at = pg_catalog.clock_timestamp()
     where team.id = v_team.id;
    update public.tournament_games as game
       set score_a = v_score_a,
           score_b = v_score_b,
           winner_team_id = p_winner_team_id::uuid,
           loser_team_id = p_non_playing_team_id::uuid,
           finalized_at = (p_game_patch->>'finalized_at')::timestamptz,
           updated_at = coalesce(
             nullif(p_game_patch->>'updated_at', '')::timestamptz,
             pg_catalog.clock_timestamp()
           )
     where game.id = v_game.id
       and game.updated_at = p_expected_game_updated_at;
    if not found then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT_STALE: target game changed while retiring the team.';
    end if;
    -- Fire the retirement trigger for every directly scheduled remaining game.
    -- Nested playoff dependency updates recursively handle games whose teams
    -- are not known until a preceding match finishes.
    update public.tournament_games as game
       set updated_at = pg_catalog.clock_timestamp()
     where game.tournament_id = v_game.tournament_id
       and game.draw_id = v_game.draw_id
       and game.id <> v_game.id
       and game.finalized_at is null
       and p_non_playing_team_id::uuid in (game.team_a_id, game.team_b_id);
    update public.tournament_event_draws as draw
       set updated_at = pg_catalog.clock_timestamp()
     where draw.id = v_draw.id
     returning * into v_draw;
    select pg_catalog.jsonb_build_object(
      'ok', true,
      'game', pg_catalog.to_jsonb(game),
      'retired_team', pg_catalog.to_jsonb(team)
    ) into v_score_result
      from public.tournament_games as game
      join public.tournament_teams as team on team.id = v_team.id
     where game.id = v_game.id;
  else
    if p_expected_team_updated_at is not null
       or p_retirement_max_score is not null
       or v_intent #> '{payload,score_evidence,retirement}' is not null then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_OUTCOME: retirement evidence is not valid for this outcome.';
    end if;
    v_score_result := public.admin_score_tournament_game_cas(
      p_club_id,
      p_tournament_id,
      p_game_id,
      p_expected_game_updated_at,
      p_expected_draw_updated_at,
      p_game_patch,
      p_dependency_updates
    );
    select draw.* into v_draw
      from public.tournament_event_draws as draw
     where draw.id = v_game.draw_id;
  end if;

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
         reserved_court_id = null,
         reserved_at = null,
         blocker_code = null,
         blocker_detail = null,
         released_at = coalesce(queue.released_at, pg_catalog.clock_timestamp()),
         completed_at = coalesce(queue.completed_at, pg_catalog.clock_timestamp()),
         version = case
           when queue.state = 'COMPLETED' then queue.version
           else queue.version + 1
         end,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where queue.id = v_queue.id;
  if v_result_type <> 'RETIREMENT'
     and (v_court_id is not null or v_reserved_court_id is not null) then
    update public.tournament_day_live_courts as court
       set version = court.version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = pg_catalog.clock_timestamp()
     where court.id in (v_court_id, v_reserved_court_id);
  end if;

  if v_result_type <> 'RETIREMENT' then
    select coalesce(
             pg_catalog.array_agg(nullif(dependency.value->>'id', '')::uuid
               order by nullif(dependency.value->>'id', '')::uuid),
             '{}'::uuid[]
           )
      into v_dependency_ids
      from pg_catalog.jsonb_array_elements(p_dependency_updates) as dependency(value);
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
  end if;

  update public.tournament_day_live_draws as day_draw
     set source_draw_updated_at = v_draw.updated_at,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where day_draw.run_id = v_run.id
     and day_draw.draw_id = v_queue.draw_id;
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
    'released_court_id', coalesce(v_court_id, v_reserved_court_id),
    'assignments', v_assignments,
    'result_type', v_result_type,
    'non_playing_team_id', p_non_playing_team_id,
    'rating_publish_eligible', false,
    'non_played_result', true,
    'team_retired', v_result_type = 'RETIREMENT'
  );
end;
$function$;

revoke all on function public.cascade_tournament_team_retirement()
  from public, anon, authenticated, service_role;
revoke all on function public.admin_record_tournament_day_non_played_result_cas(
  text, text, text, text, bigint, bigint, bigint, bigint,
  timestamptz, timestamptz, jsonb, jsonb, text, text, text, text,
  timestamptz, integer, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.cascade_tournament_team_retirement()
  to service_role;
grant execute on function public.admin_record_tournament_day_non_played_result_cas(
  text, text, text, text, bigint, bigint, bigint, bigint,
  timestamptz, timestamptz, jsonb, jsonb, text, text, text, text,
  timestamptz, integer, text, text, text, text
) to service_role;

comment on function public.admin_record_tournament_day_non_played_result_cas(
  text, text, text, text, bigint, bigint, bigint, bigint,
  timestamptz, timestamptz, jsonb, jsonb, text, text, text, text,
  timestamptz, integer, text, text, text, text
) is
  'Service-role-only atomic non-play result. The operator selects the non-playing team; retirement persists team status, preserves played rating scores, overrides round-robin standings, and cascades remaining forfeits.';

notify pgrst, 'reload schema';
