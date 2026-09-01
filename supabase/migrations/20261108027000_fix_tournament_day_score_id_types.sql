-- Tournament event options retain the legacy text tournament identifier while
-- tournament game rows use UUIDs. Compare through text at the two trigger
-- boundaries that inherit event scoring metadata.

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
  v_scoring_format text;
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
    select pg_catalog.upper(
             coalesce(
               nullif(pg_catalog.btrim(new.scoring_format), ''),
               nullif(pg_catalog.btrim(event.scoring_override), ''),
               nullif(pg_catalog.btrim(event.scoring_default), '')
             )
           )
      into v_scoring_format
      from (select 1) as anchor
      left join public.tournament_event_draws as draw
        on draw.id = new.draw_id
       and draw.tournament_id = new.tournament_id
      left join public.tournament_event_options as event
        on event.id::text = coalesce(
             nullif(new.event_option_id::text, ''),
             draw.event_option_id::text
           )
       and event.tournament_id = new.tournament_id::text;
    if coalesce(v_scoring_format, '') not in (
         'GAME_TO_11',
         'GAME_TO_15',
         'GAME_TO_21',
         'BEST_2_OF_3'
       )
       or v_scoring_format is distinct from
            v_payload #>> '{score_evidence,score_review,scoring_format}' then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_SCORE_FORMAT_STALE';
    end if;
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

revoke all on function public.apply_tournament_day_result_metadata()
  from public, anon, authenticated, service_role;
grant execute on function public.apply_tournament_day_result_metadata()
  to service_role;

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
  v_scoring_format text;
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
    if pg_catalog.upper(coalesce(new.stage, '')) = 'PLAYOFF' then
      v_scoring_format := pg_catalog.upper(
        nullif(pg_catalog.btrim(new.scoring_format), '')
      );
      if v_scoring_format is null then
        select pg_catalog.upper(
                 coalesce(
                   nullif(pg_catalog.btrim(event.scoring_override), ''),
                   nullif(pg_catalog.btrim(event.scoring_default), '')
                 )
               )
          into v_scoring_format
          from public.tournament_event_options as event
         where event.tournament_id = new.tournament_id::text
           and event.id::text = coalesce(
             nullif(new.event_option_id::text, ''),
             (
               select draw.event_option_id::text
                 from public.tournament_event_draws as draw
                where draw.id = new.draw_id
                  and draw.tournament_id = new.tournament_id
             )
           );
      end if;
      v_target := case v_scoring_format
        when 'GAME_TO_11' then 11
        when 'GAME_TO_15' then 15
        when 'GAME_TO_21' then 21
        when 'BEST_2_OF_3' then 2
        else null
      end;
      if v_target is null then
        raise exception using errcode = 'P0001',
          message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT: playoff scoring format is missing.';
      end if;
    elsif pg_catalog.upper(coalesce(new.stage, '')) = 'ROUND_ROBIN' then
      -- retirement_max_score is deliberately a round-robin standings policy;
      -- it must not override the configured target of a later playoff round.
      v_target := greatest(
        case when v_a_retired then v_team_a.retirement_max_score else 0 end,
        case when v_b_retired then v_team_b.retirement_max_score else 0 end
      );
      if v_target is null or v_target <= 0 then
        raise exception using errcode = 'P0001',
          message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT: configured round-robin max score is missing.';
      end if;
      -- The stored retirement target is the authoritative round-robin
      -- standings policy. Record its canonical format too so every synthetic
      -- result carries complete, internally consistent scoring evidence.
      v_scoring_format := case v_target
        when 11 then 'GAME_TO_11'
        when 15 then 'GAME_TO_15'
        when 21 then 'GAME_TO_21'
        when 2 then 'BEST_2_OF_3'
        else null
      end;
      if v_scoring_format is null then
        raise exception using errcode = 'P0001',
          message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT: configured round-robin max score is unsupported.';
      end if;
    else
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_RETIREMENT: game stage is unsupported.';
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
             'non_playing_team_id', v_loser::text,
             'scoring_format', v_scoring_format,
             'target_score', v_target
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

revoke all on function public.cascade_tournament_team_retirement()
  from public, anon, authenticated, service_role;
grant execute on function public.cascade_tournament_team_retirement()
  to service_role;

notify pgrst, 'reload schema';
