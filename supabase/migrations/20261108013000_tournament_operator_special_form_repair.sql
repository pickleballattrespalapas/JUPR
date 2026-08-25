-- Repair the three score-review wrappers installed by
-- 20261108000000_tournament_day_operator_safety.sql. PostgreSQL implements
-- COALESCE and NULLIF as conditional expressions, so qualifying them with
-- pg_catalog makes the stored PL/pgSQL bodies fail when their SQL paths are
-- first prepared. Keep this additive replacement because that migration was
-- already applied to staging before the portability defect was discovered.

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

notify pgrst, 'reload schema';
