-- Persist the reviewed scoring format on each generated playoff game. This
-- keeps score review and retirement progression bound to the exact bracket
-- configuration approved by the operator instead of a later event default.

alter table public.tournament_games
  add column if not exists scoring_format text null;

alter table public.tournament_games
  drop constraint if exists tournament_games_scoring_format_chk;

alter table public.tournament_games
  add constraint tournament_games_scoring_format_chk
  check (
    scoring_format is null
    or scoring_format in (
      'GAME_TO_11',
      'GAME_TO_15',
      'GAME_TO_21',
      'BEST_2_OF_3'
    )
  );

comment on column public.tournament_games.scoring_format is
  'Reviewed game-level scoring policy. New reviewed playoff flows persist it explicitly; legacy and round-robin rows may inherit the event format when null.';

-- Treat a scoring-policy change as a game derivation change so the same
-- podium, published-result, and downstream locks protect it. The shared
-- trigger remains compatible with tournament_teams and tournament_podium by
-- keeping every game-only OLD/NEW field inside the tournament_games branch.
create or replace function public.touch_tournament_draw_version_from_child()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_old_draw_id uuid;
  v_new_draw_id uuid;
  v_team_structural_change boolean := false;
  v_game_derivation_change boolean := false;
begin
  if tg_op in ('UPDATE', 'DELETE') then
    v_old_draw_id := old.draw_id;
  end if;
  if tg_op in ('INSERT', 'UPDATE') then
    v_new_draw_id := new.draw_id;
  end if;
  update public.tournament_event_draws
     set updated_at = clock_timestamp()
   where id in (v_old_draw_id, v_new_draw_id);

  if tg_table_name = 'tournament_teams' then
    v_team_structural_change := tg_op <> 'UPDATE';
    if tg_op = 'UPDATE' then
      v_team_structural_change := old.id is distinct from new.id
        or old.tournament_id is distinct from new.tournament_id
        or old.draw_id is distinct from new.draw_id
        or old.registration_day_id is distinct from new.registration_day_id
        or old.event_option_id is distinct from new.event_option_id
        or old.team_number is distinct from new.team_number
        or old.player1_id is distinct from new.player1_id
        or old.player2_id is distinct from new.player2_id;
    end if;
  end if;
  if v_team_structural_change
     and coalesce(current_setting('jupr.tournament_results_import_structural_write', true), 'off') <> 'on'
     and exists (
      select 1 from public.tournament_games g
       where g.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_GAMES';
  end if;

  if tg_table_name = 'tournament_games' then
    v_game_derivation_change := tg_op <> 'UPDATE';
    if tg_op = 'UPDATE' then
      v_game_derivation_change := old.stage is distinct from new.stage
        or old.team_a_id is distinct from new.team_a_id
        or old.team_b_id is distinct from new.team_b_id
        or old.team_a_source is distinct from new.team_a_source
        or old.team_b_source is distinct from new.team_b_source
        or old.scoring_format is distinct from new.scoring_format
        or old.score_a is distinct from new.score_a
        or old.score_b is distinct from new.score_b
        or old.winner_team_id is distinct from new.winner_team_id
        or old.loser_team_id is distinct from new.loser_team_id
        or old.finalized_at is distinct from new.finalized_at;
    end if;
    if v_game_derivation_change and exists (
      select 1 from public.tournament_podium p
       where p.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PODIUM_LOCK';
    end if;
    if v_game_derivation_change and exists (
      select 1
        from public.matches m
        join public.tournament_games published_game on published_game.id = m.tournament_game_id
       where published_game.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK';
    end if;
    if v_game_derivation_change
       and (
         (tg_op in ('UPDATE', 'DELETE') and upper(coalesce(old.stage, '')) = 'ROUND_ROBIN')
         or (tg_op in ('INSERT', 'UPDATE') and upper(coalesce(new.stage, '')) = 'ROUND_ROBIN')
       )
       and exists (
         select 1 from public.tournament_games playoff
          where playoff.draw_id in (v_old_draw_id, v_new_draw_id)
            and upper(coalesce(playoff.stage, '')) = 'PLAYOFF'
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DOWNSTREAM_SCORE_LOCK';
    end if;
  end if;
  if tg_table_name = 'tournament_podium' and exists (
    select 1
      from public.tournament_event_draws d
      join public.tournaments t on t.id = d.tournament_id
      join public.player_badges badge
        on badge.club_id = t.club_id
       and badge.context_type = 'tournament'
       and badge.context_id::text like d.tournament_id::text || ':draw:' || d.id::text || ':podium:%'
     where d.id in (v_old_draw_id, v_new_draw_id)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_ALREADY_AWARDED';
  end if;
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

revoke all on function public.touch_tournament_draw_version_from_child()
  from public, anon, authenticated;
grant execute on function public.touch_tournament_draw_version_from_child()
  to service_role;

-- The existing atomic insert seam remains the only route used by both
-- Tournament Ops and Day Live. Parse the reviewed format from the exact
-- p_games document already bound to the durable operation and persist it in
-- the same insert. Unsupported explicit values fail closed; older callers and
-- round-robin rows may continue to omit the field and inherit the event format.
create or replace function public.admin_insert_tournament_draw_games_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_mode text,
  p_expected_teams jsonb,
  p_expected_source_games jsonb,
  p_games jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_saved jsonb;
  v_mode text := upper(coalesce(p_mode, ''));
begin
  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if v_mode not in ('ROUND_ROBIN', 'PLAYOFF') then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_MODE_INVALID';
  end if;
  if jsonb_typeof(coalesce(p_games, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_games, '[]'::jsonb)) = 0 then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAMES_REQUIRED';
  end if;
  if jsonb_typeof(coalesce(p_expected_teams, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_teams, '[]'::jsonb)) = 0
     or exists (
       select 1 from jsonb_to_recordset(coalesce(p_expected_teams, '[]'::jsonb)) as x(id text, updated_at timestamptz)
        where nullif(x.id, '') is null or x.updated_at is null
     )
     or (
       select count(distinct x.id)
         from jsonb_to_recordset(coalesce(p_expected_teams, '[]'::jsonb)) as x(id text)
     ) <> jsonb_array_length(coalesce(p_expected_teams, '[]'::jsonb)) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;

  -- Direct child writers and this CAS retain the established child -> draw
  -- lock order, preventing an inverse-edge deadlock with version triggers.
  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
  perform source_game.id
    from public.tournament_games source_game
   where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
   order by source_game.id
   for update;
  perform badge.id
    from public.player_badges badge
   where badge.club_id = p_club_id
     and badge.context_type = 'tournament'
     and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
   order by badge.id
   for update;

  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and d.updated_at = p_expected_draw_updated_at
   for update of d;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  if (
       select count(*) from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_teams)
     or exists (
       select 1 from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
          and not exists (
            select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
             where x.id = team.id::text and x.updated_at = team.updated_at
          )
     )
     or exists (
       select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
        where not exists (
          select 1 from public.tournament_teams team
           where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
             and team.id::text = x.id and team.updated_at = x.updated_at
        )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(p_games) as x(stage text)
     where upper(coalesce(x.stage, '')) <> v_mode
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_STAGE_INVALID';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(p_games) as x(scoring_format text)
     where nullif(btrim(x.scoring_format), '') is not null
       and upper(btrim(x.scoring_format)) not in (
         'GAME_TO_11',
         'GAME_TO_15',
         'GAME_TO_21',
         'BEST_2_OF_3'
       )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_GAME_SCORING_FORMAT_INVALID';
  end if;
  if exists (
    select 1
      from jsonb_to_recordset(p_games) as x(team_a_id text, team_b_id text)
      left join public.tournament_teams a
        on a.id = nullif(x.team_a_id, '')::uuid
       and a.tournament_id = v_draw.tournament_id and a.draw_id = v_draw.id
      left join public.tournament_teams b
        on b.id = nullif(x.team_b_id, '')::uuid
       and b.tournament_id = v_draw.tournament_id and b.draw_id = v_draw.id
     where (nullif(x.team_a_id, '') is not null and a.id is null)
        or (nullif(x.team_b_id, '') is not null and b.id is null)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_GAME_TEAM_OUTSIDE_DRAW';
  end if;

  if v_mode = 'ROUND_ROBIN' then
    if exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_GAMES';
    end if;
  else
    if jsonb_typeof(coalesce(p_expected_source_games, '[]'::jsonb)) <> 'array'
       or jsonb_array_length(coalesce(p_expected_source_games, '[]'::jsonb)) = 0
       or exists (
         select 1 from jsonb_to_recordset(coalesce(p_expected_source_games, '[]'::jsonb)) as x(id text, updated_at timestamptz)
          where nullif(x.id, '') is null or x.updated_at is null
       )
       or (
         select count(distinct x.id)
           from jsonb_to_recordset(coalesce(p_expected_source_games, '[]'::jsonb)) as x(id text)
       ) <> jsonb_array_length(coalesce(p_expected_source_games, '[]'::jsonb)) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE';
    end if;
    if (
         select count(*) from public.tournament_games source_game
          where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
       ) <> jsonb_array_length(p_expected_source_games)
       or exists (
         select 1 from public.tournament_games source_game
          where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
            and not exists (
              select 1 from jsonb_to_recordset(p_expected_source_games) as x(id text, updated_at timestamptz)
               where x.id = source_game.id::text and x.updated_at = source_game.updated_at
            )
       )
       or exists (
         select 1 from jsonb_to_recordset(p_expected_source_games) as x(id text, updated_at timestamptz)
          where not exists (
            select 1 from public.tournament_games source_game
             where source_game.tournament_id = v_draw.tournament_id and source_game.draw_id = v_draw.id
               and source_game.id::text = x.id and source_game.updated_at = x.updated_at
          )
       ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE';
    end if;
    if exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id and g.stage = 'PLAYOFF'
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_PLAYOFFS';
    end if;
    if exists (
      select 1 from public.tournament_podium p
       where p.tournament_id = v_draw.tournament_id and p.draw_id = v_draw.id
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_PODIUM';
    end if;
    if not exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id and g.stage = 'ROUND_ROBIN'
    ) or exists (
      select 1 from public.tournament_games g
       where g.tournament_id = v_draw.tournament_id and g.draw_id = v_draw.id and g.stage = 'ROUND_ROBIN'
         and (g.score_a is null or g.score_b is null or g.winner_team_id is null)
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_INCOMPLETE';
    end if;
  end if;

  with inserted as (
    insert into public.tournament_games (
      id, tournament_id, draw_id, registration_day_id, event_option_id, stage,
      rr_round_number, rr_slot_number, playoff_game_code, playoff_round,
      scoring_format,
      team_a_id, team_b_id, team_a_source, team_b_source,
      score_a, score_b, winner_team_id, loser_team_id, finalized_at,
      created_at, updated_at
    )
    select
      coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
      v_draw.tournament_id, v_draw.id,
      coalesce(nullif(x.registration_day_id, ''), v_draw.registration_day_id),
      coalesce(nullif(x.event_option_id, ''), v_draw.event_option_id),
      v_mode, x.rr_round_number, x.rr_slot_number,
      nullif(x.playoff_game_code, ''), nullif(x.playoff_round, ''),
      upper(nullif(btrim(x.scoring_format), '')),
      nullif(x.team_a_id, '')::uuid, nullif(x.team_b_id, '')::uuid,
      x.team_a_source, x.team_b_source,
      x.score_a, x.score_b,
      nullif(x.winner_team_id, '')::uuid, nullif(x.loser_team_id, '')::uuid,
      x.finalized_at,
      coalesce(x.created_at, clock_timestamp()), clock_timestamp()
    from jsonb_to_recordset(p_games) as x(
      id text, registration_day_id text, event_option_id text, stage text,
      rr_round_number integer, rr_slot_number integer,
      playoff_game_code text, playoff_round text, scoring_format text,
      team_a_id text, team_b_id text, team_a_source jsonb, team_b_source jsonb,
      score_a integer, score_b integer, winner_team_id text, loser_team_id text,
      finalized_at timestamptz, created_at timestamptz
    )
    returning *
  )
  select coalesce(
           jsonb_agg(
             to_jsonb(inserted)
             order by inserted.stage, inserted.rr_round_number,
                      inserted.rr_slot_number, inserted.playoff_game_code
           ),
           '[]'::jsonb
         )
    into v_saved
    from inserted;
  update public.tournament_event_draws
     set updated_at = clock_timestamp()
   where id = v_draw.id;
  return jsonb_build_object('ok', true, 'games', v_saved);
end;
$$;

revoke all on function public.admin_insert_tournament_draw_games_cas(
  text, text, text, timestamptz, text, jsonb, jsonb, jsonb
) from public, anon, authenticated;
grant execute on function public.admin_insert_tournament_draw_games_cas(
  text, text, text, timestamptz, text, jsonb, jsonb, jsonb
) to service_role;

-- Day Live writes through the established base score CAS. Bind its durable
-- score review to the game override (when present) before falling back to the
-- event format, so a stale browser cannot score a configured playoff round
-- under a different policy.
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
       and event.tournament_id = new.tournament_id;
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

-- The ordinary Tournament Ops wrapper performs the same database-side stale
-- format check after its base CAS. Prefer the stored game format, then inherit
-- the event default for legacy rows.
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
             nullif(pg_catalog.btrim(game.scoring_format), ''),
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
   for share of event, game;
  if not found
     or v_scoring_format is distinct from
          p_game_patch #>> '{score_review_json,scoring_format}' then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_SCORE_FORMAT_STALE';
  end if;

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

revoke all on function public.admin_score_tournament_game_result_cas(
  text, text, text, timestamptz, timestamptz, jsonb, jsonb
) from public, anon, authenticated;
grant execute on function public.admin_score_tournament_game_result_cas(
  text, text, text, timestamptz, timestamptz, jsonb, jsonb
) to service_role;

-- Retirement is a standings override for the round robin, but a future
-- synthetic playoff result must obey that playoff game's own scoring policy.
-- Legacy playoff rows without a stored override inherit the event setting.
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
         where event.tournament_id = new.tournament_id
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

comment on column public.tournament_teams.retirement_max_score is
  'Round-robin standings maximum used for retirement losses. Future playoff forfeits derive their target from each game scoring format.';

notify pgrst, 'reload schema';
