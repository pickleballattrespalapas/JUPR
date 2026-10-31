-- Recovery must rebuild the exact within-pool pairing set, never a whole-draw RR.
create or replace function public.tournament_draw_expected_rr_game_count(p_draw_id uuid)
returns integer
language plpgsql
stable
security invoker
set search_path = ''
as $function$
declare v_config jsonb; v_count integer;
begin
  select pool_config into v_config from public.tournament_event_draws where id = p_draw_id;
  if v_config is null then
    select count(*) into v_count from public.tournament_teams where draw_id = p_draw_id;
    return v_count * (v_count - 1) / 2;
  end if;
  perform public.assert_tournament_draw_pool_config(p_draw_id, v_config);
  select sum(pg_catalog.jsonb_array_length(pool->'team_ids') * (pg_catalog.jsonb_array_length(pool->'team_ids') - 1) / 2)
    into v_count from pg_catalog.jsonb_array_elements(v_config->'pools') pool;
  return v_count;
end
$function$;
revoke all on function public.tournament_draw_expected_rr_game_count(uuid) from public, anon, authenticated;
grant execute on function public.tournament_draw_expected_rr_game_count(uuid) to service_role;

create or replace function public.admin_reconcile_tournament_round_robin_games_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_expected_source_games jsonb,
  p_games jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_team_count integer;
  v_existing_count integer;
  v_expected_game_count integer;
  v_saved jsonb;
begin
  lock table public.tournament_teams in share mode;
  -- This RPC writes tournament_games. A self-conflicting table lock prevents
  -- two recovery calls from each holding SHARE and deadlocking on lock upgrade.
  lock table public.tournament_games in share row exclusive mode;
  lock table public.tournament_podium in share mode;
  lock table public.matches in share mode;
  lock table public.player_badges in share mode;
  lock table public.tournament_day_live_draws in share mode;
  lock table public.tournament_day_live_queue in share mode;

  select draw.* into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id = p_club_id
     and pg_catalog.upper(coalesce(tournament.status, ''))
         not in ('COMPLETED', 'ARCHIVED')
     and draw.updated_at = p_expected_draw_updated_at
   for update of draw;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  perform private.assert_tournament_draw_recovery_snapshot(
    p_tournament_id, p_draw_id, p_expected_teams, p_expected_source_games
  );
  perform private.assert_tournament_draw_recovery_dependencies_clear(
    p_club_id, p_tournament_id, p_draw_id
  );

  select pg_catalog.count(*) into v_team_count
    from public.tournament_teams as team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id;
  select pg_catalog.count(*) into v_existing_count
    from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null;
  v_expected_game_count := public.tournament_draw_expected_rr_game_count(v_draw.id);

  -- Existing official-looking results are preserved exactly, but every result
  -- must be internally complete and every pair must belong to the current
  -- roster. One finalized game in a 21-game partial draw therefore survives.
  if v_existing_count = 0
     or exists (
       select 1 from public.tournament_games as game
       left join public.tournament_teams as team_a
         on team_a.id = game.team_a_id
        and team_a.tournament_id = v_draw.tournament_id and team_a.draw_id = v_draw.id
       left join public.tournament_teams as team_b
         on team_b.id = game.team_b_id
        and team_b.tournament_id = v_draw.tournament_id and team_b.draw_id = v_draw.id
       where game.tournament_id = v_draw.tournament_id
         and game.draw_id = v_draw.id
         and game.series_parent_game_id is null
         and (
           pg_catalog.upper(coalesce(game.stage, '')) <> 'ROUND_ROBIN'
           or team_a.id is null or team_b.id is null or team_a.id = team_b.id
           or not public.tournament_pool_pair_allowed(v_draw.pool_config, team_a.id::text, team_b.id::text)
           or not (
             (
               game.score_a is null and game.score_b is null
               and game.winner_team_id is null and game.loser_team_id is null
               and game.finalized_at is null
             )
             or (
               game.score_a is not null and game.score_b is not null
               and game.score_a >= 0 and game.score_b >= 0 and game.score_a <> game.score_b
               and game.finalized_at is not null
               and game.winner_team_id = case when game.score_a > game.score_b then game.team_a_id else game.team_b_id end
               and game.loser_team_id = case when game.score_a > game.score_b then game.team_b_id else game.team_a_id end
             )
           )
         )
     )
     or exists (
       select 1 from public.tournament_games as game
        where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null
        group by least(game.team_a_id::text, game.team_b_id::text),
                 greatest(game.team_a_id::text, game.team_b_id::text)
       having pg_catalog.count(*) <> 1
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED';
  end if;

  if pg_catalog.jsonb_typeof(coalesce(p_games, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_games) = 0
     or v_existing_count + pg_catalog.jsonb_array_length(p_games) <> v_expected_game_count
     or (
       select pg_catalog.count(distinct payload.id)
         from pg_catalog.jsonb_to_recordset(p_games) as payload(id text)
     ) <> pg_catalog.jsonb_array_length(p_games)
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(p_games) as payload(
           id text, stage text, rr_round_number integer, rr_slot_number integer,
           team_a_id text, team_b_id text, score_a integer, score_b integer,
           winner_team_id text, loser_team_id text, finalized_at timestamptz
         )
         left join public.tournament_teams as team_a
           on team_a.id::text = payload.team_a_id
          and team_a.tournament_id = v_draw.tournament_id and team_a.draw_id = v_draw.id
         left join public.tournament_teams as team_b
           on team_b.id::text = payload.team_b_id
          and team_b.tournament_id = v_draw.tournament_id and team_b.draw_id = v_draw.id
        where nullif(payload.id, '') is null
           or pg_catalog.upper(coalesce(payload.stage, '')) <> 'ROUND_ROBIN'
           or payload.rr_round_number is null or payload.rr_round_number < 1
           or payload.rr_slot_number is null or payload.rr_slot_number < 1
           or team_a.id is null or team_b.id is null or team_a.id = team_b.id
           or not public.tournament_pool_pair_allowed(v_draw.pool_config, team_a.id::text, team_b.id::text)
           or payload.score_a is not null or payload.score_b is not null
           or payload.winner_team_id is not null or payload.loser_team_id is not null
           or payload.finalized_at is not null
           or exists (
             select 1 from public.tournament_games as existing
              where existing.tournament_id = v_draw.tournament_id
                and existing.draw_id = v_draw.id
                and least(existing.team_a_id::text, existing.team_b_id::text) = least(payload.team_a_id, payload.team_b_id)
                and greatest(existing.team_a_id::text, existing.team_b_id::text) = greatest(payload.team_a_id, payload.team_b_id)
           )
     )
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_games)
         as payload(team_a_id text, team_b_id text)
        group by least(payload.team_a_id, payload.team_b_id),
                 greatest(payload.team_a_id, payload.team_b_id)
       having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1 from (
         select game.rr_round_number, game.rr_slot_number
           from public.tournament_games as game
          where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null
         union all
         select payload.rr_round_number, payload.rr_slot_number
           from pg_catalog.jsonb_to_recordset(p_games)
             as payload(rr_round_number integer, rr_slot_number integer)
       ) as schedule
       group by schedule.rr_round_number, schedule.rr_slot_number
       having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1 from (
         select game.rr_round_number, game.team_a_id::text as team_id
           from public.tournament_games as game
          where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null
         union all
         select game.rr_round_number, game.team_b_id::text
           from public.tournament_games as game
          where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null
         union all
         select payload.rr_round_number, participant.team_id
           from pg_catalog.jsonb_to_recordset(p_games)
             as payload(rr_round_number integer, team_a_id text, team_b_id text)
           cross join lateral (values (payload.team_a_id), (payload.team_b_id)) as participant(team_id)
       ) as appearances
       group by appearances.rr_round_number, appearances.team_id
       having pg_catalog.count(*) > 1
     )
     or exists (
       select 1
         from public.tournament_teams as team_a
         join public.tournament_teams as team_b
           on team_b.tournament_id = team_a.tournament_id
          and team_b.draw_id = team_a.draw_id
          and team_a.id < team_b.id
          and public.tournament_pool_pair_allowed(v_draw.pool_config, team_a.id::text, team_b.id::text)
        where team_a.tournament_id = v_draw.tournament_id
          and team_a.draw_id = v_draw.id
          and not exists (
            select 1 from (
              select game.team_a_id::text as team_a_id, game.team_b_id::text as team_b_id
                from public.tournament_games as game
               where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null
              union all
              select payload.team_a_id, payload.team_b_id
                from pg_catalog.jsonb_to_recordset(p_games)
                  as payload(team_a_id text, team_b_id text)
            ) as combined
            where least(combined.team_a_id, combined.team_b_id) = least(team_a.id::text, team_b.id::text)
              and greatest(combined.team_a_id, combined.team_b_id) = greatest(team_a.id::text, team_b.id::text)
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED';
  end if;

  insert into public.tournament_games (
    id, tournament_id, draw_id, registration_day_id, event_option_id, stage,
    rr_round_number, rr_slot_number, team_a_id, team_b_id, score_a, score_b,
    winner_team_id, loser_team_id, finalized_at, created_at, updated_at
  )
  select
    payload.id::uuid, v_draw.tournament_id, v_draw.id,
    v_draw.registration_day_id, v_draw.event_option_id,
    'ROUND_ROBIN', payload.rr_round_number, payload.rr_slot_number,
    payload.team_a_id::uuid, payload.team_b_id::uuid,
    null, null, null, null, null,
    coalesce(payload.created_at, pg_catalog.clock_timestamp()),
    pg_catalog.clock_timestamp()
  from pg_catalog.jsonb_to_recordset(p_games) as payload(
    id text, registration_day_id text, event_option_id text, rr_round_number integer,
    rr_slot_number integer, team_a_id text, team_b_id text, created_at timestamptz
  );

  select coalesce(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(game) order by game.rr_round_number, game.rr_slot_number),
    '[]'::jsonb
  ) into v_saved
    from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null;
  return pg_catalog.jsonb_build_object('ok', true, 'games', v_saved);
exception
  when raise_exception then
    if sqlerrm = 'JUPR_TOURNAMENT_DRAW_RECOVERY_DEPENDENCY_BLOCKED' then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED';
    end if;
    raise;
end
$function$;

create or replace function public.admin_rebuild_tournament_round_robin_games_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_expected_source_games jsonb,
  p_games jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_team_count integer;
  v_expected_game_count integer;
  v_saved jsonb;
begin
  lock table public.tournament_teams in share mode;
  -- This RPC writes tournament_games. A self-conflicting table lock prevents
  -- two recovery calls from each holding SHARE and deadlocking on lock upgrade.
  lock table public.tournament_games in share row exclusive mode;
  lock table public.tournament_podium in share mode;
  lock table public.matches in share mode;
  lock table public.player_badges in share mode;
  lock table public.tournament_day_live_draws in share mode;
  lock table public.tournament_day_live_queue in share mode;

  select draw.* into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id = p_club_id
     and pg_catalog.upper(coalesce(tournament.status, ''))
         not in ('COMPLETED', 'ARCHIVED')
     and draw.updated_at = p_expected_draw_updated_at
   for update of draw;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  perform private.assert_tournament_draw_recovery_snapshot(
    p_tournament_id, p_draw_id, p_expected_teams, p_expected_source_games
  );
  perform private.assert_tournament_draw_recovery_dependencies_clear(
    p_club_id, p_tournament_id, p_draw_id
  );

  if exists (
    select 1 from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         pg_catalog.upper(coalesce(game.stage, '')) <> 'ROUND_ROBIN'
         or game.score_a is not null or game.score_b is not null
         or game.winner_team_id is not null or game.loser_team_id is not null
         or game.finalized_at is not null
       )
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_REBUILD_BLOCKED';
  end if;

  select pg_catalog.count(*) into v_team_count
    from public.tournament_teams as team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id;
  v_expected_game_count := public.tournament_draw_expected_rr_game_count(v_draw.id);
  if pg_catalog.jsonb_typeof(coalesce(p_games, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_games) <> v_expected_game_count
     or (
       select pg_catalog.count(distinct payload.id)
         from pg_catalog.jsonb_to_recordset(p_games) as payload(id text)
     ) <> v_expected_game_count
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(p_games) as payload(
           id text, stage text, rr_round_number integer, rr_slot_number integer,
           team_a_id text, team_b_id text, score_a integer, score_b integer,
           winner_team_id text, loser_team_id text, finalized_at timestamptz
         )
         left join public.tournament_teams as team_a
           on team_a.id::text = payload.team_a_id
          and team_a.tournament_id = v_draw.tournament_id and team_a.draw_id = v_draw.id
         left join public.tournament_teams as team_b
           on team_b.id::text = payload.team_b_id
          and team_b.tournament_id = v_draw.tournament_id and team_b.draw_id = v_draw.id
        where nullif(payload.id, '') is null
           or pg_catalog.upper(coalesce(payload.stage, '')) <> 'ROUND_ROBIN'
           or payload.rr_round_number is null or payload.rr_round_number < 1
           or payload.rr_slot_number is null or payload.rr_slot_number < 1
           or team_a.id is null or team_b.id is null or team_a.id = team_b.id
           or not public.tournament_pool_pair_allowed(v_draw.pool_config, team_a.id::text, team_b.id::text)
           or payload.score_a is not null or payload.score_b is not null
           or payload.winner_team_id is not null or payload.loser_team_id is not null
           or payload.finalized_at is not null
     )
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_games)
         as payload(rr_round_number integer, rr_slot_number integer)
        group by payload.rr_round_number, payload.rr_slot_number having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_games)
         as payload(team_a_id text, team_b_id text)
        group by least(payload.team_a_id, payload.team_b_id),
                 greatest(payload.team_a_id, payload.team_b_id)
       having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(p_games)
           as payload(rr_round_number integer, team_a_id text, team_b_id text)
         cross join lateral (values (payload.team_a_id), (payload.team_b_id)) as participant(team_id)
        group by payload.rr_round_number, participant.team_id
       having pg_catalog.count(*) > 1
     )
     or exists (
       select 1
         from public.tournament_teams as team_a
         join public.tournament_teams as team_b
           on team_b.tournament_id = team_a.tournament_id
          and team_b.draw_id = team_a.draw_id
          and team_a.id < team_b.id
          and public.tournament_pool_pair_allowed(v_draw.pool_config, team_a.id::text, team_b.id::text)
        where team_a.tournament_id = v_draw.tournament_id
          and team_a.draw_id = v_draw.id
          and not exists (
            select 1 from pg_catalog.jsonb_to_recordset(p_games)
              as payload(team_a_id text, team_b_id text)
             where least(payload.team_a_id, payload.team_b_id) = least(team_a.id::text, team_b.id::text)
               and greatest(payload.team_a_id, payload.team_b_id) = greatest(team_a.id::text, team_b.id::text)
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_REBUILD_BLOCKED';
  end if;

  delete from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id;

  insert into public.tournament_games (
    id, tournament_id, draw_id, registration_day_id, event_option_id, stage,
    rr_round_number, rr_slot_number, team_a_id, team_b_id, score_a, score_b,
    winner_team_id, loser_team_id, finalized_at, created_at, updated_at
  )
  select
    payload.id::uuid, v_draw.tournament_id, v_draw.id,
    v_draw.registration_day_id, v_draw.event_option_id,
    'ROUND_ROBIN', payload.rr_round_number, payload.rr_slot_number,
    payload.team_a_id::uuid, payload.team_b_id::uuid,
    null, null, null, null, null,
    coalesce(payload.created_at, pg_catalog.clock_timestamp()),
    pg_catalog.clock_timestamp()
  from pg_catalog.jsonb_to_recordset(p_games) as payload(
    id text, registration_day_id text, event_option_id text, rr_round_number integer,
    rr_slot_number integer, team_a_id text, team_b_id text, created_at timestamptz
  );

  select coalesce(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(game) order by game.rr_round_number, game.rr_slot_number),
    '[]'::jsonb
  ) into v_saved
    from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id;
  return pg_catalog.jsonb_build_object('ok', true, 'games', v_saved);
exception
  when raise_exception then
    if sqlerrm = 'JUPR_TOURNAMENT_DRAW_RECOVERY_DEPENDENCY_BLOCKED' then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_REBUILD_BLOCKED';
    end if;
    raise;
end
$function$;

revoke all on function public.admin_reconcile_tournament_round_robin_games_cas(text, text, text, timestamptz, jsonb, jsonb, jsonb) from public, anon, authenticated;
grant execute on function public.admin_reconcile_tournament_round_robin_games_cas(text, text, text, timestamptz, jsonb, jsonb, jsonb) to service_role;
revoke all on function public.admin_rebuild_tournament_round_robin_games_cas(text, text, text, timestamptz, jsonb, jsonb, jsonb) from public, anon, authenticated;
grant execute on function public.admin_rebuild_tournament_round_robin_games_cas(text, text, text, timestamptz, jsonb, jsonb, jsonb) to service_role;

notify pgrst, 'reload schema';
