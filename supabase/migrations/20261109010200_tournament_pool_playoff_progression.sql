-- Pool playoffs retain one draw, championship, publication and podium.
-- Preserve operation-intent/CAS/roster locks while accepting reviewed pool
-- qualifier counts 2..16 and real two-/three-qualifier medal outcomes.
begin;

-- Complete, non-overlapping within-pool schedules are required before activation
-- and before playoff advancement. Series child games are rating evidence only.
create or replace function public.assert_tournament_pool_schedule_complete(p_draw_id uuid)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_config jsonb;
begin
  select draw.pool_config into v_config
    from public.tournament_event_draws draw where draw.id = p_draw_id;
  if v_config is null then return; end if;
  perform public.assert_tournament_draw_pool_config(p_draw_id, v_config);
  if (select pg_catalog.count(*) from public.tournament_games game
       where game.draw_id = p_draw_id and upper(coalesce(game.stage,'')) = 'ROUND_ROBIN'
         and game.series_parent_game_id is null)
       <> public.tournament_draw_expected_rr_game_count(p_draw_id)
     or exists (select 1 from public.tournament_games game
       where game.draw_id = p_draw_id and upper(coalesce(game.stage,'')) = 'ROUND_ROBIN'
         and game.series_parent_game_id is null
         and not public.tournament_pool_pair_allowed(v_config,game.team_a_id::text,game.team_b_id::text))
     or exists (select 1 from public.tournament_games game
       where game.draw_id = p_draw_id and upper(coalesce(game.stage,'')) = 'ROUND_ROBIN'
         and game.series_parent_game_id is null
       group by least(game.team_a_id::text,game.team_b_id::text),
                greatest(game.team_a_id::text,game.team_b_id::text)
       having pg_catalog.count(*) <> 1) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_POOL_SCHEDULE_INCOMPLETE: every same-pool pairing must appear exactly once before activation or advancement.';
  end if;
end;
$function$;

-- The exact reviewed qualifier list is durable, including a reason when
-- withdrawals require a different total than originally configured.
create or replace function public.assert_tournament_pool_playoff_selection(
  p_draw_id uuid, p_advance_count integer, p_configuration jsonb
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare v_draw public.tournament_event_draws%rowtype;
begin
  select draw.* into v_draw from public.tournament_event_draws draw where draw.id=p_draw_id;
  if not found then raise exception using errcode='22023',message='JUPR_TOURNAMENT_POOL_DRAW: draw not found.'; end if;
  if v_draw.pool_config is not null then
    if p_advance_count is null or p_advance_count not between 2 and 16
       or pg_catalog.jsonb_typeof(p_configuration) is distinct from 'object' then
      raise exception using errcode='22023', message='JUPR_TOURNAMENT_POOL_QUALIFIERS: review a supported exact playoff configuration.';
    end if;
    if p_configuration->'override_reason' is not null and (
       pg_catalog.jsonb_typeof(p_configuration->'override_reason') is distinct from 'string'
       or pg_catalog.length(p_configuration->>'override_reason')>500) then
      raise exception using errcode='22023', message='JUPR_TOURNAMENT_POOL_OVERRIDE_REASON: provide a text explanation of at most 500 characters.';
    end if;
    if p_advance_count is distinct from (
         select sum((pool->>'advance_count')::integer)::integer
           from pg_catalog.jsonb_array_elements(v_draw.pool_config->'pools') pool
       ) and (nullif(pg_catalog.btrim(p_configuration->>'override_reason'),'') is null
         or pg_catalog.length(p_configuration->>'override_reason')>500) then
      raise exception using errcode='22023',
        message='JUPR_TOURNAMENT_POOL_OVERRIDE_REASON: explain a change from the configured qualifier count before generation.';
    end if;
    if pg_catalog.jsonb_typeof(p_configuration->'seed_team_ids') is distinct from 'array'
       or pg_catalog.jsonb_array_length(p_configuration->'seed_team_ids') <> p_advance_count
       or p_configuration->>'template_code'
            is distinct from 'SINGLE_ELIMINATION_' || p_advance_count::text
       or (select count(distinct seed_id) from pg_catalog.jsonb_array_elements_text(
             p_configuration->'seed_team_ids') seed(seed_id)) <> p_advance_count
       or exists (select 1 from pg_catalog.jsonb_array_elements_text(
             p_configuration->'seed_team_ids') seed(seed_id)
          where not exists (select 1 from public.tournament_teams team
            where team.id::text=seed.seed_id and team.draw_id=v_draw.id
              and team.tournament_id=v_draw.tournament_id
              and upper(coalesce(team.competition_status,'ACTIVE')) <> 'RETIRED')) then
      raise exception using errcode='22023',
        message='JUPR_TOURNAMENT_POOL_QUALIFIERS: review the exact unique active qualifiers and configured advance count.';
    end if;
  end if;
end;
$function$;

-- A two-qualifier pool championship awards gold and silver. With three
-- qualifiers the sole semifinal loser is bronze. Larger brackets retain the
-- explicit bronze match. Never infer a small bracket merely from missing games.
create or replace function public.tournament_draw_expected_playoff_podium(
  p_tournament_id uuid, p_draw_id uuid
)
returns table(placement integer, team_id uuid)
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_config jsonb;
  v_qualifier_count integer;
  v_playoff_count integer;
  v_final_count integer;
  v_bronze_count integer;
  v_sf_count integer;
  v_final public.tournament_games%rowtype;
  v_bronze public.tournament_games%rowtype;
  v_semifinal public.tournament_games%rowtype;
  v_third uuid;
begin
  select draw.pool_config into v_config from public.tournament_event_draws draw
    where draw.id = p_draw_id and draw.tournament_id = p_tournament_id;
  if not found then return; end if;
  if v_config is not null then
    -- The reviewed qualifier total may be overridden after a retirement.
    -- Configured pool quotas remain historical, so count the actual bracket.
    select count(distinct side.team_id)::integer into v_qualifier_count
      from public.tournament_games game
      cross join lateral (values(game.team_a_id),(game.team_b_id)) side(team_id)
     where game.tournament_id=p_tournament_id and game.draw_id=p_draw_id
       and upper(coalesce(game.stage,''))='PLAYOFF';
  end if;
  select count(*), count(*) filter(where upper(game.playoff_round) = 'FINAL'),
         count(*) filter(where upper(game.playoff_round) = 'BRONZE'),
         count(*) filter(where upper(game.playoff_round) = 'SF')
    into v_playoff_count,v_final_count,v_bronze_count,v_sf_count
    from public.tournament_games game
    where game.tournament_id=p_tournament_id and game.draw_id=p_draw_id
      and upper(coalesce(game.stage,''))='PLAYOFF';
  if v_final_count <> 1 or exists (
    select 1 from public.tournament_games game
     where game.tournament_id=p_tournament_id and game.draw_id=p_draw_id
       and upper(coalesce(game.stage,''))='PLAYOFF'
       and (game.finalized_at is null or game.score_a is null or game.score_b is null
         or game.score_a = game.score_b or game.team_a_id is null or game.team_b_id is null
         or game.team_a_id = game.team_b_id or game.winner_team_id is null or game.loser_team_id is null
         or game.winner_team_id = game.loser_team_id
         or game.winner_team_id not in (game.team_a_id,game.team_b_id)
         or game.loser_team_id not in (game.team_a_id,game.team_b_id)
         or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
         or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id))
  ) then return; end if;
  select game.* into v_final from public.tournament_games game
    where game.tournament_id=p_tournament_id and game.draw_id=p_draw_id
      and upper(coalesce(game.stage,''))='PLAYOFF' and upper(game.playoff_round)='FINAL';
  if v_qualifier_count=2 then
    if v_playoff_count<>1 or v_bronze_count<>0 then return; end if;
  elsif v_qualifier_count=3 then
    if v_playoff_count<>2 or v_sf_count<>1 or v_bronze_count<>0 then return; end if;
    select game.* into v_semifinal from public.tournament_games game
      where game.tournament_id=p_tournament_id and game.draw_id=p_draw_id
        and upper(coalesce(game.stage,''))='PLAYOFF' and upper(game.playoff_round)='SF';
    if v_semifinal.winner_team_id not in(v_final.team_a_id,v_final.team_b_id) then return; end if;
    v_third := v_semifinal.loser_team_id;
  else
    if v_bronze_count<>1 then return; end if;
    if v_config is not null and (v_qualifier_count not between 4 and 16
       or v_playoff_count<>v_qualifier_count or v_sf_count<>2) then return; end if;
    select game.* into v_bronze from public.tournament_games game
      where game.tournament_id=p_tournament_id and game.draw_id=p_draw_id
        and upper(coalesce(game.stage,''))='PLAYOFF' and upper(game.playoff_round)='BRONZE';
    v_third := v_bronze.winner_team_id;
  end if;
  if v_third is not null and v_third in(v_final.winner_team_id,v_final.loser_team_id) then return; end if;
  return query select 1,v_final.winner_team_id union all select 2,v_final.loser_team_id;
  if v_third is not null then return query select 3,v_third; end if;
end;
$function$;


create or replace function public.assert_tournament_day_live_draw_ready(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_draw_id uuid,
  p_expected_draw_updated_at timestamptz
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_day public.tournament_registration_days%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_event_option_id text;
  v_player_id integer;
  v_registration_count integer;
begin
  -- Read only the setup identity first.  The final draw row is deliberately
  -- locked after its team/game children because their version triggers touch
  -- the draw and use that same child -> parent order.
  select draw.event_option_id
    into v_event_option_id
    from public.tournament_event_draws as draw
    join public.tournaments as tournament
      on tournament.id = draw.tournament_id
     and tournament.club_id::text = p_club_id
   where draw.id = p_draw_id
     and draw.tournament_id::text = p_tournament_id;
  if v_event_option_id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed draw identity is no longer available.';
  end if;

  select day.* into v_day
    from public.tournament_registration_days as day
   where day.id = p_registration_day_id
     and day.tournament_id::text = p_tournament_id
     and day.enabled is true
   for share;
  if v_day.id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: selected tournament day changed or is disabled.';
  end if;

  select event.* into v_event
    from public.tournament_event_options as event
   where event.id = v_event_option_id
     and event.tournament_id::text = p_tournament_id
   for share;
  if v_event.id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: reviewed event identity changed.';
  end if;

  perform team.id
    from public.tournament_teams as team
   where team.tournament_id::text = p_tournament_id
     and team.draw_id = p_draw_id
   order by team.id
   for share;
  perform game.id
    from public.tournament_games as game
   where game.tournament_id::text = p_tournament_id
     and game.draw_id = p_draw_id
   order by game.id
   for share;

  select draw.*
    into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament
      on tournament.id = draw.tournament_id
     and tournament.club_id::text = p_club_id
   where draw.id = p_draw_id
     and draw.tournament_id::text = p_tournament_id
   for share;

  if v_draw.id is null
     or v_draw.updated_at is distinct from p_expected_draw_updated_at
     or v_draw.event_option_id is distinct from v_event.id
     or pg_catalog.upper(coalesce(v_draw.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or coalesce(v_draw.hidden_from_primary_ops, false) is true
     or pg_catalog.upper(coalesce(v_draw.draw_kind, 'STANDARD')) <> 'STANDARD' then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: draw changed, is inactive, or is not supported by the day runner.';
  end if;

  if v_event.enabled is not true
     or pg_catalog.upper(coalesce(v_event.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or (case
       when v_draw.registration_day_id is not null
         then v_draw.registration_day_id = p_registration_day_id
          and case
            when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
              then case
                when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) > 0
                  then v_event.scheduled_day_ids ? v_draw.registration_day_id
                else v_event.registration_day_id is null
                  or v_event.registration_day_id = v_draw.registration_day_id
              end
            else false
          end
       else case
         when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
           then case
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 1
               then v_event.scheduled_day_ids ? p_registration_day_id
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
               then v_event.registration_day_id = p_registration_day_id
             else false
           end
         else false
       end
     end) is not true then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_UNSCHEDULED: draw is not scheduled on this enabled tournament day.';
  end if;

  if not exists (
    select 1 from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAMES_REQUIRED: generate games before activating this draw.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         game.registration_day_id is distinct from p_registration_day_id
         or game.event_option_id is distinct from v_draw.event_option_id
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_DAY: every draw game must belong to the selected tournament day and event.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and game.series_parent_game_id is null
       and pg_catalog.upper(coalesce(game.stage, '')) not in ('ROUND_ROBIN', 'PLAYOFF')
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_STAGE: every draw game must use a supported ROUND_ROBIN or PLAYOFF stage.';
  end if;

  if (
    select pg_catalog.count(*)
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and (
         team.registration_day_id is distinct from p_registration_day_id
         or team.event_option_id is distinct from v_draw.event_option_id
       )
  ) > 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_TEAM_DAY_SCOPE: every draw team must belong to the selected tournament day and draw event.';
  end if;

  if exists (
    select participant.player_id
      from public.tournament_teams as team
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
       and participant.player_id is not null
     group by participant.player_id
    having pg_catalog.count(*) > 1
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROSTER_PLAYER_DUPLICATE: each player may belong to only one exact team in this draw.';
  end if;

  if (
    select coalesce(
             pg_catalog.array_agg(team.id order by team.id),
             '{}'::uuid[]
           )
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) is distinct from (
    select coalesce(
             pg_catalog.array_agg(distinct side.team_id order by side.team_id)
               filter (where side.team_id is not null),
             '{}'::uuid[]
           )
      from public.tournament_games as game
      cross join lateral (values (game.team_a_id), (game.team_b_id)) as side(team_id)
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROUND_ROBIN_ROSTER: every exact draw team must appear in the reviewed round-robin schedule.';
  end if;

  if (
    select pg_catalog.count(*)
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) < 4 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_FORMAT: activate at least four exact in-draw teams so a supported playoff format can complete day closeout.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
      cross join lateral (
        select
          pg_catalog.count(participant.player_id) as player_count,
          pg_catalog.count(distinct participant.player_id) as distinct_player_count
        from (values
          (team_a.player1_id), (team_a.player2_id),
          (team_b.player1_id), (team_b.player2_id)
        ) as participant(player_id)
        where participant.player_id is not null
      ) as participant_counts
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
       and (
         game.team_a_id is null
         or game.team_b_id is null
         or game.team_a_id = game.team_b_id
         or team_a.id is null
         or team_b.id is null
         or team_a.player1_id is null
         or team_b.player1_id is null
         or participant_counts.player_count not in (2, 4)
         or participant_counts.player_count <> participant_counts.distinct_player_count
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROUND_ROBIN: every round-robin game requires two distinct in-draw teams with two or four distinct effective players.';
  end if;

  perform public.assert_tournament_pool_schedule_complete(v_draw.id);

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'PLAYOFF'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFFS_ALREADY_GENERATED: activate reviewed round-robin games first, then generate playoffs through the guarded day operation.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         (game.finalized_at is null and (
           game.score_a is not null or game.score_b is not null
           or game.winner_team_id is not null or game.loser_team_id is not null
         ))
         or
         (game.finalized_at is not null and (
           game.score_a is null or game.score_b is null
           or game.score_a = game.score_b
           or game.winner_team_id is null or game.loser_team_id is null
           or game.team_a_id is null or game.team_b_id is null
           or game.team_a_id = game.team_b_id
           or game.winner_team_id = game.loser_team_id
           or game.winner_team_id not in (game.team_a_id, game.team_b_id)
           or game.loser_team_id not in (game.team_a_id, game.team_b_id)
           or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
           or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id)
         ))
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_STATE: partially scored or tied games require reconciliation before activation.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      cross join lateral (
        select
          pg_catalog.count(participant.player_id) as player_count,
          pg_catalog.count(distinct participant.player_id) as distinct_player_count
        from public.tournament_teams as team
        cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
        where team.id in (game.team_a_id, game.team_b_id)
          and team.tournament_id = game.tournament_id
          and team.draw_id = game.draw_id
          and team.registration_day_id = p_registration_day_id
          and team.event_option_id = v_draw.event_option_id
          and participant.player_id is not null
      ) as participant_counts
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and game.team_a_id is not null
       and game.team_b_id is not null
       and (
         participant_counts.player_count not in (2, 4)
         or participant_counts.player_count <> participant_counts.distinct_player_count
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PARTICIPANTS: every playable game needs two or four distinct effective players.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         (game.team_a_id is not null and (
           team_a.id is null or team_a.player1_id is null
         ))
         or (game.team_b_id is not null and (
           team_b.id is null or team_b.player1_id is null
         ))
         or (
           game.team_a_id is not null and game.team_b_id is not null
           and team_a.id = team_b.id
         )
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PARTICIPANTS: a playable game has incomplete or foreign team evidence.';
  end if;

  for v_player_id in
    select distinct participant.player_id
      from public.tournament_games as game
      join public.tournament_teams as team
        on team.id in (game.team_a_id, game.team_b_id)
       and team.tournament_id = game.tournament_id
       and team.draw_id = game.draw_id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and participant.player_id is not null
     order by participant.player_id
  loop
    perform player.id
      from public.players as player
     where player.id = v_player_id
       and player.club_id::text = p_club_id
     for share;
    if not found then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYER_SCOPE: every participant must belong to the tournament club.';
    end if;
    perform registration.id
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED')
     order by registration.id
     for share;
    select pg_catalog.count(*)
      into v_registration_count
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED');
    if v_registration_count <> 1 then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_REGISTRATION: every player must resolve to exactly one active tournament registration.';
    end if;
    -- The exact draw roster plus canonical active registration is the
    -- live-play authority. Check-in, waiver, and payment remain visible in
    -- preflight but never block draw activation or court assignment.
  end loop;
end;
$function$;

create or replace function public.admin_generate_tournament_day_playoffs_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_draw_id text,
  p_advance_count integer,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_day_draw_version bigint,
  p_expected_draw_version timestamptz,
  p_expected_team_versions jsonb,
  p_expected_source_game_versions jsonb,
  p_games jsonb,
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
  v_day_draw public.tournament_day_live_draws%rowtype;
  v_day public.tournament_registration_days%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_event_option_id text;
  v_intent jsonb;
  v_insert_result jsonb;
  v_priority_base bigint;
  v_requested_game_ids uuid[] := '{}'::uuid[];
  v_inserted_game_ids uuid[] := '{}'::uuid[];
  v_queued_game_ids uuid[] := '{}'::uuid[];
  v_queue_insert_count integer := 0;
  v_assignments jsonb := '[]'::jsonb;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_generate_playoffs'
  );
  if v_intent #>> '{payload,action}' is distinct from 'generate_playoffs'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,payload,draw_id}' is distinct from p_draw_id
     or (v_intent #>> '{payload,payload,advance_count}')::integer
          is distinct from p_advance_count
     or v_intent #> '{payload,payload}' is distinct from
          pg_catalog.jsonb_build_object(
            'draw_id', p_draw_id,
            'advance_count', p_advance_count
          )
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,draw_version}')::bigint
          is distinct from p_expected_day_draw_version
     or nullif(v_intent #>> '{payload,playoff_evidence,source_draw_updated_at}', '')::timestamptz
          is distinct from p_expected_draw_version
     or v_intent #> '{payload,playoff_evidence,team_versions}'
          is distinct from p_expected_team_versions
     or v_intent #> '{payload,playoff_evidence,source_game_versions}'
          is distinct from p_expected_source_game_versions
     or v_intent #> '{payload,playoff_games}' is distinct from p_games then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: playoff arguments do not match durable intent.';
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
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_STALE: day run changed or is not active.';
  end if;
  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.run_id = v_run.id
     and day_draw.draw_id::text = p_draw_id
     and day_draw.state = 'ACTIVE'
     and day_draw.version = p_expected_day_draw_version
   for update;
  if v_day_draw.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_STALE: active draw membership changed.';
  end if;

  select source_draw.event_option_id
    into v_event_option_id
    from public.tournament_event_draws as source_draw
   where source_draw.id = v_day_draw.draw_id
     and source_draw.tournament_id::text = p_tournament_id;
  if v_event_option_id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_SCOPE: reviewed source draw identity is no longer available.';
  end if;
  select source_day.* into v_day
    from public.tournament_registration_days as source_day
   where source_day.id = p_registration_day_id
     and source_day.tournament_id::text = p_tournament_id
     and source_day.enabled is true
   for share;
  select source_event.* into v_event
    from public.tournament_event_options as source_event
   where source_event.id = v_event_option_id
     and source_event.tournament_id::text = p_tournament_id
   for share;
  if v_day.id is null or v_event.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_SCOPE: selected day or event changed before playoff generation.';
  end if;

  perform locked_team.id
    from public.tournament_teams as locked_team
   where locked_team.tournament_id = v_run.tournament_id
     and locked_team.draw_id = v_day_draw.draw_id
   order by locked_team.id
   for share;
  perform locked_game.id
    from public.tournament_games as locked_game
   where locked_game.tournament_id = v_run.tournament_id
     and locked_game.draw_id = v_day_draw.draw_id
   order by locked_game.id
   for share;
  select source_draw.* into v_draw
    from public.tournament_event_draws as source_draw
   where source_draw.id = v_day_draw.draw_id
     and source_draw.tournament_id = v_run.tournament_id
   for share;
  if v_draw.id is null
     or v_draw.updated_at is distinct from p_expected_draw_version
     or v_draw.event_option_id is distinct from v_event.id
     or pg_catalog.upper(coalesce(v_draw.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or coalesce(v_draw.hidden_from_primary_ops, false) is true
     or pg_catalog.upper(coalesce(v_draw.draw_kind, 'STANDARD')) <> 'STANDARD'
     or v_event.enabled is not true
     or pg_catalog.upper(coalesce(v_event.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or (case
       when v_draw.registration_day_id is not null
         then v_draw.registration_day_id = p_registration_day_id
          and case
            when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
              then case
                when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) > 0
                  then v_event.scheduled_day_ids ? v_draw.registration_day_id
                else v_event.registration_day_id is null
                  or v_event.registration_day_id = v_draw.registration_day_id
              end
            else false
          end
       else case
         when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
           then case
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 1
               then v_event.scheduled_day_ids ? p_registration_day_id
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
               then v_event.registration_day_id = p_registration_day_id
             else false
           end
         else false
       end
     end) is not true then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_SCOPE: draw is inactive, unsupported, or no longer scheduled on this enabled day.';
  end if;
  if exists (
    select 1
      from public.tournament_teams as team
     where team.tournament_id = v_run.tournament_id
       and team.draw_id = v_day_draw.draw_id
       and (
         team.registration_day_id is distinct from p_registration_day_id
         or team.event_option_id is distinct from v_draw.event_option_id
       )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_TEAM_SCOPE: every draw team must remain on this exact day and event.';
  end if;
  if exists (
    select participant.player_id
      from public.tournament_teams as team
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where team.tournament_id = v_run.tournament_id
       and team.draw_id = v_day_draw.draw_id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
       and participant.player_id is not null
     group by participant.player_id
    having pg_catalog.count(*) > 1
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ROSTER_PLAYER_DUPLICATE: each player may belong to only one exact team in this draw.';
  end if;
  if (
    select coalesce(
             pg_catalog.array_agg(team.id order by team.id),
             '{}'::uuid[]
           )
      from public.tournament_teams as team
     where team.tournament_id = v_run.tournament_id
       and team.draw_id = v_day_draw.draw_id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) is distinct from (
    select coalesce(
             pg_catalog.array_agg(distinct side.team_id order by side.team_id)
               filter (where side.team_id is not null),
             '{}'::uuid[]
           )
      from public.tournament_games as game
      cross join lateral (values (game.team_a_id), (game.team_b_id)) as side(team_id)
     where game.tournament_id = v_run.tournament_id
       and game.draw_id = v_day_draw.draw_id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ROSTER: every exact draw team must have reviewed round-robin evidence before playoff generation.';
  end if;
  if (case when v_draw.pool_config is null then p_advance_count not in (4,5,6)
       else p_advance_count not between 2 and 16 end)
     or p_advance_count > (
       select pg_catalog.count(*)
        from public.tournament_teams as team
       where team.tournament_id = v_run.tournament_id
         and team.draw_id = v_day_draw.draw_id
          and team.registration_day_id = p_registration_day_id
          and team.event_option_id = v_draw.event_option_id
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ADVANCE_COUNT: choose a supported advancing-team count available for this draw.';
  end if;
  perform public.assert_tournament_pool_schedule_complete(v_draw.id);
  perform public.assert_tournament_pool_playoff_selection(
    v_draw.id,p_advance_count,v_intent #> '{payload,playoff_configuration}'
  );
  if p_games is null
     or pg_catalog.jsonb_typeof(p_games) is distinct from 'array'
     or pg_catalog.jsonb_array_length(p_games) = 0 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_GAMES: an exact nonempty playoff plan is required.';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_games) as planned(
        id text, tournament_id text, draw_id text,
        registration_day_id text, event_option_id text, stage text
      )
     where nullif(planned.id, '') is null
        or planned.tournament_id is distinct from p_tournament_id
        or planned.draw_id is distinct from p_draw_id
        or planned.registration_day_id is distinct from p_registration_day_id
        or planned.event_option_id is distinct from v_draw.event_option_id::text
        or pg_catalog.upper(coalesce(planned.stage, '')) is distinct from 'PLAYOFF'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_GAMES: every planned game must belong to this exact draw, event, and day.';
  end if;
  select coalesce(
           pg_catalog.array_agg(nullif(planned.id, '')::uuid
             order by nullif(planned.id, '')::uuid),
           '{}'::uuid[]
         )
    into v_requested_game_ids
    from pg_catalog.jsonb_to_recordset(p_games) as planned(id text);
  if pg_catalog.cardinality(v_requested_game_ids)
       is distinct from pg_catalog.jsonb_array_length(p_games)
     or pg_catalog.cardinality(v_requested_game_ids) is distinct from pg_catalog.cardinality(
       array(select distinct game_id from pg_catalog.unnest(v_requested_game_ids) as planned(game_id))
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_GAMES: planned game identities must be present and unique.';
  end if;
  if exists (
    select 1 from public.tournament_day_live_queue as queue
     where queue.day_draw_id = v_day_draw.id
       and queue.state in ('HELD', 'CALLED', 'ON_COURT')
       and queue.released_at is null
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ASSIGNMENT: finish the draw current court assignments first.';
  end if;
  if not exists (
    select 1 from public.tournament_games as game
     where game.tournament_id::text = p_tournament_id
       and game.draw_id::text = p_draw_id
       and game.stage = 'ROUND_ROBIN'
  ) or exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
     where game.tournament_id::text = p_tournament_id
       and game.draw_id::text = p_draw_id
       and game.stage = 'ROUND_ROBIN'
       and (
         game.registration_day_id is distinct from p_registration_day_id
         or game.event_option_id is distinct from v_draw.event_option_id
         or game.finalized_at is null
         or game.score_a is null or game.score_b is null
         or game.score_a = game.score_b
         or game.team_a_id is null or game.team_b_id is null
         or game.team_a_id = game.team_b_id
         or team_a.id is null or team_b.id is null
         or game.winner_team_id is null or game.loser_team_id is null
         or game.winner_team_id = game.loser_team_id
         or game.winner_team_id not in (game.team_a_id, game.team_b_id)
         or game.loser_team_id not in (game.team_a_id, game.team_b_id)
         or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
         or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id)
       )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ROUND_ROBIN: every round-robin game must be finalized and non-tied.';
  end if;

  if exists (
    select 1 from public.tournament_admin_operations as operation
     where operation.club_id = p_club_id
       and operation.operation_key <> p_operation_key
       and operation.status in ('intent', 'mutated', 'recovery_required')
       and (
         operation.lock_scope = pg_catalog.concat_ws(':', 'tournament', p_tournament_id, 'day', p_registration_day_id)
         or (operation.entity_type = 'tournament_event_draw' and operation.entity_id = p_draw_id)
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_OPERATION: reconcile the unsettled day or draw operation first.';
  end if;

  perform pg_catalog.set_config('jupr.day_live_operation_key', p_operation_key, true);
  v_insert_result := public.admin_insert_tournament_draw_games_cas(
    p_club_id,
    p_tournament_id,
    p_draw_id,
    p_expected_draw_version,
    'PLAYOFF',
    p_expected_team_versions,
    p_expected_source_game_versions,
    p_games
  );
  if pg_catalog.jsonb_typeof(v_insert_result->'games') is distinct from 'array' then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_INSERT: atomic game insertion returned incomplete evidence.';
  end if;
  select coalesce(
           pg_catalog.array_agg(nullif(inserted.id, '')::uuid
             order by nullif(inserted.id, '')::uuid),
           '{}'::uuid[]
         )
    into v_inserted_game_ids
    from pg_catalog.jsonb_to_recordset(v_insert_result->'games') as inserted(id text);
  if v_inserted_game_ids is distinct from v_requested_game_ids then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_INSERT: inserted games do not exactly match the reviewed plan.';
  end if;

  select coalesce(pg_catalog.max(queue.priority), 0)
    into v_priority_base
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id;
  insert into public.tournament_day_live_queue (
    run_id, tournament_id, registration_day_id, day_draw_id, draw_id, game_id,
    team_a_id, team_b_id, state, priority, court_id,
    blocker_code, blocker_detail, version, last_operation_key,
    eligible_since, updated_by
  )
  select
    v_run.id, game.tournament_id, v_run.registration_day_id,
    v_day_draw.id, game.draw_id, game.id, game.team_a_id, game.team_b_id,
    case when game.team_a_id is not null and game.team_b_id is not null then 'WAITING' else 'BLOCKED' end,
    v_priority_base + pg_catalog.row_number() over (
      order by game.playoff_round nulls last, game.playoff_game_code nulls last, game.id
    ),
    null,
    case when game.team_a_id is null or game.team_b_id is null then 'DEPENDENCY_PENDING' else null end,
    case when game.team_a_id is null or game.team_b_id is null then 'Playoff source teams are not resolved yet.' else null end,
    1, p_operation_key,
    case when game.team_a_id is not null and game.team_b_id is not null then pg_catalog.clock_timestamp() else null end,
    p_actor
  from public.tournament_games as game
  where game.tournament_id::text = p_tournament_id
    and game.draw_id::text = p_draw_id
    and game.id = any(v_requested_game_ids)
    and game.stage = 'PLAYOFF'
    and game.registration_day_id = p_registration_day_id
    and game.event_option_id is not distinct from (
      select draw.event_option_id
        from public.tournament_event_draws as draw
       where draw.id = v_day_draw.draw_id
    )
    and not exists (
      select 1 from public.tournament_day_live_queue as existing
       where existing.run_id = v_run.id and existing.game_id = game.id
    )
  order by game.playoff_round nulls last, game.playoff_game_code nulls last, game.id;
  get diagnostics v_queue_insert_count = row_count;
  if v_queue_insert_count is distinct from pg_catalog.cardinality(v_requested_game_ids) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_QUEUE: every inserted playoff game must enter this day queue.';
  end if;
  select coalesce(
           pg_catalog.array_agg(queue.game_id order by queue.game_id),
           '{}'::uuid[]
         )
    into v_queued_game_ids
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
     and queue.day_draw_id = v_day_draw.id
     and queue.game_id = any(v_requested_game_ids)
     and queue.last_operation_key = p_operation_key;
  if v_queued_game_ids is distinct from v_requested_game_ids then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_QUEUE: queued games do not exactly match the inserted playoff plan.';
  end if;

  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.id = v_day_draw.id
   for update;
  update public.tournament_day_live_draws as day_draw
     set source_draw_updated_at = draw.updated_at,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
    from public.tournament_event_draws as draw
   where day_draw.id = v_day_draw.id and draw.id = day_draw.draw_id;
  v_assignments := public.fill_tournament_day_live_courts(v_run.id, p_operation_key, p_actor);
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return v_insert_result || pg_catalog.jsonb_build_object(
    'run', pg_catalog.to_jsonb(v_run),
    'assignments', v_assignments,
    'playoff_generation', true
  );
end;
$function$;

create or replace function public.admin_close_tournament_day_live_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
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
  v_intent jsonb;
  v_close_draw record;
  v_reviewed boolean;
  v_expected_podium_count integer;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_close_day'
  );
  if v_intent #>> '{payload,action}' is distinct from 'close_day'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #> '{payload,payload}' is distinct from '{}'::jsonb
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: close-day arguments do not match durable intent.';
  end if;
  if exists (
    select 1
      from public.tournament_admin_operations as operation
     where operation.club_id = p_club_id
       and operation.operation_key <> p_operation_key
       and operation.status in ('intent', 'mutated', 'recovery_required')
       and (
         operation.entity_id = pg_catalog.concat_ws(':', p_tournament_id, p_registration_day_id)
         or operation.lock_scope = pg_catalog.concat_ws(
           ':', 'tournament', p_tournament_id, 'day', p_registration_day_id
         )
         or (
           operation.entity_type = 'tournament_event_draw'
           and exists (
             select 1
               from public.tournament_day_live_draws as day_draw
              where day_draw.run_id = (
                select run.id
                  from public.tournament_day_live_runs as run
                 where run.club_id = p_club_id
                   and run.tournament_id::text = p_tournament_id
                   and run.registration_day_id = p_registration_day_id
              )
                and day_draw.draw_id::text = operation.entity_id
           )
         )
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_OPERATION: reconcile unsettled day operations before close.';
  end if;
  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state in ('ACTIVE', 'PAUSED')
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: day run changed or is not open.';
  end if;

  if not exists (
    select 1
      from public.tournament_day_live_draws as day_draw
     where day_draw.run_id = v_run.id
       and day_draw.state <> 'REMOVED'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_DRAWS: activate and complete at least one scheduled draw before closing the day.';
  end if;
  -- Close is rare and terminal.  A single deterministic relation-lock request
  -- waits out every in-flight setup writer and blocks day/event/draw phantoms
  -- until the exact supported-draw set has been rechecked and the run closes.
  -- A lock/deadlock victim fails closed and the guarded operation is retryable.
  lock table
    public.tournament_registration_days,
    public.tournament_event_options,
    public.tournament_event_draws
  in share mode;
  if not exists (
    select 1
      from public.tournament_registration_days as source_day
     where source_day.id = v_run.registration_day_id
       and source_day.tournament_id::text = p_tournament_id
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: selected day scope changed before close.';
  end if;

  perform queue.id
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
   order by queue.id
   for update;
  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.run_id = v_run.id
   order by claim.player_id, claim.id
   for update;
  if exists (
    select 1
      from public.tournament_day_live_queue as queue
     where queue.run_id = v_run.id
       and queue.state not in ('COMPLETED', 'WITHDRAWN')
  ) or exists (
    select 1
      from public.tournament_day_live_participant_claims as claim
     where claim.run_id = v_run.id and claim.released_at is null
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_BLOCKED: finish every owned game and release every player claim before closing the day.';
  end if;

  -- Close is the terminal day lifecycle step. Lock the exact draw-owned
  -- progression evidence before taking the draw rows so concurrent legacy
  -- podium/award operations cannot race the close decision.
  perform podium.id
    from public.tournament_podium as podium
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = podium.draw_id
   where podium.tournament_id = v_run.tournament_id
   order by podium.id
   for share;
  perform team.id
    from public.tournament_teams as team
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = team.draw_id
   where team.tournament_id = v_run.tournament_id
   order by team.id
   for share;
  perform game.id
    from public.tournament_games as game
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = game.draw_id
   where game.tournament_id = v_run.tournament_id
   order by game.id
   for share;
  perform badge.id
    from public.player_badges as badge
   where badge.club_id = p_club_id
     and badge.context_type = 'tournament'
     and badge.context_id::text like p_tournament_id || ':draw:%:podium:%'
   order by badge.id
   for share;
  perform draw.id
    from public.tournament_event_draws as draw
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = draw.id
   where draw.tournament_id = v_run.tournament_id
   order by draw.id
   for share;
  perform day_draw.id
    from public.tournament_day_live_draws as day_draw
   where day_draw.run_id = v_run.id
   order by day_draw.draw_id
   for update;

  if exists (
    select 1
      from public.tournament_event_draws as source_draw
      join public.tournament_event_options as source_event
        on source_event.id = source_draw.event_option_id
       and source_event.tournament_id = v_run.tournament_id::text
       and source_event.enabled is true
       and pg_catalog.upper(coalesce(source_event.status, 'DRAFT')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     where source_draw.tournament_id = v_run.tournament_id
       and pg_catalog.upper(coalesce(source_draw.status, 'DRAFT')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
       and coalesce(source_draw.hidden_from_primary_ops, false) is false
       and pg_catalog.upper(coalesce(source_draw.draw_kind, 'STANDARD')) = 'STANDARD'
       and (case
         when source_draw.registration_day_id is not null
           then source_draw.registration_day_id = v_run.registration_day_id
            and case
              when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
                then case
                  when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) > 0
                    then source_event.scheduled_day_ids ? source_draw.registration_day_id
                  else source_event.registration_day_id is null
                    or source_event.registration_day_id = source_draw.registration_day_id
                end
              else false
            end
         else case
           when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
             then case
               when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 1
                 then source_event.scheduled_day_ids ? v_run.registration_day_id
               when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 0
                 then source_event.registration_day_id = v_run.registration_day_id
               else false
             end
           else false
         end
       end) is true
       and not exists (
         select 1
           from public.tournament_day_live_draws as day_draw
          where day_draw.run_id = v_run.id
            and day_draw.draw_id = source_draw.id
            and day_draw.state <> 'REMOVED'
       )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_DRAWS: every currently supported scheduled draw must be activated and completed before close.';
  end if;

  for v_close_draw in
    select
      day_draw.draw_id,
      draw.tournament_id,
      draw.registration_day_id,
      draw.event_option_id,
      draw.name,
      draw.status,
      draw.draw_kind,
      draw.hidden_from_primary_ops
      from public.tournament_day_live_draws as day_draw
      join public.tournament_event_draws as draw on draw.id = day_draw.draw_id
     where day_draw.run_id = v_run.id and day_draw.state <> 'REMOVED'
     order by day_draw.draw_id
  loop
    if not exists (
      select 1
        from public.tournament_games as game
       where game.tournament_id = v_run.tournament_id
         and game.draw_id = v_close_draw.draw_id
         and game.stage = 'PLAYOFF'
    ) or exists (
      select 1
        from public.tournament_games as game
        left join public.tournament_teams as team_a
          on team_a.id = game.team_a_id
         and team_a.tournament_id = game.tournament_id
         and team_a.draw_id = game.draw_id
         and team_a.registration_day_id = v_run.registration_day_id
         and team_a.event_option_id = v_close_draw.event_option_id
        left join public.tournament_teams as team_b
          on team_b.id = game.team_b_id
         and team_b.tournament_id = game.tournament_id
         and team_b.draw_id = game.draw_id
         and team_b.registration_day_id = v_run.registration_day_id
         and team_b.event_option_id = v_close_draw.event_option_id
       where game.tournament_id = v_run.tournament_id
         and game.draw_id = v_close_draw.draw_id
         and (
           game.registration_day_id is distinct from v_run.registration_day_id
           or game.finalized_at is null
           or game.score_a is null or game.score_b is null
           or game.score_a = game.score_b
           or game.team_a_id is null or game.team_b_id is null
           or game.team_a_id = game.team_b_id
           or team_a.id is null or team_b.id is null
           or game.winner_team_id is null or game.loser_team_id is null
           or game.winner_team_id = game.loser_team_id
           or game.winner_team_id not in (game.team_a_id, game.team_b_id)
           or game.loser_team_id not in (game.team_a_id, game.team_b_id)
           or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
           or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id)
         )
    ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_PROGRESSION: every activated draw must finish its generated playoffs before close.';
    end if;

    select count(*) into v_expected_podium_count
      from public.tournament_draw_expected_playoff_podium(v_run.tournament_id,v_close_draw.draw_id);
    if v_expected_podium_count not in (2,3)
       or (select count(*) from public.tournament_podium podium
            where podium.tournament_id=v_run.tournament_id and podium.draw_id=v_close_draw.draw_id)
            <> v_expected_podium_count
       or exists (
         select 1 from public.tournament_draw_expected_playoff_podium(v_run.tournament_id,v_close_draw.draw_id) expected
          where not exists (
            select 1 from public.tournament_podium podium
              join public.tournament_teams team on team.id=podium.team_id
               and team.draw_id=podium.draw_id and team.tournament_id=podium.tournament_id
             where podium.tournament_id=v_run.tournament_id and podium.draw_id=v_close_draw.draw_id
               and podium.placement=expected.placement and podium.team_id=expected.team_id
          )
       ) then
      raise exception using errcode='22023',
        message='JUPR_TOURNAMENT_DAY_LIVE_CLOSE_PODIUM_RESULT: podium must exactly match the completed championship bracket, including its supported two- or three-place format.';
    end if;

    select exists (
      select 1
        from public.admin_activity_log as activity
        cross join lateral (
          select activity.after_json #> '{podium_review_evidence}' as evidence
        ) as reviewed
       where activity.club_id = p_club_id
         and activity.entity_type = 'tournament_event_draw'
         and activity.entity_id = v_close_draw.draw_id::text
         and activity.action_type = 'review_tournament_draw_podium_admin'
         and reviewed.evidence->>'contract' = 'jupr:tournament-podium-review:v1'
         and reviewed.evidence->>'tournament_id' = p_tournament_id
         and reviewed.evidence->>'draw_id' = v_close_draw.draw_id::text
         and pg_catalog.length(coalesce(reviewed.evidence->>'review_fingerprint', '')) = 64
         and reviewed.evidence #>> '{draw,id}' = v_close_draw.draw_id::text
         and reviewed.evidence #>> '{draw,tournament_id}' = v_close_draw.tournament_id::text
         and nullif(reviewed.evidence #>> '{draw,registration_day_id}', '')
               is not distinct from v_close_draw.registration_day_id
         and nullif(reviewed.evidence #>> '{draw,event_option_id}', '')
               is not distinct from v_close_draw.event_option_id::text
         and nullif(reviewed.evidence #>> '{draw,name}', '')
               is not distinct from nullif(v_close_draw.name, '')
         and nullif(reviewed.evidence #>> '{draw,status}', '')
               is not distinct from nullif(v_close_draw.status, '')
         and nullif(reviewed.evidence #>> '{draw,draw_kind}', '')
               is not distinct from nullif(v_close_draw.draw_kind, '')
         and coalesce(
               (reviewed.evidence #>> '{draw,hidden_from_primary_ops}')::boolean,
               false
             ) is not distinct from coalesce(v_close_draw.hidden_from_primary_ops, false)
         and pg_catalog.jsonb_typeof(reviewed.evidence->'team_versions') = 'array'
         and pg_catalog.jsonb_array_length(reviewed.evidence->'team_versions') = (
           select pg_catalog.count(*)
             from public.tournament_teams as team
            where team.tournament_id = v_run.tournament_id
              and team.draw_id = v_close_draw.draw_id
         )
         and not exists (
           select 1
             from public.tournament_teams as team
            where team.tournament_id = v_run.tournament_id
              and team.draw_id = v_close_draw.draw_id
              and not exists (
                select 1
                  from pg_catalog.jsonb_to_recordset(reviewed.evidence->'team_versions')
                    as version(id text, updated_at timestamptz)
                 where version.id = team.id::text
                   and version.updated_at is not distinct from team.updated_at
              )
         )
         and pg_catalog.jsonb_typeof(reviewed.evidence->'game_versions') = 'array'
         and pg_catalog.jsonb_array_length(reviewed.evidence->'game_versions') = (
           select pg_catalog.count(*)
             from public.tournament_games as game
            where game.tournament_id = v_run.tournament_id
              and game.draw_id = v_close_draw.draw_id
         )
         and not exists (
           select 1
             from public.tournament_games as game
            where game.tournament_id = v_run.tournament_id
              and game.draw_id = v_close_draw.draw_id
              and not exists (
                select 1
                  from pg_catalog.jsonb_to_recordset(reviewed.evidence->'game_versions')
                    as version(id text, updated_at timestamptz)
                 where version.id = game.id::text
                   and version.updated_at is not distinct from game.updated_at
              )
         )
         and pg_catalog.jsonb_typeof(reviewed.evidence->'podium') = 'array'
         and pg_catalog.jsonb_array_length(reviewed.evidence->'podium') = v_expected_podium_count
         and not exists (
           select 1
             from public.tournament_podium as podium
            where podium.tournament_id = v_run.tournament_id
              and podium.draw_id = v_close_draw.draw_id
              and not exists (
                select 1
                  from pg_catalog.jsonb_to_recordset(reviewed.evidence->'podium')
                    as evidence_row(id text, draw_id text, placement integer, team_id text, source text)
                 where evidence_row.id = podium.id::text
                   and evidence_row.draw_id = podium.draw_id::text
                   and evidence_row.placement = podium.placement
                   and evidence_row.team_id = podium.team_id::text
                   and pg_catalog.upper(evidence_row.source) = pg_catalog.upper(podium.source)
              )
         )
    ) into v_reviewed;
    if not v_reviewed then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_REVIEW: a current explicit podium review is required before close.';
    end if;

    if exists (
      with expected_awards as (
        select distinct
          participant.player_id,
          case podium.placement
            when 1 then 'tournament_champion'
            when 2 then 'tournament_runner_up'
            when 3 then 'tournament_third_place'
          end as badge_id,
          pg_catalog.concat_ws(
            ':', p_tournament_id, 'draw', v_close_draw.draw_id::text,
            'podium', podium.placement::text
          ) as context_id
          from public.tournament_podium as podium
          join public.tournament_teams as team
            on team.id = podium.team_id
           and team.tournament_id = podium.tournament_id
           and team.draw_id = podium.draw_id
          cross join lateral (values (team.player1_id), (team.player2_id))
            as participant(player_id)
         where podium.tournament_id = v_run.tournament_id
           and podium.draw_id = v_close_draw.draw_id
           and participant.player_id is not null
      ), actual_awards as (
        select badge.player_id, badge.badge_id, badge.context_id::text as context_id
          from public.player_badges as badge
         where badge.club_id = p_club_id
           and badge.context_type = 'tournament'
           and badge.revoked_at is null
           and badge.context_id::text like
             p_tournament_id || ':draw:' || v_close_draw.draw_id::text || ':podium:%'
      )
      select 1
       where exists (select * from expected_awards except select * from actual_awards)
          or exists (select * from actual_awards except select * from expected_awards)
    ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_AWARDS: every expected podium medal must exist exactly before close.';
    end if;
  end loop;

  update public.tournament_day_live_draws as day_draw
     set state = case when day_draw.state = 'REMOVED' then 'REMOVED' else 'PAUSED' end,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where day_draw.run_id = v_run.id;
  update public.tournament_day_live_courts as court
     set state = 'CLOSED',
         version = court.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where court.run_id = v_run.id;
  update public.tournament_day_live_runs as run
     set state = 'CLOSED',
         version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp(),
         closed_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'run', pg_catalog.to_jsonb(v_run),
    'closed', true
  );
end;
$function$;

create or replace function public.admin_award_tournament_draw_podium_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_expected_podium jsonb,
  p_expected_awards jsonb,
  p_badges jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_saved jsonb;
begin
  if jsonb_typeof(coalesce(p_expected_teams, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_teams, '[]'::jsonb)) = 0
     or jsonb_typeof(coalesce(p_expected_podium, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_podium, '[]'::jsonb)) not in (2,3)
     or jsonb_typeof(coalesce(p_expected_awards, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_awards, '[]'::jsonb)) = 0
     or jsonb_typeof(coalesce(p_badges, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_badges, '[]'::jsonb)) <> jsonb_array_length(p_expected_awards) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;

  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  perform podium.id
    from public.tournament_podium podium
   where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
   order by podium.id
   for update;
  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
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

  -- Two medal recipients are valid only for an explicitly configured
  -- two-qualifier pool championship whose real final is complete.
  if jsonb_array_length(p_expected_podium)=2 and (
       v_draw.pool_config is null
       or (select count(*) from public.tournament_draw_expected_playoff_podium(v_draw.tournament_id,v_draw.id))<>2
       or exists (
         select 1 from public.tournament_draw_expected_playoff_podium(v_draw.tournament_id,v_draw.id) expected
          where not exists (select 1 from jsonb_to_recordset(p_expected_podium) proposed(placement integer,team_id text)
            where proposed.placement=expected.placement and proposed.team_id=expected.team_id::text)
       )
     ) then
    raise exception using errcode='P0001', message='JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE';
  end if;

  if (
       select count(*) from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_teams)
     or (
       select count(distinct x.id) from jsonb_to_recordset(p_expected_teams) as x(id text)
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
        where nullif(x.id, '') is null or x.updated_at is null or not exists (
          select 1 from public.tournament_teams team
           where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
             and team.id::text = x.id and team.updated_at = x.updated_at
        )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;

  if (
       select count(*) from public.tournament_podium podium
        where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_podium)
     or (
       select count(distinct x.placement)
         from jsonb_to_recordset(p_expected_podium) as x(placement integer)
     ) <> jsonb_array_length(p_expected_podium)
     or exists (
       select 1 from public.tournament_podium podium
        where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
          and not exists (
            select 1
              from jsonb_to_recordset(p_expected_podium) as x(placement integer, team_id text, source text)
             where x.placement = podium.placement
               and x.team_id = podium.team_id::text
               and upper(coalesce(x.source, '')) = upper(coalesce(podium.source, ''))
          )
     )
     or exists (
       select 1
         from jsonb_to_recordset(p_expected_podium) as x(placement integer, team_id text, source text)
        where x.placement not between 1 and 3 or nullif(x.team_id, '') is null or not exists (
          select 1 from public.tournament_podium podium
           where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
             and podium.placement = x.placement
             and podium.team_id::text = x.team_id
             and upper(coalesce(podium.source, '')) = upper(coalesce(x.source, ''))
        )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE';
  end if;

  if (
    with derived as (
      select team.player1_id::bigint as player_id,
             case podium.placement
               when 1 then 'tournament_champion'
               when 2 then 'tournament_runner_up'
               when 3 then 'tournament_third_place'
             end as badge_id,
             p_tournament_id || ':draw:' || p_draw_id || ':podium:' || podium.placement::text as context_id
        from public.tournament_podium podium
        join public.tournament_teams team on team.id = podium.team_id
       where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
         and team.player1_id is not null
      union all
      select team.player2_id::bigint,
             case podium.placement
               when 1 then 'tournament_champion'
               when 2 then 'tournament_runner_up'
               when 3 then 'tournament_third_place'
             end,
             p_tournament_id || ':draw:' || p_draw_id || ':podium:' || podium.placement::text
        from public.tournament_podium podium
        join public.tournament_teams team on team.id = podium.team_id
       where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
         and team.player2_id is not null
    ), expected as (
      select x.player_id, x.badge_id, x.context_id
        from jsonb_to_recordset(p_expected_awards) as x(player_id bigint, badge_id text, context_id text)
    )
    select (select count(*) from derived) <> (select count(*) from expected)
        or (select count(*) from expected) <> (select count(*) from (select distinct * from expected) distinct_expected)
        or exists (select * from derived except select * from expected)
        or exists (select * from expected except select * from derived)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;

  if exists (
       select 1
         from jsonb_to_recordset(p_badges) as badge(club_id text, player_id bigint, badge_id text, context_type text, context_id text)
        where badge.club_id <> p_club_id
           or badge.context_type <> 'tournament'
           or not exists (
             select 1
               from jsonb_to_recordset(p_expected_awards) as expected(player_id bigint, badge_id text, context_id text)
              where expected.player_id = badge.player_id
                and expected.badge_id = badge.badge_id
                and expected.context_id = badge.context_id
           )
     )
     or (
       select count(distinct (badge.player_id, badge.badge_id, badge.context_id))
         from jsonb_to_recordset(p_badges) as badge(player_id bigint, badge_id text, context_id text)
     ) <> jsonb_array_length(p_badges) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;
  if exists (
    select 1 from public.player_badges badge
     where badge.club_id = p_club_id
       and badge.context_type = 'tournament'
       and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_ALREADY_EXISTS';
  end if;

  with inserted as (
    insert into public.player_badges (
      id, club_id, player_id, badge_id, earned_at, context_type,
      context_id, match_id, value_num, value_json
    )
    select
      coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
      p_club_id,
      x.player_id,
      x.badge_id,
      coalesce(x.earned_at, clock_timestamp()),
      'tournament',
      x.context_id,
      x.match_id,
      x.value_num,
      coalesce(x.value_json, '{}'::jsonb)
    from jsonb_to_recordset(p_badges) as x(
      id text, player_id bigint, badge_id text, earned_at timestamptz,
      context_id text, match_id text, value_num numeric, value_json jsonb
    )
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.context_id, inserted.badge_id, inserted.player_id), '[]'::jsonb)
    into v_saved from inserted;
  if jsonb_array_length(v_saved) <> jsonb_array_length(p_expected_awards) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;
  return jsonb_build_object('ok', true, 'badges', v_saved);
end;
$$;

revoke all on function public.assert_tournament_pool_schedule_complete(uuid),
  public.tournament_draw_expected_playoff_podium(uuid,uuid),
  public.assert_tournament_pool_playoff_selection(uuid,integer,jsonb) from public, anon, authenticated;
grant execute on function public.assert_tournament_pool_schedule_complete(uuid),
  public.tournament_draw_expected_playoff_podium(uuid,uuid),
  public.assert_tournament_pool_playoff_selection(uuid,integer,jsonb) to service_role;
-- CREATE OR REPLACE preserves the existing protected RPC ACLs; repeat the
-- explicit grants so a restored schema cannot inherit PUBLIC execution.
revoke execute on function public.assert_tournament_day_live_draw_ready(text,text,text,uuid,timestamptz),
  public.admin_generate_tournament_day_playoffs_cas(text,text,text,text,integer,bigint,bigint,bigint,timestamptz,jsonb,jsonb,jsonb,text,text,text),
  public.admin_close_tournament_day_live_cas(text,text,text,bigint,bigint,text,text,text),
  public.admin_award_tournament_draw_podium_cas(text,text,text,timestamptz,jsonb,jsonb,jsonb,jsonb)
  from public, anon, authenticated;
grant execute on function public.assert_tournament_day_live_draw_ready(text,text,text,uuid,timestamptz),
  public.admin_generate_tournament_day_playoffs_cas(text,text,text,text,integer,bigint,bigint,bigint,timestamptz,jsonb,jsonb,jsonb,text,text,text),
  public.admin_close_tournament_day_live_cas(text,text,text,bigint,bigint,text,text,text),
  public.admin_award_tournament_draw_podium_cas(text,text,text,timestamptz,jsonb,jsonb,jsonb,jsonb)
  to service_role;
notify pgrst, 'reload schema';
commit;
