-- Keep tournament-day play authoritative to the reviewed draw roster.
-- Check-in, payment, and waiver tracking remain operator information only and
-- cannot block day activation, draw activation, queue eligibility, or the
-- database-authoritative atomic court scheduler.

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

create or replace function public.tournament_day_live_players_ready(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_player_ids integer[]
)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_player_id integer;
  v_registration_count integer;
begin
  -- Retain the structural and club-player safety boundary used by the atomic
  -- scheduler. The draw roster, not check-in/payment/waiver workflow state,
  -- is authoritative for live tournament play.
  if pg_catalog.cardinality(coalesce(p_player_ids, '{}'::integer[])) not in (2, 4)
     or pg_catalog.cardinality(p_player_ids) <>
       pg_catalog.cardinality(array(
         select distinct participant.player_id
           from pg_catalog.unnest(p_player_ids) as participant(player_id)
       )) then
    return false;
  end if;

  foreach v_player_id in array p_player_ids
  loop
    perform player.id
      from public.players as player
     where player.id = v_player_id
       and player.club_id::text = p_club_id
     for share;
    if not found then
      return false;
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
      return false;
    end if;
  end loop;

  return true;
end;
$function$;

-- Attendance and waiver are operator tracking, so recording them cannot be
-- blocked merely because the player is currently on a court. Identity and
-- substitute mutations remain fenced while an active participant claim exists.
create or replace function public.guard_tournament_check_in_during_player_claim()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_old_player_id integer;
  v_new_player_id integer;
begin
  if old.attendee_identity_key is not distinct from new.attendee_identity_key
     and old.approved_substitute_player_id is not distinct from new.approved_substitute_player_id
     and old.approved_substitute_name is not distinct from new.approved_substitute_name
     and old.tournament_id is not distinct from new.tournament_id
     and old.registration_day_id is not distinct from new.registration_day_id
     and old.registration_id is not distinct from new.registration_id then
    return new;
  end if;
  if old.attendee_identity_key ~ '^player:[0-9]+$' then
    v_old_player_id := pg_catalog.split_part(old.attendee_identity_key, ':', 2)::integer;
  end if;
  if new.attendee_identity_key ~ '^player:[0-9]+$' then
    v_new_player_id := pg_catalog.split_part(new.attendee_identity_key, ':', 2)::integer;
  end if;
  if exists (
    select 1
      from public.tournament_day_live_participant_claims as claim
      join public.tournament_day_live_runs as run on run.id = claim.run_id
     where (
         (run.tournament_id = old.tournament_id and run.registration_day_id = old.registration_day_id)
         or
         (run.tournament_id = new.tournament_id and run.registration_day_id = new.registration_day_id)
       )
       and run.state in ('ACTIVE', 'PAUSED')
       and claim.released_at is null
       and claim.player_id in (v_old_player_id, v_new_player_id)
  ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYER_CLAIM: release or score the player current court before changing check-in identity or substitute assignment.';
  end if;
  return new;
end;
$function$;

revoke execute on function public.assert_tournament_day_live_draw_ready(
  text, text, text, uuid, timestamptz
) from public, anon, authenticated;
revoke execute on function public.tournament_day_live_players_ready(
  text, text, text, integer[]
) from public, anon, authenticated;
revoke execute on function public.guard_tournament_check_in_during_player_claim()
  from public, anon, authenticated;

grant execute on function public.assert_tournament_day_live_draw_ready(
  text, text, text, uuid, timestamptz
) to service_role;
grant execute on function public.tournament_day_live_players_ready(
  text, text, text, integer[]
) to service_role;
grant execute on function public.guard_tournament_check_in_during_player_claim()
  to service_role;
