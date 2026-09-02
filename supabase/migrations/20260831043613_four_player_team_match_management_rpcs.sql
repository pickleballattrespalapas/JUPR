-- Canonical source for the four-player team-match management RPCs first
-- applied to staging as migration 20260831043613. Keep these privileged RPCs
-- callable only by FastAPI's service_role data plane.

create or replace function public.admin_update_team_match_competition_v1(
  p_club_id text,
  p_competition_id text,
  p_expected_version integer,
  p_patch jsonb,
  p_actor text
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_current public.team_match_competitions%rowtype;
  v_updated public.team_match_competitions%rowtype;
begin
  select * into v_current
    from public.team_match_competitions
   where id = p_competition_id::uuid
     and club_id = p_club_id
   for update;
  if not found then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_NOT_FOUND';
  end if;
  if v_current.version <> p_expected_version then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_VERSION_CONFLICT';
  end if;
  if v_current.status in ('COMPLETE', 'ARCHIVED') then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_SCHEDULE_LOCKED';
  end if;
  if nullif(p_patch->>'roster_mode', '') is not null
     and p_patch->>'roster_mode' <> v_current.roster_mode
     and exists (
       select 1
         from public.team_match_teams team
        where team.competition_id = v_current.id
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_ROSTER_MODE_LOCKED';
  end if;
  if exists (
       select 1
         from public.team_match_matchups matchup
        where matchup.competition_id = v_current.id
     ) and (
       (
         nullif(p_patch->>'roster_mode', '') is not null
         and p_patch->>'roster_mode' <> v_current.roster_mode
       )
       or (
         nullif(p_patch->>'scoring_mode', '') is not null
         and p_patch->>'scoring_mode' <> v_current.scoring_mode
       )
       or (
         nullif(p_patch->>'target_score', '') is not null
         and (p_patch->>'target_score')::smallint <> v_current.target_score
       )
       or (
         nullif(p_patch->>'win_by', '') is not null
         and (p_patch->>'win_by')::smallint <> v_current.win_by
       )
       or (
         nullif(p_patch->>'tiebreak_mode', '') is not null
         and p_patch->>'tiebreak_mode' <> v_current.tiebreak_mode
       )
       or (
         nullif(p_patch->>'round_robin_cycles', '') is not null
         and (p_patch->>'round_robin_cycles')::smallint
           <> v_current.round_robin_cycles
       )
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_SCHEDULE_LOCKED';
  end if;

  update public.team_match_competitions
     set name = coalesce(nullif(pg_catalog.btrim(p_patch->>'name'), ''), name),
         roster_mode = coalesce(
           nullif(p_patch->>'roster_mode', ''),
           roster_mode
         ),
         scoring_mode = coalesce(
           nullif(p_patch->>'scoring_mode', ''),
           scoring_mode
         ),
         target_score = coalesce(
           nullif(p_patch->>'target_score', '')::smallint,
           target_score
         ),
         win_by = coalesce(
           nullif(p_patch->>'win_by', '')::smallint,
           win_by
         ),
         tiebreak_mode = coalesce(
           nullif(p_patch->>'tiebreak_mode', ''),
           tiebreak_mode
         ),
         round_robin_cycles = coalesce(
           nullif(p_patch->>'round_robin_cycles', '')::smallint,
           round_robin_cycles
         ),
         version = version + 1,
         updated_by = nullif(pg_catalog.btrim(p_actor), ''),
         updated_at = pg_catalog.clock_timestamp()
   where id = v_current.id
  returning * into v_updated;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'competition', pg_catalog.to_jsonb(v_updated)
  );
end
$$;

create or replace function public.admin_save_team_match_team_v1(
  p_club_id text,
  p_competition_id text,
  p_team_id text,
  p_team_name text,
  p_pairing text,
  p_members jsonb,
  p_expected_competition_version integer,
  p_expected_team_version integer,
  p_actor text
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_comp public.team_match_competitions%rowtype;
  v_team public.team_match_teams%rowtype;
  v_team_id uuid;
  v_member jsonb;
  v_slots text[];
  v_player_ids bigint[];
begin
  select * into v_comp
    from public.team_match_competitions
   where id = p_competition_id::uuid
     and club_id = p_club_id
   for update;
  if not found then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_NOT_FOUND';
  end if;
  if v_comp.version <> p_expected_competition_version then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_VERSION_CONFLICT';
  end if;
  if v_comp.status in ('COMPLETE', 'ARCHIVED')
     or exists (
       select 1
         from public.team_match_matchups
        where competition_id = v_comp.id
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_SCHEDULE_LOCKED';
  end if;
  if nullif(pg_catalog.btrim(p_team_name), '') is null
     or pg_catalog.jsonb_typeof(p_members) <> 'array'
     or pg_catalog.jsonb_array_length(p_members) <> 4 then
    raise exception using
      errcode = '22023',
      message = 'TEAM_MATCH_ROSTER_INVALID';
  end if;

  select pg_catalog.array_agg(
           upper(member->>'slot')
           order by upper(member->>'slot')
         ),
         pg_catalog.array_agg(
           (member->>'player_id')::bigint
           order by (member->>'player_id')::bigint
         )
    into v_slots, v_player_ids
    from pg_catalog.jsonb_array_elements(p_members) member;

  if pg_catalog.cardinality(v_player_ids) <> 4
     or (
       select count(distinct selected.player_id)
         from pg_catalog.unnest(v_player_ids) as selected(player_id)
     ) <> 4
     or (
       select count(distinct selected.slot_key)
         from pg_catalog.unnest(v_slots) as selected(slot_key)
     ) <> 4 then
    raise exception using
      errcode = '22023',
      message = 'TEAM_MATCH_ROSTER_INVALID';
  end if;
  if (
       v_comp.roster_mode = 'MIXED_2M2W'
       and v_slots <> array['M1', 'M2', 'W1', 'W2']::text[]
     )
     or (
       v_comp.roster_mode <> 'MIXED_2M2W'
       and v_slots <> array['P1', 'P2', 'P3', 'P4']::text[]
     ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_MATCH_ROSTER_INVALID';
  end if;
  if exists (
    select 1
      from pg_catalog.unnest(v_player_ids) selected(player_id)
      left join public.players player
        on player.id = selected.player_id
       and player.club_id = p_club_id
     where player.id is null
        or not coalesce(player.active, true)
        or player.inactive_at is not null
  ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_MATCH_PLAYER_INVALID';
  end if;
  if exists (
    select 1
      from public.team_match_team_members member
      join public.team_match_teams team on team.id = member.team_id
     where member.competition_id = v_comp.id
       and member.status = 'ACTIVE'
       and team.status = 'ACTIVE'
       and member.player_id = any(v_player_ids)
       and (p_team_id is null or team.id <> p_team_id::uuid)
  ) then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_ROSTER_CONFLICT';
  end if;

  if nullif(pg_catalog.btrim(p_team_id), '') is null then
    insert into public.team_match_teams (
      competition_id,
      club_id,
      name,
      pairing_code,
      created_by,
      updated_by
    ) values (
      v_comp.id,
      p_club_id,
      pg_catalog.btrim(p_team_name),
      upper(p_pairing),
      nullif(pg_catalog.btrim(p_actor), ''),
      nullif(pg_catalog.btrim(p_actor), '')
    ) returning * into v_team;
  else
    select * into v_team
      from public.team_match_teams
     where id = p_team_id::uuid
       and competition_id = v_comp.id
     for update;
    if not found then
      raise exception using
        errcode = 'P0001',
        message = 'TEAM_MATCH_NOT_FOUND';
    end if;
    if p_expected_team_version is null
       or v_team.version <> p_expected_team_version then
      raise exception using
        errcode = 'P0001',
        message = 'TEAM_MATCH_VERSION_CONFLICT';
    end if;
    update public.team_match_teams
       set name = pg_catalog.btrim(p_team_name),
           pairing_code = upper(p_pairing),
           status = 'ACTIVE',
           version = version + 1,
           updated_by = nullif(pg_catalog.btrim(p_actor), ''),
           updated_at = pg_catalog.clock_timestamp()
     where id = v_team.id
    returning * into v_team;
  end if;
  v_team_id := v_team.id;

  delete from public.team_match_team_members where team_id = v_team_id;
  for v_member in
    select member from pg_catalog.jsonb_array_elements(p_members) member
  loop
    insert into public.team_match_team_members (
      competition_id,
      team_id,
      club_id,
      slot_key,
      player_id,
      display_name_snapshot,
      gender_snapshot,
      status
    ) values (
      v_comp.id,
      v_team_id,
      p_club_id,
      upper(v_member->>'slot'),
      (v_member->>'player_id')::bigint,
      coalesce(
        nullif(pg_catalog.btrim(v_member->>'display_name'), ''),
        'Player'
      ),
      nullif(pg_catalog.btrim(v_member->>'gender'), ''),
      'ACTIVE'
    );
  end loop;

  update public.team_match_competitions
     set version = version + 1,
         updated_by = nullif(pg_catalog.btrim(p_actor), ''),
         updated_at = pg_catalog.clock_timestamp()
   where id = v_comp.id;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'team', pg_catalog.to_jsonb(v_team),
    'competition_version', v_comp.version + 1
  );
end
$$;

create or replace function public.admin_replace_team_match_schedule_v1(
  p_club_id text,
  p_competition_id text,
  p_expected_version integer,
  p_matchups jsonb,
  p_actor text
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_comp public.team_match_competitions%rowtype;
  v_matchup jsonb;
  v_game jsonb;
  v_matchup_id uuid;
  v_matchup_count integer := 0;
  v_game_count integer := 0;
begin
  select * into v_comp
    from public.team_match_competitions
   where id = p_competition_id::uuid
     and club_id = p_club_id
   for update;
  if not found then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_NOT_FOUND';
  end if;
  if v_comp.version <> p_expected_version then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_VERSION_CONFLICT';
  end if;
  if v_comp.status in ('COMPLETE', 'ARCHIVED')
     or exists (
       select 1
         from public.team_match_games
        where competition_id = v_comp.id
          and (
            status not in ('SCHEDULED', 'BLOCKED')
            or publish_status not in ('NOT_STARTED', 'NOT_REQUIRED')
          )
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'TEAM_MATCH_SCHEDULE_LOCKED';
  end if;
  if pg_catalog.jsonb_typeof(p_matchups) <> 'array'
     or pg_catalog.jsonb_array_length(p_matchups) < 1 then
    raise exception using
      errcode = '22023',
      message = 'TEAM_MATCH_SCHEDULE_INVALID';
  end if;

  delete from public.team_match_matchups where competition_id = v_comp.id;
  for v_matchup in
    select matchup from pg_catalog.jsonb_array_elements(p_matchups) matchup
  loop
    insert into public.team_match_matchups (
      competition_id,
      club_id,
      stage,
      round_number,
      slot_number,
      team_a_id,
      team_b_id,
      status
    ) values (
      v_comp.id,
      p_club_id,
      upper(coalesce(v_matchup->>'stage', 'ROUND_ROBIN')),
      (v_matchup->>'round_number')::integer,
      (v_matchup->>'slot_number')::integer,
      (v_matchup->>'team_a_id')::uuid,
      (v_matchup->>'team_b_id')::uuid,
      'SCHEDULED'
    ) returning id into v_matchup_id;
    v_matchup_count := v_matchup_count + 1;

    if pg_catalog.jsonb_typeof(v_matchup->'games') <> 'array'
       or pg_catalog.jsonb_array_length(v_matchup->'games') not in (4, 5) then
      raise exception using
        errcode = '22023',
        message = 'TEAM_MATCH_SCHEDULE_INVALID';
    end if;
    for v_game in
      select game from pg_catalog.jsonb_array_elements(v_matchup->'games') game
    loop
      insert into public.team_match_games (
        competition_id,
        matchup_id,
        club_id,
        game_code,
        game_order,
        label,
        team_a_player_ids,
        team_b_player_ids,
        counts_for_rating,
        status,
        publish_status
      ) values (
        v_comp.id,
        v_matchup_id,
        p_club_id,
        upper(v_game->>'game_code'),
        (v_game->>'game_order')::smallint,
        coalesce(
          nullif(pg_catalog.btrim(v_game->>'label'), ''),
          'Team game'
        ),
        array(
          select value::bigint
            from pg_catalog.jsonb_array_elements_text(
              v_game->'team_a_player_ids'
            ) value
        ),
        array(
          select value::bigint
            from pg_catalog.jsonb_array_elements_text(
              v_game->'team_b_player_ids'
            ) value
        ),
        coalesce((v_game->>'counts_for_rating')::boolean, true),
        upper(coalesce(v_game->>'status', 'SCHEDULED')),
        case
          when coalesce((v_game->>'counts_for_rating')::boolean, true)
          then 'NOT_STARTED'
          else 'NOT_REQUIRED'
        end
      );
      v_game_count := v_game_count + 1;
    end loop;
  end loop;

  update public.team_match_competitions
     set status = 'ACTIVE',
         version = version + 1,
         updated_by = nullif(pg_catalog.btrim(p_actor), ''),
         updated_at = pg_catalog.clock_timestamp()
   where id = v_comp.id;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'matchup_count', v_matchup_count,
    'game_count', v_game_count,
    'competition_version', v_comp.version + 1
  );
end
$$;

revoke execute on function public.admin_update_team_match_competition_v1(
  text,
  text,
  integer,
  jsonb,
  text
) from public, anon, authenticated;
grant execute on function public.admin_update_team_match_competition_v1(
  text,
  text,
  integer,
  jsonb,
  text
) to service_role;

revoke execute on function public.admin_save_team_match_team_v1(
  text,
  text,
  text,
  text,
  text,
  jsonb,
  integer,
  integer,
  text
) from public, anon, authenticated;
grant execute on function public.admin_save_team_match_team_v1(
  text,
  text,
  text,
  text,
  text,
  jsonb,
  integer,
  integer,
  text
) to service_role;

revoke execute on function public.admin_replace_team_match_schedule_v1(
  text,
  text,
  integer,
  jsonb,
  text
) from public, anon, authenticated;
grant execute on function public.admin_replace_team_match_schedule_v1(
  text,
  text,
  integer,
  jsonb,
  text
) to service_role;

comment on function public.admin_update_team_match_competition_v1(
  text,
  text,
  integer,
  jsonb,
  text
) is 'Service-only CAS update for a four-player team-match competition.';
comment on function public.admin_save_team_match_team_v1(
  text,
  text,
  text,
  text,
  text,
  jsonb,
  integer,
  integer,
  text
) is 'Service-only CAS create/update for a four-player team-match roster.';
comment on function public.admin_replace_team_match_schedule_v1(
  text,
  text,
  integer,
  jsonb,
  text
) is 'Service-only CAS replacement for a four-player team-match schedule.';

notify pgrst, 'reload schema';
