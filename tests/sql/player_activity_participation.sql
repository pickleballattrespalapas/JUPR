-- Staging only. All fixture identities, memberships, operations and audits roll back.
begin;
do $$
declare
  club text := 'activity-fixture-' || gen_random_uuid()::text;
  other_club text := 'activity-other-' || gen_random_uuid()::text;
  league text := 'Returning Players';
  ids bigint[] := '{}';
  pid bigint;
  i integer;
  result jsonb;
begin
  insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
  values(club,club,'Activity fixture',false,'draft','draft'),
        (other_club,other_club,'Other fixture',false,'draft','draft');
  for i in 1..5 loop
    pid := 8000000000000 + (random()*1000000000000)::bigint;
    ids := array_append(ids,pid);
    insert into public.players(id,club_id,name,normalized_name,rating,active,inactive_at,gender)
    values(pid,case when i=5 then other_club else club end,
      case when i=4 then 'Old Profile (MERGED into Remaining #1)' else 'Returning '||i end,
      'activity fixture '||i,1600,i=2,now()-interval '30 days','female');
  end loop;
  insert into public.leagues_metadata(club_id,league_name,status,is_active,match_format,league_type)
  values(club,league,'active',true,'doubles','Team');

  result := public.admin_apply_league_roster_batch_atomic_v1(
    gen_random_uuid(),club,league,'activity-roster-add',repeat('a',64),'activate',
    to_jsonb(ids[1:2]),null,'activity-fixture@example.test','administrator','activity_regression');
  if (select count(*) from public.league_ratings where club_id=club and league_name=league
      and player_id=any(ids[1:2]) and is_active) <> 2 then
    raise exception 'Inactive players were not added to the league';
  end if;
  for i in 4..5 loop
    begin
      perform public.admin_apply_league_roster_batch_atomic_v1(
        gen_random_uuid(),club,league,'activity-roster-reject-'||i,repeat('b',64),'activate',
        jsonb_build_array(ids[i]),null,'activity-fixture@example.test','administrator','activity_regression');
      raise exception 'Merged or foreign-club identity entered league roster';
    exception when invalid_parameter_value then
      if (i=4 and sqlerrm not like '%PLAYER_MERGED%')
        or (i=5 and sqlerrm not like '%PLAYER_NOT_FOUND%') then raise; end if;
    end;
  end loop;

  insert into public.team_league_settings(club_id,league_name,status,registration_open,substitute_pool_enabled,allow_substitutes)
  values(club,league,'registration_open',true,true,true);
  result := public.team_league_register_public_v1(
    gen_random_uuid(),club,league,'activity-team-signup',repeat('c',64),'team',ids[1],ids[2],
    'Returning Pair','captain@example.test','partner@example.test',null,repeat('d',64),
    now()+interval '1 day','activity_regression');
  if not exists(select 1 from public.team_league_teams where club_id=club
      and captain_player_id=ids[1] and partner_player_id=ids[2]) then
    raise exception 'Inactive players could not register a team';
  end if;
  if (select count(*) from public.team_league_team_members where club_id=club
      and player_id=any(ids[1:2]) and status in ('active','invited')) <> 2 then
    raise exception 'Team roster did not retain inactive club players';
  end if;
  insert into public.team_league_substitute_pool(club_id,league_name,player_id,status)
  values(club,league,ids[3],'available');
  begin
    insert into public.team_league_substitute_pool(club_id,league_name,player_id,status)
    values(club,league,ids[4],'available');
    raise exception 'Merged identity entered substitute pool';
  exception when foreign_key_violation then
    if sqlerrm not like '%POOL_PLAYER_UNAVAILABLE%' then raise; end if;
  end;
  begin
    perform public.team_league_register_public_v1(
      gen_random_uuid(),club,league,'activity-merged-signup',repeat('e',64),'solo',ids[4],null,
      null,'merged@example.test',null,null,null,null,'activity_regression');
    raise exception 'Merged identity registered for a team league';
  exception when foreign_key_violation then
    if sqlerrm not like '%PLAYER_UNAVAILABLE%' then raise; end if;
  end;
  if exists(select 1 from public.players where id=any(ids) and inactive_at is null)
    or (select active from public.players where id=ids[1])
    or not (select active from public.players where id=ids[2])
    or (select rating from public.players where id=ids[1]) <> 1600 then
    raise exception 'Participation changed player activity or rating';
  end if;
end
$$;
select 'PASS: inactive participation, merged identity and club guards, unchanged activity/rating' as result;
rollback;
