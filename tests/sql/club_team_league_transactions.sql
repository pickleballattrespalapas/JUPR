-- Isolated staging rehearsal: fixture identities and all writes roll back.
begin;
set local role service_role;
do $$
<<rehearsal>>
declare
  club text := 'tres_palapas';
  nonce text := gen_random_uuid()::text;
  league text;
  size integer;
  member_number integer;
  team_number integer;
  player bigint;
  players bigint[];
  teams uuid[];
  result jsonb;
  settings jsonb;
  version integer;
  fixture uuid;
  match_id bigint;
  operation uuid;
  key text;
  a1 bigint;
  a2 bigint;
  b1 bigint;
  b2 bigint;
begin
  foreach size in array array[2,4] loop
    league := '__team_release_' || nonce || '_' || size;
    key := 'rehearsal:' || nonce || ':' || size;
    insert into public.leagues_metadata(club_id,league_name,league_type,status)
      values(club,league,'Team','draft');
    settings := jsonb_build_object('team_size',size,'team_category','mixed',
      'mixed_required_men',size/2,'mixed_required_women',size/2,
      'registration_open',false,'timezone','America/Mazatlan','start_date','2026-10-05');
    perform public.team_league_save_settings_v2(gen_random_uuid(),club,league,key || ':save',
      repeat('a',64),0,settings,'fixture@invalid.example','club_owner','release_rehearsal');
    result := public.team_league_save_settings_v2(gen_random_uuid(),club,league,key || ':save',
      repeat('a',64),0,settings,'fixture@invalid.example','club_owner','release_rehearsal');
    if not (result->>'idempotent')::boolean then raise exception 'Save retry was not idempotent'; end if;
    if not exists(select 1 from public.team_league_settings s where s.club_id=club
      and s.league_name=league and s.team_size=size and s.settings_version=1) then
      raise exception 'Roster size was not saved';
    end if;
    begin
      perform public.team_league_save_settings_v2(gen_random_uuid(),club,league,key || ':stale',
        repeat('b',64),0,settings,'fixture@invalid.example','club_owner','release_rehearsal');
      raise exception 'Stale settings accepted';
    exception when serialization_failure then null; end;
    players := '{}';
    for member_number in 1..(size*2) loop
      insert into public.players(club_id,name,normalized_name,gender)
        values(club,league || ' player ' || member_number,lower(league || ' player ' || member_number),
          case when member_number%2=1 then 'male' else 'female' end) returning id into player;
      players := array_append(players,player);
    end loop;
    begin
      insert into public.team_league_teams(club_id,league_name,team_name,status,captain_player_id,
        partner_player_id,captain_contact_email,partner_contact_email)
        values(club,league,'Public closed signup','pending_partner',players[1],players[2],
          'fixture@invalid.example','partner@invalid.example');
      raise exception 'Closed public registration accepted';
    exception when object_not_in_prerequisite_state then
      if sqlerrm <> 'TEAM_LEAGUE_REGISTRATION_CLOSED' then raise; end if;
    end;
    teams := '{}';
    for team_number in 0..1 loop
      select s.roster_version into version from public.team_league_settings s
        where s.club_id=club and s.league_name=league;
      result := public.team_league_create_team_v1(gen_random_uuid(),club,league,'Fixture team ' || team_number,
        players[team_number*size+1],'fixture@invalid.example',players[team_number*size+2],
        'partner@invalid.example',version,key || ':team:' || team_number,repeat('c',64),
        'fixture@invalid.example','club_owner','release_rehearsal');
      teams := array_append(teams,(result->'team'->>'id')::uuid);
      if size=4 then
        if result->'team'->>'status' <> 'pending_partner' then raise exception 'Partial roster marked complete'; end if;
        for member_number in 3..4 loop
          select s.roster_version into version from public.team_league_settings s
            where s.club_id=club and s.league_name=league;
          perform public.team_league_apply_roster_action_v1(gen_random_uuid(),club,league,'add_member',
            teams[team_number+1],players[team_number*size+member_number],'primary','active',
            'member@invalid.example',null,version,key || ':member:' || team_number || ':' || member_number,
            repeat('d',64),'fixture@invalid.example','club_owner','release_rehearsal');
        end loop;
      end if;
    end loop;
    if (select count(*) from public.team_league_teams t where t.club_id=club and t.league_name=league and t.status='confirmed') <> 2
      or (select count(*) from public.team_league_team_members m where m.club_id=club and m.league_name=league and m.status='active') <> size*2 then
      raise exception 'Complete normalized rosters were not confirmed';
    end if;
    select s.roster_version into version from public.team_league_settings s
      where s.club_id=club and s.league_name=league;
    perform public.team_league_replace_schedule_v2(gen_random_uuid(),club,league,'regular',key || ':schedule',
      repeat('e',64),0,0,version,public.team_league_confirmed_roster_fingerprint_v1(club,league),
      jsonb_build_array(jsonb_build_object('round_number',1,'week_number',1,'bracket_slot',1,
        'scheduled_at','2026-10-06T01:00:00Z','team_a_id',teams[1],'team_b_id',teams[2],'status','scheduled')),
      'fixture@invalid.example','club_owner','release_rehearsal');
    select f.id into strict fixture from public.team_league_fixtures f where f.club_id=club and f.league_name=league;
    begin
      perform public.team_league_create_team_v1(gen_random_uuid(),club,league,'Late admin team',
        players[1],'fixture@invalid.example',players[2],'partner@invalid.example',version,
        key || ':late-team',repeat('9',64),'fixture@invalid.example','club_owner','release_rehearsal');
      raise exception 'Admin team creation bypassed schedule lock';
    exception when object_not_in_prerequisite_state then
      if sqlerrm <> 'TEAM_LEAGUE_REGISTRATION_CLOSED' then raise; end if;
    end;
    -- Use the last two primary members: this exercises the full four-player roster.
    a1:=players[size-1]; a2:=players[size]; b1:=players[size*2-1]; b2:=players[size*2];
    operation:=gen_random_uuid();
    insert into public.team_league_operations(id,club_id,league_name,idempotency_key,request_fingerprint,
      operation_type,status,request_json,actor_email,actor_role,source)
      values(operation,club,league,key || ':score',repeat('f',64),'admin_score_fixture','started',
        jsonb_build_object('fixture_id',fixture),'fixture@invalid.example','club_owner','release_rehearsal');
    perform public.team_league_reserve_fixture_score_v1(operation,club,league,fixture,teams[1],teams[2]);
    insert into public.matches(club_id,league,match_type,date,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2)
      values(club,league,'Team League','2026-10-05',a1,a2,b1,b2,11,7) returning id into match_id;
    result:=public.team_league_finalize_fixture_v2(operation,club,fixture,'complete',11,7,teams[1],match_id,
      a1,a2,b1,b2,'[]',null,'fixture@invalid.example','club_owner','release_rehearsal');
    result:=public.team_league_finalize_fixture_v2(operation,club,fixture,'complete',11,7,teams[1],match_id,
      a1,a2,b1,b2,'[]',null,'fixture@invalid.example','club_owner','release_rehearsal');
    if not (result->>'idempotent')::boolean then raise exception 'Score retry was not idempotent'; end if;
    if not exists(select 1 from public.team_league_fixtures f where f.id=fixture
      and f.status='complete' and f.official_match_id=match_id and f.winner_team_id=teams[1]) then
      raise exception 'Fixture result did not persist';
    end if;
    if (select count(*) from public.matches m where m.club_id=club and m.league=rehearsal.league) <> 1 then
      raise exception 'Duplicate canonical match';
    end if;
    if (select s.standings_version from public.team_league_settings s where s.club_id=club and s.league_name=league) <> 1 then
      raise exception 'Standings version did not advance exactly once';
    end if;
    -- Match Log corrections and exclusions must still control team standings.
    update public.matches set score_t1=7,score_t2=11 where id=match_id;
    operation:=gen_random_uuid();
    insert into public.team_league_operations(id,club_id,league_name,idempotency_key,request_fingerprint,
      operation_type,status,request_json,actor_email,actor_role,source)
      values(operation,club,league,key || ':reconcile',repeat('1',64),'admin_reconcile_fixture','started',
        jsonb_build_object('fixture_id',fixture),'fixture@invalid.example','club_owner','release_rehearsal');
    perform public.team_league_reconcile_fixture_v2(operation,club,fixture,'fixture@invalid.example','club_owner','release_rehearsal');
    if not exists(select 1 from public.team_league_fixtures f where f.id=fixture and f.winner_team_id=teams[2]) then
      raise exception 'Match Log correction did not update standings';
    end if;
    update public.matches set deleted_at=now() where id=match_id;
    operation:=gen_random_uuid();
    insert into public.team_league_operations(id,club_id,league_name,idempotency_key,request_fingerprint,
      operation_type,status,request_json,actor_email,actor_role,source)
      values(operation,club,league,key || ':excluded',repeat('2',64),'admin_reconcile_fixture','started',
        jsonb_build_object('fixture_id',fixture),'fixture@invalid.example','club_owner','release_rehearsal');
    perform public.team_league_reconcile_fixture_v2(operation,club,fixture,'fixture@invalid.example','club_owner','release_rehearsal');
    if not exists(select 1 from public.team_league_fixtures f where f.id=fixture and f.status='cancelled' and f.winner_team_id is null) then
      raise exception 'Excluded match continued to count in standings';
    end if;
  end loop;
end $$;
rollback;
select 'Two- and four-player setup, stale-save rejection, complete rosters, scheduling, results, idempotency, Match Log corrections and exclusions passed; all fixture writes rolled back' as result;
