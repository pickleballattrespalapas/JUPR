-- Run inside a transaction after the migration, then ROLLBACK the whole test.
-- Uses synthetic admin identity and a real staging club player; no writes persist.
do $$
declare
  target_club text;
  target_player bigint;
  actor uuid := gen_random_uuid();
  operation uuid := gen_random_uuid();
  recognition uuid := gen_random_uuid();
  season_id uuid := gen_random_uuid();
  payload jsonb;
  result jsonb;
  changed jsonb;
  prior_awards bigint;
begin
  select club_id,id into target_club,target_player from public.players order by id limit 1;
  select count(*) into prior_awards from public.player_badges;
  insert into public.admin_role_assignments(club_id,user_id,email,role) values(target_club,actor,'badge-test@invalid.example','administrator');
  payload:=jsonb_build_object('player_id',target_player,'badge_id','good_sport','criteria',jsonb_build_array('honest_calls'),'note','Transaction verification only','contribution_date','2026-01-01');
  result:=public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',recognition,'award_community',payload);
  if not (result->>'ok')::boolean then raise exception 'award failed'; end if;
  result:=public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',recognition,'award_community',payload);
  if not (result->>'replayed')::boolean then raise exception 'retry was not replayed'; end if;
  if (select count(*) from public.player_badges) <> prior_awards+1 then raise exception 'retry duplicated award'; end if;
  begin
    perform public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',recognition,'award_community',payload||jsonb_build_object('note','Different contribution'));
    raise exception 'changed retry accepted';
  exception when sqlstate '22023' then null; end;
  begin
    perform public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'operator',gen_random_uuid(),'award_community',payload);
    raise exception 'operator accepted';
  exception when insufficient_privilege then null; end;
  begin
    perform public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',gen_random_uuid(),'award_community',payload||jsonb_build_object('criteria',jsonb_build_array('volunteer')));
    raise exception 'invalid criteria accepted';
  exception when sqlstate '22023' then null; end;
  perform public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',gen_random_uuid(),'award_community',payload||jsonb_build_object('note','A separate contribution'));
  if (select count(*) from public.player_badges) <> prior_awards+2 then raise exception 'repeat contribution suppressed'; end if;

  payload:=jsonb_build_object('id',season_id,'name','Transaction test season','start_date','2040-11-01','end_date','2041-03-31','timezone','America/Mazatlan','expected_revision',0);
  result:=public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',operation,'save_season',payload);
  if result->'season'->>'start_date' <> '2040-11-01' then raise exception 'season date lost'; end if;
  begin
    perform public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',gen_random_uuid(),'save_season',payload||jsonb_build_object('id',gen_random_uuid()));
    raise exception 'overlap accepted';
  exception when sqlstate '22023' then null; end;
  begin
    perform public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',gen_random_uuid(),'save_season',payload);
    raise exception 'stale revision accepted';
  exception when serialization_failure then null; end;
  insert into public.player_badges(club_id,player_id,badge_id,context_type,context_id,value_json)
    values(target_club,target_player,'battle_tested','season','badge-season:'||season_id,
      jsonb_build_object('season_start','2040-11-01','season_end','2041-03-31','season_timezone','America/Mazatlan'));
  begin
    perform public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',gen_random_uuid(),'save_season',payload||jsonb_build_object('expected_revision',1,'end_date','2041-04-01'));
    raise exception 'awarded season date change accepted';
  exception when sqlstate '22023' then null; end;
  result:=public.admin_manage_badges_v1(target_club,'badge-test@invalid.example',actor,'administrator',gen_random_uuid(),'save_season',payload||jsonb_build_object('expected_revision',1,'name','Renamed season'));
  if result->'season'->>'name' <> 'Renamed season' then raise exception 'season rename failed'; end if;
  begin
    insert into public.player_badges(club_id,player_id,badge_id,context_type,context_id,value_json)
      values(target_club,target_player,'consistency','season','badge-season:'||season_id,
        jsonb_build_object('season_start','2040-11-01','season_end','2041-04-01','season_timezone','America/Mazatlan'));
    raise exception 'stale season evidence accepted';
  exception when serialization_failure then null; end;
  if has_function_privilege('anon','public.admin_manage_badges_v1(text,text,uuid,text,uuid,text,jsonb)','EXECUTE')
    or has_table_privilege('authenticated','public.badge_seasons','INSERT') then raise exception 'client write exposure'; end if;
end;
$$;
select 'badge admin transactional tests passed' as result;
