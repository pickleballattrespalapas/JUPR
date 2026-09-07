-- Run in BEGIN / ROLLBACK against staging after the program migration.
do $$
declare
 club text; player bigint; rev bigint; actor uuid:=gen_random_uuid(); operation uuid:=gen_random_uuid();
 candidate jsonb; tie jsonb; payload jsonb; response jsonb; original_id uuid; earned timestamptz;
 prior_digest text; after_digest text; test_session uuid:=gen_random_uuid();
begin
 select club_id,id into club,player from public.players order by id limit 1;
 select md5(string_agg(to_jsonb(p)::text,',' order by p.id)) into prior_digest from public.player_badges p;
 select revision into rev from public.badge_program_state where club_id=club;
 candidate:=jsonb_build_array(jsonb_build_object('player_id',player,'badge_id','round_robin_wins_1','context_type','overall',
     'context_id','transaction-test','earned_at','2026-01-01T00:00:00Z','value_num',1,'value_json',jsonb_build_object('rule_version','program-badges-v1')));
 response:=public.apply_program_badges_v1(club,rev,candidate,'[]','[]');
 if (response->>'inserted')::int <> 1 then raise exception 'first apply must insert'; end if;
 select id,earned_at into original_id,earned from public.player_badges where club_id=club and context_id='transaction-test';
 response:=public.apply_program_badges_v1(club,rev,candidate,'[]','[]');
 if (response->>'inserted')::int <> 0 or (response->>'revoked')::int <> 0 then raise exception 'retry changed awards'; end if;
 perform public.apply_program_badges_v1(club,rev,'[]','[]','[]');
 if not exists(select 1 from public.player_badges where id=original_id and revoked_at is not null and earned_at=earned) then raise exception 'correction lost identity/date'; end if;
 response:=public.apply_program_badges_v1(club,rev,candidate,'[]','[]');
 if (response->>'restored')::int <> 1 or not exists(select 1 from public.player_badges where id=original_id and revoked_at is null) then raise exception 'restoration duplicated identity'; end if;
 update public.player_badges set revoked_at=now(),revoked_by=actor where id=original_id;
 perform public.apply_program_badges_v1(club,rev,candidate,'[]','[]');
 if not exists(select 1 from public.player_badges where id=original_id and revoked_by=actor) then raise exception 'manual revocation erased'; end if;
 begin
   perform public.apply_program_badges_v1(club,rev-1,candidate,'[]','[]');
   raise exception 'stale revision accepted';
 exception when serialization_failure then null; end;
 begin
   perform public.apply_program_badges_v1(club,rev,null,'[]','[]');
   raise exception 'incomplete response accepted';
 exception when invalid_parameter_value then null; end;
 begin
   perform public.apply_program_badges_v1(club,rev,jsonb_build_array(candidate->0 || jsonb_build_object('badge_id','participant')),'[]','[]');
   raise exception 'legacy badge mutation accepted';
 exception when invalid_parameter_value then null; end;

 insert into public.admin_role_assignments(club_id,user_id,email,role) values(club,actor,'program-badge-test@invalid.example','administrator');
 insert into public.live_sessions(id,club_id,session_key,status,title,source,state)
 values(test_session,club,'badge-test-'||test_session,'completed','Badge transaction test','play_generator',
    jsonb_build_object('event',jsonb_build_object('sourceEventUid','transaction-test','type','round_robin')));
 select revision into rev from public.badge_program_state where club_id=club;
 tie:=jsonb_build_object('source_key','live:transaction-test','source_fingerprint',repeat('a',64),'display_fingerprint',repeat('b',64),
   'leaders',jsonb_build_array(jsonb_build_object('player_id',player,'participant_id','p1','name','Test player','wins',3,'differential',10,'points',30)));
 perform public.apply_program_badges_v1(club,rev,candidate,jsonb_build_array(tie),'[]');
 payload:=jsonb_build_object('source_key',tie->>'source_key','source_fingerprint',tie->>'source_fingerprint','winner_player_id',player);
 begin
   perform public.admin_resolve_round_robin_v1(club,'program-badge-test@invalid.example',actor,'operator',operation,payload);
   raise exception 'operator chose winner';
 exception when insufficient_privilege then null; end;
 begin
   perform public.admin_resolve_round_robin_v1(club,'program-badge-test@invalid.example',actor,'administrator',operation,payload||jsonb_build_object('source_fingerprint',repeat('c',64)));
   raise exception 'stale tie accepted';
 exception when serialization_failure then null; end;
 response:=public.admin_resolve_round_robin_v1(club,'program-badge-test@invalid.example',actor,'administrator',operation,payload);
 if not exists(select 1 from public.live_sessions where id=test_session and state->'event'->'round_robin_winner'->>'participant_id'='p1') then raise exception 'public winner decision missing'; end if;
 if response <> public.admin_resolve_round_robin_v1(club,'program-badge-test@invalid.example',actor,'administrator',operation,payload) then raise exception 'decision retry changed result'; end if;
 if (select count(*) from public.badge_round_robin_operations where club_id=club and operation_id=operation) <> 1 then raise exception 'duplicate decision'; end if;
 if (select revision from public.badge_program_state where club_id=club) <= rev then raise exception 'decision failed to enqueue evaluation'; end if;
 begin
   perform public.admin_resolve_round_robin_v1(club,'program-badge-test@invalid.example',actor,'administrator',operation,payload||jsonb_build_object('winner_player_id',-1));
   raise exception 'changed retry accepted';
 exception when serialization_failure then null; end;

 select revision into rev from public.badge_program_state where club_id=club;
 update public.players set name=name||' trigger-test' where id=player and club_id=club;
 if (select revision from public.badge_program_state where club_id=club) <= rev then raise exception 'source changes lost'; end if;
 if has_function_privilege('anon','public.apply_program_badges_v1(text,bigint,jsonb,jsonb,jsonb)','EXECUTE')
    or has_function_privilege('authenticated','public.admin_resolve_round_robin_v1(text,text,uuid,text,uuid,jsonb)','EXECUTE')
    or has_table_privilege('authenticated','public.badge_program_state','UPDATE') then raise exception 'private program data exposed'; end if;
 select md5(string_agg(to_jsonb(p)::text,',' order by p.id)) into after_digest from public.player_badges p where p.id<>original_id;
 if prior_digest is distinct from after_digest then raise exception 'existing awards changed'; end if;
end $$;
select 'program badge transaction tests passed' result;
