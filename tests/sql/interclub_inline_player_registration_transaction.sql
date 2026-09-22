-- Staging-only rollback rehearsal. No existing club, person or season is used.
begin;

do $$
declare
 home text:='inline-organizer-'||gen_random_uuid()::text; away text:='inline-club-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='inline-qa@example.invalid'; sid uuid:=gen_random_uuid(); share uuid; meet_id uuid;
 regular_request uuid:=gen_random_uuid(); public_request uuid:=gen_random_uuid(); late_request uuid:=gen_random_uuid();
 regular_profile jsonb:='{"name":"Inline regular QA","starting_jupr":3.25,"gender":"female"}';
 public_profile jsonb:='{"name":"Inline public QA","starting_jupr":3.5,"gender":"female","email":"inline-public@example.invalid"}';
 late_profile jsonb:='{"name":"Inline late QA","starting_jupr":3.25,"gender":"female","email":"inline-late@example.invalid"}';
 failed_profile jsonb:='{"name":"Inline rollback QA","starting_jupr":3.25}';
 existing_id bigint:=8600000000000+(random()*1000000000)::bigint;
 regular_id bigint; public_id bigint; late_id bigint; regular_member uuid; public_member uuid; late_member uuid;
 result jsonb; replay jsonb; public_payload jsonb; initial_count integer;
begin
 if has_table_privilege('anon','public.pcs_interclub_player_creation_requests','SELECT')
  or has_table_privilege('authenticated','public.pcs_interclub_player_creation_requests','INSERT')
  or has_table_privilege('service_role','public.pcs_interclub_player_creation_requests','UPDATE')
  or has_function_privilege('authenticated','public.pcs_interclub_register_new_player(uuid,text,text,uuid,boolean,uuid,jsonb,jsonb,text)','EXECUTE')
  or has_function_privilege('anon','public.pcs_interclub_create_directory_player(text,jsonb)','EXECUTE')
  or not(select relrowsecurity from pg_class where oid='public.pcs_interclub_player_creation_requests'::regclass) then
  raise exception 'Creation receipts or RPC privileges are incorrect'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
  values(home,home,'Inline organizer',false,'draft','draft'),(away,away,'Inline registering club',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
  values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 insert into public.players(id,club_id,name,normalized_name,rating,starting_rating,active,gender)
  values(existing_id,away,'Existing rating QA','existing rating qa',1376,1200,true,'female');
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Inline Registration QA',
  'start_date',(current_date+1)::text,'end_date',(current_date+90)::text,'timezone','America/Mazatlan',
  'divisions','["3.5"]'::jsonb,'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
   jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,'{"3.5":{"min_rating":null,"max_rating":3.999,"women_required":2}}');
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 select id into meet_id from public.pcs_interclub_meets where season_id=sid and plan_index=0;
 select share_id into share from public.pcs_interclub_pool_settings where season_id=sid and club_id=away;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,regular_request,regular_profile,'[]',null);
  raise exception 'Unconfigured registration created a player';
 exception when sqlstate 'PT423' then null; end;
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now()-interval '1 day',now()+interval '1 day');
 begin
  perform public.pcs_interclub_register_new_player(gen_random_uuid(),actor_email,away,sid,false,regular_request,regular_profile,'[]',null);
  raise exception 'Forged actor created a player';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=now() where club_id=away;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,regular_request,regular_profile,'[]',null);
  raise exception 'Revoked administrator created a player';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=null where club_id=away;
 set local role service_role;
 result:=public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,regular_request,regular_profile,'[]',null);
 reset role;
 regular_member:=(result->>'id')::uuid; regular_id:=(result->>'player_id')::bigint;
 if regular_id is null or result->>'approval_status'<>'approved' or (result->>'late_join')::boolean
  or result->>'email'<>'' or result->>'consent_at' is not null or result->'divisions'<>'["3.5"]'::jsonb then
  raise exception 'Regular creation did not produce the linked, eligible signup without consent'; end if;
 if not exists(select 1 from public.players where id=regular_id and club_id=away and active and rating=1300 and starting_rating=1300 and gender='female')
  or not exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=regular_id and starting_rating=3.25) then
  raise exception 'Regular creation did not preserve the reviewed starting JUPR'; end if;
 replay:=public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,regular_request,regular_profile,'[]',null);
 if replay->>'id'<>regular_member::text or replay->>'player_id'<>regular_id::text then raise exception 'Exact admin replay changed identity'; end if;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,regular_request,regular_profile||'{"starting_jupr":4.0}', '[]',null);
  raise exception 'Changed replay payload was accepted';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,gen_random_uuid(),regular_profile,'[]',null);
  raise exception 'Fresh request duplicated an existing signup';
 exception when sqlstate 'P4092' then null; end;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,gen_random_uuid(),
   '{"name":"  EXISTING   rating QA  ","starting_jupr":7}', '[]',null);
  raise exception 'Existing directory profile was recreated or its rating overwritten';
 exception when sqlstate 'P4091' then null; end;
 perform public.pcs_interclub_pool_bulk_add(actor,actor_email,away,sid,'[{"name":"Unlinked collision QA","player_id":null}]');
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,gen_random_uuid(),
   '{"name":"Unlinked collision QA","starting_jupr":3.25}', '[]',null);
  raise exception 'Unlinked signup acquired a duplicate profile';
 exception when sqlstate 'P4092' then null; end;
 if exists(select 1 from public.players where club_id=away and name='Unlinked collision QA') then raise exception 'Duplicate precheck left an orphan'; end if;
 select count(*) into initial_count from public.pcs_interclub_entries where season_id=sid;
 -- The API rejects a missing request ID. A direct trusted call reaches the
 -- pool NOT NULL constraint after creating its profile, proving atomic rollback.
 begin
  perform public.pcs_interclub_pool_public_action('signup',jsonb_build_object('share_id',share,
   'request_fingerprint','inline-rollback','name','Inline rollback QA','email','rollback@example.invalid',
   'email_consent',true,'divisions','["3.5"]'::jsonb,'notes','',
   'new_player',failed_profile||'{"email":"rollback@example.invalid"}'),repeat('i',64));
  raise exception 'Downstream NOT NULL failure did not abort registration';
 exception when not_null_violation then null; end;
 if exists(select 1 from public.players where club_id=away and name='Inline rollback QA')
  or exists(select 1 from public.pcs_interclub_pool_members where season_id=sid and name='Inline rollback QA')
  or (select count(*) from public.pcs_interclub_player_creation_requests where season_id=sid)<>1
  or (select count(*) from public.pcs_interclub_entries where season_id=sid)<>initial_count then
  raise exception 'Failed registration left a profile, signup, seed or replay receipt'; end if;

 public_payload:=jsonb_build_object('share_id',share,'request_id',public_request,'request_fingerprint','inline-public-v1',
  'name',public_profile->>'name','email',public_profile->>'email','email_consent',true,'divisions','["3.5"]'::jsonb,'notes','','new_player',public_profile);
 set local role service_role;
 result:=public.pcs_interclub_pool_public_action('signup',public_payload,repeat('i',64));
 reset role;
 public_member:=(result->'member'->>'id')::uuid; public_id:=(result->'member'->>'player_id')::bigint;
 if result->>'status'<>'registered' or public_id is null or result->'member'->>'approval_status'<>'approved'
  or result->'member'->>'consent_at' is null
  or not exists(select 1 from public.players where id=public_id and club_id=away and rating=1400 and starting_rating=1400 and gender='female')
  or not exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=public_id and starting_rating=3.5) then
  raise exception 'Public creation did not link its reviewed rating and explicit consent'; end if;
 replay:=public.pcs_interclub_pool_public_action('signup',public_payload,repeat('i',64));
 if replay->'member'->>'id'<>public_member::text then raise exception 'Exact public replay duplicated signup'; end if;
 begin
  perform public.pcs_interclub_pool_public_action('signup',public_payload||'{"request_fingerprint":"inline-public-changed"}',repeat('i',64));
  raise exception 'Changed public replay accepted';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('signup',public_payload||jsonb_build_object('request_id',gen_random_uuid()),repeat('i',64));
  raise exception 'Public creation duplicated an existing signup';
 exception when sqlstate 'P4092' then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('signup',public_payload||jsonb_build_object('request_id',gen_random_uuid(),'name','Different name QA'),repeat('i',64));
  raise exception 'Public profile identity did not match the signup';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,true,late_request,late_profile,'[]',null);
  raise exception 'Open season created a late request';
 exception when sqlstate 'PT423' then null; end;

 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,1,now()-interval '1 day',now()-interval '1 hour');
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,false,gen_random_uuid(),failed_profile,'[]',null);
  raise exception 'Closed season created a regular player';
 exception when sqlstate 'PT423' then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('signup',public_payload||jsonb_build_object('request_id',gen_random_uuid()),repeat('i',64));
  raise exception 'Closed season allowed public creation';
 exception when sqlstate 'PT423' then null; end;
 set local role service_role;
 result:=public.pcs_interclub_register_new_player(actor,actor_email,away,sid,true,late_request,late_profile,'[]',null);
 reset role;
 late_member:=(result->>'id')::uuid; late_id:=(result->>'player_id')::bigint;
 if late_id is null or result->>'approval_status'<>'pending' or (result->>'late_join')::boolean is not true
  or result->>'email'<>'inline-late@example.invalid' or result->>'consent_at' is not null
  or result->>'notes'<>'' or result->>'approved_at' is not null
  or not exists(select 1 from public.pcs_interclub_late_player_requests where member_id=late_member and reason='' and contact_email='inline-late@example.invalid')
  or exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=late_id) then
  raise exception 'No-note late request autoapproved or invented contact consent'; end if;
 replay:=public.pcs_interclub_register_new_player(actor,actor_email,away,sid,true,late_request,late_profile,'[]',null);
 if replay->>'id'<>late_member::text or replay->>'player_id'<>late_id::text then raise exception 'Late replay changed identity'; end if;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,true,late_request,late_profile,'[]','Changed note');
  raise exception 'Late retry changed its immutable request';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_interclub_register_new_player(actor,actor_email,away,sid,true,gen_random_uuid(),late_profile,'[]',null);
  raise exception 'Late player acquired a duplicate profile';
 exception when sqlstate 'P4092' then null; end;
 begin
  perform public.pcs_save_interclub_meet_roster(actor,actor_email,away,sid,meet_id,1,gen_random_uuid(),0,'Pending inline pair','3.5',array[regular_id,late_id]);
  raise exception 'Pending inline player entered a roster';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_review_interclub_pool_member(actor,actor_email,away,sid,late_member,1,true,'Club self-approval');
  raise exception 'Requesting club approved its own late player';
 exception when insufficient_privilege then null; end;
 perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,late_member,1,true,'Commissioner accepts the reviewed profile');
 if not exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=late_id and starting_rating=3.25) then
  raise exception 'Approved late profile lacks its reviewed league rating'; end if;
 perform public.pcs_save_interclub_meet_roster(actor,actor_email,away,sid,meet_id,1,gen_random_uuid(),0,'Approved inline pair','3.5',array[regular_id,late_id]);
 -- Existing-profile late requests also accept blank notes without bypassing approval.
 result:=public.pcs_request_interclub_late_player(actor,actor_email,away,sid,existing_id,'["3.5"]','');
 if result->>'approval_status'<>'pending' or (result->>'late_join')::boolean is not true
  or not exists(select 1 from public.pcs_interclub_late_player_requests where member_id=(result->>'id')::uuid and reason='') then
  raise exception 'Existing-profile blank-note request lost its approval guard'; end if;
 if (select count(*) from public.players where club_id=away)<>4
  or (select count(*) from public.pcs_interclub_player_creation_requests where season_id=sid)<>2
  or (select count(*) from public.pcs_interclub_pool_members where season_id=sid and player_id=regular_id)<>1
  or (select count(*) from public.pcs_interclub_pool_members where season_id=sid and player_id=public_id)<>1
  or (select count(*) from public.pcs_interclub_pool_members where season_id=sid and player_id=late_id)<>1
  or not exists(select 1 from public.players where id=existing_id and rating=1376 and starting_rating=1200)
  or not exists(select 1 from public.players where id=public_id and rating=1400 and starting_rating=1400) then
  raise exception 'Replay, rejection or approval duplicated a profile or changed saved club ratings'; end if;
end $$;
rollback;
