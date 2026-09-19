-- Staging-only rollback rehearsal. No user profiles, rosters or sign-ups are retained.
begin;
do $$
<<pool_test>>
declare
 home text:='pool-home-'||gen_random_uuid()::text; away text:='pool-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='pool-qa@example.invalid'; sid uuid:=gen_random_uuid();
 mid uuid; mid2 uuid; member_id uuid; member_nonce uuid; response_id uuid; response_nonce uuid;
 home_player bigint:=8000000000000+(random()*1000000000000)::bigint;
 away_player bigint:=8000000000000+(random()*1000000000000)::bigint;
 result jsonb; retry jsonb; signup jsonb; settings jsonb; share uuid; entry_before jsonb;
 requester text:=repeat('a',64); table_name text;
begin
 foreach table_name in array array['pcs_interclub_pool_settings','pcs_interclub_pool_members','pcs_interclub_availability_settings','pcs_interclub_availability_responses','pcs_interclub_pool_rate_buckets'] loop
  if has_table_privilege('anon','public.'||table_name,'SELECT') or has_table_privilege('authenticated','public.'||table_name,'SELECT')
   or not (select relrowsecurity from pg_class where oid=('public.'||table_name)::regclass) then raise exception 'Pool browser grants or RLS incorrect'; end if;
 end loop;
 if has_function_privilege('anon','public.pcs_interclub_pool_public_action(text,jsonb,text)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_interclub_pool_action(uuid,text,text,uuid,text,jsonb)','EXECUTE') then raise exception 'Pool RPC browser grants incorrect'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(home,home,'Pool Home',false,'draft','draft'),(away,away,'Pool Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 insert into public.players(id,club_id,name,normalized_name,rating,active,gender)
 values(home_player,home,'Pool QA','pool qa',1400,true,'male'),(away_player,away,'Pool QA','pool qa',1600,true,'male');
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Pool QA League','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.5'),'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
  jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4),
  jsonb_build_object('host_club_id',away,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '40 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,'{"3.5":{"min_rating":null,"max_rating":3.99,"women_required":null}}');
 begin
  perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'settings','{"expected_revision":0,"open":true}');
  raise exception 'Unaccepted club opened pool';
 exception when insufficient_privilege then null; end;
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 settings:=public.pcs_interclub_pool_action(actor,actor_email,home,sid,'settings','{"expected_revision":0,"open":true}');
 share:=(settings->>'share_id')::uuid;
 signup:=jsonb_build_object('share_id',share,'name','Pool QA','email','pool-qa@example.invalid','divisions',jsonb_build_array('3.5'),
  'notes','Visiting for part of the season','request_id',gen_random_uuid(),'request_fingerprint','fixture-payload','email_consent',true);
 result:=public.pcs_interclub_pool_public_action('signup',signup,requester);
 member_id:=(result->'member'->>'id')::uuid; member_nonce:=(result->'member'->>'token_nonce')::uuid;
 retry:=public.pcs_interclub_pool_public_action('signup',signup,requester);
 if result<>retry then raise exception 'Signup retry did not return original member'; end if;
 retry:=public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid()),requester);
 if retry->>'status'<>'already_registered' or retry ? 'member' then raise exception 'Duplicate signup exposed private member'; end if;
 if exists(select 1 from public.pcs_interclub_entries where season_id=sid) or exists(select 1 from public.pcs_interclub_teams where season_id=sid) then
  raise exception 'Interest signup created ratings or roster'; end if;
 begin
  perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'member',jsonb_build_object('member_id',member_id,'expected_revision',1,'player_id',away_player,'status','active'));
  raise exception 'Cross-club player linked';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_interclub_pool_action(actor,actor_email,away,sid,'member',jsonb_build_object('member_id',member_id,'expected_revision',1,'player_id',away_player,'status','active'));
  raise exception 'Another club changed pool member';
 exception when no_data_found then null; end;
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'member',jsonb_build_object('member_id',member_id,'expected_revision',1,'player_id',home_player,'status','active'));
 select to_jsonb(e) into entry_before from public.pcs_interclub_entries e where season_id=sid and player_id=home_player;
 if entry_before is null then raise exception 'Approved linked player did not receive a season identity'; end if;
 select id into mid from public.pcs_interclub_meets where season_id=sid and plan_index=0;
 select id into mid2 from public.pcs_interclub_meets where season_id=sid and plan_index=1;
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'availability',jsonb_build_object('meet_id',mid,'expected_revision',0,'open',true,'deadline',now()+interval '18 days'));
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'availability',jsonb_build_object('meet_id',mid2,'expected_revision',0,'open',true,'deadline',now()+interval '38 days'));
 result:=public.pcs_interclub_pool_action(actor,actor_email,home,sid,'invite',jsonb_build_object('meet_id',mid,'member_ids',jsonb_build_array(member_id)));
 response_id:=(result->'responses'->0->>'id')::uuid; response_nonce:=(result->'responses'->0->>'token_nonce')::uuid;
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'invite',jsonb_build_object('meet_id',mid2,'member_ids',jsonb_build_array(member_id)));
 perform public.pcs_interclub_pool_public_action('respond_meet',jsonb_build_object('id',response_id,'nonce',response_nonce,'season_id',sid,'club_id',home,'expected_revision',1,'status','available'),requester);
 begin
  perform public.pcs_interclub_pool_public_action('respond_meet',jsonb_build_object('id',response_id,'nonce',response_nonce,'season_id',sid,'club_id',home,'expected_revision',1,'status','maybe'),requester);
  raise exception 'Stale response overwritten';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('respond_meet',jsonb_build_object('id',response_id,'nonce',gen_random_uuid(),'season_id',sid,'club_id',home,'expected_revision',2,'status','maybe'),requester);
  raise exception 'Wrong capability changed response';
 exception when no_data_found then null; end;
 result:=public.pcs_interclub_pool_action(actor,actor_email,home,sid,'invite',jsonb_build_object('meet_id',mid,'member_ids',jsonb_build_array(member_id)));
 if result->'responses'->0->>'status'<>'available' then raise exception 'Repeat invitation reset response'; end if;
 if (select r.status from public.pcs_interclub_availability_responses r where r.member_id=pool_test.member_id and r.meet_id=mid2 limit 1)<>'invited' then raise exception 'One meet response changed another'; end if;
 perform public.pcs_interclub_pool_public_action('update_season',signup||jsonb_build_object('id',member_id,'nonce',member_nonce,'season_id',sid,'club_id',home,'expected_revision',2,'status','withdrawn'),requester);
 begin
  perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'invite',jsonb_build_object('meet_id',mid,'member_ids',jsonb_build_array(member_id)));
  raise exception 'Withdrawn member invited';
 exception when invalid_parameter_value then null; end;
 if (select count(*) from public.pcs_interclub_availability_responses where season_id=sid)<>2 then raise exception 'History was lost'; end if;
 if (select rating from public.players where id=home_player)<>1400
  or (select to_jsonb(e) from public.pcs_interclub_entries e where season_id=sid and player_id=home_player) is distinct from entry_before
  or exists(select 1 from public.pcs_interclub_teams where season_id=sid) then raise exception 'Availability altered ratings or lineups'; end if;
 settings:=public.pcs_interclub_pool_action(actor,actor_email,home,sid,'settings','{"expected_revision":1,"open":true,"rotate_link":true}');
 if (settings->>'share_id')::uuid=share then raise exception 'Share link did not rotate'; end if;
 begin
  perform public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid()),requester);
  raise exception 'Rotated share link still accepted';
 exception when no_data_found then null; end;
end $$;
rollback;
