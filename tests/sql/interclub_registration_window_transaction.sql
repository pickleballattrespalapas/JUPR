-- Staging-only rollback rehearsal: every club, player and season is synthetic.
begin;
do $$
<<registration_window_test>>
declare
 home text:='window-home-'||gen_random_uuid()::text; away text:='window-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='window-qa@example.invalid'; sid uuid:=gen_random_uuid();
 p1 bigint:=8200000000000+(random()*1000000000)::bigint; p2 bigint; p3 bigint;
 mid uuid; batch_id uuid:=gen_random_uuid(); member_id uuid; guest_id uuid; share uuid; old_share uuid;
 result jsonb; signup jsonb; phase text; window_revision integer:=0; blocked integer:=0;
 saved public.pcs_interclub_pool_members; initial_count integer; window_audits integer;
begin
 p2:=p1+1;p3:=p1+2;
 if public.pcs_interclub_registration_phase(null,null,now())<>'unconfigured'
  or public.pcs_interclub_registration_phase(now()+interval '1 hour',now()+interval '2 hours',now())<>'scheduled'
  or public.pcs_interclub_registration_phase(now(),now()+interval '1 hour',now())<>'open'
  or public.pcs_interclub_registration_phase(now()-interval '1 hour',now(),now())<>'closed' then
  raise exception 'Registration phase boundary is incorrect'; end if;
 if has_function_privilege('anon','public.pcs_set_interclub_registration_window(uuid,text,text,uuid,integer,timestamptz,timestamptz)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_require_interclub_registration_phase(uuid,text)','EXECUTE') then
  raise exception 'Browser can bypass trusted API registration boundaries'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(home,home,'Window Home',false,'draft','draft'),(away,away,'Window Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 insert into public.players(id,club_id,name,normalized_name,rating,active,gender) values
 (p1,home,'Window Person','window person',1480,true,'male'),
 (p2,home,'Window Verbal','window verbal',1480,true,'female'),
 (p3,home,'Window Guest','window guest',1480,true,'female');
 -- Initial commissioner planning and scheduled meet creation work without a window.
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Window QA League','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.5'),'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
  jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,'{"3.5":{"min_rating":3.5,"max_rating":3.999,"women_required":2}}');
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 select id into mid from public.pcs_interclub_meets where season_id=sid;
 if (select count(*) from public.pcs_interclub_pool_settings where season_id=sid)<>2 then raise exception 'Acceptance did not create both club signup links'; end if;
 if exists(select 1 from public.pcs_interclub_seasons where id=sid and
  (registration_opens_at is not null or registration_closes_at is not null or registration_revision<>0)) then
  raise exception 'Window dates were invented'; end if;
 select share_id into share from public.pcs_interclub_pool_settings where season_id=sid and club_id=home;
 signup:=jsonb_build_object('share_id',share,'name','Window Person','email','window-person@example.invalid','divisions',jsonb_build_array('3.5'),
  'notes','','request_id',gen_random_uuid(),'request_fingerprint','window-signup','email_consent',true);
 -- Schedule-only publishing is still a commissioner planning task.
 perform public.pcs_write_interclub_publication(actor,actor_email,home,sid,0,'save','{"results":[]}');
 perform public.pcs_write_interclub_publication(actor,actor_email,home,sid,1,'publish','{"results":[],"schedule":[]}');
 perform public.pcs_write_interclub_publication(actor,actor_email,home,sid,2,'unpublish',null);

 begin
  perform public.pcs_set_interclub_registration_window(actor,actor_email,away,sid,0,now(),now()+interval '1 day');
  raise exception 'Participating club changed league-wide registration';
 exception when insufficient_privilege then null; end;
 begin
  perform public.pcs_set_interclub_registration_window(gen_random_uuid(),actor_email,home,sid,0,now(),now()+interval '1 day');
  raise exception 'Forged commissioner identity changed window';
 exception when insufficient_privilege then null; end;
 begin
  perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now()+interval '2 days',now()+interval '1 day');
  raise exception 'Reversed window accepted';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,null,now()+interval '1 day');
  raise exception 'Partial window accepted';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now(),'infinity'::timestamptz);
  raise exception 'Infinite window accepted';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now(),now()+interval '21 days');
  raise exception 'Registration closed after earliest meet';
 exception when invalid_parameter_value then null; end;

 -- Synthetic prior result proves result-side RPCs cannot bypass a reopened or
 -- unconfigured window. This fixture setup is not an application permission.
 insert into public.pcs_interclub_competition_batches(id,season_id,meet_id,phase,state,document,roster_sources,approved_document,approved_revision,ratings_status)
 values(batch_id,sid,mid,'regular','approved','{}','[]','{}',1,'pending');
 foreach phase in array array['unconfigured','scheduled','open'] loop
  if phase='scheduled' then
   result:=public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,window_revision,now()+interval '1 day',now()+interval '2 days');
   window_revision:=(result->>'registration_revision')::integer;
   begin
    perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now(),now()+interval '1 day');
    raise exception 'Stale commissioner revision accepted';
   exception when sqlstate 'PT409' then null; end;
  elsif phase='open' then
   result:=public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,window_revision,now(),now()+interval '1 day');
   window_revision:=(result->>'registration_revision')::integer;
  end if;
  if (select public.pcs_interclub_registration_phase(registration_opens_at,registration_closes_at) from public.pcs_interclub_seasons where id=sid)<>phase then
   raise exception 'Incorrect phase %',phase; end if;
  if exists(select 1 from public.pcs_interclub_meet_workspaces where id=mid and (roster_open or deadline_editable)) then
   raise exception 'Meet workspace permission hint escaped % phase',phase; end if;
  begin
   perform public.pcs_set_interclub_meet_deadline(actor,actor_email,home,sid,mid,1,now()+interval '3 days');
   raise exception 'Meet deadline escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,gen_random_uuid(),0,'Team','3.5',array[p1,p2],false);
   raise exception 'Meet roster escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_review_interclub_meet_roster(actor,actor_email,home,sid,mid,1,gen_random_uuid(),1,true,'Fixture exception');
   raise exception 'Roster exception escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_create_interclub_competition_meet(actor,actor_email,home,sid,'{}');
   raise exception 'Meet creation escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','generate',0,'{}','[]');
   raise exception 'Competition escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'availability',jsonb_build_object('meet_id',mid,'expected_revision',0,'open',true,'deadline',now()+interval '3 days'));
   raise exception 'Meet availability escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'invite',jsonb_build_object('meet_id',mid,'member_ids','[]'::jsonb));
   raise exception 'Meet invitation escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_interclub_pool_public_action('respond_meet',jsonb_build_object('season_id',sid,'club_id',home),repeat('w',64));
   raise exception 'Meet RSVP escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_interclub_meet_player_ratings(sid,mid,home);
   raise exception 'Meet rating snapshot escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_assert_interclub_competition_eligibility(sid,mid,'{}','regular');
   raise exception 'Competition eligibility escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_apply_interclub_rating_projection(array[home],'fixture','{}',batch_id,1);
   raise exception 'Explicit rating retry escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_fail_interclub_ratings(batch_id,1,'Fixture');
   raise exception 'Rating status escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_write_interclub_publication(actor,actor_email,home,sid,3,'publish','{"results":[{}]}');
   raise exception 'Result publication escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_write_interclub_publication(actor,actor_email,home,sid,3,'unpublish',null);
   raise exception 'Result removal escaped % lock',phase;
  exception when sqlstate 'PT423' then blocked:=blocked+1; end;
  begin
   perform public.pcs_interclub_pool_action(actor,actor_email,away,sid,'settings','{"expected_revision":1,"open":true}');
   raise exception 'Club independently changed season registration';
  exception when insufficient_privilege then null; end;
  if phase<>'open' then
   begin
    perform public.pcs_interclub_pool_public_action('signup',signup,repeat('w',64));
    raise exception 'Public intake escaped % lock',phase;
   exception when sqlstate 'PT423' then null; end;
   begin
    perform public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(jsonb_build_object('player_id',p2)));
    raise exception 'Bulk intake escaped % lock',phase;
   exception when sqlstate 'PT423' then null; end;
  else
   -- Legacy false never closes the shared season window.
   if (select open from public.pcs_interclub_pool_settings where season_id=sid and club_id=home) then raise exception 'Fixture requires retired flag false'; end if;
   result:=public.pcs_interclub_pool_public_action('signup',signup,repeat('w',64));
   member_id:=(result->'member'->>'id')::uuid;
   result:=public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(jsonb_build_object('player_id',p2),jsonb_build_object('name','Window Guest')));
   if result->>'added_count'<>'2' then raise exception 'Open registration did not accept bulk commitments'; end if;
   select id into guest_id from public.pcs_interclub_pool_members where season_id=sid and name='Window Guest';
  end if;
 end loop;
 if blocked<>42 then raise exception 'Expected 42 phase-locked direct operations, got %',blocked; end if;

 result:=public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,window_revision,now()-interval '1 hour',now());
 window_revision:=(result->>'registration_revision')::integer;
 if result->>'registration_phase'<>'closed' then raise exception 'Close boundary failed'; end if;
 select * into saved from public.pcs_interclub_pool_members where id=member_id;
 begin
  perform public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid(),'name','Another Person'),repeat('w',64));
  raise exception 'Closed registration accepted new player';
 exception when sqlstate 'PT423' then null; end;
 begin
  perform public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,'[{"name":"Late bulk"}]');
  raise exception 'Closed registration accepted bulk addition';
 exception when sqlstate 'PT423' then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('update_season',signup||jsonb_build_object('id',saved.id,'nonce',saved.token_nonce,
   'season_id',sid,'club_id',home,'expected_revision',saved.revision,'status','active','notes','Changed after closing'),repeat('w',64));
  raise exception 'Active public edit escaped close';
 exception when sqlstate 'PT423' then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('update_season',signup||jsonb_build_object('id',saved.id,'nonce',saved.token_nonce,
   'season_id',sid,'club_id',home,'expected_revision',saved.revision,'status','withdrawn','notes','Smuggled edit'),repeat('w',64));
  raise exception 'Withdrawal smuggled a closed contact edit';
 exception when sqlstate 'PT423' then null; end;
 perform public.pcs_interclub_pool_public_action('update_season',signup||jsonb_build_object('id',saved.id,'nonce',saved.token_nonce,
  'season_id',sid,'club_id',home,'expected_revision',saved.revision,'status','withdrawn'),repeat('w',64));
 begin
  perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'member',jsonb_build_object('member_id',member_id,'expected_revision',saved.revision+1,'player_id',p1,'status','active'));
  raise exception 'Club restored a withdrawn player after close';
 exception when sqlstate 'PT423' then null; end;
 -- Existing active entries can still be linked/corrected by their club.
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'member',jsonb_build_object('member_id',guest_id,'expected_revision',1,'player_id',p3,'status','active'));
 perform public.pcs_set_interclub_meet_deadline(actor,actor_email,home,sid,mid,1,now()+interval '3 days');
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'availability',jsonb_build_object('meet_id',mid,'expected_revision',0,'open',true,'deadline',now()+interval '2 days'));
 if not (select roster_open from public.pcs_interclub_meet_workspaces where id=mid) then raise exception 'Closed enrollment did not unlock meet workspace'; end if;

 -- Commissioner may reopen before the first meet; operational data survives,
 -- every meet operation relocks, and pool restores become available again.
 result:=public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,window_revision,now()-interval '1 hour',now()+interval '1 hour');
 window_revision:=(result->>'registration_revision')::integer;
 begin
  perform public.pcs_set_interclub_meet_deadline(actor,actor_email,home,sid,mid,2,now()+interval '4 days');
  raise exception 'Reopened registration did not relock meet';
 exception when sqlstate 'PT423' then null; end;
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'member',jsonb_build_object('member_id',member_id,'expected_revision',saved.revision+1,'player_id',p1,'status','active'));
 if (select count(*) from public.pcs_interclub_competition_batches where id=batch_id)<>1 then raise exception 'Reopening deleted operational data'; end if;
 -- Rotation remains possible, independently of all registration dates.
 select share_id into old_share from public.pcs_interclub_pool_settings where season_id=sid and club_id=home;
 result:=public.pcs_interclub_pool_action(actor,actor_email,home,sid,'settings','{"expected_revision":1,"rotate_link":true}');
 if (result->>'share_id')::uuid=old_share or (select registration_revision from public.pcs_interclub_seasons where id=sid)<>window_revision then
  raise exception 'Link rotation changed registration window or failed to rotate'; end if;
 select count(*) into window_audits from public.pcs_interclub_registration_audit where season_id=sid and action='season_registration_window';
 if window_audits<>window_revision then raise exception 'Commissioner date revisions lack audit rows'; end if;
end $$;
rollback;
