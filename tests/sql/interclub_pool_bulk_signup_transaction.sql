-- Staging-only rollback rehearsal. All synthetic clubs and people are discarded.
begin;
do $$
<<pool_bulk_test>>
declare
 home text:='pool-bulk-home-'||gen_random_uuid()::text; away text:='pool-bulk-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='pool-bulk-qa@example.invalid'; sid uuid:=gen_random_uuid();
 p1 bigint:=8100000000000+(random()*1000000000)::bigint; p2 bigint; p3 bigint; p4 bigint; p5 bigint; away_player bigint;
 result jsonb; retry jsonb; signup jsonb; settings jsonb; share uuid; saved public.pcs_interclub_pool_members;
 requester text:=repeat('b',64); before_count integer; late_player bigint; details jsonb;
begin
 p2:=p1+1;p3:=p1+2;p4:=p1+3;p5:=p1+4;away_player:=p1+5;late_player:=p1+6;
 if has_function_privilege('anon','public.pcs_interclub_pool_bulk_add(uuid,text,text,uuid,jsonb)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_interclub_pool_player_details(uuid,text,bigint[])','EXECUTE') then
  raise exception 'Browser can execute privileged pool routines'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(home,home,'Pool Bulk Home',false,'draft','draft'),(away,away,'Pool Bulk Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 insert into public.players(id,club_id,name,normalized_name,rating,active,gender) values
 (p1,home,'Pool Unique','pool unique',1480,true,'male'),
 (p2,home,'Pool Namesake','pool namesake',1400,true,'female'),
 (p3,home,'POOL  NAMESAKE','pool namesake',1620,true,'male'),
 (p4,home,'Pool Unrated','pool unrated',0,true,'female'),
 (p5,home,'Pool Verbal','pool verbal',1500,true,'female'),
 (away_player,away,'Pool Unique','pool unique',1800,true,'male'),
 (late_player,home,'Pool Late','pool late',1450,true,'female');
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Pool Bulk QA League','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.5','4.0'),'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
  jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,
  '{"3.5":{"min_rating":3.5,"max_rating":3.999,"women_required":2},"4.0":{"min_rating":4.0,"max_rating":4.499,"women_required":2}}');
 begin
  perform public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(jsonb_build_object('player_id',p1)));
  raise exception 'Unaccepted club added players';
 exception when insufficient_privilege then null; end;
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');

 -- Bulk addition works before public signup opens and accepts missing emails.
 result:=public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(
  jsonb_build_object('player_id',p5,'divisions',jsonb_build_array('3.5')),
  jsonb_build_object('name','Unknown Verbal','divisions',jsonb_build_array('3.5'))));
 if result->>'added_count'<>'2' or result->>'skipped_count'<>'0' then raise exception 'Bulk add count incorrect'; end if;
 if (select open from public.pcs_interclub_pool_settings where season_id=sid and club_id=home) then raise exception 'Bulk addition opened public signup'; end if;
 select * into saved from public.pcs_interclub_pool_members where season_id=sid and player_id=p5;
 if saved.email<>'' or saved.consent_at is not null or saved.approval_status<>'approved' then raise exception 'Verbal addition invents email/consent or misses profile'; end if;
 if exists(select 1 from public.pcs_interclub_pool_members where season_id=sid and name='Unknown Verbal' and (player_id is not null or approval_status<>'pending')) then raise exception 'Unmatched member gained identity'; end if;
 retry:=public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(
  jsonb_build_object('player_id',p5,'email','different@example.invalid'),jsonb_build_object('name','Unknown Verbal')));
 if retry->>'added_count'<>'0' or retry->>'skipped_count'<>'2' then raise exception 'Duplicate retry inserted or reset member'; end if;
 if (select email from public.pcs_interclub_pool_members where id=saved.id)<>'' then raise exception 'Duplicate overwrote email'; end if;
 -- Account-free capability withdrawal also works for a no-email commitment.
 perform public.pcs_interclub_pool_public_action('update_season',jsonb_build_object('id',saved.id,'nonce',saved.token_nonce,
  'season_id',sid,'club_id',home,'expected_revision',1,'status','withdrawn','name',saved.name,'email','','divisions',saved.divisions,'notes',''),requester);
 if (select status from public.pcs_interclub_pool_members where id=saved.id)<>'withdrawn' then raise exception 'No-email withdrawal failed'; end if;

 -- All-or-nothing batch validation prevents the first row surviving a later error.
 select count(*) into before_count from public.pcs_interclub_pool_members where season_id=sid;
 begin
  perform public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(
   jsonb_build_object('name','Must Roll Back'),jsonb_build_object('player_id',away_player)));
  raise exception 'Cross-club bulk selection accepted';
 exception when invalid_parameter_value then null; end;
 if (select count(*) from public.pcs_interclub_pool_members where season_id=sid)<>before_count then raise exception 'Invalid batch partially committed'; end if;
 begin
  perform public.pcs_interclub_pool_bulk_add(gen_random_uuid(),actor_email,home,sid,jsonb_build_array(jsonb_build_object('name','Forged Actor')));
  raise exception 'Wrong account added players';
 exception when insufficient_privilege then null; end;

 settings:=public.pcs_interclub_pool_action(actor,actor_email,home,sid,'settings','{"expected_revision":1,"open":true}');
 share:=(settings->>'share_id')::uuid;
 signup:=jsonb_build_object('share_id',share,'name','  Pool   Unique ','email','unique@example.invalid','divisions',jsonb_build_array('3.5'),
  'notes','','request_id',gen_random_uuid(),'request_fingerprint','unique-signup','email_consent',true);
 result:=public.pcs_interclub_pool_public_action('signup',signup,requester);
 if (result->'member'->>'player_id')::bigint<>p1 or result->'member'->>'approval_status'<>'approved' then raise exception 'Unique name did not link own club profile'; end if;
 retry:=public.pcs_interclub_pool_public_action('signup',signup,requester);
 if retry<>result then raise exception 'Retry lost private capability'; end if;
 retry:=public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid(),'email','different@example.invalid'),requester);
 if retry->>'status'<>'already_registered' or retry ? 'member' then raise exception 'Duplicate linked profile leaked member'; end if;
 begin
  perform public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid(),'player_id',away_player),requester);
  raise exception 'Public signup linked other club profile';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid(),'player_id',p4),requester);
  raise exception 'Public signup linked mismatched name';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid(),'name','Pool Namesake','email','namesake@example.invalid'),requester);
  raise exception 'Ambiguous name auto-linked';
 exception when sqlstate 'PT422' then null; end;
 result:=public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid(),
  'name','Pool Namesake','email','new-namesake@example.invalid','player_id',null),requester);
 if result->'member'->>'player_id' is not null or result->'member'->>'approval_status'<>'pending' then raise exception 'Explicit no-profile choice was ignored'; end if;

 -- Different selected directory identities can share both name and household email.
 result:=public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(
  jsonb_build_object('player_id',p2,'email','household@example.invalid'),jsonb_build_object('player_id',p3,'email','household@example.invalid')));
 if result->>'added_count'<>'2' then raise exception 'Distinct namesake profiles were collapsed'; end if;
 result:=public.pcs_interclub_pool_public_action('signup',signup||jsonb_build_object('request_id',gen_random_uuid(),
  'name','Pool Unrated','email','unrated@example.invalid','player_id',p4),requester);
 if (result->'member'->>'player_id')::bigint<>p4 or result->'member'->>'approval_status'<>'pending' then raise exception 'Unrated profile cannot join pending'; end if;
 if exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=p4) then raise exception 'Unrated profile received a rating entry'; end if;

 -- The same rating authority used for lineups determines displayed division.
 details:=public.pcs_interclub_pool_search_players(sid,home,'Pool   Unique');
 if jsonb_array_length(details)<>1 or (details->0->>'id')::bigint<>p1 then raise exception 'Normalized search missed own club profile'; end if;
 details:=public.pcs_interclub_pool_player_details(sid,home,array[p1,p2,p3,away_player]);
 if jsonb_array_length(details)<>3 then raise exception 'Rating summary crossed club scope'; end if;
 if not exists(select 1 from jsonb_array_elements(details) d where (d->>'player_id')::bigint=p3 and (d->>'league_rating')::numeric=4.05 and d->'eligible_divisions'='["4.0"]'::jsonb) then raise exception 'Rating/eligibility summary incorrect'; end if;
 -- Change only club rating: the established league seed must remain in authority.
 update public.players set rating=1700 where id=p1;
 details:=public.pcs_interclub_pool_player_details(sid,home,array[p1]);
 if (details->0->>'league_rating')::numeric<>3.7 or details->0->'eligible_divisions'<>'["3.5"]'::jsonb then raise exception 'Club rating replaced established league eligibility'; end if;

 -- Late additions remain pending even when an administrator selects their profile.
 update public.pcs_interclub_seasons set details=jsonb_set(pcs_interclub_seasons.details,'{start_date}',to_jsonb((current_date-1)::text)) where id=sid;
 result:=public.pcs_interclub_pool_bulk_add(actor,actor_email,home,sid,jsonb_build_array(jsonb_build_object('player_id',late_player)));
 select * into saved from public.pcs_interclub_pool_members where season_id=sid and player_id=late_player;
 if saved.late_join is not true or saved.approval_status<>'pending' then raise exception 'Bulk addition bypassed late approval'; end if;
 if exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=late_player) then raise exception 'Late addition bypassed approval seed'; end if;
 perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,saved.id,saved.revision,true,'Approved late arrival');
 if not exists(select 1 from public.pcs_interclub_entries where season_id=sid and player_id=late_player) then raise exception 'Organizer approval did not seed entry'; end if;
end $$;
rollback;
