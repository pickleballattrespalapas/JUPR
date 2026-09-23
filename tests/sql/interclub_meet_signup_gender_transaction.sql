-- Synthetic staging fixtures; the complete rehearsal rolls back.
begin;
do $$
declare
 home text:='meet-signup-home-'||gen_random_uuid()::text; away text:='meet-signup-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); email text:='meet-signup-qa@example.invalid'; sid uuid:=gen_random_uuid(); mid uuid;
 first_player bigint:=8200000000000+(random()*1000000000)::bigint; pid bigint; i integer;
 result jsonb; retry jsonb; payload jsonb; scope jsonb; first_request jsonb; first_entry jsonb; share uuid; old_order bigint;
 cfg public.pcs_interclub_meet_signup_settings; meet public.pcs_interclub_meets; team public.pcs_interclub_teams; entry public.pcs_interclub_meet_signups;
begin
 if has_function_privilege('anon','public.pcs_interclub_meet_signup_action(text,jsonb,uuid,text,text)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_reconcile_interclub_meet_signups(uuid,text,uuid)','EXECUTE')
  or has_table_privilege('anon','public.pcs_interclub_meet_signups','SELECT') then raise exception 'Browser has private queue privileges'; end if;
 if public.pcs_interclub_signup_priority(2.93,'3.0')<>'play_up' or public.pcs_interclub_signup_priority(3,'3.0')<>'in_band'
  or public.pcs_interclub_signup_priority(3.4999,'3.0')<>'in_band' or public.pcs_interclub_signup_priority(3.5,'3.0')<>'review'
  or public.pcs_interclub_signup_priority(4.49,'Open')<>'play_up' or public.pcs_interclub_signup_priority(4.5,'Open')<>'in_band' then raise exception 'Exact rating band priority failed'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(home,home,'Meet Signup Home',false,'draft','draft'),(away,away,'Meet Signup Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id) values(home,email,'administrator',actor),(away,email,'administrator',actor);
 for i in 0..8 loop
  insert into public.players(id,club_id,name,normalized_name,rating,active,gender)
  values(first_player+i,case when i=8 then away else home end,'Meet Signup Player '||i,'meet signup player '||i,
   case when i=0 then 1172 when i=7 then 1400 else 1280 end,true,
   case when i=6 then null when i in (0,1,2,3) then 'female' else 'male' end);
 end loop;
 perform public.pcs_save_interclub_draft(actor,email,home,sid,0,jsonb_build_object('name','Meet Signup QA','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.0','3.5'),'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
  jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,email,home,sid,1,
  '{"3.0":{"min_rating":3.0,"max_rating":3.499,"women_required":2},"3.5":{"min_rating":3.5,"max_rating":3.999,"women_required":2}}');
 perform public.pcs_set_interclub_registration_window(actor,email,home,sid,0,now()-interval '2 days',now()+interval '1 day');
 perform public.pcs_interclub_participation(actor,email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,email,away,sid,away,1,'accept');
 for i in 0..7 loop
  perform public.pcs_interclub_pool_bulk_add(actor,email,home,sid,jsonb_build_array(jsonb_build_object('player_id',first_player+i)));
 end loop;
 perform public.pcs_set_interclub_registration_window(actor,email,home,sid,1,now()-interval '2 days',now()-interval '1 day');
 select * into meet from public.pcs_interclub_meets where season_id=sid limit 1; mid:=meet.id;
 scope:=jsonb_build_object('season_id',sid,'club_id',home,'meet_id',mid);
 perform public.pcs_interclub_meet_signup_action('settings',scope||jsonb_build_object('expected_revision',0,'expected_meet_revision',meet.revision,'open',true,'deadline',meet.roster_deadline),actor,email);
 select * into cfg from public.pcs_interclub_meet_signup_settings where season_id=sid and club_id=home and meet_id=mid; share:=cfg.share_id;
 for i in 1..6 loop
  payload:=jsonb_build_object('share_id',share,'player_id',first_player+i,'name','Meet Signup Player '||i,'division','3.0','email','','request_id',gen_random_uuid(),'fingerprint','gender-qa-'||i,
   'gender',case when i in (1,4) then 'male' when i in (2,6) then 'female' when i=3 then 'non_binary' else 'prefer_not_to_say' end);
  result:=public.pcs_interclub_meet_signup_action('join',payload,null,null,repeat('b',64));
  if i in (3,5) then
   if result->'entry'->>'placement' is distinct from 'review' or result->'entry'->>'gender' is distinct from 'unknown' or result->'entry'->>'reason' not like '%An admin will review%' then raise exception 'Declared gender did not receive admin review'; end if;
  elsif result->'entry'->>'placement' is distinct from 'confirmed' then raise exception 'Selected gender did not fill open place'; end if;
 end loop;
 if (select gender from public.players where id=first_player+1) is distinct from 'female' then raise exception 'Meet gender overwrote club profile'; end if;
 select * into team from public.pcs_interclub_teams where season_id=sid and club_id=home and not withdrawn;
 if team.id is null or (select count(*) from public.pcs_interclub_team_players where team_id=team.id)<>4 then raise exception 'Declared genders did not create real roster'; end if;
 select * into entry from public.pcs_interclub_meet_signups where season_id=sid and player_id=first_player+3;
 old_order:=entry.registration_order;
 begin
  perform public.pcs_interclub_meet_signup_action('review_gender',scope||jsonb_build_object('id',entry.id,'expected_revision',entry.revision,'lineup_gender','female'),gen_random_uuid(),email);
  raise exception 'Forged reviewer accepted';
 exception when insufficient_privilege then null; end;
 result:=public.pcs_interclub_meet_signup_action('review_gender',scope||jsonb_build_object('id',entry.id,'expected_revision',entry.revision,'lineup_gender','female'),actor,email);
 if result->'entry'->>'declared_gender' is distinct from 'non_binary' or result->'entry'->>'reviewed_gender' is distinct from 'female'
  or result->'entry'->>'placement' is distinct from 'confirmed' or (result->'entry'->>'registration_order')::bigint is distinct from old_order then raise exception 'Review changed declaration or original order'; end if;
 if (select placement from public.pcs_interclub_meet_signups where season_id=sid and player_id=first_player+6) is distinct from 'waitlist' then raise exception 'Review lost original FIFO priority'; end if;
 select * into entry from public.pcs_interclub_meet_signups where season_id=sid and player_id=first_player+5;
 result:=public.pcs_interclub_meet_signup_action('review_gender',scope||jsonb_build_object('id',entry.id,'expected_revision',entry.revision,'lineup_gender','male'),actor,email);
 if result->'entry'->>'declared_gender' is distinct from 'prefer_not_to_say' or result->'entry'->>'placement' is distinct from 'waitlist' then raise exception 'Review overfilled a full category'; end if;
 result:=public.pcs_interclub_meet_signup_action('add',scope||jsonb_build_object('player_id',first_player,'division','3.0','gender','non_binary','request_id',gen_random_uuid(),'fingerprint','admin-gender'),actor,email);
 if result->'entry'->>'placement' is distinct from 'review' then raise exception 'Admin-selected gender bypassed review'; end if;
 result:=public.pcs_interclub_meet_signup_action('review_gender',scope||jsonb_build_object('id',result->'entry'->>'id','expected_revision',result->'entry'->>'revision','lineup_gender','female'),actor,email);
 if result->'entry'->>'placement' is distinct from 'waitlist' or result->'entry'->>'priority' is distinct from 'play_up' then raise exception 'Review bypassed rating priority'; end if;
 update public.pcs_interclub_meets set roster_deadline=now() where id=mid;
 perform public.pcs_lock_interclub_meet_eligibility(mid);
 if (select gender from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and player_id=first_player+3) is distinct from 'female' then raise exception 'Roster cutoff lost reviewed category'; end if;
 if has_function_privilege('anon','public.pcs_interclub_meet_gender(uuid,uuid,text,bigint,text)','EXECUTE') then raise exception 'Private gender helper exposed'; end if;
end $$;
rollback;
