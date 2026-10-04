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
 for i in 0..6 loop
  payload:=jsonb_build_object('share_id',share,'player_id',first_player+i,'name','Meet Signup Player '||i,'division','3.0','email','','request_id',gen_random_uuid(),'fingerprint','qa-'||i);
  result:=public.pcs_interclub_meet_signup_action('join',payload,null,null,repeat('a',64));
  if i=0 then
   first_request:=payload;first_entry:=result->'entry';
   if first_entry->>'placement'<>'waitlist' or first_entry->>'priority'<>'play_up' or (first_entry->>'rating')::numeric<>2.93 then raise exception '2.93 player took an automatic spot'; end if;
  elsif i in (1,2,4,5) and result->'entry'->>'placement'<>'confirmed' then raise exception 'First two eligible players did not get their spot';
  elsif i=3 and result->'entry'->>'placement'<>'waitlist' then raise exception 'Third woman overfilled roster';
  elsif i=6 and result->'entry'->>'placement'<>'review' then raise exception 'Unknown gender took a spot'; end if;
 end loop;
 if (select count(*) from public.pcs_interclub_meet_signups where season_id=sid and placement='confirmed')<>4 then raise exception 'Four spots not enforced'; end if;
 select * into team from public.pcs_interclub_teams where season_id=sid and club_id=home and not withdrawn;
 if team.id is null or (select count(*) from public.pcs_interclub_team_players where team_id=team.id)<>4 then raise exception 'Full signup did not create the real team'; end if;
 retry:=public.pcs_interclub_meet_signup_action('join',first_request,null,null,repeat('a',64));
 if retry->'entry'->>'id'<>first_entry->>'id' or retry->'entry'->>'registration_order'<>first_entry->>'registration_order' then raise exception 'Network retry lost original place'; end if;
 retry:=public.pcs_interclub_meet_signup_action('join',first_request||jsonb_build_object('request_id',gen_random_uuid()),null,null,repeat('a',64));
 if retry->>'duplicate'<>'true' or retry ? 'entry' then raise exception 'Duplicate reset order or leaked private capability'; end if;
 begin
  perform public.pcs_interclub_meet_signup_action('join',first_request||jsonb_build_object('request_id',gen_random_uuid(),'player_id',first_player+7,'name','Meet Signup Player 7'),null,null,repeat('a',64));
  raise exception 'Above-band player was admitted';
 exception when sqlstate 'PT422' then null; end;
 begin
  perform public.pcs_interclub_meet_signup_action('join',first_request||jsonb_build_object('request_id',gen_random_uuid(),'player_id',first_player+8,'name','Meet Signup Player 8'),null,null,repeat('a',64));
  raise exception 'Foreign-club player was admitted';
 exception when sqlstate 'PT422' then null; end;
 begin
  perform public.pcs_save_interclub_meet_roster(actor,email,home,sid,mid,meet.revision,team.id,team.revision,team.name,team.division,'{}',true);
  raise exception 'Manual edit bypassed active signup';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_interclub_meet_signup_action('promote',scope||jsonb_build_object('id',first_entry->>'id','expected_revision',1),actor,email);
  raise exception 'Playing-up promotion displaced an eligible player';
 exception when sqlstate 'PT422' then null; end;
 select * into entry from public.pcs_interclub_meet_signups where season_id=sid and player_id=first_player+1;
 perform public.pcs_interclub_meet_signup_action('withdraw',jsonb_build_object('id',entry.id,'nonce',entry.token_nonce,'expected_revision',entry.revision));
 if not exists(select 1 from public.pcs_interclub_team_players where team_id=team.id and player_id=first_player+3)
  or exists(select 1 from public.pcs_interclub_team_players where team_id=team.id and player_id=first_player+1) then raise exception 'Withdrawal did not promote the next rated woman into actual roster'; end if;
 if (select placement from public.pcs_interclub_meet_signups where id=(first_entry->>'id')::uuid)<>'waitlist' then raise exception 'Lower-rated first signup jumped the rated substitute'; end if;
 begin
  perform public.pcs_interclub_meet_signup_action('withdraw',jsonb_build_object('id',entry.id,'nonce',entry.token_nonce,'expected_revision',entry.revision));
  raise exception 'Stale revision accepted';
 exception when sqlstate 'PT409' then null; end;
 select * into entry from public.pcs_interclub_meet_signups where season_id=sid and player_id=first_player+2;
 perform public.pcs_interclub_meet_signup_action('withdraw',jsonb_build_object('id',entry.id,'nonce',entry.token_nonce,'expected_revision',entry.revision));
 if (select withdrawn from public.pcs_interclub_teams where id=team.id) is not true then raise exception 'Incomplete lineup still marked ready'; end if;
 perform public.pcs_interclub_meet_signup_action('promote',scope||jsonb_build_object('id',first_entry->>'id','expected_revision',1),actor,email);
 if not exists(select 1 from public.pcs_interclub_team_players where team_id=team.id and player_id=first_player) then raise exception 'Admin could not fill vacancy with play-up player'; end if;
 -- Rejoining gets a new place and private nonce; rated players retain priority.
 old_order:=entry.registration_order;
 result:=public.pcs_interclub_meet_signup_action('join',first_request||jsonb_build_object('request_id',gen_random_uuid(),'fingerprint','rejoin','player_id',first_player+2,'name','Meet Signup Player 2'),null,null,repeat('a',64));
 if (result->'entry'->>'registration_order')::bigint<=old_order or result->'entry'->>'token_nonce'=entry.token_nonce::text then raise exception 'Rejoin reused old order or private link'; end if;
 if exists(select 1 from public.pcs_interclub_team_players where team_id=team.id and player_id=first_player) then raise exception 'Admin play-up exception overrode rated signup priority'; end if;
 begin
  perform public.pcs_interclub_meet_signup_action('refresh',scope,gen_random_uuid(),email);
  raise exception 'Forged admin accepted';
 exception when insufficient_privilege then null; end;
 -- Cutoff is checked inside the same transaction that would allocate a place.
 update public.pcs_interclub_meet_signup_settings set deadline=now()-interval '1 second' where season_id=sid;
 begin
  perform public.pcs_interclub_meet_signup_action('join',first_request||jsonb_build_object('request_id',gen_random_uuid()),null,null,repeat('a',64));
  raise exception 'Signup passed the deadline';
 exception when sqlstate 'PT409' then null; end;
 update public.pcs_interclub_meet_signup_settings set deadline=meet.roster_deadline where season_id=sid;
 perform public.pcs_interclub_meet_signup_action('settings',scope||jsonb_build_object('expected_revision',cfg.revision,'expected_meet_revision',meet.revision,'open',false,'deadline',meet.roster_deadline),actor,email);
 select * into team from public.pcs_interclub_teams where id=team.id;
 perform public.pcs_save_interclub_meet_roster(actor,email,home,sid,mid,meet.revision,team.id,team.revision,team.name,team.division,'{}',true);
 begin
  perform public.pcs_interclub_meet_signup_action('settings',scope||jsonb_build_object('expected_revision',cfg.revision+1,'expected_meet_revision',meet.revision,'open',true,'deadline',meet.roster_deadline),actor,email);
  raise exception 'Reopen overwrote a manual lineup change';
 exception when sqlstate 'PT409' then null; end;
end $$;
rollback;
