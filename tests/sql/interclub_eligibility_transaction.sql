-- Staging-only rollback rehearsal: approved pool, exact rating, deadline and representation.
begin;
do $$
declare
 home text:='elig-home-'||gen_random_uuid()::text; away text:='elig-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='elig-qa@example.invalid'; sid uuid:=gen_random_uuid(); mid uuid;
 member uuid; late_member uuid; linked_player bigint; players bigint[]:='{}'; entries uuid[]:='{}';
 i integer; next_player bigint:=8100000000000+(random()*10000000000)::bigint;
 team uuid:=gen_random_uuid(); partial_team uuid:=gen_random_uuid(); result jsonb; cutoff timestamptz:=now()-interval '1 minute';
 first_entry uuid; another_player bigint; snapshot numeric; t text;
begin
 foreach t in array array['pcs_interclub_represented_players','pcs_interclub_appearances','pcs_interclub_meet_eligibility_snapshots'] loop
  if has_table_privilege('anon','public.'||t,'SELECT') or has_table_privilege('authenticated','public.'||t,'SELECT')
   or not (select relrowsecurity from pg_class where oid=('public.'||t)::regclass) then raise exception 'Eligibility RLS or grants wrong'; end if;
 end loop;
 if has_function_privilege('anon','public.pcs_review_interclub_pool_member(uuid,text,text,uuid,uuid,integer,boolean,text)','EXECUTE') then raise exception 'Review RPC is exposed'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status) values(home,home,'Eligibility Home',false,'draft','draft'),(away,away,'Eligibility Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id) values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Eligibility QA','start_date',(current_date-3)::text,
 'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.5','4.0'),'club_ids',jsonb_build_array(home,away),
 'meets',jsonb_build_array(jsonb_build_object('host_club_id',home,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '10 days','duration_minutes',180,'courts',4))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,'{"3.5":{"min_rating":3.5,"max_rating":3.999,"women_required":2},"4.0":{"min_rating":4.0,"max_rating":4.499,"women_required":2}}');
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 perform public.pcs_interclub_pool_action(actor,actor_email,home,sid,'settings','{"expected_revision":0,"open":true}');
 perform public.pcs_interclub_pool_action(actor,actor_email,away,sid,'settings','{"expected_revision":0,"open":true}');
 select id into mid from public.pcs_interclub_meets where season_id=sid;
 for i in 1..5 loop
  linked_player:=next_player+i;
  insert into public.players(id,club_id,name,normalized_name,rating,active,gender) values(linked_player,home,'Player '||i,'player '||i,1480,true,case when i<=2 then 'female' else 'male' end);
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,request_id,request_fingerprint,created_at)
   values(sid,home,'Player '||i,'elig-'||i||'@example.invalid',gen_random_uuid(),'qa',case when i<=4 then now()-interval '5 days' else now() end) returning id into member;
  result:=public.pcs_interclub_pool_action(actor,actor_email,home,sid,'member',jsonb_build_object('member_id',member,'expected_revision',1,'player_id',linked_player,'status','active'));
  if i<=4 then
   if result->>'approval_status'<>'approved' then raise exception 'Timely linked player not approved'; end if;
   players:=array_append(players,linked_player);
   select id into first_entry from public.pcs_interclub_entries where season_id=sid and player_id=linked_player;
   entries:=array_append(entries,first_entry);
  else
   late_member:=member;
   if result->>'approval_status'<>'pending' then raise exception 'Late signup bypassed organizer approval'; end if;
  end if;
 end loop;
 begin
  perform public.pcs_review_interclub_pool_member(actor,actor_email,away,sid,late_member,2,true,'Late traveler');
  raise exception 'Nonorganizer approved late player';
 exception when insufficient_privilege then null; end;
 perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,late_member,2,true,'Late traveler approved');
 begin
  perform public.pcs_review_interclub_pool_member(actor,actor_email,home,sid,late_member,2,true,'Stale retry');
  raise exception 'Stale approval accepted';
 exception when sqlstate 'PT409' then null; end;
 result:=public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,partial_team,0,'Women only','3.5',players[1:2],false);
 if jsonb_array_length(result->'roster'->'roster')<>2 or result->'roster'->>'status'<>'eligible' then raise exception 'Two-player regular roster requires ghosts or exception'; end if;
 perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,partial_team,1,'','',array[]::bigint[],true);
 update public.pcs_interclub_meets set competition_phase='final' where id=mid;
 begin
  perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,gen_random_uuid(),0,'Incomplete final','3.5',players[1:2],false);
  raise exception 'Championship partial team accepted';
 exception when invalid_parameter_value then null; end;
 update public.pcs_interclub_meets set competition_phase='regular' where id=mid;
 perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,team,0,'Home 3.5','3.5',players,false);
 begin
  perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,gen_random_uuid(),0,'Wrong level','4.0',players,false);
  raise exception 'Wrong skill level accepted';
 exception when invalid_parameter_value then null; end;
 update public.pcs_interclub_entries set entered_at=now()-interval '2 days' where season_id=sid;
 update public.pcs_interclub_pool_members set approved_at=now()-interval '2 days' where season_id=sid;
 update public.pcs_interclub_meets set roster_deadline=cutoff where id=mid;
 perform public.pcs_lock_interclub_meet_eligibility(mid);
 select rating into snapshot from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and entry_id=entries[1] and deadline=cutoff;
 if snapshot<>3.7 then raise exception 'Deadline snapshot not seeded from club rating'; end if;
 update public.pcs_interclub_entries set starting_rating=4.2 where id=entries[1];
 perform public.pcs_lock_interclub_meet_eligibility(mid);
 if (select rating from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and entry_id=entries[1] and deadline=cutoff)<>3.7 then raise exception 'Frozen rating changed'; end if;
 insert into public.pcs_interclub_appearances(season_id,meet_id,entry_id,club_id,player_id,division,batch_id,revision,phase,game_id)
 values(sid,mid,entries[1],home,players[1],'3.5',gen_random_uuid(),1,'regular','qa-game');
 begin
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,request_id,request_fingerprint)
   values(sid,away,'Player 1','elig-1@example.invalid',gen_random_uuid(),'different-club');
  raise exception 'Played player switched clubs';
 exception when invalid_parameter_value then null; end;
 select pool_member_id into member from public.pcs_interclub_entries where id=entries[1];
 update public.pcs_interclub_pool_members set status='withdrawn' where id=member;
 begin
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,request_id,request_fingerprint,created_at,player_id)
   values(sid,home,'A new identity','another@example.invalid',gen_random_uuid(),'identity-replacement',now()-interval '5 days',players[1]);
  raise exception 'Played entry identity replaced via a new signup';
 exception when invalid_parameter_value then null; end;

 begin
  update public.pcs_interclub_pool_members set player_id=next_player+5 where id=member;
  raise exception 'Played pool identity reassigned';
 exception when invalid_parameter_value then null; end;

 -- Lazy freeze happens before a post-deadline withdrawal, retaining eligibility.
 delete from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and entry_id=entries[2];
 select pool_member_id into member from public.pcs_interclub_entries where id=entries[2];
 update public.pcs_interclub_pool_members set status='withdrawn' where id=member;
 if not exists(select 1 from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and entry_id=entries[2]) then raise exception 'Post-deadline withdrawal erased eligibility'; end if;
 if not public.pcs_interclub_rating_in_division(4.0,'4.0') or public.pcs_interclub_rating_in_division(4.0,'3.5')
  or not public.pcs_interclub_rating_in_division(3.999,'3.5') then raise exception 'Rating boundary incorrect'; end if;
end $$;
rollback;
