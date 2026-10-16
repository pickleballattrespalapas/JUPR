-- Staging-only rollback rehearsal: all fixture rows and edits are rolled back.
begin;
do $$
declare
 home text:='schedule-home-'||gen_random_uuid()::text; away text:='schedule-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='schedule-qa@example.invalid'; sid uuid:=gen_random_uuid(); mid uuid; added uuid;
 roster_id uuid:=gen_random_uuid(); request_id uuid:=gen_random_uuid(); member_id uuid; response_id uuid; old_nonce uuid;
 player_ids bigint[]:='{}'; pid bigint; i integer; out jsonb; payload jsonb; before_roster jsonb; before_doc jsonb;
 start_at timestamptz:=now()+interval '20 days'; earlier timestamptz:=now()+interval '10 days'; old_deadline timestamptz;
 batch_id uuid:=gen_random_uuid(); first_publication jsonb;
begin
 if has_function_privilege('anon','public.pcs_update_interclub_meet_schedule(uuid,text,text,uuid,uuid,integer,timestamptz,timestamptz,integer,integer)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_interclub_schedule_deadline_editable(uuid)','EXECUTE') then
  raise exception 'Schedule RPC exposed to browsers'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(home,home,'Schedule Home',false,'draft','draft'),(away,away,'Schedule Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Schedule QA',
  'start_date',(current_date+1)::text,'end_date',(current_date+90)::text,'timezone','America/Mazatlan',
  'divisions',jsonb_build_array('3.5'),'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
   jsonb_build_object('host_club_id',away,'club_ids',jsonb_build_array(home,away),'starts_at',start_at,'duration_minutes',180,'courts',2))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,'{"3.5":{"min_rating":3.5,"max_rating":3.999,"women_required":2}}');
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now()-interval '3 days',now()+interval '1 day');
 select id into mid from public.pcs_interclub_meets where season_id=sid;
 first_publication:=jsonb_build_object('name','Schedule QA','meets',jsonb_build_array(jsonb_build_object('id',mid,'starts_at',start_at)));
 insert into public.pcs_interclub_publications(season_id,published,published_at) values(sid,first_publication,now());
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,1,earlier,earlier,180,2);
  raise exception 'Schedule changed during registration';
 exception when sqlstate 'PT423' then null; end;
 for i in 1..4 loop
  pid:=8000000000000+(random()*1000000000000)::bigint; player_ids:=array_append(player_ids,pid);
  insert into public.players(id,club_id,name,normalized_name,rating,active,gender)
   values(pid,home,'Schedule QA '||i,'schedule qa '||i,1440,true,case when i<=2 then 'female' else 'male' end);
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,player_id,request_id,request_fingerprint)
   values(sid,home,'Schedule QA '||i,'schedule-'||i||'@example.invalid',pid,gen_random_uuid(),'fixture') returning id into member_id;
 end loop;
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,1,now()-interval '3 days',now()-interval '2 days');
 perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,roster_id,0,'Home 3.5','3.5',player_ids);
 select roster into before_roster from public.pcs_interclub_current_rosters where id=roster_id;
 -- A prior exception decision belongs only to its reviewed roster version.
 -- Use an explicit alias to avoid confusing PL/pgSQL variables with columns.
 update public.pcs_interclub_roster_versions v set status='exception_approved',decision_reason='Prior cutoff review',decided_at=now() where v.team_id=roster_id and v.revision=1;
 insert into public.pcs_interclub_availability_settings(season_id,club_id,meet_id,open,deadline)
  values(sid,home,mid,true,start_at-interval '1 day');
 insert into public.pcs_interclub_availability_responses(season_id,club_id,meet_id,member_id,status,responded_at)
  values(sid,home,mid,member_id,'available',now()) returning id,token_nonce into response_id,old_nonce;
 if (select deadline_editable from public.pcs_interclub_meet_workspaces where id=mid)
  or not(select schedule_deadline_editable from public.pcs_interclub_meet_workspaces where id=mid) then raise exception 'Deadline editor policies disagree'; end if;
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,away,sid,mid,1,earlier,earlier,180,2);
  raise exception 'Host changed commissioner schedule';
 exception when insufficient_privilege then null; end;
 out:=public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,1,earlier,earlier,120,2);
 if out->'meet'->>'revision'<>'2' or out->>'availability_reset_count'<>'1' or out->>'rosters_refreshed'<>'1'
  or out->>'publication_review_required'<>'true' then raise exception 'Schedule update response incomplete'; end if;
 if (select roster from public.pcs_interclub_roster_versions v where v.team_id=roster_id and v.revision=1)<>before_roster then raise exception 'Historical roster rewritten'; end if;
 if (select revision from public.pcs_interclub_teams where id=roster_id)<>2
  or (select count(*) from public.pcs_interclub_team_players p where p.team_id=roster_id)<>4
  or exists(select 1 from public.pcs_interclub_current_rosters r,lateral jsonb_array_elements(r.roster) p
   where r.id=roster_id and ((p->>'rating_deadline')::timestamptz<>earlier or (p->>'rating_locked')::boolean))
  or (select status from public.pcs_interclub_current_rosters where id=roster_id)<>'needs_exception'
  or (select decided_at from public.pcs_interclub_current_rosters where id=roster_id) is not null then raise exception 'Provisional roster refresh lost choices or copied old approval'; end if;
 if (select status from public.pcs_interclub_availability_responses where id=response_id)<>'invited'
  or (select token_nonce from public.pcs_interclub_availability_responses where id=response_id)=old_nonce
  or (select responded_at from public.pcs_interclub_availability_responses where id=response_id) is not null
  or (select open from public.pcs_interclub_availability_settings where meet_id=mid) then raise exception 'Old RSVP/link remained valid for changed time'; end if;
 if (select (details->'meets'->0->>'starts_at')::timestamptz from public.pcs_interclub_seasons where id=sid)<>earlier then raise exception 'Season schedule projection stale'; end if;
 if not exists(select 1 from public.pcs_interclub_registration_audit where season_id=sid and action='meet_schedule_updated'
  and details->'responses_before'->0->>'status'='available' and details->>'rosters_refreshed'='1') then raise exception 'Schedule history missing'; end if;
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,1,start_at,earlier,120,2);
  raise exception 'Stale schedule revision accepted';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,2,now()-interval '3 days',earlier+interval '1 hour');
  raise exception 'Registration allowed past revised first meet';
 exception when invalid_parameter_value then null; end;

 payload:=jsonb_build_object('request_id',request_id,'competition_phase','regular','host_club_id',away,'club_ids',jsonb_build_array(home,away),
  'starts_at',now()+interval '30 days','roster_deadline',now()+interval '29 days','duration_minutes',120,'courts',2);
 out:=public.pcs_create_interclub_competition_meet(actor,actor_email,home,sid,payload); added:=(out->>'id')::uuid;
 if (public.pcs_create_interclub_competition_meet(actor,actor_email,home,sid,payload)->>'id')::uuid<>added
  or (select count(*) from public.pcs_interclub_meets where season_id=sid)<>2 then raise exception 'Create retry duplicated meet'; end if;
 if jsonb_array_length((select details->'meets' from public.pcs_interclub_seasons where id=sid))<>2 then raise exception 'Added meet missing from season details'; end if;
 if (select published from public.pcs_interclub_publications where season_id=sid)<>first_publication then raise exception 'Public schedule changed without explicit publication review'; end if;
 begin
  perform public.pcs_create_interclub_competition_meet(actor,actor_email,home,sid,payload||jsonb_build_object('courts',3));
  raise exception 'Changed create retry accepted';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,added,1,earlier,earlier,120,2);
  raise exception 'Overlapping schedule accepted';
 exception when invalid_parameter_value then null; end;
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,added,1,now()+interval '91 days',now()+interval '90 days',120,2);
  raise exception 'Out of season schedule accepted';
 exception when invalid_parameter_value then null; end;
 begin
  -- UTC next day is still the last league date, but the meet end crosses it.
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,added,1,
   ((current_date+90)::timestamp+interval '23 hours') at time zone 'America/Mazatlan',now()+interval '29 days',120,2);
  raise exception 'Meet ending beyond local season boundary accepted';
 exception when invalid_parameter_value then null; end;

 -- Even a pre-existing snapshot at a future cutoff freezes deadline editing.
 begin
  insert into public.pcs_interclub_meet_eligibility_snapshots(season_id,meet_id,deadline,entry_id,club_id,player_id,rating,gender)
   select sid,mid,earlier,e.id,e.club_id,e.player_id,e.starting_rating,'female' from public.pcs_interclub_entries e where e.season_id=sid limit 1;
  if (select schedule_deadline_editable from public.pcs_interclub_meet_workspaces where id=mid) then raise exception 'Snapshot did not freeze future cutoff'; end if;
  begin
   perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,2,start_at,start_at,120,2);
   raise exception 'Snapshot eligibility cutoff changed';
  exception when sqlstate 'PT409' then null; end;
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,2,start_at,earlier,120,2);
  if not exists(select 1 from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and deadline=earlier) then raise exception 'Date change removed snapshot'; end if;
  raise exception 'Rollback snapshot probe' using errcode='Z0001';
 exception when sqlstate 'Z0001' then null; end;
 -- A passed eligibility cutoff can never be rewritten; moving only the future
 -- start keeps the cutoff and every historical rating/roster version intact.
 update public.pcs_interclub_meets set roster_deadline=now()-interval '1 hour' where id=mid;
 select roster_deadline into old_deadline from public.pcs_interclub_meets where id=mid;
 perform public.pcs_lock_interclub_meet_eligibility(mid);
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,2,start_at,start_at,120,2);
  raise exception 'Frozen eligibility cutoff changed';
 exception when sqlstate 'PT409' then null; end;
 out:=public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,2,start_at,old_deadline,120,2);
 if (out->'meet'->>'roster_deadline')::timestamptz<>old_deadline
  or (select revision from public.pcs_interclub_teams where id=roster_id)<>2 then raise exception 'Date-only change rewrote eligibility'; end if;

 -- Unplayed generated sheets can move with fixed courts/cutoff, without touching
 -- any document or source revision. Real play and published work remain locked.
 before_doc:=jsonb_build_object('weather','normal','encounters',jsonb_build_array(jsonb_build_object('tiebreak',null,'pairings',jsonb_build_array(
  jsonb_build_object('players_a',jsonb_build_array('a','b'),'players_b',jsonb_build_array('c','d'),'games',jsonb_build_array(jsonb_build_object('status','pending')))))));
 insert into public.pcs_interclub_competition_batches(id,season_id,meet_id,phase,document,roster_sources)
 values(batch_id,sid,mid,'regular',before_doc,'[]');
 out:=public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,3,start_at+interval '1 day',old_deadline,120,2);
 if (select document from public.pcs_interclub_competition_batches where id=batch_id)<>before_doc
  or (select revision from public.pcs_interclub_competition_batches where id=batch_id)<>1 then raise exception 'Prepared score sheet rewritten'; end if;
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,4,start_at+interval '2 days',old_deadline,120,3);
  raise exception 'Generated court assignments changed';
 exception when sqlstate 'PT409' then null; end;
 update public.pcs_interclub_competition_batches set document=jsonb_set(document,'{encounters,0,pairings,0,games,0,a}','11') where id=batch_id;
 begin
  perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,mid,4,start_at+interval '2 days',old_deadline,120,2);
  raise exception 'Scored meet changed through ordinary scheduling';
 exception when sqlstate 'PT409' then null; end;
 update public.pcs_interclub_competition_batches set document=before_doc,approved_document=before_doc,approved_revision=1 where id=batch_id;
 if (select schedule_editable from public.pcs_interclub_meet_workspaces where id=mid) then raise exception 'Approved history did not lock editing'; end if;
 -- Simulate a transaction/season-lock wait crossing both boundaries. This
 -- cutoff remains after transaction-start now(), even after wall time passes.
 begin
  old_deadline:=clock_timestamp()+interval '20 milliseconds';
  update public.pcs_interclub_meets set starts_at=old_deadline,roster_deadline=old_deadline where id=added;
  if old_deadline<=now() then raise exception 'Clock-boundary fixture must be after transaction start'; end if;
  perform pg_sleep(0.03);
  if public.pcs_interclub_schedule_deadline_editable(added)
   or public.pcs_interclub_meet_schedule_lock_reason(added) is null then
   raise exception 'Transaction-start clock allowed editing after a real-time cutoff'; end if;
  begin
   perform public.pcs_update_interclub_meet_schedule(actor,actor_email,home,sid,added,1,
    now()+interval '31 days',now()+interval '30 days',120,2);
   raise exception 'Ordinary schedule edit moved a meet after its start boundary';
  exception when sqlstate 'PT409' then null; end;
  raise exception 'Rollback clock-boundary probe' using errcode='Z0001';
 exception when sqlstate 'Z0001' then null; end;
 update public.pcs_interclub_meets set starts_at=now(),roster_deadline=now() where id=added;
 if (select schedule_editable from public.pcs_interclub_meet_workspaces where id=added) then raise exception 'Started meet remained editable'; end if;
 raise notice 'Meet schedule adjustments: provisional rosters, RSVP reset, frozen history, permissions, revisions, date bounds and idempotency passed';
end $$;
rollback;
