-- Staging-only rollback rehearsal: no fixture data, ratings or account changes persist.
begin;
do $$
declare
 home text:='competition-home-'||gen_random_uuid()::text; away text:='competition-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='competition-qa@example.invalid'; sid uuid:=gen_random_uuid(); mid uuid;
 team_home uuid:=gen_random_uuid(); team_away uuid:=gen_random_uuid(); ids bigint[]:='{}'; entries uuid[]:='{}'; pid bigint;
 out jsonb; doc jsonb; completed jsonb; replay jsonb; pairings jsonb:='[]'; games jsonb; lineup_a jsonb; lineup_b jsonb;
 sources jsonb; first_official jsonb; first_pair jsonb; i integer; j integer; k integer; c text; bid uuid; forbidden boolean;
begin
 if has_table_privilege('anon','public.pcs_interclub_competition_batches','SELECT')
  or has_table_privilege('authenticated','public.pcs_interclub_competition_audit','SELECT')
  or not(select relrowsecurity from pg_class where oid='public.pcs_interclub_competition_batches'::regclass)
  or has_function_privilege('authenticated','public.pcs_write_interclub_competition(uuid,text,text,uuid,uuid,text,text,integer,jsonb,jsonb,text,timestamptz,timestamptz,jsonb)','EXECUTE') then
  raise exception 'Competition data or RPC exposed to browsers'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(home,home,'Competition Home',false,'draft','draft'),(away,away,'Competition Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(home,actor_email,'administrator',actor),(away,actor_email,'administrator',actor);
 perform public.pcs_save_interclub_draft(actor,actor_email,home,sid,0,jsonb_build_object('name','Competition QA','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.5'),'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
  jsonb_build_object('host_club_id',away,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',2))));
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,home,sid,1,'{"3.5":{"min_rating":3.5,"max_rating":3.999,"women_required":2}}');
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,actor_email,away,sid,away,1,'accept');
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,0,now()-interval '3 days',now()+interval '1 day');
 select id into mid from public.pcs_interclub_meets where season_id=sid;
 for i in 1..8 loop
  c:=case when i<=4 then home else away end;
  pid:=8000000000000+(random()*1000000000000)::bigint; ids:=array_append(ids,pid);
  insert into public.players(id,club_id,name,normalized_name,rating,active,gender)
   values(pid,c,'Competition QA '||i,'competition qa '||i,1440,true,case when (i-1)%4<2 then 'female' else 'male' end);
  insert into public.pcs_interclub_pool_members(season_id,club_id,name,email,player_id,request_id,request_fingerprint)
   values(sid,c,'Competition QA '||i,'competition-'||i||'@example.invalid',pid,gen_random_uuid(),'fixture');
  entries:=array_append(entries,(select id from public.pcs_interclub_entries where season_id=sid and club_id=c and player_id=pid));
 end loop;
 perform public.pcs_set_interclub_registration_window(actor,actor_email,home,sid,1,now()-interval '3 days',now()-interval '2 days');
 begin
  perform public.pcs_create_interclub_competition_meet(actor,actor_email,home,sid,jsonb_build_object(
   'competition_phase','final','host_club_id',away,'club_ids',jsonb_build_array(home,away),
   'starts_at',now()+interval '20 days','roster_deadline',now()+interval '19 days','duration_minutes',180,'courts',2));
  raise exception 'Overlapping meet accepted';
 exception when invalid_parameter_value then null; end;
 perform public.pcs_save_interclub_meet_roster(actor,actor_email,home,sid,mid,1,team_home,0,'Home 3.5','3.5',ids[1:4]);
 perform public.pcs_save_interclub_meet_roster(actor,actor_email,away,sid,mid,1,team_away,0,'Away 3.5','3.5',ids[5:8]);
 -- The fixtures reach the deadline in this rollback-only transaction.
 update public.pcs_interclub_meets set roster_deadline=now() where id=mid;
 sources:=jsonb_build_array(jsonb_build_object('team_id',team_home,'revision',1),jsonb_build_object('team_id',team_away,'revision',1));
 for i in 0..1 loop
  lineup_a:=to_jsonb(entries[(i*2+1):(i*2+2)]);lineup_b:=to_jsonb(entries[(i*2+5):(i*2+6)]); games:='[]';
  for j in 1..3 loop
   games:=games||jsonb_build_array(jsonb_build_object('id',gen_random_uuid(),'status','pending','a',null,'b',null,'winner',null,'players_a','[]'::jsonb,'players_b','[]'::jsonb,'played_at',now(),'injury_reason',null));
  end loop;
  pairings:=pairings||jsonb_build_array(jsonb_build_object('id',gen_random_uuid(),'kind',case when i=0 then 'women' else 'men' end,'court',i+1,
   'eligibility_deadline',now(),'players_a',lineup_a,'players_b',lineup_b,'games',games));
 end loop;
 doc:=jsonb_build_object('schema_version',1,'meet_id',mid,'phase','regular','format','gender','weather','normal','encounters',jsonb_build_array(
  jsonb_build_object('id',gen_random_uuid(),'division','3.5','club_a',home,'club_b',away,'rotation',1,'pairings',pairings,'tiebreak',null)));
 out:=public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','generate',0,doc,sources);
 bid:=(out->>'id')::uuid;
 if out->>'state'<>'draft' or (out->>'revision')::integer<>1 then raise exception 'Generation failed'; end if;
 -- Positive refresh path runs in a deliberate subtransaction so the main
 -- approval fixture keeps its revision numbers after this independent check.
 begin
  out:=public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','refresh_lineups',1,doc,sources);
  if out->>'revision'<>'2' or out->'document'<>doc then raise exception 'Pre-start refresh failed' using errcode='XX001'; end if;
  raise exception 'Rollback positive refresh rehearsal' using errcode='Z0001';
 exception when sqlstate 'Z0001' then null; end;
 update public.pcs_interclub_meets set starts_at=now() where id=mid;
 begin
  perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','refresh_lineups',1,doc,sources);
  raise exception 'Post-start lineup refresh accepted';
 exception when sqlstate 'PT409' then null; end;
 if (select revision from public.pcs_interclub_competition_batches where id=bid)<>1 then raise exception 'Rejected refresh was not atomic'; end if;
 update public.pcs_interclub_meets set starts_at=now()+interval '20 days' where id=mid;

 begin
  perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','save',0,doc,sources);
  raise exception 'Stale revision accepted';
 exception when sqlstate 'PT409' then null; end;
 begin
  perform public.pcs_write_interclub_competition(gen_random_uuid(),actor_email,away,sid,mid,'regular','save',1,doc,sources);
  raise exception 'Forged actor accepted';
 exception when insufficient_privilege then null; end;
 begin
  perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','submit',1);
  raise exception 'Incomplete whole meet submitted';
 exception when invalid_parameter_value then null; end;
 completed:=doc;
 for i in 0..1 loop
  for j in 0..2 loop
   completed:=jsonb_set(completed,array['encounters','0','pairings',i::text,'games',j::text,'status'],'"completed"');
   completed:=jsonb_set(completed,array['encounters','0','pairings',i::text,'games',j::text,'a'],'11');
   completed:=jsonb_set(completed,array['encounters','0','pairings',i::text,'games',j::text,'b'],'7');
   completed:=jsonb_set(completed,array['encounters','0','pairings',i::text,'games',j::text,'winner'],'"a"');
  end loop;
 end loop;
 begin
  perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','save',1,completed,jsonb_set(sources,'{0,revision}','2'));
  raise exception 'Stale roster source accepted';
 exception when sqlstate 'PT409' then null; end;
 perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','save',1,completed,sources);
 perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','submit',2);
 begin
  perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','approve',3);
  raise exception 'Host approved official ratings';
 exception when insufficient_privilege then null; end;
 out:=public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','approve',3);
 if out->>'ratings_status'<>'pending' or out->>'state'<>'approved' or out->'approved_document'<>completed then raise exception 'Official approval failed'; end if;
 if (select count(*) from public.pcs_interclub_appearances where batch_id=bid)<>24 then raise exception 'Actual player appearances missing'; end if;
 first_official:=out->'approved_document';
 out:=public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','reopen',4,null,null,'Correct pencil score');
 if out->'approved_document'<>first_official or out->>'approved_revision'<>'4' then raise exception 'Reopening removed official scores'; end if;
 begin
  perform public.pcs_write_interclub_competition(actor,actor_email,away,sid,mid,'regular','save',5,completed,sources);
  raise exception 'Host changed published correction';
 exception when insufficient_privilege then null; end;
 completed:=jsonb_set(completed,'{encounters,0,pairings,0,games,0,b}','8');
 perform public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','save',5,completed,sources);
 perform public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','submit',6);
 out:=public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','approve',7);
 if (select count(*) from public.pcs_interclub_appearances where batch_id=bid)<>24
  or exists(select 1 from public.pcs_interclub_appearances where batch_id=bid and revision<>8) then raise exception 'Correction duplicated appearances'; end if;
 perform public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','reopen',8,null,null,'Weather cancellation correction');
 -- One completed pairing stands. The other unfinished pairing starts over.
 replay:=jsonb_set(completed,'{encounters,0,pairings,1,games}',doc#>'{encounters,0,pairings,1,games}');
 perform public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','save',9,replay,sources);
 replay:=jsonb_set(replay,'{weather}','"rescheduled"');
 replay:=jsonb_set(replay,'{encounters,0,pairings,1,eligibility_deadline}',to_jsonb(now()+interval '10 days'));
 first_pair:=completed#>'{encounters,0,pairings,0}';
 out:=public.pcs_write_interclub_competition(actor,actor_email,home,sid,mid,'regular','reschedule',10,replay,null,'Weather replay',now()+interval '11 days',now()+interval '10 days');
 if out#>'{document,encounters,0,pairings,0}'<>first_pair or out->'approved_document'<>completed then raise exception 'Replay changed completed or previously official scores'; end if;
 if (select count(*) from public.pcs_interclub_competition_audit where batch_id=bid)<>11 then raise exception 'Missing competition audit revisions'; end if;
 if not exists(select 1 from public.pcs_interclub_meet_eligibility_snapshots where meet_id=mid and deadline=now()) then raise exception 'Original eligibility not frozen before replay'; end if;
end $$;
rollback;
