-- Staging only. Every fixture and projection is rolled back.
begin;
do $$
declare
 a text:='rating-a-'||gen_random_uuid(); b text:='rating-b-'||gen_random_uuid(); unrelated text:='rating-other-'||gen_random_uuid();
 sid uuid:=gen_random_uuid(); mid uuid:=gen_random_uuid(); bid uuid:=gen_random_uuid(); ea uuid:=gen_random_uuid(); eb uuid:=gen_random_uuid();
 pa bigint:=8000000000000+(random()*1000000000000)::bigint; pb bigint:=9000000000000+(random()*1000000000000)::bigint;
 po bigint:=10000000000000+(random()*1000000000000)::bigint;
 snapshot jsonb; projection jsonb; output jsonb; bad jsonb; original numeric;
begin
 if has_table_privilege('anon','public.pcs_interclub_rating_effects','SELECT')
 or has_table_privilege('authenticated','public.pcs_interclub_current_ratings','SELECT')
 or has_function_privilege('authenticated','public.pcs_apply_interclub_rating_projection(text[],text,jsonb,uuid,integer)','EXECUTE')
 then raise exception 'Rating data exposed to browser roles'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status) values
  (a,a,'Rating A',false,'draft','draft'),(b,b,'Rating B',false,'draft','draft'),(unrelated,unrelated,'Unrelated',false,'draft','draft');
 insert into public.players(id,club_id,name,normalized_name,rating,starting_rating,active,gender,wins,losses,matches_played)
 values(pa,a,'Rating A','rating a',1400,1400,true,'female',0,0,0),(pb,b,'Rating B','rating b',1400,1400,true,'female',0,0,0),
 (po,unrelated,'Unrelated','unrelated',1500,1500,true,'male',0,0,0);
 insert into public.pcs_interclub_drafts(id,organizer_club_id,draft) values(sid,a,'{}');
 insert into public.pcs_interclub_seasons(id,organizer_club_id,source_revision,details,rules) values(sid,a,1,
  jsonb_build_object('name','Rating regression','start_date',(current_date-10)::text,'end_date',(current_date+30)::text,'timezone','America/Mazatlan'),'{}');
 insert into public.pcs_interclub_participations(season_id,club_id,status) values(sid,a,'accepted'),(sid,b,'accepted');
 insert into public.pcs_interclub_entries(id,season_id,club_id,player_id,starting_rating,entered_at)
 values(ea,sid,a,pa,3.5,now()-interval '2 days'),(eb,sid,b,pb,3.5,now()-interval '2 days');
 insert into public.pcs_interclub_meets(id,season_id,plan_index,host_club_id,club_ids,starts_at,duration_minutes,courts,roster_deadline)
 values(mid,sid,0,a,jsonb_build_array(a,b),now()-interval '1 day',180,2,now()-interval '2 days');
 insert into public.pcs_interclub_competition_batches(id,season_id,meet_id,phase,revision,state,document,roster_sources,ratings_status,approved_at)
 values(bid,sid,mid,'regular',2,'approved','{}','[]','pending',now());
 snapshot:=public.pcs_interclub_rating_snapshot(array[a]);
 if jsonb_array_length(snapshot->'snapshot'->'players')<>2 then raise exception 'Club closure leaked unrelated players'; end if;
 projection:=jsonb_build_object('players',jsonb_build_array(
  jsonb_build_object('id',pa,'club_id',a,'rating',1410,'wins',1,'losses',0,'matches_played',1),
  jsonb_build_object('id',pb,'club_id',b,'rating',1390,'wins',0,'losses',1,'matches_played',1)),
  'matches','[]'::jsonb,'effects',jsonb_build_array(jsonb_build_object('season_id',sid,'entry_id',ea,'batch_id',bid,'game_id','regression-game',
   'stream','league','played_at',now()-interval '1 day','ordinal',0,'before_elo',1400,'after_elo',1410)),
  'league_ratings',jsonb_build_array(jsonb_build_object('entry_id',ea,'rating',3.525)),
  'sources',jsonb_build_array(jsonb_build_object('batch_id',bid,'season_id',sid,'meet_id',mid,'revision',2,'document','{}'::jsonb,'approved_at',now())),
  'rated_games',1);
 -- An error after player writes must roll back the entire projection.
 bad:=jsonb_set(projection,'{effects,0,entry_id}',to_jsonb(gen_random_uuid()::text));
 begin
  perform public.pcs_apply_interclub_rating_projection(array[a],snapshot->>'fingerprint',bad,bid,2);
  raise exception 'Invalid effect was committed';
 exception when foreign_key_violation then null; end;
 if (select rating from public.players where id=pa)<>1400 or exists(select 1 from public.pcs_interclub_rating_generations where sid=any(seasons)) then
  raise exception 'Partial projection survived rollback'; end if;
 -- A plan cannot replace one expected player with an unrelated club identity.
 bad:=jsonb_set(jsonb_set(projection,'{players,0,id}',to_jsonb(po)),'{players,0,club_id}',to_jsonb(unrelated));
 begin
  perform public.pcs_apply_interclub_rating_projection(array[a],snapshot->>'fingerprint',bad,bid,2);
  raise exception 'Cross-club injection accepted';
 exception when raise_exception then if sqlerrm='Cross-club injection accepted' then raise; end if; end;
 output:=public.pcs_apply_interclub_rating_projection(array[a],snapshot->>'fingerprint',projection,bid,2);
 if output->>'status'<>'completed' or (select rating from public.players where id=pa)<>1410
 or (select rating from public.players where id=po)<>1500 then raise exception 'Atomic owned projection failed'; end if;
 if public.pcs_interclub_rating_at(sid,ea,now()-interval '1 day')<>3.5
 or public.pcs_interclub_rating_at(sid,ea,'infinity')<>3.525 then raise exception 'Deadline history changed retroactively'; end if;
 output:=public.pcs_apply_interclub_rating_projection(array[a],snapshot->>'fingerprint',projection,bid,2);
 if output->>'status'<>'conflict' or (select rating from public.players where id=pa)<>1410 then raise exception 'Stale replay applied twice'; end if;
 -- Local edits durably request a repair before the HTTP worker runs.
 update public.players set rating=1420 where id=pa;
 if not exists(select 1 from public.pcs_interclub_rating_repairs where club_id=a and pending)
 or (select ratings_status from public.pcs_interclub_competition_batches where id=bid)<>'pending' then raise exception 'Local rating repair was not durable'; end if;
 snapshot:=public.pcs_interclub_rating_snapshot(array[a]);
 update public.pcs_interclub_competition_batches set revision=3,state='draft' where id=bid;
 output:=public.pcs_apply_interclub_rating_projection(array[a],snapshot->>'fingerprint',projection,bid,2);
 if output->>'status'<>'conflict' then raise exception 'Changed approval revision accepted'; end if;
end $$;
rollback;
