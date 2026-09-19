-- Staging-only rehearsal. Every fixture and publication is rolled back.
begin;
do $$
declare home text:='publication-home-'||gen_random_uuid()::text;
 away text:='publication-away-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); email text:='publication-qa@example.invalid';
 sid uuid:=gen_random_uuid(); mid uuid; sources jsonb; doc jsonb; result jsonb;
begin
 if has_function_privilege('anon','public.pcs_publish_reviewed_interclub_publication(uuid,text,text,uuid,integer,jsonb,jsonb)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_publish_reviewed_interclub_publication(uuid,text,text,uuid,integer,jsonb,jsonb)','EXECUTE') then
  raise exception 'Reviewed publication RPC is exposed to browser roles'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(home,home,'Publication Home',false,'draft','draft'),(away,away,'Publication Away',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(home,email,'administrator',actor),(away,email,'administrator',actor);
 perform public.pcs_save_interclub_draft(actor,email,home,sid,0,jsonb_build_object('name','Publication QA','start_date',(current_date+1)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.5'),'club_ids',jsonb_build_array(home,away),'meets',jsonb_build_array(
  jsonb_build_object('host_club_id',away,'club_ids',jsonb_build_array(home,away),'starts_at',now()+interval '20 days','duration_minutes',180,'courts',2))));
 perform public.pcs_open_interclub_meet_registration(actor,email,home,sid,1,'{"3.5":{"min_rating":3.5,"max_rating":3.999,"women_required":2}}');
 perform public.pcs_interclub_participation(actor,email,home,sid,home,1,'accept');
 perform public.pcs_interclub_participation(actor,email,away,sid,away,1,'accept');
 select id into mid from public.pcs_interclub_meets where season_id=sid;
 perform public.pcs_write_interclub_publication(actor,email,home,sid,0,'save','{"results":[]}');
 select jsonb_build_object('approved','[]'::jsonb,
  'season',(select details from public.pcs_interclub_seasons where id=sid),
  'clubs',(select jsonb_agg(jsonb_build_object('id',id,'name',name) order by id) from public.clubs where id in(home,away)),
  'meets',(select jsonb_agg(jsonb_build_object('id',id,'revision',revision) order by id) from public.pcs_interclub_meets where season_id=sid)) into sources;
 doc:='{"scoring_version":1,"competition_results":[],"results":[]}';
 begin
  perform public.pcs_publish_reviewed_interclub_publication(actor,email,away,sid,1,doc,sources);
  raise exception 'Participant published organizer website';
 exception when insufficient_privilege then null; end;
 result:=public.pcs_publish_reviewed_interclub_publication(actor,email,home,sid,1,doc,sources);
 if result->'published'<>doc or result->>'revision'<>'2' then raise exception 'Exact reviewed publication failed'; end if;
 update public.pcs_interclub_meets set revision=revision+1 where id=mid;
 begin
  perform public.pcs_publish_reviewed_interclub_publication(actor,email,home,sid,2,doc,sources);
  raise exception 'Changed meet published without review';
 exception when sqlstate 'PT409' then null; end;
 update public.pcs_interclub_meets set revision=revision-1 where id=mid;
 update public.clubs set name='Changed name' where id=away;
 begin
  perform public.pcs_publish_reviewed_interclub_publication(actor,email,home,sid,2,doc,sources);
  raise exception 'Changed club published without review';
 exception when sqlstate 'PT409' then null; end;
 update public.clubs set name='Publication Away' where id=away;
 insert into public.pcs_interclub_competition_batches(season_id,meet_id,phase,state,document,roster_sources,approved_document,approved_revision)
 values(sid,mid,'regular','approved','{}','[]','{}',1);
 begin
  perform public.pcs_publish_reviewed_interclub_publication(actor,email,home,sid,2,doc,sources);
  raise exception 'New official result published without review';
 exception when sqlstate 'PT409' then null; end;
 if (select revision from public.pcs_interclub_publications where season_id=sid)<>2 then
  raise exception 'A failed review guard changed the publication'; end if;
 raise notice 'Reviewed publication, isolation, source changes and browser permissions passed';
end $$;
rollback;
