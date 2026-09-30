-- Staging-only transaction smoke. Every synthetic row is rolled back.
begin;
set local plpgsql.check_asserts=on;
do $$
declare cid text:='qa-event-history-'||gen_random_uuid()::text; actor uuid:=gen_random_uuid();
 kind text; original text; fresh uuid; rid uuid; s jsonb; input jsonb; template jsonb; result jsonb; again jsonb;
 tid uuid; op text; config jsonb; before_count integer; failed boolean;
begin
 insert into public.clubs(id,name,slug,status,is_active,public_site_status)
  values(cid,'QA event history',cid,'draft',false,'draft');
 insert into public.admin_role_assignments(user_id,email,role,club_id)
  values(actor,'event-history@example.invalid','administrator',cid);

 foreach kind in array array['individual','team','interclub','tournament'] loop
  fresh:=gen_random_uuid(); rid:=gen_random_uuid();
  if kind in ('individual','team') then
   original:=kind||' 2026';
   insert into public.leagues_metadata(club_id,league_name,league_type,is_active,status,match_format,rules_config,awards_config)
    values(cid,original,case when kind='team' then 'Team' else 'Individual' end,false,'ended','doubles',
     '{"overview":{"league_format":"ladder"}}','{"categories":{"most_wins":{"enabled":true}}}');
   if kind='team' then insert into public.team_league_settings(club_id,league_name,team_size,allow_substitutes)
    values(cid,original,2,true); end if;
   template:='{"schedule_config":{"start_date":"2027-01-01","end_date":"2027-03-01"}}';
  elsif kind='interclub' then
   original:=gen_random_uuid()::text;
   config:=jsonb_build_object('name','Interclub 2026','start_date','2026-01-01','end_date','2026-03-01',
    'timezone','America/Mazatlan','club_ids',jsonb_build_array(cid),'divisions',jsonb_build_array('3.5'),
    'registration_rules','{}'::jsonb,'meets','[]'::jsonb,'setup_step',0);
   perform public.pcs_save_interclub_draft(actor,'event-history@example.invalid',cid,original::uuid,0,config);
   insert into public.pcs_interclub_seasons(id,organizer_club_id,source_revision,details,rules)
    values(original::uuid,cid,1,config,'{}');
   insert into public.pcs_interclub_publications(season_id,draft,published)
    values(original::uuid,'{}','{"season_complete":true,"club_cup":{"status":"complete"}}');
   template:=config||'{"name":"Interclub 2027","start_date":"2027-01-01","end_date":"2027-03-01"}';
  else
   tid:=gen_random_uuid(); original:=tid::text;
   insert into public.tournaments(id,club_id,name,status,team_count) values(tid,cid,'Classic 2026','COMPLETED',4);
   s:=public.pcs_event_season_source(cid,'tournament',original);
   assert s->>'complete'='false','Completion requires the official lifecycle receipt';
   op:=md5(tid::text)||md5(cid);
   insert into public.tournament_admin_operations(operation_key,request_fingerprint,club_id,surface,action,entity_type,entity_id,
    lock_scope,expected_state,status,created_by,updated_by)
    values(op,op,cid,'tournament','complete','tournament',original,'qa:'||original,op,'completed','event-history@example.invalid','event-history@example.invalid');
   insert into public.tournament_lifecycle_receipts(club_id,tournament_id,action,from_status,to_status,operation_key,request_fingerprint,
    evidence_fingerprint,evidence_json,created_by)
    values(cid,tid,'complete','ACTIVE','COMPLETED',op,op,op,'{}','event-history@example.invalid');
   template:='{"version":3,"saved_step":"basics","days":[],"divisions":[],"event_families":[],"published_at":null,"published_event_families":[],"settings":{"locale":"en","timezone":"America/Mazatlan"},"basics":{"name":"Classic 2027"}}';
  end if;
  s:=public.pcs_event_season_source(cid,case when kind in ('individual','team') then 'league' else kind end,original);
  assert s->>'complete'='true','The source is completed';
  input:=jsonb_build_object('fingerprint',s->>'fingerprint','series_name',kind||' series','current_label','2026','label','2027',
   'name',kind||' 2027','start_date','2027-01-01','end_date','2027-03-01');
  result:=public.pcs_start_event_season(actor,'event-history@example.invalid',cid,
   case when kind in ('individual','team') then 'league' else kind end,original,rid,input,fresh,template);
  again:=public.pcs_start_event_season(actor,'event-history@example.invalid',cid,
   case when kind in ('individual','team') then 'league' else kind end,original,rid,input,fresh,template);
  assert result=again,'Retry returns the same new season';
  assert (select count(*)=2 from public.pcs_event_editions where series_id=(result->>'series_id')::uuid),'Exactly two editions';
  assert public.pcs_event_season_source(cid,case when kind in ('individual','team') then 'league' else kind end,original)->>'fingerprint'=s->>'fingerprint',
   'Rollover preserves the previous event';
  s:=public.pcs_event_season_source(cid,case when kind in ('individual','team') then 'league' else kind end,result->>'source_id');
  assert s->>'public'='false' and s->>'complete'='false','The new season stays a private draft';
  if kind='team' then
   assert s->'team'->>'allow_substitutes'='true' and s->'team'->>'registration_open'='false','Team settings carry forward with closed registration';
   assert (select count(*)=0 from public.team_league_teams where club_id=cid and league_name=result->>'source_id'),'No copied teams';
  elsif kind='tournament' then
   assert s->'settings'->'builder_draft_json'->>'published_at' is null,'No copied tournament publication';
   assert (select count(*)=0 from public.tournament_event_draws where tournament_id=fresh),'No copied tournament draws';
  elsif kind='interclub' then
   assert s->'setup'->'meets'='[]'::jsonb,'No copied meet dates';
   assert (select count(*)=0 from public.pcs_interclub_entries where season_id=fresh),'No copied player entries';
  end if;
  failed:=false;
  begin
   perform public.pcs_start_event_season(actor,'event-history@example.invalid',cid,
    case when kind in ('individual','team') then 'league' else kind end,original,gen_random_uuid(),input,gen_random_uuid(),template);
  exception when sqlstate 'PT409' then failed:=true; end;
  assert failed,'A second new season from the same predecessor is blocked';
  if kind='tournament' then
   delete from public.tournament_registration_settings where tournament_id=fresh::text;
   delete from public.tournaments where id=fresh and club_id=cid;
   assert not exists(select 1 from public.pcs_event_editions where source_id=fresh::text),
    'Deleting an unused draft clears its history membership';
   result:=public.pcs_start_event_season(actor,'event-history@example.invalid',cid,'tournament',original,
    gen_random_uuid(),input,gen_random_uuid(),template);
   assert (select count(*)=2 from public.pcs_event_editions where series_id=(result->>'series_id')::uuid),
    'A deleted draft can be replaced without losing the previous edition';
  end if;
 end loop;

 -- Linking historical results changes order/identity only, never old records.
 insert into public.leagues_metadata(club_id,league_name,league_type,is_active,status)
  values(cid,'Earlier 2025','Individual',false,'ended');
 s:=public.pcs_event_season_source(cid,'league','individual 2026');
 input:=jsonb_build_object('fingerprint',s->>'fingerprint','series_name','ignored existing series','current_label','2026','label','2025',
  'past_source_id','Earlier 2025','past_fingerprint',public.pcs_event_season_source(cid,'league','Earlier 2025')->>'fingerprint','before_source_id','individual 2026');
 result:=public.pcs_link_event_season(actor,'event-history@example.invalid',cid,'league','individual 2026',gen_random_uuid(),input);
 assert (select array_agg(label order by position)=array['2025','2026','2027'] from public.pcs_event_editions where series_id=(result->>'series_id')::uuid),'Historical ordering is durable';
 failed:=false;
 begin
  perform public.pcs_start_event_season(gen_random_uuid(),'outsider@example.invalid',cid,'league','individual 2026',gen_random_uuid(),input,gen_random_uuid(),'{}');
 exception when insufficient_privilege then failed:=true; end;
 assert failed,'Unauthorized staff cannot create seasons';
 assert not has_table_privilege('anon','public.pcs_event_editions','select'),'History ledger is server-only';
 assert not has_function_privilege('authenticated','public.pcs_start_event_season(uuid,text,text,text,text,uuid,jsonb,uuid,jsonb)','execute'),'Public clients cannot call the privileged write';
end $$;
set constraints all immediate;
rollback;
