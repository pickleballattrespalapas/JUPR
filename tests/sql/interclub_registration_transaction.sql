-- Staging only; fixtures, rosters, identity changes and audits all roll back.
begin;
do $$
declare
 org text:='interclub-org-'||gen_random_uuid()::text; home text:='interclub-home-'||gen_random_uuid()::text;
 other text:='interclub-other-'||gen_random_uuid()::text;
 actor uuid:=gen_random_uuid(); actor_email text:='interclub-fixture@example.test';
 sid uuid:=gen_random_uuid(); tid uuid:=gen_random_uuid(); another_team uuid:=gen_random_uuid();
 ids bigint[]:='{}'; pid bigint; i integer; out jsonb; first_roster jsonb; entry_seed numeric; count_before bigint;
 rules jsonb:='{"3.5":{"min_rating":null,"max_rating":3.75,"women_required":2}}';
begin
 if has_function_privilege('anon','public.pcs_save_interclub_roster(uuid,text,text,uuid,uuid,integer,text,text,bigint[],boolean)','EXECUTE')
 or has_function_privilege('authenticated','public.pcs_review_interclub_roster(uuid,text,text,uuid,uuid,integer,boolean,text)','EXECUTE')
 or has_table_privilege('authenticated','public.pcs_interclub_current_rosters','SELECT')
 or has_table_privilege('anon','public.pcs_interclub_entries','SELECT') then raise exception 'Interclub browser privileges incorrect'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(org,org,'Organizer',false,'draft','draft'),(home,home,'Home club',false,'draft','draft'),(other,other,'Other club',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(org,actor_email,'administrator',actor),(home,actor_email,'administrator',actor),(other,actor_email,'administrator',actor);
 -- Use isolated IDs without consuming the real players identity sequence.
 for i in 1..9 loop
  pid:=8000000000000 + (random()*1000000000000)::bigint;
  ids:=array_append(ids,pid);
  insert into public.players(id,club_id,name,normalized_name,rating,active,gender)
  values(pid,case when i=9 then other else home end,'Fixture '||i,'fixture '||i,
   case when i=7 then null when i in (5,6) then 1680 else 1400 end,i<>8,case when i in (1,2) then 'male' else 'female' end);
 end loop;
 perform public.pcs_save_interclub_draft(actor,actor_email,org,sid,0,jsonb_build_object('name','Coastal League','start_date',(current_date+10)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('3.5'),'club_ids',jsonb_build_array(home,other),'meets','[]'::jsonb));
 begin
  perform public.pcs_open_interclub_registration(actor,actor_email,home,sid,1,rules,now()+interval '7 days');
  raise exception 'Host opened organizer registration';
 exception when insufficient_privilege then null; end;
 begin
  perform public.pcs_open_interclub_registration(actor,actor_email,org,sid,2,rules,now()+interval '7 days');
  raise exception 'Stale draft opened';
 exception when serialization_failure then null; end;
 out:=public.pcs_open_interclub_registration(actor,actor_email,org,sid,1,rules,now()+interval '7 days');
 perform public.pcs_open_interclub_registration(actor,actor_email,org,sid,1,rules,now()+interval '7 days');
 if (select count(*) from public.pcs_interclub_participations where season_id=sid)<>2
 or (select count(*) from public.pcs_interclub_registration_audit where season_id=sid)<>1 then raise exception 'Open retry duplicated invitations'; end if;
 perform public.pcs_save_interclub_draft(actor,actor_email,org,sid,1,jsonb_build_object('name','Edited draft','start_date',(current_date+10)::text,
  'end_date',(current_date+90)::text,'timezone','America/Mazatlan','divisions',jsonb_build_array('Open'),'club_ids',jsonb_build_array(home,other),'meets','[]'::jsonb));
 if (select details->>'name' from public.pcs_interclub_seasons where id=sid)<>'Coastal League' then raise exception 'Planning edit changed accepted terms'; end if;
 begin
  perform public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,0,'Home Blue','3.5',ids[1:4]);
  raise exception 'Unaccepted club submitted';
 exception when insufficient_privilege then null; end;
 begin
  perform public.pcs_interclub_participation(actor,actor_email,org,sid,home,1,'accept');
  raise exception 'Organizer accepted for another club';
 exception when insufficient_privilege then null; end;
 perform public.pcs_interclub_participation(actor,actor_email,org,sid,other,1,'cancel');
 begin
  perform public.pcs_interclub_participation(actor,actor_email,other,sid,other,2,'accept');
  raise exception 'Cancelled club accepted';
 exception when serialization_failure then null; end;
 perform public.pcs_interclub_participation(actor,actor_email,org,sid,other,2,'reinvite');
 perform public.pcs_interclub_participation(actor,actor_email,other,sid,other,3,'decline');
 perform public.pcs_interclub_participation(actor,actor_email,home,sid,home,1,'accept');
 -- Cross-club references fail both through the transaction and the FK.
 begin
  perform public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,0,'Home Blue','3.5',array[ids[1],ids[2],ids[3],ids[9]]);
  raise exception 'Cross-club player accepted';
 exception when invalid_parameter_value then null; end;
 begin
  insert into public.pcs_interclub_entries(season_id,club_id,player_id,starting_rating) values(sid,home,ids[9],3.5);
  raise exception 'Cross-club FK bypassed';
 exception when foreign_key_violation then null; end;
 if exists(select 1 from public.pcs_interclub_entries where season_id=sid) then raise exception 'Failed roster left partial entries'; end if;
 for i in 7..8 loop
  begin
   perform public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,0,'Home Blue','3.5',array[ids[1],ids[2],ids[3],ids[i]]);
   raise exception 'Unrated or inactive player accepted';
  exception when invalid_parameter_value then null; end;
 end loop;
 -- Exercise mutations through the real API role as well as fixture ownership.
 set local role service_role;
 out:=public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,0,'Home Blue','3.5',ids[1:4]);
 reset role;
 if out->'roster'->>'status'<>'eligible' then raise exception 'Valid roster not eligible'; end if;
 first_roster:=out->'roster'->'roster';
 begin
  perform public.pcs_save_interclub_roster(actor,actor_email,home,sid,another_team,0,'Home Red','3.5',ids[1:4]);
  raise exception 'Duplicate division lineup accepted';
 exception when unique_violation then null; end;
 update public.players set rating=2400 where id=ids[1];
 -- Simulate time passing while retaining creation before the original deadline.
 update public.pcs_interclub_teams set created_at=now()-interval '2 days' where id=tid;
 update public.pcs_interclub_seasons set roster_deadline=now()-interval '1 day' where id=sid;
 out:=public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,1,'Home Blue','3.5',ids[1:4]);
 if out->'roster'->>'status'<>'eligible' or not (out->'roster'->>'late_change')::boolean then raise exception 'Valid late substitution blocked'; end if;
 select starting_rating into entry_seed from public.pcs_interclub_entries where season_id=sid and club_id=home and player_id=ids[1];
 if entry_seed<>3.5 then raise exception 'Starting league rating changed'; end if;
 out:=public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,2,'Home Blue','3.5',array[ids[1],ids[2],ids[3],ids[5]]);
 if out->'roster'->>'status'<>'needs_exception' then raise exception 'Over-limit player not blocked'; end if;
 begin
  perform public.pcs_review_interclub_roster(actor,actor_email,home,sid,tid,3,true,'Host approves');
  raise exception 'Host approved eligibility';
 exception when insufficient_privilege then null; end;
 out:=public.pcs_review_interclub_roster(actor,actor_email,org,sid,tid,3,true,'Organizer approved for this roster');
 if out->'roster'->>'status'<>'exception_approved' then raise exception 'Organizer decision failed'; end if;
 out:=public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,3,'Home Blue','3.5',array[ids[1],ids[2],ids[3],ids[6]]);
 if out->'roster'->>'status'<>'needs_exception' or out->'roster'->>'decision_reason' is not null then raise exception 'Prior exception carried into new roster'; end if;
 begin
  perform public.pcs_review_interclub_roster(actor,actor_email,org,sid,tid,3,true,'Stale review');
  raise exception 'Stale roster approved';
 exception when serialization_failure then null; end;
 perform public.pcs_review_interclub_roster(actor,actor_email,org,sid,tid,4,false,'Use an eligible substitute');
 if (select roster from public.pcs_interclub_roster_versions where team_id=tid and revision=1) is distinct from first_roster then raise exception 'Historical lineup changed'; end if;
 update public.admin_role_assignments set revoked_at=now() where club_id=home;
 begin
  perform public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,4,'Home Blue','3.5',ids[1:4]);
  raise exception 'Revoked admin changed roster';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=null where club_id=home;
 out:=public.pcs_save_interclub_roster(actor,actor_email,home,sid,tid,4,'','',array[]::bigint[],true);
 if out->'roster'->>'status'<>'withdrawn' or exists(select 1 from public.pcs_interclub_team_players where team_id=tid) then raise exception 'Withdrawal retained active places'; end if;
 out:=public.pcs_save_interclub_roster(actor,actor_email,home,sid,another_team,0,'Late new team','3.5',ids[1:4]);
 if out->'roster'->>'status'<>'needs_exception' or not (out->'roster'->'issues' @> '[{"code":"late_new_team"}]') then raise exception 'Late new team bypassed review'; end if;
 if (select count(*) from public.pcs_interclub_roster_versions where team_id=tid)<>5 then raise exception 'Roster history missing'; end if;
 if (select count(*) from public.pcs_interclub_current_rosters where season_id=sid)<>2 then raise exception 'Current-roster projection inconsistent'; end if;
 if (select rating from public.players where id=ids[1])<>2400 then raise exception 'Roster operation wrote a home-club rating'; end if;
 select count(*) into count_before from public.pcs_interclub_registration_audit where season_id=sid;
 if count_before<>13 then raise exception 'Unexpected audit count: %',count_before; end if;
end $$;
rollback;
