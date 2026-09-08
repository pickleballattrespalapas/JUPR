-- Staging only. Every fixture and audit entry rolls back.
begin;
do $$
declare
 actor uuid:=gen_random_uuid(); sid uuid:=gen_random_uuid();
 actor_email text:='wizard-fixture@example.test';
 org text:='wizard-org-'||gen_random_uuid()::text;
 guest text:='wizard-guest-'||gen_random_uuid()::text;
 draft jsonb; saved jsonb; rules jsonb:='{"3.5":{"min_rating":null,"max_rating":3.999,"women_required":2}}';
begin
 if has_function_privilege('anon','public.pcs_save_interclub_draft(uuid,text,text,uuid,integer,jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.pcs_save_interclub_draft(uuid,text,text,uuid,integer,jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.pcs_save_interclub_draft(uuid,text,text,uuid,integer,jsonb)','EXECUTE') then
  raise exception 'Draft function privileges changed';
 end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(org,org,'Wizard organizer',false,'draft','draft'),(guest,guest,'Wizard guest',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
 values(org,actor_email,'administrator',actor),(guest,actor_email,'administrator',actor);
 draft:=jsonb_build_object('name','Coastal season','start_date',null,'end_date',null,'timezone','America/Mazatlan',
  'club_ids','[]'::jsonb,'divisions',jsonb_build_array('3.5'),'meets','[]'::jsonb,'registration_rules',rules,'setup_step',0);
 set local role service_role;
 saved:=public.pcs_save_interclub_draft(actor,actor_email,org,sid,0,draft);
 reset role;
 if saved->'draft' is distinct from draft or (saved->>'revision')::int<>1 then raise exception 'Partial draft did not round-trip'; end if;
 draft:=draft||jsonb_build_object('club_ids',jsonb_build_array(org,guest),'setup_step',3,'meets',jsonb_build_array(
  jsonb_build_object('host_club_id','','club_ids','[]'::jsonb,'starts_at',null,'duration_minutes',180,'courts',4)));
 saved:=public.pcs_save_interclub_draft(actor,actor_email,org,sid,1,draft);
 if saved->'draft' is distinct from draft or (saved->>'revision')::int<>2 then raise exception 'Unfinished meet or eligibility rules lost'; end if;
 begin
  perform public.pcs_save_interclub_draft(actor,actor_email,org,sid,1,draft);
  raise exception 'Stale save accepted';
 exception when serialization_failure then null; end;
 begin
  perform public.pcs_save_interclub_draft(actor,actor_email,guest,sid,2,draft);
  raise exception 'Other club overwrote draft';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=now() where club_id=org;
 begin
  perform public.pcs_save_interclub_draft(actor,actor_email,org,sid,2,draft);
  raise exception 'Revoked administrator saved draft';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=null where club_id=org;
 draft:=draft||jsonb_build_object('start_date',(current_date+10)::text,'end_date',(current_date+40)::text,'setup_step',4,
  'meets',jsonb_build_array(jsonb_build_object('host_club_id',org,'club_ids',jsonb_build_array(org,guest),
   'starts_at',now()+interval '20 days','duration_minutes',180,'courts',4)));
 saved:=public.pcs_save_interclub_draft(actor,actor_email,org,sid,2,draft);
 perform public.pcs_open_interclub_meet_registration(actor,actor_email,org,sid,3,rules);
 begin
  perform public.pcs_save_interclub_draft(actor,actor_email,org,sid,3,draft||'{"name":"Misleading edit"}'::jsonb);
  raise exception 'Opened season still accepts planning edits';
 exception when serialization_failure then null; end;
 if (select revision from public.pcs_interclub_drafts where id=sid)<>3
 or (select count(*) from public.pcs_interclub_draft_audit where season_id=sid)<>3
 or (select details->>'name' from public.pcs_interclub_seasons where id=sid)<>'Coastal season'
 or (select s.rules from public.pcs_interclub_seasons s where id=sid) is distinct from rules then
  raise exception 'Failed save changed draft, audit, or invitation terms';
 end if;
end $$;
rollback;
