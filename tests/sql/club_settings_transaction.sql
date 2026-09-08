-- Run against staging only. All fixtures and writes roll back.
begin;
do $$
declare
 actor uuid := gen_random_uuid();
 club_id text := 'settings-test-' || gen_random_uuid()::text;
 other_id text := 'settings-other-' || gen_random_uuid()::text;
 c public.clubs;
 saved jsonb;
 stamp timestamptz;
 count_before bigint;
begin
 if has_function_privilege('anon','public.pcs_save_club_settings(uuid,text,text,timestamptz,text,text,text,boolean)','EXECUTE')
 or has_function_privilege('authenticated','public.pcs_save_club_settings(uuid,text,text,timestamptz,text,text,text,boolean)','EXECUTE')
 or not has_function_privilege('service_role','public.pcs_save_club_settings(uuid,text,text,timestamptz,text,text,text,boolean)','EXECUTE') then
  raise exception 'Settings RPC privileges are incorrect';
 end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(club_id,club_id,'Test club',false,'draft','draft') returning * into c;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(other_id,other_id,'Other club',false,'draft','draft');
 -- Use an unbound assignment, as created by the real onboarding workflow.
 insert into public.admin_role_assignments(club_id,email,role)
 values(club_id,'settings-fixture@example.test','administrator');
 stamp := c.updated_at;
 saved := public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'New name','Local play','',false);
 if saved->>'onboarding_status' <> 'in_progress' or saved->>'name' <> 'New name' then raise exception 'Draft save failed'; end if;
 begin
  perform public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'Stale','Local play','',false);
  raise exception 'Stale save was accepted';
 exception when serialization_failure then null; end;
 stamp := (saved->>'updated_at')::timestamptz;
 begin
  perform public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'New name','Local play','',true);
  raise exception 'Missing contact was accepted';
 exception when invalid_parameter_value then null; end;
 saved := public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'New name','Local play','CONTACT@EXAMPLE.TEST',true);
 if saved->>'onboarding_status' <> 'ready_for_review' or saved->>'support_email' <> 'contact@example.test'
 or (saved->>'is_active')::boolean then raise exception 'Submission changed account activation or failed'; end if;
 stamp := (saved->>'updated_at')::timestamptz;
 begin
  perform public.pcs_save_club_settings(actor,'settings-fixture@example.test',other_id,stamp,'Intruder','','',false);
  raise exception 'Cross-club save was accepted';
 exception when insufficient_privilege then null; end;
 saved := public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'New name','Updated description','contact@example.test',false);
 if saved->>'onboarding_status' <> 'in_progress' then raise exception 'Edited submission stayed ready'; end if;
 stamp := (saved->>'updated_at')::timestamptz;
 select count(*) into count_before from public.pcs_platform_audit where pcs_platform_audit.club_id = c.id;
 if count_before <> 3 then raise exception 'Missing atomic audit records'; end if;
 update public.admin_role_assignments set revoked_at = now() where admin_role_assignments.club_id = c.id;
 begin
  perform public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'Revoked','','',false);
  raise exception 'Revoked staff save was accepted';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at = null, role='operator', expires_at=now()-interval '1 hour' where admin_role_assignments.club_id = c.id;
 begin
  perform public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'Expired','','',false);
  raise exception 'Expired operator save was accepted';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set role='administrator',expires_at=null where admin_role_assignments.club_id = c.id;
 update public.clubs set is_active = true, onboarding_status = 'ready' where clubs.id = c.id;
 select updated_at into stamp from public.clubs where clubs.id = c.id;
 saved := public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'Active update','','contact@example.test',false);
 if not (saved->>'is_active')::boolean or saved->>'onboarding_status' <> 'ready' then raise exception 'Active club status changed'; end if;
 stamp := (saved->>'updated_at')::timestamptz;
 begin
  perform public.pcs_save_club_settings(actor,'settings-fixture@example.test',club_id,stamp,'Active update','','contact@example.test',true);
  raise exception 'Active club was resubmitted';
 exception when invalid_parameter_value then null; end;
 if exists(select 1 from public.clubs where id=other_id and name <> 'Other club') then raise exception 'Other club changed'; end if;
 if (select count(*) from public.pcs_platform_audit where pcs_platform_audit.club_id = c.id) <> count_before+1 then raise exception 'Rejected writes created audits'; end if;
end $$;
rollback;
