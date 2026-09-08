-- Staging only. No mail, persistent Auth users, clubs, or assignments are created.
begin;
do $$
declare
 actor uuid := gen_random_uuid(); recipient uuid := gen_random_uuid(); wrong_user uuid := gen_random_uuid();
 club text := 'invitation-test-' || gen_random_uuid()::text;
 other_club text := 'invitation-other-' || gen_random_uuid()::text;
 recipient_email text := recipient::text || '@example.test';
 actor_email text := actor::text || '@example.test';
 invitation_id uuid := gen_random_uuid(); second_id uuid := gen_random_uuid();
 snapshot jsonb; result jsonb; audit_count bigint;
 requested_scopes jsonb := '[{"kind":"program_type","program_type":"leagues","resource_id":""}]';
begin
 if has_function_privilege('anon','public.pcs_staff_invitation(text,uuid,text,uuid,text,text,text,jsonb,timestamptz)','EXECUTE')
 or has_function_privilege('authenticated','public.pcs_staff_invitation(text,uuid,text,uuid,text,text,text,jsonb,timestamptz)','EXECUTE')
 or not has_function_privilege('service_role','public.pcs_staff_invitation(text,uuid,text,uuid,text,text,text,jsonb,timestamptz)','EXECUTE')
 or has_table_privilege('authenticated','public.club_staff_invitations','SELECT')
 or has_function_privilege('authenticated','public.pcs_staff_verified_email(uuid,text)','EXECUTE') then raise exception 'Invitation privileges incorrect'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
 values(club,club,'Invitation test',false,'draft','draft'),(other_club,other_club,'Other club',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role) values(club,actor_email,'administrator');
 insert into auth.users(id,email,email_confirmed_at,aud,role)
 values(recipient,recipient_email,now(),'authenticated','authenticated'),(wrong_user,wrong_user::text||'@example.test',now(),'authenticated','authenticated');
 result := public.pcs_staff_invitation('create',invitation_id,club,actor,actor_email,recipient_email,'operator',requested_scopes,now()+interval '30 days');
 if exists(select 1 from public.admin_role_assignments where club_id=club and email=recipient_email) then raise exception 'Creation granted access'; end if;
 snapshot := result;
 result := public.pcs_staff_invitation('create',invitation_id,club,actor,actor_email,recipient_email,'operator',requested_scopes,now()+interval '30 days');
 if result is distinct from snapshot then raise exception 'Creation retry changed invitation'; end if;
 select count(*) into audit_count from public.club_staff_audit where club_id=club;
 if audit_count<>1 then raise exception 'Creation retry added audit'; end if;
 begin
  perform public.pcs_staff_invitation('create',second_id,club,actor,actor_email,recipient_email,'administrator');
  raise exception 'Duplicate pending invitation allowed';
 exception when serialization_failure then null; end;
 begin
  perform public.pcs_staff_invitation('cancel',invitation_id,other_club,actor,actor_email);
  raise exception 'Cross-club cancellation allowed';
 exception when no_data_found then null; end;
 begin
  perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>wrong_user,p_actor_email=>recipient_email);
  raise exception 'Forged email accepted';
 exception when insufficient_privilege then null; end;
 update auth.users set email_confirmed_at=null where id=recipient;
 begin
  perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
  raise exception 'Unconfirmed account accepted';
 exception when insufficient_privilege then null; end;
 update auth.users set email_confirmed_at=now() where id=recipient;
 if public.pcs_staff_invitation('email_claim',invitation_id,p_email=>'wrong@example.test') is not null then raise exception 'Wrong email claimed'; end if;
 if public.pcs_staff_invitation('email_claim',invitation_id,p_email=>recipient_email) is null then raise exception 'First email claim failed'; end if;
 if public.pcs_staff_invitation('email_claim',invitation_id,p_email=>recipient_email) is not null then raise exception 'Email throttle bypass'; end if;
 update public.club_staff_invitations set sign_in_count=5,last_sign_in_at=now()-interval '1 day' where id=invitation_id;
 if public.pcs_staff_invitation('email_claim',invitation_id,p_email=>recipient_email) is not null then raise exception 'Email cap bypass'; end if;
 update public.admin_role_assignments set revoked_at=now() where club_id=club and email=actor_email;
 begin
  perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
  raise exception 'Revoked inviter accepted';
 exception when serialization_failure then null; end;
 update public.admin_role_assignments set revoked_at=null where club_id=club and email=actor_email;
 update public.club_staff_invitations set expires_at=now()-interval '1 second' where id=invitation_id;
 begin
  perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
  raise exception 'Expired invitation accepted';
 exception when serialization_failure then null; end;
 -- A fresh invitation replaces only an expired pending invitation.
 result := public.pcs_staff_invitation('create',second_id,club,actor,actor_email,recipient_email,'operator',requested_scopes,now()+interval '30 days');
 if (select status from public.club_staff_invitations where id=invitation_id)<>'cancelled' then raise exception 'Expired invitation not superseded'; end if;
 invitation_id := second_id;
 -- A staff change after invitation creation must not be overwritten.
 perform public.pcs_save_staff(club,actor_email,actor,recipient_email,'operator',requested_scopes,null,false);
 begin
  perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
  raise exception 'Newer staff assignment overwritten';
 exception when serialization_failure then null; end;
 perform public.pcs_staff_invitation('cancel',invitation_id,club,actor,actor_email);
 begin
  perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
  raise exception 'Cancelled invitation accepted';
 exception when serialization_failure then null; end;
 perform public.pcs_save_staff(club,actor_email,actor,recipient_email,'operator',requested_scopes,null,true);
 invitation_id := gen_random_uuid();
 result := public.pcs_staff_invitation('create',invitation_id,club,actor,actor_email,recipient_email,'operator',requested_scopes,now()+interval '30 days');
 -- Execute the acceptance as the same restricted role the API uses.
 set local role service_role;
 result := public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
 reset role;
 if result->>'status'<>'accepted' or (result->>'accepted_by')::uuid<>recipient then raise exception 'Acceptance not recorded'; end if;
 if not exists(select 1 from public.admin_role_assignments where club_id=club and email=recipient_email and user_id=recipient
  and revoked_at is null and role='operator' and admin_role_assignments.scopes=requested_scopes and expires_at=now()+interval '30 days') then raise exception 'Grant lost scope, expiry or identity'; end if;
 select count(*) into audit_count from public.club_staff_audit where club_id=club;
 perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
 if (select count(*) from public.club_staff_audit where club_id=club)<>audit_count then raise exception 'Acceptance retry granted again'; end if;
 perform public.pcs_save_staff(club,actor_email,actor,recipient_email,'operator',requested_scopes,null,true);
 begin
  perform public.pcs_staff_invitation('accept',invitation_id,p_actor_id=>recipient,p_actor_email=>recipient_email);
  raise exception 'Replay restored removed access';
 exception when serialization_failure then null; end;
 if exists(select 1 from public.admin_role_assignments where club_id=other_club) then raise exception 'Another club changed'; end if;
 if (select count(*) from public.club_staff_invitations where club_id=club and status='accepted')<>1 then raise exception 'Unexpected accepted invitations'; end if;
end $$;
rollback;
