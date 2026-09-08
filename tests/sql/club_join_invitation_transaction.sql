-- Staging transaction only; all Auth users, clubs, invitations and grants roll back.
begin;
do $$
declare
 actor uuid:=gen_random_uuid(); recipient uuid:=gen_random_uuid(); wrong_user uuid:=gen_random_uuid();
 actor_email text:=actor::text||'@example.test'; recipient_email text:=recipient::text||'@example.test';
 organizer text:='join-organizer-'||gen_random_uuid()::text;
 new_club text:='join-club-'||gen_random_uuid()::text;
 other_club text:='join-other-'||gen_random_uuid()::text;
 season_id uuid:=gen_random_uuid(); invitation_id uuid:=gen_random_uuid();
 draft jsonb; result jsonb; snapshot jsonb; audit_count bigint;
begin
 if has_function_privilege('anon','public.pcs_create_interclub_club_invitation(uuid,text,text,uuid,uuid,integer,jsonb,text,text,text)','EXECUTE')
  or has_function_privilege('authenticated','public.pcs_interclub_club_invitation(text,uuid,uuid,text,text,uuid,integer,text)','EXECUTE')
  or has_table_privilege('authenticated','public.pcs_club_join_invitations','SELECT') then raise exception 'Invitation exposed to browser roles'; end if;
 insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
  values(organizer,organizer,'Join test organizer',false,'draft','draft'),(other_club,other_club,'Join other club',false,'draft','draft');
 insert into public.admin_role_assignments(club_id,email,role,user_id)
  values(organizer,actor_email,'administrator',actor),(other_club,recipient_email,'administrator',recipient);
 insert into auth.users(id,email,email_confirmed_at,aud,role)
  values(recipient,recipient_email,now(),'authenticated','authenticated'),(wrong_user,wrong_user::text||'@example.test',now(),'authenticated','authenticated');
 draft:=jsonb_build_object('name','Join season','club_ids',jsonb_build_array(organizer),'setup_step',1,'divisions',jsonb_build_array('3.5'));
 perform public.pcs_save_interclub_draft(actor,actor_email,organizer,season_id,0,draft);
 begin
  perform public.pcs_create_interclub_club_invitation(actor,actor_email,other_club,season_id,invitation_id,1,draft,new_club,new_club,recipient_email);
  raise exception 'Cross-club creation allowed';
 exception when insufficient_privilege then null; end;
 begin
  perform public.pcs_create_interclub_club_invitation(actor,actor_email,organizer,season_id,invitation_id,2,draft,new_club,new_club,recipient_email);
  raise exception 'Stale draft allowed';
 exception when serialization_failure then null; end;
 if exists(select 1 from public.clubs where id=new_club) then raise exception 'Failed creation leaked a club'; end if;
 set local role service_role;
 result:=public.pcs_create_interclub_club_invitation(actor,actor_email,organizer,season_id,invitation_id,1,draft,new_club,new_club,recipient_email);
 reset role;
 snapshot:=result;
 if result->'season'->'draft'->'club_ids' is distinct from jsonb_build_array(organizer,new_club)
  or (result->'season'->>'revision')::integer<>2 then raise exception 'Club selection not saved atomically'; end if;
 if exists(select 1 from public.admin_role_assignments where club_id=new_club)
  or exists(select 1 from public.clubs where id=new_club and (is_active or plan_status<>'free')) then raise exception 'Creation granted access or activated billing'; end if;
 result:=public.pcs_create_interclub_club_invitation(actor,actor_email,organizer,season_id,invitation_id,1,draft,new_club,new_club,recipient_email);
 if result is distinct from snapshot then raise exception 'Creation retry changed data'; end if;
 select count(*) into audit_count from public.pcs_platform_audit where club_id=new_club;
 if audit_count<>1 then raise exception 'Creation retry repeated audit'; end if;
 begin
  perform public.pcs_create_interclub_club_invitation(actor,actor_email,organizer,season_id,gen_random_uuid(),2,draft,new_club,new_club,recipient_email);
  raise exception 'Existing club could be claimed';
 exception when unique_violation then null; end;
 begin
  perform public.pcs_interclub_club_invitation('cancel',invitation_id,actor,actor_email,other_club,season_id,1,recipient_email);
  raise exception 'Cross-club cancellation allowed';
 exception when no_data_found then null; end;
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,wrong_user,recipient_email);
  raise exception 'Forged identity accepted';
 exception when insufficient_privilege then null; end;
 update auth.users set email_confirmed_at=null where id=recipient;
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
  raise exception 'Unverified email accepted';
 exception when insufficient_privilege then null; end;
 update auth.users set email_confirmed_at=now() where id=recipient;
 if public.pcs_interclub_club_invitation('email_claim',invitation_id,p_email=>'wrong@example.test') is not null then raise exception 'Wrong email claimed'; end if;
 if public.pcs_interclub_club_invitation('email_claim',invitation_id,p_email=>recipient_email) is null then raise exception 'First email claim failed'; end if;
 if public.pcs_interclub_club_invitation('email_claim',invitation_id,p_email=>recipient_email) is not null then raise exception 'Email throttle bypass'; end if;
 update public.pcs_club_join_invitations set sign_in_count=5,last_sign_in_at=now()-interval '1 day' where id=invitation_id;
 if public.pcs_interclub_club_invitation('email_claim',invitation_id,p_email=>recipient_email) is not null then raise exception 'Email cap bypass'; end if;
 update public.admin_role_assignments set revoked_at=now() where club_id=organizer;
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
  raise exception 'Revoked inviter accepted';
 exception when insufficient_privilege then null; end;
 update public.admin_role_assignments set revoked_at=null where club_id=organizer;
 update public.pcs_club_join_invitations set expires_at=now()-interval '1 second' where id=invitation_id;
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
  raise exception 'Expired invitation accepted';
 exception when serialization_failure then null; end;
 result:=public.pcs_interclub_club_invitation('renew',invitation_id,actor,actor_email,organizer,season_id,1,'corrected@example.test');
 if result->>'email'<>'corrected@example.test' or (result->>'revision')::integer<>2 then raise exception 'Email correction not saved'; end if;
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
  raise exception 'Previous recipient accepted';
 exception when insufficient_privilege then null; end;
 begin
  perform public.pcs_interclub_club_invitation('cancel',invitation_id,actor,actor_email,organizer,season_id,1,recipient_email);
  raise exception 'Stale invitation update allowed';
 exception when serialization_failure then null; end;
 perform public.pcs_interclub_club_invitation('renew',invitation_id,actor,actor_email,organizer,season_id,2,recipient_email);
 perform public.pcs_interclub_club_invitation('cancel',invitation_id,actor,actor_email,organizer,season_id,3,recipient_email);
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
  raise exception 'Cancelled invitation accepted';
 exception when serialization_failure then null; end;
 perform public.pcs_interclub_club_invitation('renew',invitation_id,actor,actor_email,organizer,season_id,4,recipient_email);
 -- A concurrent platform staff assignment must never be replaced.
 insert into public.admin_role_assignments(club_id,email,role) values(new_club,'other-admin@example.test','administrator');
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
  raise exception 'Existing staff overwritten';
 exception when serialization_failure then null; end;
 delete from public.admin_role_assignments where club_id=new_club;
 set local role service_role;
 result:=public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
 reset role;
 if result->>'status'<>'accepted' then raise exception 'Acceptance not recorded'; end if;
 if not exists(select 1 from public.admin_role_assignments where club_id=new_club and email=recipient_email and user_id=recipient and role='administrator' and revoked_at is null)
  or exists(select 1 from public.admin_role_assignments where club_id=new_club and email=actor_email)
  or not exists(select 1 from public.admin_role_assignments where club_id=other_club and email=recipient_email) then raise exception 'Incorrect club grants'; end if;
 select count(*) into audit_count from public.pcs_platform_audit where club_id=new_club;
 perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
 if (select count(*) from public.pcs_platform_audit where club_id=new_club)<>audit_count then raise exception 'Retry repeated acceptance'; end if;
 update public.admin_role_assignments set revoked_at=now() where club_id=new_club;
 begin
  perform public.pcs_interclub_club_invitation('accept',invitation_id,recipient,recipient_email);
  raise exception 'Replay restored revoked access';
 exception when serialization_failure then null; end;
 if exists(select 1 from public.pcs_interclub_seasons where id=season_id) then raise exception 'Account invitation opened the season'; end if;
end $$;
rollback;
