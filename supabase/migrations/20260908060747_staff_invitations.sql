begin;

-- Invitation IDs locate records; they are not bearer credentials. Only a
-- confirmed account with the invited email can accept an invitation.
create table public.club_staff_invitations (
 id uuid primary key,
 club_id text not null references public.clubs(id),
 email text not null check (email = lower(trim(email)) and length(email) between 3 and 254),
 role text not null check (role in ('administrator','operator')),
 scopes jsonb not null check (jsonb_typeof(scopes) = 'array'),
 access_expires_at timestamptz,
 expires_at timestamptz not null default now() + interval '7 days',
 status text not null default 'pending' check (status in ('pending','accepted','cancelled')),
 invited_by uuid not null,
 invited_by_email text not null,
 target_before jsonb,
 target_after jsonb,
 created_at timestamptz not null default now(),
 accepted_at timestamptz,
 accepted_by uuid,
 cancelled_at timestamptz,
 last_sign_in_at timestamptz,
 sign_in_count integer not null default 0,
 check (role <> 'administrator' or access_expires_at is null)
);
create index club_staff_invitations_club_created_idx on public.club_staff_invitations(club_id,created_at desc);
create unique index club_staff_invitations_pending_email_idx on public.club_staff_invitations(club_id,email) where status='pending';
alter table public.club_staff_invitations enable row level security;
revoke all on public.club_staff_invitations from public,anon,authenticated;
grant all on public.club_staff_invitations to service_role;

-- All invitation changes share the existing staff lock. Recipient identity is
-- checked again against auth.users, not user metadata or an unverified address.
create function public.pcs_staff_invitation(
 p_action text, p_id uuid, p_club_id text default null,
 p_actor_id uuid default null, p_actor_email text default null,
 p_email text default null, p_role text default null,
 p_scopes jsonb default '[]', p_access_expires_at timestamptz default null
) returns jsonb language plpgsql security invoker set search_path = public as $$
declare
 invitation public.club_staff_invitations;
 actor public.admin_role_assignments;
 target public.admin_role_assignments;
 old_state jsonb;
 current_target jsonb;
 normalized_email text := lower(trim(p_email));
 actual_email text;
 actual_confirmed timestamptz;
begin
 if p_action not in ('create','cancel','accept','email_claim') or p_id is null then
  raise exception 'Invalid invitation action' using errcode='22023';
 end if;
 if p_action <> 'create' then
  select * into invitation from public.club_staff_invitations where id=p_id;
  if not found or (p_club_id is not null and invitation.club_id<>p_club_id) then
   raise exception 'Invitation unavailable' using errcode='P0002';
  end if;
  p_club_id := invitation.club_id;
 end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-staff:' || p_club_id,0));
 select * into invitation from public.club_staff_invitations where id=p_id for update;
 old_state := to_jsonb(invitation);

 if p_action in ('create','cancel') then
  select * into actor from public.admin_role_assignments
  where club_id=p_club_id and email=lower(trim(p_actor_email)) and revoked_at is null
   and (user_id is null or user_id=p_actor_id) and (expires_at is null or expires_at>now()) for share;
  if actor.role is null or actor.role not in ('super_admin','club_owner','administrator') then
   raise exception 'Administrator access required' using errcode='42501';
  end if;
 end if;

 if p_action='create' then
  if invitation.id is not null then
   if invitation.club_id is distinct from p_club_id or invitation.invited_by is distinct from p_actor_id
    or invitation.email is distinct from normalized_email or invitation.role is distinct from p_role
    or invitation.scopes is distinct from p_scopes or invitation.access_expires_at is distinct from p_access_expires_at then
    raise exception 'Invitation request changed' using errcode='40001';
   end if;
   return to_jsonb(invitation);
  end if;
  if p_role is null or p_role not in ('administrator','operator') or p_scopes is null or jsonb_typeof(p_scopes)<>'array'
   or normalized_email is null or normalized_email !~ '^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$'
   or (p_role='operator' and jsonb_array_length(p_scopes)=0)
   or (p_role='administrator' and (p_scopes<>'[]'::jsonb or p_access_expires_at is not null))
   or p_access_expires_at<=now() then
   raise exception 'Invalid invitation' using errcode='22023';
  end if;
  select * into target from public.admin_role_assignments where club_id=p_club_id and email=normalized_email for update;
  if target.role='super_admin' or (target.role is not null and target.revoked_at is null and (target.expires_at is null or target.expires_at>now())) then
   raise exception 'Edit existing staff access' using errcode='40001';
  end if;
  -- Expired invitations stay in the history but no longer block a new one.
  update public.club_staff_invitations set status='cancelled',cancelled_at=now()
   where club_id=p_club_id and email=normalized_email and status='pending' and (expires_at<=now() or access_expires_at<=now());
  if exists(select 1 from public.club_staff_invitations where club_id=p_club_id and email=normalized_email and status='pending') then
   raise exception 'Cancel the pending invitation first' using errcode='40001';
  end if;
  insert into public.club_staff_invitations(id,club_id,email,role,scopes,access_expires_at,invited_by,invited_by_email,target_before)
  values(p_id,p_club_id,normalized_email,p_role,p_scopes,p_access_expires_at,p_actor_id,lower(trim(p_actor_email)),to_jsonb(target))
  returning * into invitation;
 elsif p_action='cancel' then
  if invitation.status='accepted' then raise exception 'Edit accepted staff access' using errcode='40001'; end if;
  if invitation.status='cancelled' then return to_jsonb(invitation); end if;
  update public.club_staff_invitations set status='cancelled',cancelled_at=now() where id=p_id returning * into invitation;
 else
  if p_action='accept' then
   select lower(email),email_confirmed_at into actual_email,actual_confirmed from auth.users where id=p_actor_id for share;
   if actual_email is distinct from invitation.email or actual_confirmed is null
    or lower(trim(p_actor_email)) is distinct from invitation.email then
    raise exception 'Sign in with the invited verified email' using errcode='42501';
   end if;
  elsif normalized_email is distinct from invitation.email then
   return null;
  end if;
  select * into target from public.admin_role_assignments where club_id=p_club_id and email=invitation.email for update;
  current_target := to_jsonb(target);
  -- A retried acceptance never reapplies a grant, including after removal.
  if p_action='accept' and invitation.status='accepted' then
   if invitation.accepted_by=p_actor_id and current_target is not distinct from invitation.target_after
    and target.revoked_at is null and (target.expires_at is null or target.expires_at>now()) then return to_jsonb(invitation); end if;
   raise exception 'Staff access changed' using errcode='40001';
  end if;
  if invitation.status<>'pending' or invitation.expires_at<=now() or invitation.access_expires_at<=now() then
   raise exception 'Invitation no longer available' using errcode='40001';
  end if;
  select * into actor from public.admin_role_assignments where club_id=p_club_id and email=invitation.invited_by_email
   and revoked_at is null and (user_id is null or user_id=invitation.invited_by)
   and (expires_at is null or expires_at>now()) for share;
  if actor.role is null or actor.role not in ('super_admin','club_owner','administrator') then
   raise exception 'Inviter no longer has administrator access' using errcode='40001';
  end if;
  if current_target is distinct from invitation.target_before
   or (target.user_id is not null and p_action='accept' and target.user_id<>p_actor_id) then
   raise exception 'Staff access changed' using errcode='40001';
  end if;
  if p_action='email_claim' then
   if invitation.sign_in_count>=5 or invitation.last_sign_in_at>now()-interval '1 minute' then return null; end if;
   update public.club_staff_invitations set last_sign_in_at=now(),sign_in_count=sign_in_count+1 where id=p_id returning * into invitation;
   return to_jsonb(invitation);
  end if;
  perform public.pcs_save_staff(p_club_id,invitation.invited_by_email,invitation.invited_by,
   invitation.email,invitation.role,invitation.scopes,invitation.access_expires_at,false);
  update public.admin_role_assignments set user_id=p_actor_id where club_id=p_club_id and email=invitation.email returning * into target;
  update public.club_staff_invitations set status='accepted',accepted_at=now(),accepted_by=p_actor_id,target_after=to_jsonb(target)
   where id=p_id returning * into invitation;
 end if;
 insert into public.club_staff_audit(club_id,actor_email,target_email,before_state,after_state)
 values(p_club_id,lower(trim(p_actor_email)),invitation.email,
  jsonb_build_object('invitation',old_state),jsonb_build_object('invitation',to_jsonb(invitation),'action',p_action));
 return to_jsonb(invitation);
end $$;
revoke all on function public.pcs_staff_invitation(text,uuid,text,uuid,text,text,text,jsonb,timestamptz) from public,anon,authenticated;
grant execute on function public.pcs_staff_invitation(text,uuid,text,uuid,text,text,text,jsonb,timestamptz) to service_role;
commit;
