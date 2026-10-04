begin;

-- A league organizer can invite a NEW club, but cannot assign staff at an
-- existing club. The recipient receives access only after verified acceptance.
create table public.pcs_club_join_invitations (
 id uuid primary key,
 organizer_club_id text not null references public.clubs(id),
 season_id uuid not null references public.pcs_interclub_drafts(id),
 club_id text not null unique references public.clubs(id),
 email text not null check (email=lower(trim(email)) and length(email) between 3 and 254),
 club_name text not null,
 invited_by uuid not null,
 invited_by_email text not null,
 status text not null default 'pending' check(status in ('pending','accepted','cancelled')),
 revision integer not null default 1,
 source_revision integer not null,
 expires_at timestamptz not null default now()+interval '7 days',
 created_at timestamptz not null default now(),
 accepted_at timestamptz,
 accepted_by uuid,
 target_after jsonb,
 last_sign_in_at timestamptz,
 sign_in_count integer not null default 0
);
create index pcs_club_join_invitations_season_idx on public.pcs_club_join_invitations(season_id,created_at);
create index pcs_club_join_invitations_organizer_idx on public.pcs_club_join_invitations(organizer_club_id);
alter table public.pcs_club_join_invitations enable row level security;
revoke all on public.pcs_club_join_invitations from public,anon,authenticated;
grant all on public.pcs_club_join_invitations to service_role;

create function public.pcs_create_interclub_club_invitation(
 p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_id uuid,p_revision integer,p_draft jsonb,p_name text,p_slug text,p_email text
) returns jsonb language plpgsql security invoker set search_path=public as $$
declare
 invitation public.pcs_club_join_invitations;
 saved public.pcs_interclub_drafts;
 c public.clubs;
 next_draft jsonb;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into saved from public.pcs_interclub_drafts where id=p_season_id and organizer_club_id=p_club_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if exists(select 1 from public.pcs_interclub_seasons where id=p_season_id) then
  raise exception 'Season setup already opened' using errcode='40001'; end if;
 if p_id is null or p_name is null or length(trim(p_name)) not between 1 and 120
  or p_slug is null or p_slug !~ '^[a-z0-9]+(-[a-z0-9]+)*$' or length(p_slug) not between 3 and 60
  or p_email is null or length(p_email)>254 or p_email !~ '^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$'
  or p_draft is null or jsonb_typeof(p_draft->'club_ids') is distinct from 'array'
  or jsonb_array_length(p_draft->'club_ids')>=32 then
  raise exception 'Invalid club invitation' using errcode='22023'; end if;
 next_draft := jsonb_set(p_draft,'{club_ids}',(p_draft->'club_ids')||jsonb_build_array(p_slug));
 select * into invitation from public.pcs_club_join_invitations where id=p_id;
 if found then
  if invitation.organizer_club_id is distinct from p_club_id or invitation.season_id is distinct from p_season_id
   or invitation.invited_by is distinct from p_actor_id or invitation.club_name is distinct from trim(p_name)
   or invitation.club_id is distinct from p_slug or invitation.email is distinct from lower(trim(p_email))
   or invitation.source_revision is distinct from p_revision or invitation.status<>'pending'
   or saved.revision is distinct from p_revision+1 or saved.draft is distinct from next_draft then
   raise exception 'Invitation request changed; reload' using errcode='40001'; end if;
 else
  if saved.revision is distinct from p_revision then raise exception 'Stale season draft' using errcode='40001'; end if;
  if (select count(*) from public.pcs_club_join_invitations where season_id=p_season_id)>=32 then
   raise exception 'Season invitation limit reached' using errcode='22023'; end if;
  perform pg_advisory_xact_lock(hashtextextended('pcs-club:'||p_slug,0));
  if exists(select 1 from public.clubs where id=p_slug or slug=p_slug or lower(trim(name))=lower(trim(p_name))) then
   raise exception 'Club already exists; select it' using errcode='23505'; end if;
  insert into public.clubs(id,slug,name,is_active,status,plan_status,onboarding_status)
   values(p_slug,p_slug,trim(p_name),false,'draft','free','draft');
  insert into public.pcs_club_join_invitations(id,organizer_club_id,season_id,club_id,email,club_name,invited_by,invited_by_email,source_revision)
   values(p_id,p_club_id,p_season_id,p_slug,lower(trim(p_email)),trim(p_name),p_actor_id,lower(trim(p_actor_email)),p_revision)
   returning * into invitation;
  -- Creation, selection and saved progress commit together, using the same
  -- revision check as normal wizard saves. No staff assignment is created here.
  perform public.pcs_save_interclub_draft(p_actor_id,p_actor_email,p_club_id,p_season_id,p_revision,next_draft);
  select * into saved from public.pcs_interclub_drafts where id=p_season_id;
  insert into public.pcs_platform_audit(actor_id,club_id,action,details)
   values(p_actor_id,p_slug,'invite_interclub_club',jsonb_build_object('invitation_id',p_id,'organizer_club_id',p_club_id,'season_id',p_season_id));
 end if;
 select * into c from public.clubs where id=invitation.club_id;
 return jsonb_build_object('invitation',to_jsonb(invitation),'season',to_jsonb(saved),
  'club',jsonb_build_object('id',c.id,'name',c.name,'slug',c.slug));
end $$;

create function public.pcs_interclub_club_invitation(
 p_action text,p_id uuid,p_actor_id uuid default null,p_actor_email text default null,
 p_club_id text default null,p_season_id uuid default null,p_revision integer default null,p_email text default null
) returns jsonb language plpgsql security invoker set search_path=public as $$
declare
 invitation public.pcs_club_join_invitations;
 target public.admin_role_assignments;
 old_state jsonb;
begin
 if p_action is null or p_action not in ('accept','cancel','renew','email_claim') then
  raise exception 'Invalid invitation action' using errcode='22023'; end if;
 select * into invitation from public.pcs_club_join_invitations where id=p_id;
 if not found or (p_club_id is not null and p_club_id<>invitation.organizer_club_id)
  or (p_season_id is not null and p_season_id<>invitation.season_id) then
  raise exception 'Invitation unavailable' using errcode='P0002'; end if;
 -- Serialize with changes to the inviting club's authority and target staff.
 perform pg_advisory_xact_lock(hashtextextended('pcs-staff:'||invitation.organizer_club_id,0));
 perform pg_advisory_xact_lock(hashtextextended('pcs-staff:'||invitation.club_id,0));
 select * into invitation from public.pcs_club_join_invitations where id=p_id for update;
 old_state := to_jsonb(invitation);
 if p_action='accept' then
  if lower(trim(p_actor_email)) is distinct from invitation.email
   or not public.pcs_staff_verified_email(p_actor_id,invitation.email) then
   raise exception 'Use the invited verified email' using errcode='42501'; end if;
  select * into target from public.admin_role_assignments where club_id=invitation.club_id and email=invitation.email for update;
  if invitation.status='accepted' then
   if invitation.accepted_by=p_actor_id and to_jsonb(target) is not distinct from invitation.target_after
    and target.revoked_at is null and (target.expires_at is null or target.expires_at>now()) then return to_jsonb(invitation); end if;
   raise exception 'Club access changed' using errcode='40001'; end if;
 elsif p_action='email_claim' and lower(trim(p_email)) is distinct from invitation.email then return null;
 end if;
 if p_action in ('cancel','renew') then
  perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,invitation.organizer_club_id);
  if p_revision is distinct from invitation.revision or invitation.status='accepted' then
   raise exception 'Invitation changed' using errcode='40001'; end if;
 else
  perform public.pcs_require_interclub_admin(invitation.invited_by,invitation.invited_by_email,invitation.organizer_club_id);
  if invitation.status<>'pending' or invitation.expires_at<=now() then
   raise exception 'Invitation no longer available' using errcode='40001'; end if;
 end if;
 -- A platform administrator may have onboarded this club in the meantime.
 -- Never replace any staff history, even a revoked assignment.
 if p_action<>'cancel' and exists(select 1 from public.admin_role_assignments where club_id=invitation.club_id) then
  raise exception 'Club already has staff; use its existing account' using errcode='40001'; end if;
 if p_action='email_claim' then
  if invitation.sign_in_count>=5 or invitation.last_sign_in_at>now()-interval '1 minute' then return null; end if;
  update public.pcs_club_join_invitations set last_sign_in_at=now(),sign_in_count=sign_in_count+1 where id=p_id returning * into invitation;
  return to_jsonb(invitation);
 elsif p_action='accept' then
  insert into public.admin_role_assignments(club_id,email,user_id,role,scopes)
   values(invitation.club_id,invitation.email,p_actor_id,'administrator','[]') returning * into target;
  update public.clubs set onboarding_status='in_progress',updated_at=now() where id=invitation.club_id and onboarding_status='draft';
  update public.pcs_club_join_invitations set status='accepted',accepted_by=p_actor_id,accepted_at=now(),
   target_after=to_jsonb(target),revision=revision+1 where id=p_id returning * into invitation;
  insert into public.club_staff_audit(club_id,actor_email,target_email,before_state,after_state)
   values(invitation.club_id,invitation.email,invitation.email,null,to_jsonb(target));
 elsif p_action='cancel' then
  update public.pcs_club_join_invitations set status='cancelled',revision=revision+1 where id=p_id returning * into invitation;
 else
  if p_email is null or length(p_email)>254 or p_email !~ '^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$' then
   raise exception 'Invalid invitation email' using errcode='22023'; end if;
  update public.pcs_club_join_invitations set status='pending',email=lower(trim(p_email)),expires_at=now()+interval '7 days',
   invited_by=p_actor_id,invited_by_email=lower(trim(p_actor_email)),last_sign_in_at=null,sign_in_count=0,revision=revision+1
   where id=p_id returning * into invitation;
 end if;
 insert into public.pcs_platform_audit(actor_id,club_id,action,details)
  values(p_actor_id,invitation.club_id,'club_invitation_'||p_action,
   jsonb_build_object('before',old_state,'after',to_jsonb(invitation)));
 return to_jsonb(invitation);
end $$;
revoke all on function public.pcs_create_interclub_club_invitation(uuid,text,text,uuid,uuid,integer,jsonb,text,text,text),
 public.pcs_interclub_club_invitation(text,uuid,uuid,text,text,uuid,integer,text) from public,anon,authenticated;
grant execute on function public.pcs_create_interclub_club_invitation(uuid,text,text,uuid,uuid,integer,jsonb,text,text,text),
 public.pcs_interclub_club_invitation(text,uuid,uuid,text,text,uuid,integer,text) to service_role;
commit;
