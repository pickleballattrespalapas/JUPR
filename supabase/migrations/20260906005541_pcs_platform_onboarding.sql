begin;
create table public.pcs_platform_admins (
 user_id uuid primary key references auth.users(id),
 revoked_at timestamptz,
 created_at timestamptz not null default now()
);
alter table public.pcs_platform_admins enable row level security;
revoke all on public.pcs_platform_admins from public,anon,authenticated;
grant all on public.pcs_platform_admins to service_role;
-- Bootstrap only the existing Tres platform administrator, binding to verified
-- Auth identity. Future club role assignments cannot grant platform access.
insert into public.pcs_platform_admins(user_id)
select distinct u.id from public.admin_role_assignments a join auth.users u
 on lower(u.email)=lower(a.email) and (a.user_id is null or a.user_id=u.id)
where a.club_id='tres_palapas' and a.role='super_admin' and a.revoked_at is null
 and (a.expires_at is null or a.expires_at>now());
create table public.pcs_platform_audit (
 id bigint generated always as identity primary key,
 actor_id uuid not null, club_id text not null, action text not null,
 details jsonb not null default '{}', created_at timestamptz not null default now()
);
alter table public.pcs_platform_audit enable row level security;
revoke all on public.pcs_platform_audit from public,anon,authenticated;
grant all on public.pcs_platform_audit to service_role;
grant usage,select on sequence public.pcs_platform_audit_id_seq to service_role;
create function public.pcs_onboard_club(p_actor_id uuid,p_slug text,p_name text,p_admin_email text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare c public.clubs;
begin
 if not exists(select 1 from public.pcs_platform_admins where user_id=p_actor_id and revoked_at is null) then
  raise exception 'Platform access required' using errcode='42501';
 end if;
 if p_slug !~ '^[a-z0-9]+(-[a-z0-9]+)*$' or length(p_slug) not between 3 and 60
 or length(trim(p_name)) not between 1 and 120 or p_admin_email !~ '^[^\s@]+@[^\s@]+\.[^\s@]+$' then
  raise exception 'Invalid club details' using errcode='22023';
 end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-club:'||p_slug,0));
 if exists(select 1 from public.clubs where slug=p_slug or id=p_slug) then
  raise exception 'Club address already exists' using errcode='23505';
 end if;
 insert into public.clubs(id,slug,name,is_active,status,plan_status,onboarding_status)
 values(p_slug,p_slug,trim(p_name),false,'draft','free','draft') returning * into c;
 insert into public.admin_role_assignments(club_id,email,role,scopes)
 values(c.id,lower(trim(p_admin_email)),'administrator','[]');
 insert into public.pcs_platform_audit(actor_id,club_id,action,details)
 values(p_actor_id,c.id,'create_club',jsonb_build_object('administrator_email',lower(trim(p_admin_email))));
 return to_jsonb(c);
end $$;
create function public.pcs_review_onboarding(p_actor_id uuid,p_club_id text,p_status text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare c public.clubs;
begin
 if not exists(select 1 from public.pcs_platform_admins where user_id=p_actor_id and revoked_at is null) then
  raise exception 'Platform access required' using errcode='42501';
 end if;
 if p_status not in ('draft','in_progress','ready_for_review') then raise exception 'Invalid status' using errcode='22023'; end if;
 select * into c from public.clubs where id=p_club_id for update;
 if not found then raise exception 'Club not found' using errcode='P0002'; end if;
 if p_status='ready_for_review' and not exists(select 1 from public.admin_role_assignments where club_id=p_club_id and role in ('administrator','club_owner','super_admin') and revoked_at is null and (expires_at is null or expires_at>now())) then
  raise exception 'Club administrator required' using errcode='22023';
 end if;
 insert into public.pcs_platform_audit(actor_id,club_id,action,details)
 values(p_actor_id,p_club_id,'onboarding_status',jsonb_build_object('before',c.onboarding_status,'after',p_status));
 update public.clubs set onboarding_status=p_status,updated_at=now() where id=p_club_id returning * into c;
 return to_jsonb(c);
end $$;
revoke all on function public.pcs_onboard_club(uuid,text,text,text),public.pcs_review_onboarding(uuid,text,text) from public,anon,authenticated;
grant execute on function public.pcs_onboard_club(uuid,text,text,text),public.pcs_review_onboarding(uuid,text,text) to service_role;
commit;
