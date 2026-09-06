begin;
alter table public.admin_role_assignments drop constraint admin_role_assignments_role_check;
alter table public.admin_role_assignments add constraint admin_role_assignments_role_check
 check (role in ('super_admin','club_owner','organizer','scorekeeper','read_only','administrator','operator'));
alter table public.admin_role_assignments add column scopes jsonb not null default '[]',
 add column expires_at timestamptz, add column revoked_at timestamptz;
alter table public.admin_role_assignments add constraint staff_scopes_array check (jsonb_typeof(scopes) = 'array');
alter table public.admin_role_assignments add constraint staff_admin_no_expiry check (role <> 'administrator' or expires_at is null);

create table public.club_staff_audit (
 id bigint generated always as identity primary key,
 club_id text not null, actor_email text not null, target_email text not null,
 before_state jsonb, after_state jsonb, created_at timestamptz not null default now()
);
alter table public.club_staff_audit enable row level security;
revoke all on public.club_staff_audit from public, anon, authenticated;
grant all on public.club_staff_audit to service_role;
grant usage, select on sequence public.club_staff_audit_id_seq to service_role;

-- Service-only RPC: serialize membership edits, recheck actor authority under the
-- lock, and record the audit in the same transaction as the assignment.
create or replace function public.pcs_save_staff(
 p_club_id text, p_actor_email text, p_actor_id uuid, p_email text,
 p_role text, p_scopes jsonb, p_expires_at timestamptz, p_revoke boolean default false
) returns jsonb language plpgsql security invoker set search_path = public as $$
declare actor public.admin_role_assignments; old_row public.admin_role_assignments;
 new_row public.admin_role_assignments;
begin
 perform pg_advisory_xact_lock(hashtextextended('pcs-staff:' || p_club_id, 0));
 select * into actor from public.admin_role_assignments where club_id=p_club_id
 and email=lower(trim(p_actor_email)) and revoked_at is null
 and (user_id is null or user_id=p_actor_id) and (expires_at is null or expires_at>now());
 if actor.role is null or actor.role not in ('super_admin','club_owner','administrator') then
  raise exception 'Administrator access required' using errcode='42501';
 end if;
 if p_role not in ('administrator','operator') or jsonb_typeof(p_scopes)<>'array' then
  raise exception 'Invalid staff assignment';
 end if;
 if p_role='operator' and not p_revoke and jsonb_array_length(p_scopes)=0 then
  raise exception 'Operator scope required';
 end if;
 select * into old_row from public.admin_role_assignments where club_id=p_club_id and email=lower(trim(p_email));
 if old_row.role='super_admin' then raise exception 'Platform access cannot be changed here' using errcode='42501'; end if;
 if old_row.role in ('administrator','club_owner') and old_row.revoked_at is null
 and (p_revoke or p_role='operator') and not exists (
  select 1 from public.admin_role_assignments where club_id=p_club_id and email<>lower(trim(p_email))
  and role in ('administrator','club_owner','super_admin') and revoked_at is null
  and (expires_at is null or expires_at>now())
 ) then raise exception 'Keep at least one administrator'; end if;
 insert into public.admin_role_assignments(club_id,email,role,scopes,expires_at,revoked_at)
 values (p_club_id,lower(trim(p_email)),p_role,p_scopes,
 case when p_role='administrator' then null else p_expires_at end,
 case when p_revoke then now() else null end)
 on conflict (club_id,email) do update set role=excluded.role,scopes=excluded.scopes,
 expires_at=excluded.expires_at,revoked_at=excluded.revoked_at
 returning * into new_row;
 insert into public.club_staff_audit(club_id,actor_email,target_email,before_state,after_state)
 values(p_club_id,lower(trim(p_actor_email)),lower(trim(p_email)),to_jsonb(old_row),to_jsonb(new_row));
 return to_jsonb(new_row);
end $$;
revoke all on function public.pcs_save_staff(text,text,uuid,text,text,jsonb,timestamptz,boolean) from public,anon,authenticated;
grant execute on function public.pcs_save_staff(text,text,uuid,text,text,jsonb,timestamptz,boolean) to service_role;
commit;
