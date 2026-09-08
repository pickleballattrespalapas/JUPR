begin;

-- Called only by the API service after JWT verification. Club identity and
-- actor identity are supplied by the authorized route, never the request body.
create function public.pcs_save_club_settings(
 p_actor_id uuid, p_actor_email text, p_club_id text,
 p_expected_updated_at timestamptz, p_name text, p_tagline text,
 p_support_email text, p_submit boolean default false
) returns jsonb language plpgsql security invoker set search_path = public as $$
declare
 c public.clubs;
 old_details jsonb;
 new_details jsonb;
 next_status text;
 contact text := lower(trim(coalesce(p_support_email, '')));
 description text := trim(coalesce(p_tagline, ''));
begin
 -- Share the membership lock with pcs_save_staff, so revocation cannot race
 -- this authorization check and the save that follows it.
 perform pg_advisory_xact_lock(hashtextextended('pcs-staff:' || p_club_id, 0));
 perform 1 from public.admin_role_assignments
 where club_id = p_club_id and email = lower(trim(p_actor_email))
 and (user_id is null or user_id = p_actor_id)
 and role in ('administrator', 'club_owner', 'super_admin')
 and revoked_at is null and (expires_at is null or expires_at > now())
 for share;
 if not found or p_actor_id is null then
  raise exception 'Club administrator required' using errcode = '42501';
 end if;
 if p_name is null or length(trim(p_name)) not between 1 and 120
 or length(description) > 240 or length(contact) > 254
 or (contact <> '' and contact !~ '^[^\s@]+@[^\s@]+\.[^\s@]+$')
 or p_submit is null then
  raise exception 'Invalid club details' using errcode = '22023';
 end if;
 select * into c from public.clubs where id = p_club_id for update;
 if not found then raise exception 'Club not found' using errcode = 'P0002'; end if;
 if p_expected_updated_at is null or c.updated_at is distinct from p_expected_updated_at then
  raise exception 'Stale club settings' using errcode = '40001';
 end if;
 next_status := c.onboarding_status;
 if p_submit then
  if c.is_active or c.onboarding_status not in ('draft', 'in_progress', 'ready_for_review') or contact = '' then
   raise exception 'Club cannot be submitted for setup review' using errcode = '22023';
  end if;
  next_status := 'ready_for_review';
 elsif not c.is_active and c.onboarding_status in ('draft', 'ready_for_review') then
  if c.onboarding_status = 'draft' or
    (c.name, coalesce(c.tagline,''), coalesce(c.support_email,'')) is distinct from
    (trim(p_name), description, contact) then
   next_status := 'in_progress';
  end if;
 end if;
 old_details := jsonb_build_object('name', c.name, 'tagline', c.tagline,
   'support_email', c.support_email, 'onboarding_status', c.onboarding_status, 'updated_at', c.updated_at);
 update public.clubs set name = trim(p_name), tagline = nullif(description,''),
   support_email = nullif(contact,''), onboarding_status = next_status,
   updated_at = clock_timestamp()
 where id = p_club_id returning * into c;
 new_details := jsonb_build_object('name', c.name, 'tagline', c.tagline,
   'support_email', c.support_email, 'onboarding_status', c.onboarding_status, 'updated_at', c.updated_at);
 insert into public.pcs_platform_audit(actor_id, club_id, action, details)
 values (p_actor_id, p_club_id, case when p_submit then 'club_setup_submitted' else 'club_settings_saved' end,
   jsonb_build_object('before', old_details, 'after', new_details));
 return to_jsonb(c);
end $$;

revoke all on function public.pcs_save_club_settings(uuid,text,text,timestamptz,text,text,text,boolean) from public, anon, authenticated;
grant execute on function public.pcs_save_club_settings(uuid,text,text,timestamptz,text,text,text,boolean) to service_role;
commit;
