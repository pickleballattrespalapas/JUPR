-- A named partner can be the same person who already accepted an invitation.
-- Complete that reservation inside the canonical team transaction instead of
-- rejecting it or leaving a second, pending roster entry behind.
create or replace function private.guard_partner_invitation_reservation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_invitation public.tournament_partner_invitations%rowtype;
  v_requester public.tournament_registrations%rowtype;
  v_requester_selection text;
  v_emails text[];
begin
  if new.status not in ('CONFIRMED', 'ADMIN_CONFIRMED') then
    return new;
  end if;
  perform private.lock_tournament_registration_selection_scope(
    array[new.selection1_id, new.selection2_id]
  );
  select array_agg(lower(btrim(r.email))) into v_emails
    from public.tournament_registrations r
   where r.id in (new.registration1_id, new.registration2_id)
     and r.tournament_id = new.tournament_id::text;

  -- Include email-only requesters: their selection does not exist until they
  -- register, but their accepted reservation must still protect the division.
  for v_invitation in
    select i.* from public.tournament_partner_invitations i
     where i.tournament_id = new.tournament_id
       and i.event_option_id = new.event_option_id
       and i.status = 'RESERVED' and i.expires_at > now()
       and (i.target_selection_id in (new.selection1_id, new.selection2_id)
         or i.requester_selection_id in (new.selection1_id, new.selection2_id)
         or lower(btrim(i.requester_email)) = any(v_emails))
     order by i.id for update
  loop
    v_requester_selection := case v_invitation.target_selection_id
      when new.selection1_id then new.selection2_id
      when new.selection2_id then new.selection1_id
      else null end;
    select r.* into v_requester
      from public.tournament_registration_selections s
      join public.tournament_registrations r on r.id = s.registration_id
     where s.id = v_requester_selection
       and s.tournament_id = new.tournament_id::text
       and s.event_option_id = new.event_option_id
       and r.tournament_id = new.tournament_id::text
       and r.id in (new.registration1_id, new.registration2_id)
       and upper(r.status) not in ('CANCELLED', 'WITHDRAWN');
    if v_requester.id is null
       or lower(btrim(v_requester.email)) <> lower(btrim(v_invitation.requester_email))
       or lower(regexp_replace(btrim(v_requester.display_name), '\s+', ' ', 'g'))
          <> lower(regexp_replace(btrim(v_invitation.requester_name), '\s+', ' ', 'g'))
       or (v_invitation.requester_selection_id is not null
         and v_invitation.requester_selection_id <> v_requester_selection)
       or not exists (select 1 from public.tournaments t
         where t.id = new.tournament_id and t.club_id = v_invitation.club_id) then
      raise exception 'JUPR_INVITATION: An accepted partnership already reserves this division. Open your registration or partner-request email to review it before choosing another partner.';
    end if;

    -- Legacy accepted invitations may have no verified_at. The same tournament,
    -- division, target selection, requester email AND name must all match.
    -- A later failure rolls back this completion with the registration/team.
    update public.tournament_partner_invitations
       set status = 'COMPLETED', completed_at = now(),
           requester_selection_id = v_requester_selection,
           partner_request_id = new.accepted_request_id
     where id = v_invitation.id;
  end loop;

  update public.tournament_partner_invitations i set status = 'CANCELLED'
   where i.tournament_id = new.tournament_id
     and i.event_option_id = new.event_option_id
     and i.status in ('UNVERIFIED', 'PENDING', 'RESERVED')
     and (i.target_selection_id in (new.selection1_id, new.selection2_id)
       or i.requester_selection_id in (new.selection1_id, new.selection2_id)
       or lower(btrim(i.requester_email)) = any(v_emails));
  return new;
end;
$function$;

revoke all on function private.guard_partner_invitation_reservation()
  from public, anon, authenticated;
grant execute on function private.guard_partner_invitation_reservation() to service_role;
