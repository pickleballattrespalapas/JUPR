-- The original Send request action now delivers to the target immediately.
-- Keep verified_at truthful: it is set only for an actual email/edit capability.
-- Existing verification links remain valid; no stored request is sent by migration.
create or replace function public.create_tournament_partner_invitation(p_invitation jsonb)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  v_row public.tournament_partner_invitations%rowtype;
  v_target public.tournament_registration_selections%rowtype;
  v_email text := lower(trim(p_invitation->>'requester_email'));
begin
  -- A single email scope serializes both the throttle and idempotency check.
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('partner-email:' || v_email, 0));
  select * into v_row from public.tournament_partner_invitations where request_key = p_invitation->>'request_key';
  if found then
    if v_row.requester_email <> v_email or v_row.target_selection_id <> p_invitation->>'target_selection_id'
       or v_row.requester_name <> p_invitation->>'requester_name' or v_row.message <> p_invitation->>'message'
       or v_row.club_id <> p_invitation->>'club_id' or v_row.tournament_id::text <> p_invitation->>'tournament_id' then
      raise exception 'JUPR_INVITATION: Start a new request after changing its details.';
    end if;
    if v_row.status = 'UNVERIFIED' and coalesce((p_invitation->>'send_directly')::boolean, false) then
      if v_row.expires_at <= now() or not private.partner_invitation_target_available(v_row.target_selection_id) then
        raise exception 'JUPR_INVITATION: This request is no longer available. Start a new request from the Partner Board.';
      end if;
      update public.tournament_partner_invitations set status = 'PENDING'
        where id = v_row.id and status = 'UNVERIFIED' returning * into v_row;
      if not found then
        select * into v_row from public.tournament_partner_invitations where request_key = p_invitation->>'request_key';
      end if;
    end if;
    return to_jsonb(v_row) || jsonb_build_object('idempotent', true);
  end if;
  if (select count(*) from public.tournament_partner_invitations
      where requester_email = v_email and created_at > now() - interval '1 hour') >= 5 then
    raise exception 'JUPR_INVITATION_RATE: Please wait an hour before sending more partner requests.';
  end if;
  perform private.lock_tournament_registration_selection_scope(array[p_invitation->>'target_selection_id']);
  select * into v_target from public.tournament_registration_selections where id = p_invitation->>'target_selection_id';
  if v_target.id is null or v_target.tournament_id <> p_invitation->>'tournament_id'
     or not private.partner_invitation_target_available(v_target.id) then
    raise exception 'JUPR_INVITATION: This player is no longer looking for a partner in this division.';
  end if;
  update public.tournament_partner_invitations set status = 'EXPIRED'
    where target_selection_id = v_target.id and status = 'RESERVED' and expires_at <= now();
  if exists (select 1 from public.tournament_partner_invitations where target_selection_id = v_target.id and status = 'RESERVED') then
    raise exception 'JUPR_INVITATION: This player has already accepted a partner for this division.';
  end if;
  insert into public.tournament_partner_invitations
    (id, club_id, tournament_id, event_option_id, target_selection_id, requester_name, requester_email, message, request_key, status, verified_at)
  values (p_invitation->>'id', p_invitation->>'club_id', (p_invitation->>'tournament_id')::uuid,
    v_target.event_option_id, v_target.id, p_invitation->>'requester_name', v_email,
    p_invitation->>'message', p_invitation->>'request_key',
    case when coalesce((p_invitation->>'verified')::boolean, false)
      or coalesce((p_invitation->>'send_directly')::boolean, false) then 'PENDING' else 'UNVERIFIED' end,
    case when coalesce((p_invitation->>'verified')::boolean, false) then now() else null end)
  returning * into v_row;
  return to_jsonb(v_row) || jsonb_build_object('idempotent', false);
end;
$$;

create or replace function public.transition_tournament_partner_invitation(
  p_invitation_id text, p_action text, p_requester_selection_id text default null,
  p_versions jsonb default '{}'::jsonb
)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  v_row public.tournament_partner_invitations%rowtype;
  v_target public.tournament_registration_selections%rowtype;
  v_requester public.tournament_registration_selections%rowtype;
  v_reg public.tournament_registrations%rowtype;
  v_created jsonb;
  v_result jsonb;
  v_keys text[];
begin
  select * into v_row from public.tournament_partner_invitations where id = p_invitation_id;
  if not found then raise exception 'JUPR_INVITATION: This partner request was not found.'; end if;
  -- Lock registrations/selections before invitations, in the same order as the
  -- canonical edit/pairing engine. Lock the tournament scope because acceptance
  -- cancels the entire competing-request graph (including older requests).
  select array_agg(id order by id) into v_keys from public.tournament_registration_selections
    where tournament_id = v_row.tournament_id::text;
  perform private.lock_tournament_registration_selection_scope(coalesce(v_keys, array[]::text[]));
  select * into v_row from public.tournament_partner_invitations where id = p_invitation_id for update;
  if p_action not in ('verify','accept','decline','cancel','complete') then
    raise exception 'JUPR_INVITATION: Invalid partner request action.';
  end if;
  if v_row.status = 'COMPLETED' and p_action in ('accept','complete')
     or v_row.status = 'RESERVED' and p_action = 'accept'
     or v_row.status = 'DECLINED' and p_action = 'decline'
     or v_row.status = 'CANCELLED' and p_action = 'cancel'
     or v_row.status in ('PENDING','RESERVED','COMPLETED') and p_action = 'verify' then
    return to_jsonb(v_row) || jsonb_build_object('idempotent', true);
  end if;
  if v_row.status in ('COMPLETED','DECLINED','CANCELLED','EXPIRED') then
    return to_jsonb(v_row) || jsonb_build_object('stale', true);
  end if;
  if v_row.expires_at <= now() then
    update public.tournament_partner_invitations set status = 'EXPIRED' where id = v_row.id returning * into v_row;
    return to_jsonb(v_row) || jsonb_build_object('stale', true);
  end if;
  if p_action in ('cancel','decline') then
    if p_action = 'decline' and v_row.status <> 'PENDING' then
      raise exception 'JUPR_INVITATION: This request cannot be declined now.';
    end if;
    update public.tournament_partner_invitations set status = case when p_action = 'cancel' then 'CANCELLED' else 'DECLINED' end
      where id = v_row.id returning * into v_row;
    return to_jsonb(v_row);
  end if;
  if not private.partner_invitation_target_available(v_row.target_selection_id) then
    update public.tournament_partner_invitations set status = 'CANCELLED' where id = v_row.id returning * into v_row;
    return to_jsonb(v_row) || jsonb_build_object('stale', true);
  end if;
  if p_action = 'verify' then
    update public.tournament_partner_invitations set status = 'PENDING', verified_at = now()
      where id = v_row.id returning * into v_row;
    return to_jsonb(v_row);
  end if;
  -- Sending a message does not verify ownership of the typed email address.
  -- The target's private email capability authorizes acceptance of this request.
  if v_row.status not in ('PENDING','RESERVED') then
    raise exception 'JUPR_INVITATION: The sender must confirm their email first.';
  end if;
  if p_action = 'complete' and v_row.status <> 'RESERVED' then
    -- The guest may register before the target has accepted. Store no pairing:
    -- the later accept will resolve that registration from the submitted email/name.
    return to_jsonb(v_row);
  end if;
  select * into v_target from public.tournament_registration_selections where id = v_row.target_selection_id;
  update public.tournament_partner_invitations set status = 'EXPIRED'
    where target_selection_id = v_target.id and status = 'RESERVED' and expires_at <= now();
  if exists (select 1 from public.tournament_partner_invitations
      where id <> v_row.id and status = 'RESERVED'
        and (target_selection_id = v_target.id or target_selection_id = p_requester_selection_id
          or requester_selection_id in (v_target.id, p_requester_selection_id))) then
    raise exception 'JUPR_INVITATION: One player has already accepted another partner for this division.';
  end if;
  if nullif(p_requester_selection_id, '') is not null then
    select * into v_requester from public.tournament_registration_selections where id = p_requester_selection_id;
    select * into v_reg from public.tournament_registrations where id = v_requester.registration_id;
    if v_requester.id is null or v_reg.id is null or v_requester.event_option_id <> v_row.event_option_id
       or v_requester.tournament_id <> v_row.tournament_id::text or lower(trim(v_reg.email)) <> v_row.requester_email
       or v_reg.id = v_target.registration_id or upper(v_reg.status) in ('CANCELLED','WITHDRAWN')
       or v_requester.partner_mode <> 'NEEDS_PARTNER' then
      raise exception 'JUPR_INVITATION: The sender must have an available registration for this division.';
    end if;
    if v_row.verified_at is null and
       lower(regexp_replace(trim(v_reg.display_name), '[[:space:]]+', ' ', 'g')) is distinct from
       lower(regexp_replace(trim(v_row.requester_name), '[[:space:]]+', ' ', 'g')) then
      raise exception 'JUPR_INVITATION: The name on this request does not match the sender registration.';
    end if;
    -- The API checks canonical gender/age/rating eligibility. Reject any edited
    -- registration/selection/event since that check instead of pairing stale data.
    if (p_versions->>'requester_registration')::timestamptz is distinct from v_reg.updated_at
       or (p_versions->>'requester_selection')::timestamptz is distinct from v_requester.updated_at
       or (p_versions->>'target_selection')::timestamptz is distinct from v_target.updated_at
       or (p_versions->>'event')::timestamptz is distinct from
          (select updated_at from public.tournament_event_options where id = v_row.event_option_id)
       or (p_versions->>'target_registration')::timestamptz is distinct from
          (select updated_at from public.tournament_registrations where id = v_target.registration_id) then
      raise exception 'JUPR_INVITATION: Registration changed. Refresh this page and try again.';
    end if;
    update public.tournament_partner_invitations set status = 'COMPLETED', accepted_at = coalesce(accepted_at, now()),
      completed_at = now(), requester_selection_id = v_requester.id where id = v_row.id;
    v_created := public.create_tournament_partner_request('preq_' || replace(gen_random_uuid()::text, '-', ''),
      v_row.tournament_id::text, v_row.event_option_id, v_requester.id, v_target.id,
      (select display_name from public.tournament_registrations where id = v_target.registration_id), 'PUBLIC_PARTNER_BOARD');
    v_result := public.transition_tournament_partner_request(v_created->>'id', v_target.id, 'accept');
    if v_result->>'status' <> 'ACCEPTED' then
      raise exception 'JUPR_INVITATION: This partner request is no longer available.';
    end if;
    update public.tournament_partner_invitations set partner_request_id = v_created->>'id'
      where id = v_row.id returning * into v_row;
  else
    if p_action = 'complete' then raise exception 'JUPR_INVITATION: Complete registration for this division first.'; end if;
    update public.tournament_partner_invitations set status = 'RESERVED', accepted_at = now()
      where id = v_row.id returning * into v_row;
  end if;
  update public.tournament_partner_invitations set status = 'CANCELLED'
    where id <> v_row.id and event_option_id = v_row.event_option_id and status in ('UNVERIFIED','PENDING')
      and (target_selection_id in (v_target.id, v_requester.id)
        or requester_selection_id in (v_target.id, v_requester.id)
        or requester_email = v_row.requester_email
        or requester_email = (select lower(trim(email)) from public.tournament_registrations where id = v_target.registration_id));
  update public.tournament_registration_partner_requests set status = 'CANCELLED', responded_at = now(), updated_at = now()
    where event_option_id = v_row.event_option_id and status = 'PENDING'
      and (target_selection_id in (v_target.id, v_requester.id) or requester_selection_id in (v_target.id, v_requester.id));
  return to_jsonb(v_row);
end;
$$;

revoke all on function public.create_tournament_partner_invitation(jsonb),
  public.transition_tournament_partner_invitation(text,text,text,jsonb) from public, anon, authenticated;
grant execute on function public.create_tournament_partner_invitation(jsonb),
  public.transition_tournament_partner_invitation(text,text,text,jsonb) to service_role;
notify pgrst, 'reload schema';
