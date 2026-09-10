-- Public email invitations are separate from registrations until the sender
-- completes registration. Only the API's service role can read or mutate them.
create table public.tournament_partner_invitations (
  id text primary key,
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  target_selection_id text not null references public.tournament_registration_selections(id) on delete cascade,
  requester_name text not null check (length(requester_name) between 1 and 160),
  requester_email text not null check (length(requester_email) between 3 and 320),
  message text not null check (length(message) between 1 and 2000),
  request_key text not null unique,
  status text not null default 'UNVERIFIED' check (status in
    ('UNVERIFIED','PENDING','RESERVED','COMPLETED','DECLINED','CANCELLED','EXPIRED')),
  requester_selection_id text references public.tournament_registration_selections(id) on delete set null,
  partner_request_id text references public.tournament_registration_partner_requests(id) on delete set null,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null default now() + interval '14 days',
  verified_at timestamptz,
  accepted_at timestamptz,
  completed_at timestamptz
);
alter table public.tournament_partner_invitations enable row level security;
revoke all on public.tournament_partner_invitations from public, anon, authenticated;
grant select, insert, update, delete on public.tournament_partner_invitations to service_role;
create index partner_invitation_tournament on public.tournament_partner_invitations(tournament_id);
create index partner_invitation_event on public.tournament_partner_invitations(event_option_id);
create index partner_invitation_target on public.tournament_partner_invitations(target_selection_id, status);
create index partner_invitation_requester on public.tournament_partner_invitations(requester_selection_id);
create index partner_invitation_request on public.tournament_partner_invitations(partner_request_id);
create index partner_invitation_email_rate on public.tournament_partner_invitations(requester_email, created_at);
create unique index partner_invitation_one_reservation on public.tournament_partner_invitations(target_selection_id)
  where status = 'RESERVED';

create table public.tournament_partner_invitation_deliveries (
  invitation_id text not null references public.tournament_partner_invitations(id) on delete cascade,
  kind text not null,
  status text not null default 'sending',
  attempt_id text not null,
  updated_at timestamptz not null default now(),
  primary key (invitation_id, kind)
);
alter table public.tournament_partner_invitation_deliveries enable row level security;
revoke all on public.tournament_partner_invitation_deliveries from public, anon, authenticated;
grant select, insert, update, delete on public.tournament_partner_invitation_deliveries to service_role;

create function private.partner_invitation_target_available(p_selection_id text)
returns boolean language sql stable security invoker set search_path = '' as $$
  select exists (
    select 1 from public.tournament_registration_selections s
    join public.tournament_registrations r on r.id = s.registration_id
    join public.tournament_event_options e on e.id = s.event_option_id
    join public.tournament_registration_settings cfg on cfg.tournament_id::text = s.tournament_id
    where s.id = p_selection_id and s.partner_mode = 'NEEDS_PARTNER'
      and s.show_on_partner_board and r.wants_partner_board_contact
      and upper(r.status) not in ('CANCELLED','WITHDRAWN')
      and coalesce(e.enabled, true) and e.partner_board_enabled and cfg.partner_board_enabled
      and lower(e.status) in ('open','tentative','confirmed','published','active')
      and not exists (select 1 from public.tournament_registration_team_members m
        where m.selection_id = s.id and m.status = 'ACTIVE')
  );
$$;

create function public.create_tournament_partner_invitation(p_invitation jsonb)
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
    case when coalesce((p_invitation->>'verified')::boolean, false) then 'PENDING' else 'UNVERIFIED' end,
    case when coalesce((p_invitation->>'verified')::boolean, false) then now() else null end)
  returning * into v_row;
  return to_jsonb(v_row) || jsonb_build_object('idempotent', false);
end;
$$;

create function public.transition_tournament_partner_invitation(
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
  if v_row.verified_at is null or v_row.status not in ('PENDING','RESERVED') then
    raise exception 'JUPR_INVITATION: The sender must confirm their email first.';
  end if;
  if p_action = 'complete' and v_row.status <> 'RESERVED' then
    -- The guest may register before the target has accepted. Store no pairing:
    -- the later accept will resolve that registration from verified email/name.
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

-- Existing public/admin/manual pairing paths must also respect reservations.
create function private.guard_partner_invitation_reservation()
returns trigger language plpgsql security invoker set search_path = '' as $$
begin
  if new.status in ('CONFIRMED','ADMIN_CONFIRMED') then
    perform private.lock_tournament_registration_selection_scope(array[new.selection1_id, new.selection2_id]);
    if exists (select 1 from public.tournament_partner_invitations
        where status = 'RESERVED' and expires_at > now()
          and (target_selection_id in (new.selection1_id,new.selection2_id)
            or requester_selection_id in (new.selection1_id,new.selection2_id))) then
      raise exception 'JUPR_INVITATION: This player has a reserved partnership. Cancel it from the partnership email before choosing another partner.';
    end if;
    update public.tournament_partner_invitations set status = 'CANCELLED'
      where status in ('UNVERIFIED','PENDING','RESERVED')
        and (target_selection_id in (new.selection1_id,new.selection2_id)
          or requester_selection_id in (new.selection1_id,new.selection2_id));
  end if;
  return new;
end;
$$;
create trigger guard_partner_invitation_reservation before insert or update on public.tournament_registration_team_links
  for each row execute function private.guard_partner_invitation_reservation();

create function public.claim_partner_invitation_delivery(p_invitation_id text, p_kind text, p_attempt_id text)
returns boolean language plpgsql security invoker set search_path = '' as $$
begin
  insert into public.tournament_partner_invitation_deliveries(invitation_id, kind, attempt_id)
    values (p_invitation_id, p_kind, p_attempt_id)
    on conflict (invitation_id, kind) do update set status = 'sending', attempt_id = p_attempt_id, updated_at = now()
      where tournament_partner_invitation_deliveries.status = 'failed'
        or (tournament_partner_invitation_deliveries.status = 'sending'
          and tournament_partner_invitation_deliveries.updated_at < now() - interval '5 minutes');
  return found;
end;
$$;

revoke all on function private.partner_invitation_target_available(text), private.guard_partner_invitation_reservation(),
  public.create_tournament_partner_invitation(jsonb), public.transition_tournament_partner_invitation(text,text,text,jsonb),
  public.claim_partner_invitation_delivery(text,text,text) from public, anon, authenticated;
grant execute on function private.partner_invitation_target_available(text), private.guard_partner_invitation_reservation(),
  public.create_tournament_partner_invitation(jsonb), public.transition_tournament_partner_invitation(text,text,text,jsonb),
  public.claim_partner_invitation_delivery(text,text,text) to service_role;
notify pgrst, 'reload schema';
