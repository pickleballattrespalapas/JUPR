-- Transactional public tournament Partner Board lifecycle.
--
-- FastAPI verifies the registration edit token and club before invoking these
-- functions with the server-side Supabase client. The functions provide the
-- database transaction/locking boundary only; they are deliberately unavailable
-- to browser Data API roles.

do $migration_preflight$
begin
  if to_regclass('public.tournament_registration_partner_requests') is null
     or to_regclass('public.tournament_registration_selections') is null
     or to_regprocedure('private.lock_tournament_registration_selection_scope(text[])') is null then
    raise exception using
      errcode = '42P01',
      message = 'Partner-link tables and the canonical selection lock helper must exist before applying the public Partner Board lifecycle.';
  end if;

  if exists (
    select 1
      from public.tournament_registration_partner_requests as request
     where request.status = 'PENDING'
       and request.target_selection_id is not null
     group by request.event_option_id, request.requester_selection_id, request.target_selection_id
    having count(*) > 1
  ) then
    raise exception using
      errcode = '23505',
      message = 'JUPR_PARTNER_DUPLICATE_PENDING: resolve exact duplicate pending partner requests before applying this migration.';
  end if;
end
$migration_preflight$;

alter table if exists public.tournament_registration_partner_requests
  drop constraint if exists tournament_registration_partner_requests_source_chk;

alter table if exists public.tournament_registration_partner_requests
  add constraint tournament_registration_partner_requests_source_chk
  check (
    source in (
      'PROFILE_SEARCH',
      'NEEDS_PARTNER_LIST',
      'PUBLIC_PARTNER_BOARD',
      'ADMIN_RECONCILIATION',
      'ADMIN_CREATED',
      'INVITE_LINK',
      'LEGACY_TEXT_MATCH'
    )
  );

create unique index if not exists uq_tournament_partner_pending_pair
  on public.tournament_registration_partner_requests (
    event_option_id,
    requester_selection_id,
    target_selection_id
  )
  where status = 'PENDING' and target_selection_id is not null;

create or replace function public.create_tournament_partner_request(
  p_request_id text,
  p_tournament_id text,
  p_event_option_id text,
  p_requester_selection_id text,
  p_target_selection_id text,
  p_target_display_name_snapshot text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_requester public.tournament_registration_selections%rowtype;
  v_target public.tournament_registration_selections%rowtype;
  v_requester_registration public.tournament_registrations%rowtype;
  v_target_registration public.tournament_registrations%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_settings public.tournament_registration_settings%rowtype;
  v_existing public.tournament_registration_partner_requests%rowtype;
  v_created public.tournament_registration_partner_requests%rowtype;
begin
  if nullif(pg_catalog.btrim(p_request_id), '') is null
     or nullif(pg_catalog.btrim(p_tournament_id), '') is null
     or nullif(pg_catalog.btrim(p_event_option_id), '') is null
     or nullif(pg_catalog.btrim(p_requester_selection_id), '') is null
     or nullif(pg_catalog.btrim(p_target_selection_id), '') is null then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner request identifiers are required.';
  end if;
  if p_requester_selection_id = p_target_selection_id then
    raise exception 'JUPR_PARTNER_TRANSACTION: a player cannot request themselves as a partner.';
  end if;
  if pg_catalog.upper(coalesce(p_source, '')) <> 'PUBLIC_PARTNER_BOARD' then
    raise exception 'JUPR_PARTNER_TRANSACTION: public requests require PUBLIC_PARTNER_BOARD source.';
  end if;

  -- Reuse the relationship guard's universal registration/selection lock order.
  -- This keeps public pairing writes compatible with selection edits and admin
  -- reconciliation while the partial unique index provides a final duplicate
  -- backstop.
  perform private.lock_tournament_registration_selection_scope(
    array[p_requester_selection_id, p_target_selection_id]
  );
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      p_event_option_id || ':' || p_requester_selection_id || ':' || p_target_selection_id,
      0
    )
  );

  select * into v_requester
    from public.tournament_registration_selections
   where id = p_requester_selection_id;
  select * into v_target
    from public.tournament_registration_selections
   where id = p_target_selection_id;
  if v_requester.id is null or v_target.id is null then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner selection is no longer available.';
  end if;
  if v_requester.tournament_id::text <> p_tournament_id
     or v_target.tournament_id::text <> p_tournament_id then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner selections must be in the same tournament.';
  end if;
  if v_requester.event_option_id <> p_event_option_id
     or v_target.event_option_id <> p_event_option_id then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner selections must be in the same division.';
  end if;

  select * into v_requester_registration
    from public.tournament_registrations
   where id = v_requester.registration_id;
  select * into v_target_registration
    from public.tournament_registrations
   where id = v_target.registration_id;
  if v_requester_registration.id is null or v_target_registration.id is null then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner registration is no longer available.';
  end if;
  if pg_catalog.upper(coalesce(v_requester_registration.status, 'CONFIRMED')) in ('CANCELLED', 'WITHDRAWN')
     or pg_catalog.upper(coalesce(v_target_registration.status, 'CONFIRMED')) in ('CANCELLED', 'WITHDRAWN') then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner registration is no longer available.';
  end if;

  select * into v_event
    from public.tournament_event_options
   where id = p_event_option_id;
  select * into v_settings
    from public.tournament_registration_settings
   where tournament_id::text = p_tournament_id
   limit 1;
  if v_event.id is null
     or not coalesce(v_event.enabled, true)
     or not coalesce(v_event.partner_board_enabled, false)
     or v_settings.id is null
     or not coalesce(v_settings.partner_board_enabled, false)
     or pg_catalog.lower(coalesce(v_event.status, 'draft')) not in ('open', 'tentative', 'confirmed', 'published', 'active') then
    raise exception 'JUPR_PARTNER_TRANSACTION: target partner-board entry is no longer available.';
  end if;
  if pg_catalog.upper(coalesce(v_target.partner_mode, 'NONE')) <> 'NEEDS_PARTNER'
     or not coalesce(v_target.show_on_partner_board, false)
     or not coalesce(v_target_registration.wants_partner_board_contact, false) then
    raise exception 'JUPR_PARTNER_TRANSACTION: target partner-board entry is no longer available.';
  end if;

  if exists (
    select 1
      from public.tournament_registration_team_members as member
     where member.event_option_id = p_event_option_id
       and member.selection_id in (p_requester_selection_id, p_target_selection_id)
       and member.status = 'ACTIVE'
  ) then
    raise exception 'JUPR_PARTNER_TRANSACTION: selection is already on a confirmed team for this division.';
  end if;

  select * into v_existing
    from public.tournament_registration_partner_requests
   where event_option_id = p_event_option_id
     and requester_selection_id = p_requester_selection_id
     and target_selection_id = p_target_selection_id
     and status = 'PENDING'
   limit 1;
  if v_existing.id is not null then
    return pg_catalog.to_jsonb(v_existing) || pg_catalog.jsonb_build_object(
      'outcome', 'existing',
      'idempotent', true
    );
  end if;

  insert into public.tournament_registration_partner_requests (
    id,
    tournament_id,
    event_option_id,
    requester_selection_id,
    requester_registration_id,
    requester_player_id,
    target_selection_id,
    target_registration_id,
    target_player_id,
    target_display_name_snapshot,
    status,
    source,
    created_by_registration_id
  ) values (
    p_request_id,
    p_tournament_id::uuid,
    p_event_option_id,
    p_requester_selection_id,
    v_requester.registration_id,
    v_requester_registration.player_id,
    p_target_selection_id,
    v_target.registration_id,
    v_target_registration.player_id,
    nullif(pg_catalog.btrim(p_target_display_name_snapshot), ''),
    'PENDING',
    p_source,
    v_requester.registration_id
  )
  returning * into v_created;

  return pg_catalog.to_jsonb(v_created) || pg_catalog.jsonb_build_object(
    'outcome', 'created',
    'idempotent', false
  );
end
$function$;

create or replace function public.transition_tournament_partner_request(
  p_request_id text,
  p_actor_selection_id text,
  p_action text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_request public.tournament_registration_partner_requests%rowtype;
  v_requester public.tournament_registration_selections%rowtype;
  v_target public.tournament_registration_selections%rowtype;
  v_requester_registration public.tournament_registrations%rowtype;
  v_target_registration public.tournament_registrations%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_settings public.tournament_registration_settings%rowtype;
  v_lock_selection_ids text[];
  v_action text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_action, '')));
  v_desired_status text;
  v_team_link_id text;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_cancelled_ids jsonb := '[]'::jsonb;
begin
  if v_action not in ('accept', 'decline', 'cancel') then
    raise exception 'JUPR_PARTNER_TRANSACTION: unsupported partner request action.';
  end if;

  -- Read first for actor authorization, then lock selections before the request.
  -- This common lock order serializes accepts of competing requests safely.
  select * into v_request
    from public.tournament_registration_partner_requests
   where id = p_request_id;
  if v_request.id is null then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner request was not found.';
  end if;
  if v_action in ('accept', 'decline')
     and v_request.target_selection_id is distinct from p_actor_selection_id then
    raise exception 'JUPR_PARTNER_TRANSACTION: only the requested partner can respond to this request.';
  end if;
  if v_action = 'cancel'
     and v_request.requester_selection_id is distinct from p_actor_selection_id then
    raise exception 'JUPR_PARTNER_TRANSACTION: only the requester can cancel this request.';
  end if;

  -- For acceptance, lock the complete competing-request graph in the existing
  -- universal scope order. That makes cancellation deterministic even when two
  -- independent pairs have cross-linked pending requests.
  select pg_catalog.array_agg(distinct candidate.selection_id order by candidate.selection_id)
    into v_lock_selection_ids
    from (
      select v_request.requester_selection_id as selection_id
      union all
      select v_request.target_selection_id
      union all
      select competing.requester_selection_id
        from public.tournament_registration_partner_requests as competing
       where v_action = 'accept'
         and competing.event_option_id = v_request.event_option_id
         and competing.status = 'PENDING'
         and (
           competing.requester_selection_id in (v_request.requester_selection_id, v_request.target_selection_id)
           or competing.target_selection_id in (v_request.requester_selection_id, v_request.target_selection_id)
         )
      union all
      select competing.target_selection_id
        from public.tournament_registration_partner_requests as competing
       where v_action = 'accept'
         and competing.event_option_id = v_request.event_option_id
         and competing.status = 'PENDING'
         and (
           competing.requester_selection_id in (v_request.requester_selection_id, v_request.target_selection_id)
           or competing.target_selection_id in (v_request.requester_selection_id, v_request.target_selection_id)
         )
    ) as candidate
   where candidate.selection_id is not null;
  perform private.lock_tournament_registration_selection_scope(v_lock_selection_ids);

  select * into v_request
    from public.tournament_registration_partner_requests
   where id = p_request_id
   for update;
  if v_request.id is null then
    raise exception 'JUPR_PARTNER_TRANSACTION: partner request was not found.';
  end if;
  if v_action in ('accept', 'decline')
     and v_request.target_selection_id is distinct from p_actor_selection_id then
    raise exception 'JUPR_PARTNER_TRANSACTION: only the requested partner can respond to this request.';
  end if;
  if v_action = 'cancel'
     and v_request.requester_selection_id is distinct from p_actor_selection_id then
    raise exception 'JUPR_PARTNER_TRANSACTION: only the requester can cancel this request.';
  end if;
  v_desired_status := case v_action
    when 'accept' then 'ACCEPTED'
    when 'decline' then 'DECLINED'
    else 'CANCELLED'
  end;

  if v_request.status = v_desired_status then
    select link.id into v_team_link_id
      from public.tournament_registration_team_links as link
     where link.accepted_request_id = v_request.id
     limit 1;
    return pg_catalog.jsonb_build_object(
      'outcome', 'idempotent',
      'idempotent', true,
      'status', v_request.status,
      'partner_request_id', v_request.id,
      'team_link_id', v_team_link_id,
      'cancelled_request_ids', '[]'::jsonb
    );
  end if;
  if v_request.status <> 'PENDING' then
    return pg_catalog.jsonb_build_object(
      'outcome', 'stale',
      'idempotent', false,
      'status', v_request.status,
      'partner_request_id', v_request.id,
      'team_link_id', null,
      'cancelled_request_ids', '[]'::jsonb
    );
  end if;

  if v_action = 'accept' then
    select * into v_requester
      from public.tournament_registration_selections
     where id = v_request.requester_selection_id;
    select * into v_target
      from public.tournament_registration_selections
     where id = v_request.target_selection_id;
    if v_requester.id is null or v_target.id is null
       or v_requester.tournament_id::text <> v_request.tournament_id::text
       or v_target.tournament_id::text <> v_request.tournament_id::text
       or v_requester.event_option_id <> v_request.event_option_id
       or v_target.event_option_id <> v_request.event_option_id then
      raise exception 'JUPR_PARTNER_TRANSACTION: partner request selections are stale.';
    end if;
    if exists (
      select 1
        from public.tournament_registration_team_members as member
       where member.event_option_id = v_request.event_option_id
         and member.selection_id in (v_request.requester_selection_id, v_request.target_selection_id)
         and member.status = 'ACTIVE'
    ) then
      update public.tournament_registration_partner_requests
         set status = 'CANCELLED', responded_at = v_now, updated_at = v_now
       where id = v_request.id;
      return pg_catalog.jsonb_build_object(
        'outcome', 'stale',
        'idempotent', false,
        'status', 'CANCELLED',
        'partner_request_id', v_request.id,
        'team_link_id', null,
        'cancelled_request_ids', '[]'::jsonb
      );
    end if;

    select * into v_requester_registration
      from public.tournament_registrations
     where id = v_requester.registration_id;
    select * into v_target_registration
      from public.tournament_registrations
     where id = v_target.registration_id;
    select * into v_event
      from public.tournament_event_options
     where id = v_request.event_option_id;
    select * into v_settings
      from public.tournament_registration_settings
     where tournament_id::text = v_request.tournament_id::text
     limit 1;
    if v_requester_registration.id is null
       or v_target_registration.id is null
       or pg_catalog.upper(coalesce(v_requester_registration.status, 'CONFIRMED')) in ('CANCELLED', 'WITHDRAWN')
       or pg_catalog.upper(coalesce(v_target_registration.status, 'CONFIRMED')) in ('CANCELLED', 'WITHDRAWN')
       or v_event.id is null
       or not coalesce(v_event.enabled, true)
       or not coalesce(v_event.partner_board_enabled, false)
       or pg_catalog.lower(coalesce(v_event.status, 'draft')) not in ('open', 'tentative', 'confirmed', 'published', 'active')
       or v_settings.id is null
       or not coalesce(v_settings.partner_board_enabled, false)
       or pg_catalog.upper(coalesce(v_target.partner_mode, 'NONE')) <> 'NEEDS_PARTNER'
       or not coalesce(v_target.show_on_partner_board, false)
       or not coalesce(v_target_registration.wants_partner_board_contact, false) then
      update public.tournament_registration_partner_requests
         set status = 'CANCELLED', responded_at = v_now, updated_at = v_now
       where id = v_request.id;
      return pg_catalog.jsonb_build_object(
        'outcome', 'stale',
        'idempotent', false,
        'status', 'CANCELLED',
        'partner_request_id', v_request.id,
        'team_link_id', null,
        'cancelled_request_ids', '[]'::jsonb
      );
    end if;
    v_team_link_id := 'tlink_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', '');

    insert into public.tournament_registration_team_links (
      id, tournament_id, event_option_id,
      registration1_id, registration2_id,
      selection1_id, selection2_id,
      player1_id, player2_id,
      status, accepted_request_id, created_at, updated_at
    ) values (
      v_team_link_id, v_request.tournament_id, v_request.event_option_id,
      v_requester.registration_id, v_target.registration_id,
      v_requester.id, v_target.id,
      v_requester_registration.player_id,
      v_target_registration.player_id,
      'CONFIRMED', v_request.id, v_now, v_now
    );

    insert into public.tournament_registration_team_members (
      id, team_link_id, tournament_id, event_option_id,
      selection_id, registration_id, player_id, player_order, status, created_at
    ) values
      (
        'tmem_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', ''),
        v_team_link_id, v_request.tournament_id, v_request.event_option_id,
        v_requester.id, v_requester.registration_id,
        v_requester_registration.player_id,
        1, 'ACTIVE', v_now
      ),
      (
        'tmem_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', ''),
        v_team_link_id, v_request.tournament_id, v_request.event_option_id,
        v_target.id, v_target.registration_id,
        v_target_registration.player_id,
        2, 'ACTIVE', v_now
      );

    update public.tournament_registration_partner_requests
       set status = 'ACCEPTED', responded_at = v_now, updated_at = v_now
     where id = v_request.id;
    update public.tournament_registration_selections
       set partner_mode = 'HAS_PARTNER', show_on_partner_board = false
     where id in (v_requester.id, v_target.id);

    with cancelled as (
      update public.tournament_registration_partner_requests as competing
         set status = 'CANCELLED', responded_at = v_now, updated_at = v_now
       where competing.event_option_id = v_request.event_option_id
         and competing.status = 'PENDING'
         and competing.id <> v_request.id
         and (
           competing.requester_selection_id in (v_requester.id, v_target.id)
           or competing.target_selection_id in (v_requester.id, v_target.id)
         )
       returning competing.id
    )
    select coalesce(pg_catalog.jsonb_agg(cancelled.id order by cancelled.id), '[]'::jsonb)
      into v_cancelled_ids
      from cancelled;
  elsif v_action = 'decline' then
    update public.tournament_registration_partner_requests
       set status = 'DECLINED', responded_at = v_now, updated_at = v_now
     where id = v_request.id;
  else
    update public.tournament_registration_partner_requests
       set status = 'CANCELLED', responded_at = v_now, updated_at = v_now
     where id = v_request.id;
  end if;

  return pg_catalog.jsonb_build_object(
    'outcome', 'applied',
    'idempotent', false,
    'status', v_desired_status,
    'partner_request_id', v_request.id,
    'team_link_id', v_team_link_id,
    'cancelled_request_ids', v_cancelled_ids
  );
end
$function$;

revoke execute on function public.create_tournament_partner_request(text, text, text, text, text, text, text)
  from public, anon, authenticated;
grant execute on function public.create_tournament_partner_request(text, text, text, text, text, text, text)
  to service_role;

revoke execute on function public.transition_tournament_partner_request(text, text, text)
  from public, anon, authenticated;
grant execute on function public.transition_tournament_partner_request(text, text, text)
  to service_role;

notify pgrst, 'reload schema';
