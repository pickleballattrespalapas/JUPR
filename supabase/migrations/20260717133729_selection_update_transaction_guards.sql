-- Serialize tournament registration selection edits with partner relationship
-- mutations. The public RPC is server-only and is the only supported path for
-- moving an existing selection between event options/days.

do $migration_preflight$
begin
  if to_regclass('public.tournaments') is null
     or to_regclass('public.tournament_registrations') is null
     or to_regclass('public.tournament_registration_selections') is null
     or to_regclass('public.tournament_registration_days') is null
     or to_regclass('public.tournament_event_options') is null
     or to_regclass('public.tournament_registration_partner_requests') is null
     or to_regclass('public.tournament_registration_team_links') is null
     or to_regclass('public.tournament_registration_team_members') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament registration relationship tables must exist before applying selection transaction guards.';
  end if;

  if exists (
    select 1
    from public.tournament_registration_selections as selection
    left join public.tournament_registrations as registration
      on registration.id = selection.registration_id
    left join public.tournament_event_options as event_option
      on event_option.id = selection.event_option_id
    left join public.tournament_registration_days as registration_day
      on registration_day.id = selection.registration_day_id
    where registration.id is null
       or registration.tournament_id <> selection.tournament_id
       or event_option.id is null
       or event_option.tournament_id <> selection.tournament_id
       or event_option.registration_day_id <> selection.registration_day_id
       or registration_day.id is null
       or registration_day.tournament_id <> selection.tournament_id
  ) then
    raise exception using
      errcode = '23514',
      message = 'JUPR_SELECTION_INVALID_TARGET: clean inconsistent legacy selection registration, event, or day references before applying this migration.';
  end if;

  if exists (
    select 1
    from public.tournament_registration_selections as selection
    join public.tournament_event_options as event_option
      on event_option.id = selection.event_option_id
    group by
      selection.registration_id,
      selection.registration_day_id,
      lower(
        regexp_replace(
          btrim(
            coalesce(
              nullif(event_option.event_family_label, ''),
              nullif(event_option.label, ''),
              'Event'
            )
          ),
          '\s+',
          ' ',
          'g'
        )
      )
    having count(*) > 1
  ) then
    raise exception using
      errcode = '23505',
      message = 'JUPR_SELECTION_DUPLICATE_FAMILY: clean duplicate registration event-family selections before applying this migration.';
  end if;

  if exists (
    select 1
    from public.tournament_registration_partner_requests as request
    left join public.tournament_registration_selections as requester
      on requester.id = request.requester_selection_id
    left join public.tournament_registration_selections as target
      on target.id = request.target_selection_id
    where requester.id is null
       or requester.tournament_id <> request.tournament_id::text
       or requester.event_option_id <> request.event_option_id
       or requester.registration_id <> request.requester_registration_id
       or (
         request.target_selection_id is not null
         and (
           target.id is null
           or target.tournament_id <> request.tournament_id::text
           or target.event_option_id <> request.event_option_id
           or target.registration_id is distinct from request.target_registration_id
         )
       )
  ) then
    raise exception using
      errcode = '23514',
      message = 'JUPR_RELATION_INVALID: clean inconsistent tournament partner requests before applying this migration.';
  end if;

  if exists (
    select 1
    from public.tournament_registration_team_links as team_link
    left join public.tournament_registration_selections as selection_one
      on selection_one.id = team_link.selection1_id
    left join public.tournament_registration_selections as selection_two
      on selection_two.id = team_link.selection2_id
    where selection_one.id is null
       or selection_two.id is null
       or selection_one.tournament_id <> team_link.tournament_id::text
       or selection_two.tournament_id <> team_link.tournament_id::text
       or selection_one.event_option_id <> team_link.event_option_id
       or selection_two.event_option_id <> team_link.event_option_id
       or selection_one.registration_id <> team_link.registration1_id
       or selection_two.registration_id <> team_link.registration2_id
  ) then
    raise exception using
      errcode = '23514',
      message = 'JUPR_RELATION_INVALID: clean inconsistent tournament team links before applying this migration.';
  end if;

  if exists (
    select 1
    from public.tournament_registration_team_members as team_member
    left join public.tournament_registration_selections as selection
      on selection.id = team_member.selection_id
    left join public.tournament_registration_team_links as team_link
      on team_link.id = team_member.team_link_id
    where selection.id is null
       or team_link.id is null
       or selection.tournament_id <> team_member.tournament_id::text
       or selection.event_option_id <> team_member.event_option_id
       or selection.registration_id <> team_member.registration_id
       or team_link.tournament_id <> team_member.tournament_id
       or team_link.event_option_id <> team_member.event_option_id
       or team_member.selection_id not in (team_link.selection1_id, team_link.selection2_id)
       or (
         team_member.status = 'ACTIVE'
         and team_link.status not in ('CONFIRMED', 'ADMIN_CONFIRMED')
       )
  ) then
    raise exception using
      errcode = '23514',
      message = 'JUPR_RELATION_INVALID: clean inconsistent tournament team members before applying this migration.';
  end if;
end
$migration_preflight$;

create schema if not exists private;

revoke all on schema private from public;
revoke all on schema private from anon, authenticated;
grant usage on schema private to service_role;

create or replace function private.lock_tournament_registration_selection_scope(
  p_selection_ids text[]
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_selection_ids text[];
  v_registration_ids text[];
  v_id text;
  v_expected_count integer;
  v_found_count integer;
begin
  select array_agg(selection_id order by selection_id)
  into v_selection_ids
  from (
    select distinct btrim(raw_selection_id) as selection_id
    from unnest(coalesce(p_selection_ids, array[]::text[])) as input(raw_selection_id)
    where nullif(btrim(raw_selection_id), '') is not null
  ) as normalized_selection_ids;

  v_expected_count := coalesce(cardinality(v_selection_ids), 0);
  if v_expected_count = 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_RELATION_SELECTION_NOT_FOUND: at least one selection ID is required.';
  end if;

  -- registration_id is immutable (enforced below), so it is safe to discover
  -- the parent lock keys before acquiring the locks.
  select
    array_agg(distinct selection.registration_id order by selection.registration_id),
    count(*)
  into v_registration_ids, v_found_count
  from public.tournament_registration_selections as selection
  where selection.id = any(v_selection_ids);

  if v_found_count <> v_expected_count then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_RELATION_SELECTION_NOT_FOUND: one or more registration selections do not exist.';
  end if;

  -- Universal lock order: parent registrations first, then selections. Within
  -- each scope, lock lexicographically to avoid pair-operation deadlocks.
  foreach v_id in array v_registration_ids loop
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended('jupr:tournament-registration:' || v_id, 0)
    );
  end loop;

  foreach v_id in array v_selection_ids loop
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended('jupr:tournament-selection:' || v_id, 0)
    );
  end loop;

  perform 1
  from public.tournament_registrations as registration
  where registration.id = any(v_registration_ids)
  order by registration.id
  for update;

  select count(*)
  into v_found_count
  from public.tournament_registrations as registration
  where registration.id = any(v_registration_ids);

  if v_found_count <> cardinality(v_registration_ids) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_RELATION_SELECTION_NOT_FOUND: a parent tournament registration no longer exists.';
  end if;

  perform 1
  from public.tournament_registration_selections as selection
  where selection.id = any(v_selection_ids)
  order by selection.id
  for update;

  select count(*)
  into v_found_count
  from public.tournament_registration_selections as selection
  where selection.id = any(v_selection_ids);

  if v_found_count <> v_expected_count then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_RELATION_SELECTION_NOT_FOUND: a registration selection changed while its scope was being locked.';
  end if;
end
$function$;

create or replace function private.guard_tournament_registration_relationship_change()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_selection_ids text[];
  v_requester public.tournament_registration_selections%rowtype;
  v_target public.tournament_registration_selections%rowtype;
  v_selection_one public.tournament_registration_selections%rowtype;
  v_selection_two public.tournament_registration_selections%rowtype;
  v_team_link public.tournament_registration_team_links%rowtype;
begin
  if tg_table_name = 'tournament_registration_partner_requests' then
    if tg_op = 'INSERT' then
      v_selection_ids := array[new.requester_selection_id, new.target_selection_id];
    else
      v_selection_ids := array[
        old.requester_selection_id,
        old.target_selection_id,
        new.requester_selection_id,
        new.target_selection_id
      ];
    end if;
  elsif tg_table_name = 'tournament_registration_team_links' then
    if tg_op = 'INSERT' then
      v_selection_ids := array[new.selection1_id, new.selection2_id];
    else
      v_selection_ids := array[
        old.selection1_id,
        old.selection2_id,
        new.selection1_id,
        new.selection2_id
      ];
    end if;
  elsif tg_table_name = 'tournament_registration_team_members' then
    if tg_op = 'INSERT' then
      v_selection_ids := array[new.selection_id];
    else
      v_selection_ids := array[old.selection_id, new.selection_id];
    end if;
  else
    raise exception using
      errcode = '0A000',
      message = 'JUPR_RELATION_INVALID: unsupported relationship trigger table.';
  end if;

  perform private.lock_tournament_registration_selection_scope(v_selection_ids);

  if tg_table_name = 'tournament_registration_partner_requests' then
    select selection.*
    into strict v_requester
    from public.tournament_registration_selections as selection
    where selection.id = new.requester_selection_id;

    if v_requester.tournament_id <> new.tournament_id::text
       or v_requester.event_option_id <> new.event_option_id
       or v_requester.registration_id <> new.requester_registration_id then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: requester selection no longer matches the partner request.';
    end if;

    if new.target_selection_id is not null then
      select selection.*
      into strict v_target
      from public.tournament_registration_selections as selection
      where selection.id = new.target_selection_id;

      if v_target.tournament_id <> new.tournament_id::text
         or v_target.event_option_id <> new.event_option_id
         or v_target.registration_id is distinct from new.target_registration_id
         or v_target.id = v_requester.id then
        raise exception using
          errcode = '23514',
          message = 'JUPR_RELATION_INVALID: target selection no longer matches the partner request.';
      end if;

      if new.status = 'PENDING'
         and new.source = 'PUBLIC_PARTNER_BOARD'
         and (
           v_target.partner_mode <> 'NEEDS_PARTNER'
           or not v_target.show_on_partner_board
         ) then
        raise exception using
          errcode = '23514',
          message = 'JUPR_RELATION_INVALID: public partner-board target is no longer available.';
      end if;
    end if;

    if new.status = 'PENDING' and exists (
      select 1
      from public.tournament_registration_team_links as existing_link
      where existing_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
        and (
          existing_link.selection1_id = any(array_remove(v_selection_ids, null))
          or existing_link.selection2_id = any(array_remove(v_selection_ids, null))
        )
    ) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: a selection is already on a confirmed partner team.';
    end if;

    if new.status = 'PENDING' and exists (
      select 1
      from public.tournament_registration_partner_requests as existing_request
      where existing_request.id <> new.id
        and existing_request.status = 'PENDING'
        and existing_request.requester_selection_id = new.requester_selection_id
        and existing_request.target_selection_id is not distinct from new.target_selection_id
        and existing_request.target_player_id is not distinct from new.target_player_id
    ) then
      raise exception using
        errcode = '23505',
        message = 'JUPR_RELATION_INVALID: the same pending partner request already exists.';
    end if;

  elsif tg_table_name = 'tournament_registration_team_links' then
    select selection.*
    into strict v_selection_one
    from public.tournament_registration_selections as selection
    where selection.id = new.selection1_id;

    select selection.*
    into strict v_selection_two
    from public.tournament_registration_selections as selection
    where selection.id = new.selection2_id;

    if v_selection_one.id = v_selection_two.id
       or v_selection_one.tournament_id <> new.tournament_id::text
       or v_selection_two.tournament_id <> new.tournament_id::text
       or v_selection_one.event_option_id <> new.event_option_id
       or v_selection_two.event_option_id <> new.event_option_id
       or v_selection_one.registration_id <> new.registration1_id
       or v_selection_two.registration_id <> new.registration2_id then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: partner team link no longer matches its selections.';
    end if;

    if new.status in ('CONFIRMED', 'ADMIN_CONFIRMED') and exists (
      select 1
      from public.tournament_registration_team_links as existing_link
      where existing_link.id <> new.id
        and existing_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
        and (
          existing_link.selection1_id in (new.selection1_id, new.selection2_id)
          or existing_link.selection2_id in (new.selection1_id, new.selection2_id)
        )
    ) then
      raise exception using
        errcode = '23505',
        message = 'JUPR_RELATION_INVALID: a selection is already on another confirmed partner team.';
    end if;

    if new.accepted_request_id is not null and not exists (
      select 1
      from public.tournament_registration_partner_requests as accepted_request
      where accepted_request.id = new.accepted_request_id
        and accepted_request.tournament_id = new.tournament_id
        and accepted_request.event_option_id = new.event_option_id
        and accepted_request.requester_selection_id in (new.selection1_id, new.selection2_id)
        and accepted_request.target_selection_id in (new.selection1_id, new.selection2_id)
    ) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: accepted request does not match the partner team link.';
    end if;

  elsif tg_table_name = 'tournament_registration_team_members' then
    select selection.*
    into strict v_selection_one
    from public.tournament_registration_selections as selection
    where selection.id = new.selection_id;

    select team_link.*
    into strict v_team_link
    from public.tournament_registration_team_links as team_link
    where team_link.id = new.team_link_id;

    if v_selection_one.tournament_id <> new.tournament_id::text
       or v_selection_one.event_option_id <> new.event_option_id
       or v_selection_one.registration_id <> new.registration_id
       or v_team_link.tournament_id <> new.tournament_id
       or v_team_link.event_option_id <> new.event_option_id
       or new.selection_id not in (v_team_link.selection1_id, v_team_link.selection2_id) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: partner team member no longer matches its selection and link.';
    end if;

    if new.status = 'ACTIVE'
       and v_team_link.status not in ('CONFIRMED', 'ADMIN_CONFIRMED') then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: an active team member requires a confirmed partner link.';
    end if;
  end if;

  return new;
exception
  when no_data_found then
    raise exception using
      errcode = '23514',
      message = 'JUPR_RELATION_INVALID: a referenced tournament registration relationship row no longer exists.';
end
$function$;

create or replace function private.guard_tournament_registration_selection_identity()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_registration_tournament_id text;
  v_target_event public.tournament_event_options%rowtype;
  v_target_family text;
begin
  if tg_op = 'INSERT' then
    if nullif(btrim(new.id), '') is null
       or nullif(btrim(new.tournament_id), '') is null
       or nullif(btrim(new.registration_id), '') is null
       or nullif(btrim(new.event_option_id), '') is null
       or nullif(btrim(new.registration_day_id), '') is null then
      raise exception using
        errcode = '23514',
        message = 'JUPR_SELECTION_INVALID_TARGET: selection identity, tournament, registration, day, and event are required.';
    end if;

    -- New rows do not exist yet, so acquire the same parent-registration lock
    -- directly before the per-selection lock used by relationship mutations.
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'jupr:tournament-registration:' || btrim(new.registration_id),
        0
      )
    );
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'jupr:tournament-selection:' || btrim(new.id),
        0
      )
    );

    select registration.tournament_id
    into v_registration_tournament_id
    from public.tournament_registrations as registration
    where registration.id = btrim(new.registration_id)
    for update;

    if not found
       or v_registration_tournament_id <> btrim(new.tournament_id)
       or not exists (
         select 1
         from public.tournaments as tournament
         where tournament.id::text = btrim(new.tournament_id)
       ) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_SELECTION_INVALID_TARGET: selection does not match an existing tournament registration.';
    end if;
  else
    if new.id is distinct from old.id
       or new.tournament_id is distinct from old.tournament_id
       or new.registration_id is distinct from old.registration_id then
      raise exception using
        errcode = '23514',
        message = 'JUPR_SELECTION_IDENTITY_IMMUTABLE: selection identity columns cannot be changed.';
    end if;

    if (
      new.event_option_id is distinct from old.event_option_id
      or new.registration_day_id is distinct from old.registration_day_id
    ) and coalesce(
      pg_catalog.current_setting('jupr.selection_edit_rpc', true),
      ''
    ) <> 'on' then
      raise exception using
        errcode = '42501',
        message = 'JUPR_SELECTION_EVENT_UPDATE_REQUIRES_RPC: use admin_update_tournament_registration_selection.';
    end if;
  end if;

  select event_option.*
  into v_target_event
  from public.tournament_event_options as event_option
  where event_option.id = new.event_option_id
    and event_option.tournament_id = new.tournament_id;

  if not found
     or new.registration_day_id is distinct from v_target_event.registration_day_id then
    raise exception using
      errcode = '23514',
      message = 'JUPR_SELECTION_INVALID_TARGET: selection event, day, and tournament are inconsistent.';
  end if;

  v_target_family := lower(
    regexp_replace(
      btrim(
        coalesce(
          nullif(v_target_event.event_family_label, ''),
          nullif(v_target_event.label, ''),
          'Event'
        )
      ),
      '\s+',
      ' ',
      'g'
    )
  );

  if exists (
    select 1
    from public.tournament_registration_selections as sibling
    join public.tournament_event_options as sibling_event
      on sibling_event.id = sibling.event_option_id
    where sibling.registration_id = new.registration_id
      and sibling.id <> new.id
      and sibling.registration_day_id = new.registration_day_id
      and lower(
        regexp_replace(
          btrim(
            coalesce(
              nullif(sibling_event.event_family_label, ''),
              nullif(sibling_event.label, ''),
              'Event'
            )
          ),
          '\s+',
          ' ',
          'g'
        )
      ) = v_target_family
  ) then
    raise exception using
      errcode = '23505',
      message = 'JUPR_SELECTION_WRITE_CONFLICT: duplicate event family for this registration day.';
  end if;

  return new;
end
$function$;

create or replace function private.advance_tournament_registration_selection_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  -- Ignore caller-supplied timestamps. Every mutation receives a strictly
  -- increasing write version, including canonical partner workflow updates.
  new.updated_at := greatest(
    pg_catalog.clock_timestamp(),
    old.updated_at + interval '1 microsecond'
  );
  return new;
end
$function$;

create index if not exists idx_tournament_team_links_confirmed_selection1
  on public.tournament_registration_team_links (selection1_id)
  where status in ('CONFIRMED', 'ADMIN_CONFIRMED');

create index if not exists idx_tournament_team_links_confirmed_selection2
  on public.tournament_registration_team_links (selection2_id)
  where status in ('CONFIRMED', 'ADMIN_CONFIRMED');

create index if not exists idx_tournament_team_members_active_selection
  on public.tournament_registration_team_members (selection_id)
  where status = 'ACTIVE';

drop trigger if exists guard_tournament_partner_request_change
  on public.tournament_registration_partner_requests;
create trigger guard_tournament_partner_request_change
before insert or update
on public.tournament_registration_partner_requests
for each row
execute function private.guard_tournament_registration_relationship_change();

drop trigger if exists guard_tournament_team_link_change
  on public.tournament_registration_team_links;
create trigger guard_tournament_team_link_change
before insert or update
on public.tournament_registration_team_links
for each row
execute function private.guard_tournament_registration_relationship_change();

drop trigger if exists guard_tournament_team_member_change
  on public.tournament_registration_team_members;
create trigger guard_tournament_team_member_change
before insert or update
on public.tournament_registration_team_members
for each row
execute function private.guard_tournament_registration_relationship_change();

drop trigger if exists guard_tournament_selection_identity
  on public.tournament_registration_selections;
create trigger guard_tournament_selection_identity
before insert or update of id, tournament_id, registration_id, registration_day_id, event_option_id
on public.tournament_registration_selections
for each row
execute function private.guard_tournament_registration_selection_identity();

drop trigger if exists advance_tournament_selection_updated_at
  on public.tournament_registration_selections;
create trigger advance_tournament_selection_updated_at
before update
on public.tournament_registration_selections
for each row
execute function private.advance_tournament_registration_selection_updated_at();

create or replace function public.admin_update_tournament_registration_selection(
  p_tournament_id text,
  p_selection_id text,
  p_expected_updated_at timestamptz,
  p_patch jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_before public.tournament_registration_selections%rowtype;
  v_candidate public.tournament_registration_selections%rowtype;
  v_after public.tournament_registration_selections%rowtype;
  v_target_event public.tournament_event_options%rowtype;
  v_target_day public.tournament_registration_days%rowtype;
  v_bad_key text;
  v_target_family text;
  v_event_changed boolean;
  v_relationship_sensitive boolean;
  v_updated_at timestamptz;
begin
  if nullif(btrim(p_tournament_id), '') is null
     or nullif(btrim(p_selection_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: tournament and selection IDs are required.';
  end if;

  if p_patch is null or jsonb_typeof(p_patch) <> 'object' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: patch must be a JSON object.';
  end if;

  select patch_key
  into v_bad_key
  from jsonb_object_keys(p_patch) as patch_keys(patch_key)
  where patch_key not in (
    'registration_day_id',
    'event_option_id',
    'partner_mode',
    'partner_name',
    'partner_email',
    'partner_phone',
    'partner_dupr_id',
    'partner_skill',
    'partner_age',
    'partner_note',
    'show_on_partner_board'
  )
  limit 1;

  if v_bad_key is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: unsupported field ' || v_bad_key || '.';
  end if;

  if not exists (
    select 1
    from public.tournament_registration_selections as selection
    where selection.id = btrim(p_selection_id)
      and selection.tournament_id = btrim(p_tournament_id)
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_NOT_FOUND'
    );
  end if;

  begin
    perform private.lock_tournament_registration_selection_scope(
      array[btrim(p_selection_id)]
    );
  exception
    when others then
      if sqlerrm like 'JUPR_RELATION_SELECTION_NOT_FOUND:%' then
        return jsonb_build_object(
          'ok', false,
          'code', 'SELECTION_NOT_FOUND'
        );
      end if;
      raise;
  end;

  select selection.*
  into v_before
  from public.tournament_registration_selections as selection
  where selection.id = btrim(p_selection_id)
    and selection.tournament_id = btrim(p_tournament_id)
  for update;

  if not found then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_NOT_FOUND'
    );
  end if;

  if p_expected_updated_at is null
     or v_before.updated_at is distinct from p_expected_updated_at then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'STALE_VERSION'
    );
  end if;

  select populated.*
  into v_candidate
  from jsonb_populate_record(v_before, p_patch) as populated;

  v_candidate.event_option_id := nullif(btrim(v_candidate.event_option_id), '');
  v_candidate.partner_mode := upper(nullif(btrim(v_candidate.partner_mode), ''));

  if v_candidate.event_option_id is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: event option is required.';
  end if;

  if v_candidate.partner_mode is null
     or v_candidate.partner_mode not in ('NONE', 'HAS_PARTNER', 'NEEDS_PARTNER') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: invalid partner mode.';
  end if;

  select event_option.*
  into v_target_event
  from public.tournament_event_options as event_option
  where event_option.id = v_candidate.event_option_id
    and event_option.tournament_id = btrim(p_tournament_id)
  for share;

  if not found then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: event option is not in this tournament.';
  end if;

  if p_patch ? 'registration_day_id'
     and nullif(btrim(p_patch ->> 'registration_day_id'), '')
       is distinct from v_target_event.registration_day_id then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: registration day does not match the event option.';
  end if;

  v_candidate.registration_day_id := v_target_event.registration_day_id;
  v_event_changed :=
    v_candidate.event_option_id is distinct from v_before.event_option_id;

  if v_event_changed then
    select registration_day.*
    into v_target_day
    from public.tournament_registration_days as registration_day
    where registration_day.id = v_target_event.registration_day_id
      and registration_day.tournament_id = btrim(p_tournament_id)
    for share;

    if not found
       or not v_target_day.enabled
       or not v_target_event.enabled
       or lower(coalesce(nullif(btrim(v_target_event.status), ''), 'draft'))
         not in ('open', 'tentative', 'confirmed') then
      raise exception using
        errcode = '22023',
        message = 'JUPR_SELECTION_INVALID_TARGET: target event is not open on an enabled registration day.';
    end if;
  end if;

  v_target_family := lower(
    regexp_replace(
      btrim(
        coalesce(
          nullif(v_target_event.event_family_label, ''),
          nullif(v_target_event.label, ''),
          'Event'
        )
      ),
      '\s+',
      ' ',
      'g'
    )
  );

  if exists (
    select 1
    from public.tournament_registration_selections as sibling
    join public.tournament_event_options as sibling_event
      on sibling_event.id = sibling.event_option_id
    where sibling.registration_id = v_before.registration_id
      and sibling.id <> v_before.id
      and sibling.registration_day_id = v_target_event.registration_day_id
      and lower(
        regexp_replace(
          btrim(
            coalesce(
              nullif(sibling_event.event_family_label, ''),
              nullif(sibling_event.label, ''),
              'Event'
            )
          ),
          '\s+',
          ' ',
          'g'
        )
      ) = v_target_family
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'DUPLICATE_EVENT_FAMILY'
    );
  end if;

  v_relationship_sensitive :=
    v_candidate.event_option_id is distinct from v_before.event_option_id
    or v_candidate.registration_day_id is distinct from v_before.registration_day_id
    or v_candidate.partner_mode is distinct from v_before.partner_mode;

  if v_relationship_sensitive and (
    exists (
      select 1
      from public.tournament_registration_partner_requests as request
      where request.status in ('PENDING', 'ADMIN_CONFIRMED')
        and (
          request.requester_selection_id = v_before.id
          or request.target_selection_id = v_before.id
        )
    )
    or exists (
      select 1
      from public.tournament_registration_team_links as team_link
      where team_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
        and (
          team_link.selection1_id = v_before.id
          or team_link.selection2_id = v_before.id
        )
    )
    or exists (
      select 1
      from public.tournament_registration_team_members as team_member
      where team_member.status = 'ACTIVE'
        and team_member.selection_id = v_before.id
    )
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'PARTNER_RELATIONSHIP_CHANGED'
    );
  end if;

  if v_candidate.partner_mode = 'HAS_PARTNER'
     and (
       v_before.partner_mode <> 'HAS_PARTNER'
       or v_candidate.event_option_id <> v_before.event_option_id
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: canonical partner links must create HAS_PARTNER state.';
  end if;

  v_updated_at := greatest(
    pg_catalog.clock_timestamp(),
    v_before.updated_at + interval '1 microsecond'
  );

  perform pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true);

  update public.tournament_registration_selections as selection
  set
    registration_day_id = v_candidate.registration_day_id,
    event_option_id = v_candidate.event_option_id,
    partner_mode = v_candidate.partner_mode,
    partner_name = v_candidate.partner_name,
    partner_email = v_candidate.partner_email,
    partner_phone = v_candidate.partner_phone,
    partner_dupr_id = v_candidate.partner_dupr_id,
    partner_skill = v_candidate.partner_skill,
    partner_age = v_candidate.partner_age,
    partner_note = v_candidate.partner_note,
    show_on_partner_board = v_candidate.show_on_partner_board,
    updated_at = v_updated_at
  where selection.id = v_before.id
    and selection.tournament_id = btrim(p_tournament_id)
    and selection.updated_at = p_expected_updated_at
  returning selection.* into v_after;

  if not found then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'STALE_VERSION'
    );
  end if;

  return jsonb_build_object(
    'ok', true,
    'selection', to_jsonb(v_after)
  );
end
$function$;

revoke all on function private.lock_tournament_registration_selection_scope(text[])
  from public, anon, authenticated;
grant execute on function private.lock_tournament_registration_selection_scope(text[])
  to service_role;

revoke all on function private.guard_tournament_registration_relationship_change()
  from public, anon, authenticated;
grant execute on function private.guard_tournament_registration_relationship_change()
  to service_role;

revoke all on function private.guard_tournament_registration_selection_identity()
  from public, anon, authenticated;
grant execute on function private.guard_tournament_registration_selection_identity()
  to service_role;

revoke all on function private.advance_tournament_registration_selection_updated_at()
  from public, anon, authenticated;
grant execute on function private.advance_tournament_registration_selection_updated_at()
  to service_role;

revoke all on function public.admin_update_tournament_registration_selection(
  text,
  text,
  timestamptz,
  jsonb
) from public, anon, authenticated;
grant execute on function public.admin_update_tournament_registration_selection(
  text,
  text,
  timestamptz,
  jsonb
) to service_role;

notify pgrst, 'reload schema';
