-- Scope durable tournament attendance to an authoritative enabled event day.
--
-- FastAPI remains the only caller. The browser cannot access this table or
-- RPC directly. Existing dayless rows are migrated only when their selected
-- active events resolve to exactly one canonical scheduled day; any ambiguous
-- legacy state stops the migration for operator review instead of guessing.

do $migration_preflight$
begin
  if to_regclass('public.tournament_registration_check_ins') is null
     or to_regclass('public.tournament_registration_days') is null
     or to_regclass('public.tournament_registration_selections') is null
     or to_regclass('public.tournament_event_options') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament check-in, day, selection, and event tables must exist before day-scoped attendance is installed.';
  end if;
  if not exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'tournament_event_options'
      and column_name = 'scheduled_day_ids'
  ) then
    raise exception using
      errcode = '42703',
      message = 'Canonical tournament event scheduled_day_ids must exist before day-scoped attendance is installed.';
  end if;
end
$migration_preflight$;

alter table public.tournament_registration_check_ins
  add column if not exists registration_day_id text null,
  add column if not exists attendance_status text null,
  add column if not exists last_operation_key uuid null,
  add column if not exists last_request_fingerprint text null;

do $legacy_check_in_preflight$
begin
  if exists (
    select 1
    from public.tournament_registration_check_ins as check_in
    left join lateral (
      select count(distinct scheduled.day_id) as day_count
      from public.tournament_registration_selections as selection
      join public.tournament_event_options as event
        on event.id = selection.event_option_id
       and event.tournament_id::text = check_in.tournament_id::text
       and event.enabled is true
       and pg_catalog.upper(coalesce(event.status, '')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
      cross join lateral (
        select nullif(pg_catalog.btrim(day_value.value), '') as day_id
        from pg_catalog.jsonb_array_elements_text(
          case
            when pg_catalog.jsonb_typeof(event.scheduled_day_ids) = 'array' then
              case
                when pg_catalog.jsonb_array_length(event.scheduled_day_ids) > 0
                  then event.scheduled_day_ids
                else pg_catalog.jsonb_build_array(event.registration_day_id)
              end
            else '[]'::jsonb
          end
        ) as day_value(value)
      ) as scheduled
      join public.tournament_registration_days as registration_day
        on registration_day.id = scheduled.day_id
       and registration_day.tournament_id::text = check_in.tournament_id::text
       and registration_day.enabled is true
      where selection.tournament_id::text = check_in.tournament_id::text
        and selection.registration_id = check_in.registration_id
        and scheduled.day_id is not null
    ) as mapping on true
    where check_in.registration_day_id is null
      and coalesce(mapping.day_count, 0) <> 1
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_LEGACY_DAY_AMBIGUOUS: every legacy check-in must resolve to exactly one enabled canonical scheduled day.';
  end if;
end
$legacy_check_in_preflight$;

update public.tournament_registration_check_ins as check_in
set registration_day_id = (
      select min(scheduled.day_id)
      from public.tournament_registration_selections as selection
      join public.tournament_event_options as event
        on event.id = selection.event_option_id
       and event.tournament_id::text = check_in.tournament_id::text
       and event.enabled is true
       and pg_catalog.upper(coalesce(event.status, '')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
      cross join lateral (
        select nullif(pg_catalog.btrim(day_value.value), '') as day_id
        from pg_catalog.jsonb_array_elements_text(
          case
            when pg_catalog.jsonb_typeof(event.scheduled_day_ids) = 'array' then
              case
                when pg_catalog.jsonb_array_length(event.scheduled_day_ids) > 0
                  then event.scheduled_day_ids
                else pg_catalog.jsonb_build_array(event.registration_day_id)
              end
            else '[]'::jsonb
          end
        ) as day_value(value)
      ) as scheduled
      join public.tournament_registration_days as registration_day
        on registration_day.id = scheduled.day_id
       and registration_day.tournament_id::text = check_in.tournament_id::text
       and registration_day.enabled is true
      where selection.tournament_id::text = check_in.tournament_id::text
        and selection.registration_id = check_in.registration_id
        and scheduled.day_id is not null
    ),
    attendance_status = case
      when check_in.checked_in is true then 'CHECKED_IN'
      else 'EXPECTED'
    end,
    last_operation_key = check_in.id,
    last_request_fingerprint = pg_catalog.jsonb_build_object(
      'attendance_status', case
        when check_in.checked_in is true then 'CHECKED_IN'
        else 'EXPECTED'
      end,
      'waiver_verified', check_in.waiver_verified,
      'approved_substitute_player_id', check_in.approved_substitute_player_id,
      'notes', check_in.notes,
      'attendee_identity_key', check_in.attendee_identity_key
    )::text
where check_in.registration_day_id is null;

update public.tournament_registration_check_ins as check_in
set attendance_status = coalesce(
      check_in.attendance_status,
      case when check_in.checked_in is true then 'CHECKED_IN' else 'EXPECTED' end
    ),
    last_operation_key = coalesce(check_in.last_operation_key, check_in.id),
    last_request_fingerprint = coalesce(
      nullif(pg_catalog.btrim(check_in.last_request_fingerprint), ''),
      pg_catalog.jsonb_build_object(
        'attendance_status', coalesce(
          check_in.attendance_status,
          case when check_in.checked_in is true then 'CHECKED_IN' else 'EXPECTED' end
        ),
        'waiver_verified', check_in.waiver_verified,
        'approved_substitute_player_id', check_in.approved_substitute_player_id,
        'notes', check_in.notes,
        'attendee_identity_key', check_in.attendee_identity_key
      )::text
    );

alter table public.tournament_registration_check_ins
  alter column registration_day_id set not null,
  alter column attendance_status set default 'EXPECTED',
  alter column attendance_status set not null,
  alter column last_operation_key set not null,
  alter column last_request_fingerprint set not null,
  drop constraint if exists tournament_registration_check_ins_one_state,
  drop constraint if exists tournament_registration_check_ins_day_fk,
  drop constraint if exists tournament_registration_check_ins_attendance_status_chk,
  drop constraint if exists tournament_registration_check_ins_attendance_consistency_chk,
  drop constraint if exists tournament_registration_check_ins_request_fingerprint_chk,
  add constraint tournament_registration_check_ins_one_state
    unique (tournament_id, registration_day_id, registration_id),
  add constraint tournament_registration_check_ins_day_fk
    foreign key (registration_day_id)
    references public.tournament_registration_days(id)
    on delete restrict,
  add constraint tournament_registration_check_ins_attendance_status_chk
    check (attendance_status in ('EXPECTED', 'CHECKED_IN', 'ABSENT')),
  add constraint tournament_registration_check_ins_attendance_consistency_chk
    check (checked_in = (attendance_status = 'CHECKED_IN')),
  add constraint tournament_registration_check_ins_request_fingerprint_chk
    check (nullif(pg_catalog.btrim(last_request_fingerprint), '') is not null);

create index if not exists idx_tournament_registration_check_ins_day_status
  on public.tournament_registration_check_ins (
    tournament_id,
    registration_day_id,
    attendance_status
  );

create unique index if not exists idx_tournament_registration_check_ins_operation_key
  on public.tournament_registration_check_ins (last_operation_key);

alter table public.tournament_registration_check_ins enable row level security;
alter table public.tournament_registration_check_ins force row level security;

revoke all on table public.tournament_registration_check_ins
  from public, anon, authenticated, service_role;
grant select, insert, update on table public.tournament_registration_check_ins
  to service_role;

create or replace function public.admin_upsert_tournament_registration_check_in(
  p_club_id text,
  p_tournament_id text,
  p_registration_id text,
  p_registration_day_id text,
  p_expected_updated_at timestamptz,
  p_attendance_status text,
  p_operation_key uuid,
  p_waiver_verified boolean,
  p_approved_substitute_player_id integer,
  p_approved_substitute_name text,
  p_notes text,
  p_updated_by text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_tournament public.tournaments%rowtype;
  v_registration_day public.tournament_registration_days%rowtype;
  v_registration public.tournament_registrations%rowtype;
  v_selection public.tournament_registration_selections%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_operation_existing public.tournament_registration_check_ins%rowtype;
  v_existing public.tournament_registration_check_ins%rowtype;
  v_after public.tournament_registration_check_ins%rowtype;
  v_requested_substitute_name text := nullif(
    pg_catalog.btrim(p_approved_substitute_name),
    ''
  );
  v_substitute_name text := null;
  v_notes text := nullif(pg_catalog.btrim(p_notes), '');
  v_actor text := nullif(pg_catalog.btrim(p_updated_by), '');
  v_attendance_status text := pg_catalog.upper(
    nullif(pg_catalog.btrim(p_attendance_status), '')
  );
  v_attendee_identity_key text;
  v_attendee_identity_changed boolean := false;
  v_effective_attendance_status text;
  v_waiver_verified boolean;
  v_selected_event_count integer := 0;
  v_request_fingerprint text;
begin
  if nullif(pg_catalog.btrim(p_club_id), '') is null
     or nullif(pg_catalog.btrim(p_tournament_id), '') is null
     or nullif(pg_catalog.btrim(p_registration_id), '') is null
     or nullif(pg_catalog.btrim(p_registration_day_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_INVALID: club, tournament, registration, and day identifiers are required.';
  end if;
  if v_actor is null or p_operation_key is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_INVALID: an authenticated operator and UUID operation key are required.';
  end if;
  if v_attendance_status not in ('EXPECTED', 'CHECKED_IN', 'ABSENT') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_INVALID: attendance status must be EXPECTED, CHECKED_IN, or ABSENT.';
  end if;
  if p_approved_substitute_player_id is null
     and v_requested_substitute_name is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_SUBSTITUTE_INVALID: a substitute must be selected by active club player id.';
  end if;

  -- Lock order is tournament, enabled day, registration, selected events,
  -- then the one day-scoped attendance row.
  select tournament.*
    into v_tournament
    from public.tournaments as tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for share;

  if v_tournament.id is null then
    raise exception using
      errcode = 'P0002',
      message = 'JUPR_CHECK_IN_NOT_FOUND: tournament is not part of this club.';
  end if;

  select registration_day.*
    into v_registration_day
    from public.tournament_registration_days as registration_day
   where registration_day.id = p_registration_day_id
     and registration_day.tournament_id::text = p_tournament_id
     and registration_day.enabled is true
   for share;

  if v_registration_day.id is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_DAY_INVALID: selected day is not an enabled day in this tournament.';
  end if;

  select registration.*
    into v_registration
    from public.tournament_registrations as registration
   where registration.id = p_registration_id
     and registration.tournament_id::text = p_tournament_id
   for update;

  if v_registration.id is null then
    raise exception using
      errcode = 'P0002',
      message = 'JUPR_CHECK_IN_NOT_FOUND: registration is not part of this tournament.';
  end if;
  if pg_catalog.upper(coalesce(v_registration.status, '')) not in
       ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_INACTIVE: only active, approved, confirmed, or registered entries can be checked in.';
  end if;

  for v_selection in
    select selection.*
      from public.tournament_registration_selections as selection
     where selection.tournament_id::text = p_tournament_id
       and selection.registration_id = p_registration_id
     order by selection.event_option_id, selection.id
     for share
  loop
    select event.*
      into v_event
      from public.tournament_event_options as event
     where event.id = v_selection.event_option_id
       and event.tournament_id::text = p_tournament_id
       and event.enabled is true
       and pg_catalog.upper(coalesce(event.status, '')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
       and case
         when pg_catalog.jsonb_typeof(event.scheduled_day_ids) = 'array' then
           case
             when pg_catalog.jsonb_array_length(event.scheduled_day_ids) > 0
               then event.scheduled_day_ids ? p_registration_day_id
             else event.registration_day_id = p_registration_day_id
           end
         else false
       end
     for share;
    if found then
      v_selected_event_count := v_selected_event_count + 1;
      if p_approved_substitute_player_id is not null
         and v_event.team_allow_substitutes is not true then
        raise exception using
          errcode = '22023',
          message = 'JUPR_CHECK_IN_SUBSTITUTE_POLICY: every selected-day event must explicitly allow substitutes.';
      end if;
    end if;
  end loop;

  if v_selected_event_count = 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_DAY_UNSCHEDULED: registration has no active event scheduled for this day.';
  end if;
  if p_approved_substitute_player_id is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_SUBSTITUTE_ATOMICITY: atomic substitute eligibility and uniqueness cannot be proven by the current registration schema.';
  end if;

  v_attendee_identity_key := case
    when p_approved_substitute_player_id is not null then
      pg_catalog.concat_ws(':', 'player', p_approved_substitute_player_id::text)
    when v_registration.player_id is not null then
      pg_catalog.concat_ws(':', 'player', v_registration.player_id::text)
    else
      pg_catalog.concat_ws(
        ':',
        'registration',
        v_registration.id,
        pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.display_name, ''))),
        pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.first_name, ''))),
        pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.last_name, ''))),
        pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.email, '')))
      )
  end;

  v_request_fingerprint := pg_catalog.jsonb_build_object(
    'attendance_status', v_attendance_status,
    'waiver_verified', coalesce(p_waiver_verified, false),
    'approved_substitute_player_id', p_approved_substitute_player_id,
    'notes', v_notes,
    'attendee_identity_key', v_attendee_identity_key
  )::text;

  select check_in.*
    into v_operation_existing
    from public.tournament_registration_check_ins as check_in
   where check_in.last_operation_key = p_operation_key
   for update;

  if v_operation_existing.id is not null
     and (
       v_operation_existing.tournament_id is distinct from v_tournament.id
       or v_operation_existing.registration_day_id is distinct from p_registration_day_id
       or v_operation_existing.registration_id is distinct from p_registration_id
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_IDEMPOTENCY_CONFLICT: operation key is already bound to a different tournament attendance row.';
  end if;

  select check_in.*
    into v_existing
    from public.tournament_registration_check_ins as check_in
   where check_in.tournament_id = v_tournament.id
     and check_in.registration_day_id = p_registration_day_id
     and check_in.registration_id = p_registration_id
   for update;

  if v_existing.id is not null
     and v_existing.last_operation_key = p_operation_key then
    if v_existing.last_request_fingerprint is distinct from v_request_fingerprint then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_IDEMPOTENCY_CONFLICT: operation key was reused with a different request.';
    end if;
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'check_in', pg_catalog.to_jsonb(v_existing),
      'attendee_identity_changed', false,
      'attendance_reset', false,
      'idempotent_replay', true
    );
  end if;

  if v_existing.id is not null then
    if p_expected_updated_at is null
       or v_existing.updated_at is distinct from p_expected_updated_at then
      raise exception using
        errcode = '40001',
        message = 'JUPR_CHECK_IN_STALE: check-in changed after it was loaded.';
    end if;

    v_attendee_identity_changed :=
      v_existing.attendee_identity_key is distinct from v_attendee_identity_key
      or v_existing.approved_substitute_player_id is distinct from p_approved_substitute_player_id;
    v_effective_attendance_status := case
      when v_attendee_identity_changed then 'EXPECTED'
      else v_attendance_status
    end;
    v_waiver_verified := case
      when v_attendee_identity_changed then false
      else coalesce(p_waiver_verified, false)
    end;

    begin
      update public.tournament_registration_check_ins as check_in
         set attendance_status = v_effective_attendance_status,
             checked_in = (v_effective_attendance_status = 'CHECKED_IN'),
             waiver_verified = v_waiver_verified,
             attendee_identity_key = v_attendee_identity_key,
             approved_substitute_player_id = p_approved_substitute_player_id,
             approved_substitute_name = v_substitute_name,
             notes = v_notes,
             updated_by = v_actor,
             last_operation_key = p_operation_key,
             last_request_fingerprint = v_request_fingerprint,
             updated_at = greatest(
               pg_catalog.clock_timestamp(),
               check_in.updated_at + interval '1 microsecond'
             )
       where check_in.id = v_existing.id
         and check_in.updated_at = p_expected_updated_at
      returning check_in.* into v_after;
    exception when unique_violation then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_IDEMPOTENCY_CONFLICT: operation key is already bound to another attendance row.';
    end;

    if v_after.id is null then
      raise exception using
        errcode = '40001',
        message = 'JUPR_CHECK_IN_STALE: check-in changed after it was loaded.';
    end if;
  else
    if p_expected_updated_at is not null then
      raise exception using
        errcode = '40001',
        message = 'JUPR_CHECK_IN_STALE: check-in changed after it was loaded.';
    end if;

    begin
      insert into public.tournament_registration_check_ins (
        tournament_id,
        registration_id,
        registration_day_id,
        attendance_status,
        checked_in,
        waiver_verified,
        attendee_identity_key,
        approved_substitute_player_id,
        approved_substitute_name,
        notes,
        created_by,
        updated_by,
        last_operation_key,
        last_request_fingerprint
      ) values (
        v_tournament.id,
        p_registration_id,
        p_registration_day_id,
        v_attendance_status,
        v_attendance_status = 'CHECKED_IN',
        coalesce(p_waiver_verified, false),
        v_attendee_identity_key,
        p_approved_substitute_player_id,
        v_substitute_name,
        v_notes,
        v_actor,
        v_actor,
        p_operation_key,
        v_request_fingerprint
      )
      returning * into v_after;
    exception when unique_violation then
      select check_in.*
        into v_operation_existing
        from public.tournament_registration_check_ins as check_in
       where check_in.last_operation_key = p_operation_key
       for update;
      if v_operation_existing.id is not null then
        if v_operation_existing.tournament_id = v_tournament.id
           and v_operation_existing.registration_day_id = p_registration_day_id
           and v_operation_existing.registration_id = p_registration_id
           and v_operation_existing.last_request_fingerprint = v_request_fingerprint then
          return pg_catalog.jsonb_build_object(
            'ok', true,
            'check_in', pg_catalog.to_jsonb(v_operation_existing),
            'attendee_identity_changed', false,
            'attendance_reset', false,
            'idempotent_replay', true
          );
        end if;
        raise exception using
          errcode = '22023',
          message = 'JUPR_CHECK_IN_IDEMPOTENCY_CONFLICT: operation key is already bound to another attendance row.';
      end if;
      raise exception using
        errcode = '40001',
        message = 'JUPR_CHECK_IN_STALE: another operator created this day check-in first.';
    end;
  end if;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'check_in', pg_catalog.to_jsonb(v_after),
    'attendee_identity_changed', v_attendee_identity_changed,
    'attendance_reset', v_attendee_identity_changed,
    'idempotent_replay', false
  );
end
$function$;

-- The legacy dayless signature remains installed only for migration-history
-- compatibility. It is deliberately not executable by the service role.
revoke all on function public.admin_upsert_tournament_registration_check_in(
  text, text, text, timestamptz, boolean, boolean, integer, text, text, text
) from public, anon, authenticated, service_role;

revoke all on function public.admin_upsert_tournament_registration_check_in(
  text, text, text, text, timestamptz, text, uuid, boolean, integer, text, text, text
) from public, anon, authenticated, service_role;
grant execute on function public.admin_upsert_tournament_registration_check_in(
  text, text, text, text, timestamptz, text, uuid, boolean, integer, text, text, text
) to service_role;
