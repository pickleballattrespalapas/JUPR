-- Durable, server-only tournament registration check-in state.
--
-- The browser never talks to this table or RPC directly. FastAPI authenticates
-- the tournament operator, verifies club scope, and invokes the compare-and-
-- swap RPC with the service-role client. Keeping this state separate from the
-- registration record avoids turning event-day attendance into registration
-- profile truth.

do $migration_preflight$
begin
  if to_regclass('public.tournaments') is null
     or to_regclass('public.tournament_registrations') is null
     or to_regclass('public.tournament_registration_selections') is null
     or to_regclass('public.tournament_event_options') is null
     or to_regclass('public.players') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament, registration, and player tables must exist before check-in storage is installed.';
  end if;
end
$migration_preflight$;

create table if not exists public.tournament_registration_check_ins (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  registration_id text not null references public.tournament_registrations(id) on delete cascade,
  checked_in boolean not null default false,
  waiver_verified boolean not null default false,
  attendee_identity_key text not null,
  approved_substitute_player_id integer null references public.players(id) on delete restrict,
  approved_substitute_name text null,
  notes text null,
  created_by text not null,
  updated_by text not null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint tournament_registration_check_ins_one_state
    unique (tournament_id, registration_id),
  constraint tournament_registration_check_ins_attendee_identity_present
    check (nullif(pg_catalog.btrim(attendee_identity_key), '') is not null),
  constraint tournament_registration_check_ins_substitute_atomicity_guard
    check (
      approved_substitute_player_id is null
      and approved_substitute_name is null
    ),
  constraint tournament_registration_check_ins_substitute_name_chk
    check (
      (
        approved_substitute_player_id is null
        and approved_substitute_name is null
      )
      or (
        approved_substitute_player_id is not null
        and nullif(pg_catalog.btrim(approved_substitute_name), '') is not null
      )
    )
);

create index if not exists idx_tournament_registration_check_ins_tournament
  on public.tournament_registration_check_ins (tournament_id);

create index if not exists idx_tournament_registration_check_ins_registration
  on public.tournament_registration_check_ins (registration_id);

create index if not exists idx_tournament_registration_check_ins_substitute
  on public.tournament_registration_check_ins (approved_substitute_player_id)
  where approved_substitute_player_id is not null;

alter table public.tournament_registration_check_ins enable row level security;
alter table public.tournament_registration_check_ins force row level security;

revoke all on table public.tournament_registration_check_ins from public, anon, authenticated, service_role;
grant select, insert, update on table public.tournament_registration_check_ins to service_role;

create or replace function public.admin_upsert_tournament_registration_check_in(
  p_club_id text,
  p_tournament_id text,
  p_registration_id text,
  p_expected_updated_at timestamptz,
  p_checked_in boolean,
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
  v_registration public.tournament_registrations%rowtype;
  v_selection public.tournament_registration_selections%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_existing public.tournament_registration_check_ins%rowtype;
  v_after public.tournament_registration_check_ins%rowtype;
  v_requested_substitute_name text := nullif(
    pg_catalog.btrim(p_approved_substitute_name),
    ''
  );
  v_substitute_name text := null;
  v_notes text := nullif(pg_catalog.btrim(p_notes), '');
  v_actor text := nullif(pg_catalog.btrim(p_updated_by), '');
  v_attendee_identity_key text;
  v_attendee_identity_changed boolean := false;
  v_checked_in boolean;
  v_waiver_verified boolean;
  v_selection_count integer := 0;
begin
  if nullif(pg_catalog.btrim(p_club_id), '') is null
     or nullif(pg_catalog.btrim(p_tournament_id), '') is null
     or nullif(pg_catalog.btrim(p_registration_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_INVALID: club, tournament, and registration identifiers are required.';
  end if;
  if v_actor is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_INVALID: an authenticated operator is required.';
  end if;
  if p_approved_substitute_player_id is null
     and v_requested_substitute_name is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_SUBSTITUTE_INVALID: a substitute must be selected by active club player id.';
  end if;

  -- Shared tournament locks keep club scope stable while allowing independent
  -- registration check-ins to proceed concurrently.
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

  -- Lock order is always tournament, registration, event policy, check-in.
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

  if p_approved_substitute_player_id is not null then
    for v_selection in
      select selection.*
        from public.tournament_registration_selections as selection
       where selection.tournament_id::text = p_tournament_id
         and selection.registration_id = p_registration_id
       order by selection.event_option_id, selection.id
       for share
    loop
      v_selection_count := v_selection_count + 1;
      select event.*
        into v_event
        from public.tournament_event_options as event
       where event.id = v_selection.event_option_id
         and event.tournament_id::text = p_tournament_id
       for share;
      if not found or v_event.team_allow_substitutes is not true then
        raise exception using
          errcode = '22023',
          message = 'JUPR_CHECK_IN_SUBSTITUTE_POLICY: every selected event must explicitly allow substitutes.';
      end if;
    end loop;
    if v_selection_count = 0 then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_SUBSTITUTE_POLICY: selected event policy is unavailable.';
    end if;

    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_SUBSTITUTE_ATOMICITY: atomic substitute eligibility and uniqueness cannot be proven by the current registration schema.';
  end if;

  v_attendee_identity_key := case
    when p_approved_substitute_player_id is not null then
      pg_catalog.concat_ws(
        ':',
        'player',
        p_approved_substitute_player_id::text
      )
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

  select check_in.*
    into v_existing
    from public.tournament_registration_check_ins as check_in
   where check_in.tournament_id = v_tournament.id
     and check_in.registration_id = p_registration_id
   for update;

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
    v_checked_in := case
      when v_attendee_identity_changed then false
      else coalesce(p_checked_in, false)
    end;
    v_waiver_verified := case
      when v_attendee_identity_changed then false
      else coalesce(p_waiver_verified, false)
    end;

    update public.tournament_registration_check_ins as check_in
       set checked_in = v_checked_in,
           waiver_verified = v_waiver_verified,
           attendee_identity_key = v_attendee_identity_key,
           approved_substitute_player_id = p_approved_substitute_player_id,
           approved_substitute_name = v_substitute_name,
           notes = v_notes,
           updated_by = v_actor,
           updated_at = greatest(
             pg_catalog.clock_timestamp(),
             check_in.updated_at + interval '1 microsecond'
           )
     where check_in.id = v_existing.id
       and check_in.updated_at = p_expected_updated_at
    returning check_in.* into v_after;

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
        checked_in,
        waiver_verified,
        attendee_identity_key,
        approved_substitute_player_id,
        approved_substitute_name,
        notes,
        created_by,
        updated_by
      ) values (
        v_tournament.id,
        p_registration_id,
        coalesce(p_checked_in, false),
        coalesce(p_waiver_verified, false),
        v_attendee_identity_key,
        p_approved_substitute_player_id,
        v_substitute_name,
        v_notes,
        v_actor,
        v_actor
      )
      returning * into v_after;
    exception when unique_violation then
      raise exception using
        errcode = '40001',
        message = 'JUPR_CHECK_IN_STALE: another operator created this check-in first.';
    end;
  end if;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'check_in', pg_catalog.to_jsonb(v_after),
    'attendee_identity_changed', v_attendee_identity_changed,
    'attendance_reset', v_attendee_identity_changed
  );
end
$function$;

revoke all on function public.admin_upsert_tournament_registration_check_in(
  text, text, text, timestamptz, boolean, boolean, integer, text, text, text
) from public, anon, authenticated, service_role;
grant execute on function public.admin_upsert_tournament_registration_check_in(
  text, text, text, timestamptz, boolean, boolean, integer, text, text, text
) to service_role;
