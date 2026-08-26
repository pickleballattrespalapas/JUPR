-- Apply a selected tournament day's check-in edits as one durable transaction.
--
-- FastAPI authenticates the operator and invokes this RPC with the service
-- role. The browser supplies one UUID idempotency key for the entire batch;
-- the operation ledger binds that key to one canonical, sorted request and
-- stores the exact committed result for response-loss replay.

do $migration_preflight$
begin
  if pg_catalog.to_regclass('public.tournaments') is null
     or pg_catalog.to_regclass('public.tournament_registration_days') is null
     or pg_catalog.to_regclass('public.tournament_registrations') is null
     or pg_catalog.to_regclass('public.tournament_registration_selections') is null
     or pg_catalog.to_regclass('public.tournament_event_options') is null
     or pg_catalog.to_regclass('public.tournament_event_draws') is null
     or pg_catalog.to_regclass('public.tournament_teams') is null
     or pg_catalog.to_regclass('public.tournament_registration_check_ins') is null
     or pg_catalog.to_regclass('public.tournament_admin_operations') is null
     or pg_catalog.to_regclass('public.admin_activity_log') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament check-in, roster, operation-ledger, and audit tables must exist before bulk check-in is installed.';
  end if;
  if pg_catalog.to_regprocedure('extensions.digest(text,text)') is null then
    raise exception using
      errcode = '42883',
      message = 'The extensions.digest(text,text) SHA-256 function must exist before bulk check-in is installed.';
  end if;
end
$migration_preflight$;

create or replace function public.admin_bulk_upsert_tournament_registration_check_ins(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_operation_key uuid,
  p_updates jsonb,
  p_actor_email text,
  p_actor_role text
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
  v_existing public.tournament_registration_check_ins%rowtype;
  v_after public.tournament_registration_check_ins%rowtype;
  v_operation public.tournament_admin_operations%rowtype;
  v_patch jsonb;
  v_canonical_updates jsonb;
  v_canonical_request jsonb;
  v_registration_ids text[];
  v_request_fingerprint text;
  v_internal_operation_key text;
  v_actor_email text := nullif(pg_catalog.btrim(p_actor_email), '');
  v_actor_role text := nullif(pg_catalog.btrim(p_actor_role), '');
  v_attendee_identity_key text;
  v_expected_updated_at timestamptz;
  v_attendance_status text;
  v_waiver_verified boolean;
  v_notes text;
  v_rostered_registration_count integer;
  v_requested_count integer;
  v_before_rows jsonb := '[]'::jsonb;
  v_after_rows jsonb := '[]'::jsonb;
  v_result jsonb;
  v_bad_key text;
begin
  if nullif(pg_catalog.btrim(p_club_id), '') is null
     or nullif(pg_catalog.btrim(p_tournament_id), '') is null
     or nullif(pg_catalog.btrim(p_registration_day_id), '') is null
     or p_operation_key is null
     or v_actor_email is null
     or v_actor_role is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_INVALID: club, tournament, enabled day, UUID operation key, and authenticated actor are required.';
  end if;
  if p_updates is null
     or pg_catalog.jsonb_typeof(p_updates) <> 'array'
     or pg_catalog.jsonb_array_length(p_updates) < 1
     or pg_catalog.jsonb_array_length(p_updates) > 100 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_INVALID: updates must contain between 1 and 100 rows.';
  end if;

  for v_patch in
    select requested.element
      from pg_catalog.jsonb_array_elements(p_updates) with ordinality
        as requested(element, ordinal)
     order by requested.ordinal
  loop
    if pg_catalog.jsonb_typeof(v_patch) <> 'object' then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: every update row must be an object.';
    end if;
    select key_name.key
      into v_bad_key
      from pg_catalog.jsonb_object_keys(v_patch) as key_name(key)
     where key_name.key not in (
       'registration_id',
       'expected_updated_at',
       'attendance_status',
       'waiver_verified',
       'notes'
     )
     order by key_name.key
     limit 1;
    if v_bad_key is not null then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: substitutions and unrecognized row fields are not supported.';
    end if;
    if not (v_patch ? 'registration_id')
       or pg_catalog.jsonb_typeof(v_patch -> 'registration_id') <> 'string'
       or nullif(pg_catalog.btrim(v_patch ->> 'registration_id'), '') is null
       or pg_catalog.char_length(pg_catalog.btrim(v_patch ->> 'registration_id')) > 160
       or not (v_patch ? 'expected_updated_at')
       or pg_catalog.jsonb_typeof(v_patch -> 'expected_updated_at') not in ('string', 'null') then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: every row needs a registration id and expected updated-at version or null.';
    end if;
    if pg_catalog.jsonb_typeof(v_patch -> 'expected_updated_at') = 'string' then
      begin
        perform (v_patch ->> 'expected_updated_at')::timestamptz;
      exception when others then
        raise exception using
          errcode = '22023',
          message = 'JUPR_CHECK_IN_BULK_INVALID: expected updated-at versions must be valid timestamps.';
      end;
    end if;
    if not (v_patch ? 'attendance_status')
       and not (v_patch ? 'waiver_verified')
       and not (v_patch ? 'notes') then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: every row needs an attendance, waiver, or note patch.';
    end if;
    if v_patch ? 'attendance_status'
       and (
         pg_catalog.jsonb_typeof(v_patch -> 'attendance_status') <> 'string'
         or pg_catalog.upper(pg_catalog.btrim(v_patch ->> 'attendance_status'))
           not in ('EXPECTED', 'CHECKED_IN', 'ABSENT')
       ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: attendance status must be EXPECTED, CHECKED_IN, or ABSENT.';
    end if;
    if v_patch ? 'waiver_verified'
       and pg_catalog.jsonb_typeof(v_patch -> 'waiver_verified') <> 'boolean' then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: waiver verification must be true or false.';
    end if;
    if v_patch ? 'notes'
       and pg_catalog.jsonb_typeof(v_patch -> 'notes') not in ('string', 'null') then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: operator notes must be text or null.';
    end if;
    if v_patch ? 'notes'
       and pg_catalog.jsonb_typeof(v_patch -> 'notes') = 'string'
       and pg_catalog.char_length(v_patch ->> 'notes') > 1000 then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_INVALID: operator notes may contain at most 1000 characters.';
    end if;
    v_bad_key := null;
  end loop;

  if (
    select pg_catalog.count(*)
      from pg_catalog.jsonb_array_elements(p_updates) as requested(element)
  ) <> (
    select pg_catalog.count(distinct pg_catalog.btrim(requested.element ->> 'registration_id'))
      from pg_catalog.jsonb_array_elements(p_updates) as requested(element)
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_INVALID: each registration may appear only once.';
  end if;

  select coalesce(
           pg_catalog.jsonb_agg(
             pg_catalog.jsonb_build_object(
               'registration_id', pg_catalog.btrim(requested.element ->> 'registration_id'),
               'expected_updated_at', case
                 when pg_catalog.jsonb_typeof(requested.element -> 'expected_updated_at') = 'null'
                   then 'null'::jsonb
                 else pg_catalog.to_jsonb(
                   (requested.element ->> 'expected_updated_at')::timestamptz
                 )
               end
             )
             || case when requested.element ? 'attendance_status'
               then pg_catalog.jsonb_build_object(
                 'attendance_status',
                 pg_catalog.upper(pg_catalog.btrim(requested.element ->> 'attendance_status'))
               ) else '{}'::jsonb end
             || case when requested.element ? 'waiver_verified'
               then pg_catalog.jsonb_build_object(
                 'waiver_verified', (requested.element ->> 'waiver_verified')::boolean
               ) else '{}'::jsonb end
             || case when requested.element ? 'notes'
               then pg_catalog.jsonb_build_object(
                 'notes', case
                   when pg_catalog.jsonb_typeof(requested.element -> 'notes') = 'null'
                     then null::text
                   else nullif(pg_catalog.btrim(requested.element ->> 'notes'), '')
                 end
               ) else '{}'::jsonb end
             order by pg_catalog.btrim(requested.element ->> 'registration_id') collate "C"
           ),
           '[]'::jsonb
         )
    into v_canonical_updates
    from pg_catalog.jsonb_array_elements(p_updates) as requested(element);

  select pg_catalog.array_agg(
           canonical.element ->> 'registration_id'
           order by canonical.ordinal
         )
    into v_registration_ids
    from pg_catalog.jsonb_array_elements(v_canonical_updates) with ordinality
      as canonical(element, ordinal);
  v_requested_count := pg_catalog.cardinality(v_registration_ids);

  v_canonical_request := pg_catalog.jsonb_build_object(
    'version', 1,
    'club_id', pg_catalog.btrim(p_club_id),
    'tournament_id', pg_catalog.btrim(p_tournament_id),
    'registration_day_id', pg_catalog.btrim(p_registration_day_id),
    'operation_key', p_operation_key::text,
    'actor_email', v_actor_email,
    'actor_role', v_actor_role,
    'updates', v_canonical_updates
  );
  v_request_fingerprint := pg_catalog.encode(
    extensions.digest(v_canonical_request::text, 'sha256'),
    'hex'
  );
  v_internal_operation_key := pg_catalog.encode(
    extensions.digest(
      'tournament-check-in-bulk:' || pg_catalog.btrim(p_club_id) || ':' ||
      pg_catalog.btrim(p_tournament_id) || ':' ||
      pg_catalog.btrim(p_registration_day_id) || ':' || p_operation_key::text,
      'sha256'
    ),
    'hex'
  );

  -- Serialize the exact unique ledger identity before consulting its durable
  -- row. The lock namespace deliberately matches club + surface + client UUID,
  -- so reuse across tournaments or days conflicts deterministically instead of
  -- racing the unique idempotency index.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:tournament-admin-operation:' || pg_catalog.btrim(p_club_id) ||
        ':tournament_live:' || p_operation_key::text,
      0
    )
  );
  select operation.*
    into v_operation
    from public.tournament_admin_operations as operation
   where operation.club_id = pg_catalog.btrim(p_club_id)
     and operation.surface = 'tournament_live'
     and operation.client_idempotency_key = p_operation_key::text
   for update;
  if v_operation.operation_key is not null then
    if v_operation.operation_key is distinct from v_internal_operation_key
       or v_operation.request_fingerprint is distinct from v_request_fingerprint
       or v_operation.action is distinct from 'tournament_check_in_bulk_update'
       or v_operation.entity_type is distinct from 'tournament_registration_day'
       or v_operation.entity_id is distinct from (
         pg_catalog.btrim(p_tournament_id) || ':' || pg_catalog.btrim(p_registration_day_id)
       )
       or v_operation.request_json is distinct from v_canonical_request then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_IDEMPOTENCY_CONFLICT: operation key was already used for a different request.';
    end if;
    if v_operation.status <> 'completed'
       or pg_catalog.jsonb_typeof(v_operation.result_json) <> 'object'
       or coalesce((v_operation.result_json ->> 'ok')::boolean, false) is not true then
      raise exception using
        errcode = '22023',
        message = 'JUPR_CHECK_IN_BULK_IDEMPOTENCY_CONFLICT: operation needs recovery before this key can be replayed.';
    end if;
    return v_operation.result_json || pg_catalog.jsonb_build_object(
      'idempotent_replay', true
    );
  end if;

  -- Deterministic lock order: tournament, enabled day, sorted registrations,
  -- sorted selection/event/draw dependencies, then attendance rows.
  select tournament.*
    into v_tournament
    from public.tournaments as tournament
   where tournament.id::text = pg_catalog.btrim(p_tournament_id)
     and tournament.club_id::text = pg_catalog.btrim(p_club_id)
   for share;
  if v_tournament.id is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_ROSTER: tournament is not part of this club.';
  end if;

  select registration_day.*
    into v_registration_day
    from public.tournament_registration_days as registration_day
   where registration_day.id = pg_catalog.btrim(p_registration_day_id)
     and registration_day.tournament_id::text = pg_catalog.btrim(p_tournament_id)
     and registration_day.enabled is true
   for share;
  if v_registration_day.id is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_DAY: selected day is not enabled in this tournament.';
  end if;

  perform registration.id
    from public.tournament_registrations as registration
   where registration.tournament_id::text = pg_catalog.btrim(p_tournament_id)
     and registration.id = any(v_registration_ids)
   order by registration.id collate "C"
   for update;
  if (
    select pg_catalog.count(*)
      from public.tournament_registrations as registration
     where registration.tournament_id::text = pg_catalog.btrim(p_tournament_id)
       and registration.id = any(v_registration_ids)
  ) <> v_requested_count then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_ROSTER: every registration must belong to this tournament.';
  end if;
  if exists (
    select 1
      from public.tournament_registrations as registration
     where registration.tournament_id::text = pg_catalog.btrim(p_tournament_id)
       and registration.id = any(v_registration_ids)
       and pg_catalog.upper(coalesce(registration.status, '')) not in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED')
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_INACTIVE: selected registrations must be active, approved, confirmed, or registered.';
  end if;

  perform selection.id
    from public.tournament_registration_selections as selection
   where selection.tournament_id::text = pg_catalog.btrim(p_tournament_id)
     and selection.registration_id = any(v_registration_ids)
   order by selection.id collate "C"
   for share;
  perform event.id
    from public.tournament_event_options as event
   where event.tournament_id::text = pg_catalog.btrim(p_tournament_id)
     and event.id in (
       select selection.event_option_id
         from public.tournament_registration_selections as selection
        where selection.tournament_id::text = pg_catalog.btrim(p_tournament_id)
          and selection.registration_id = any(v_registration_ids)
     )
   order by event.id collate "C"
   for share;
  -- The draw row is the universal team commit gate. Every direct team child
  -- mutation locks its parent draw in the version trigger; taking only this
  -- draw lock linearizes the roster read without creating a team/draw lock
  -- inversion with writers that use different row-lock orders.
  perform draw.id
    from public.tournament_event_draws as draw
   where draw.tournament_id::text = pg_catalog.btrim(p_tournament_id)
     and draw.event_option_id in (
       select selection.event_option_id
         from public.tournament_registration_selections as selection
        where selection.tournament_id::text = pg_catalog.btrim(p_tournament_id)
          and selection.registration_id = any(v_registration_ids)
     )
   order by draw.id
   for share;
  perform player.id
    from public.players as player
   where player.id in (
     select registration.player_id
       from public.tournament_registrations as registration
      where registration.tournament_id::text = pg_catalog.btrim(p_tournament_id)
        and registration.id = any(v_registration_ids)
        and registration.player_id is not null
   )
   order by player.id
   for share;
  perform check_in.id
    from public.tournament_registration_check_ins as check_in
   where check_in.tournament_id = v_tournament.id
     and check_in.registration_day_id = pg_catalog.btrim(p_registration_day_id)
     and check_in.registration_id = any(v_registration_ids)
   order by check_in.registration_id collate "C"
   for update;

  -- Resolve roster authority for the complete canonical registration set in
  -- one SQL statement and therefore one READ COMMITTED statement snapshot.
  -- A concurrent direct draw insert can linearize before or after this query,
  -- but it cannot produce a mixed before/after roster across selected players.
  select pg_catalog.count(*)
    into v_rostered_registration_count
    from public.tournament_registrations as requested_registration
   where requested_registration.tournament_id::text = v_tournament.id::text
     and requested_registration.id = any(v_registration_ids)
     and exists (
       select 1
         from public.tournament_registration_selections as selection
         join public.tournament_event_options as event
           on event.id = selection.event_option_id
          and event.tournament_id::text = pg_catalog.btrim(p_tournament_id)
          and event.enabled is true
          and pg_catalog.upper(coalesce(event.status, '')) not in
            ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
          and case
            when pg_catalog.jsonb_typeof(event.scheduled_day_ids) = 'array' then
              case
                when pg_catalog.jsonb_array_length(event.scheduled_day_ids) > 0
                  then event.scheduled_day_ids ? pg_catalog.btrim(p_registration_day_id)
                else event.registration_day_id = pg_catalog.btrim(p_registration_day_id)
              end
            else event.registration_day_id = pg_catalog.btrim(p_registration_day_id)
          end
         cross join lateral (
           select pg_catalog.count(*)::integer as draw_count
             from public.tournament_event_draws as draw
            where draw.tournament_id = v_tournament.id
              and draw.event_option_id = event.id
              and pg_catalog.upper(coalesce(draw.status, 'DRAFT')) not in
                ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
              and coalesce(draw.hidden_from_primary_ops, false) is false
              and pg_catalog.upper(coalesce(draw.draw_kind, 'STANDARD')) = 'STANDARD'
              and (
                draw.registration_day_id = pg_catalog.btrim(p_registration_day_id)
                or (
                  draw.registration_day_id is null
                  and case
                    when pg_catalog.jsonb_typeof(event.scheduled_day_ids) = 'array' then
                      case
                        when pg_catalog.jsonb_array_length(event.scheduled_day_ids) > 0
                          then pg_catalog.jsonb_array_length(event.scheduled_day_ids) = 1
                        else event.registration_day_id = pg_catalog.btrim(p_registration_day_id)
                      end
                    else event.registration_day_id = pg_catalog.btrim(p_registration_day_id)
                  end
                )
              )
         ) as primary_scope
        where selection.tournament_id::text = pg_catalog.btrim(p_tournament_id)
          and selection.registration_id = requested_registration.id
          and (
            primary_scope.draw_count = 0
            or (
              primary_scope.draw_count = 1
              and requested_registration.player_id is not null
              and exists (
                select 1
                  from public.tournament_event_draws as draw
                  join public.tournament_teams as team
                    on team.tournament_id = draw.tournament_id
                   and team.draw_id = draw.id
                   and requested_registration.player_id in (
                     team.player1_id,
                     team.player2_id
                   )
                 where draw.tournament_id = v_tournament.id
                   and draw.event_option_id = event.id
                   and pg_catalog.upper(coalesce(draw.status, 'DRAFT')) not in
                     ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
                   and coalesce(draw.hidden_from_primary_ops, false) is false
                   and pg_catalog.upper(coalesce(draw.draw_kind, 'STANDARD')) = 'STANDARD'
                   and (
                     draw.registration_day_id = pg_catalog.btrim(p_registration_day_id)
                     or (
                       draw.registration_day_id is null
                       and case
                         when pg_catalog.jsonb_typeof(event.scheduled_day_ids) = 'array' then
                           case
                             when pg_catalog.jsonb_array_length(event.scheduled_day_ids) > 0
                               then pg_catalog.jsonb_array_length(event.scheduled_day_ids) = 1
                             else event.registration_day_id = pg_catalog.btrim(p_registration_day_id)
                           end
                         else event.registration_day_id = pg_catalog.btrim(p_registration_day_id)
                       end
                     )
                   )
              )
              and 1 = (
                select pg_catalog.count(*)
                  from public.tournament_registration_selections as exact_selection
                  join public.tournament_registrations as exact_registration
                    on exact_registration.id = exact_selection.registration_id
                   and exact_registration.tournament_id::text = v_tournament.id::text
                   and exact_registration.player_id = requested_registration.player_id
                   and pg_catalog.upper(coalesce(exact_registration.status, '')) in
                     ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED')
                 where exact_selection.tournament_id::text = pg_catalog.btrim(p_tournament_id)
                   and exact_selection.event_option_id = event.id
              )
              and exists (
                select 1
                  from public.players as player
                 where player.id = requested_registration.player_id
                   and player.club_id::text = pg_catalog.btrim(p_club_id)
                   and player.active is true
              )
            )
          )
     );
  if v_rostered_registration_count <> v_requested_count then
    raise exception using
      errcode = '22023',
      message = 'JUPR_CHECK_IN_BULK_ROSTER: every selected registration must resolve to the authoritative roster for this day.';
  end if;

  -- Preflight every selected row's CAS version and attendee identity before
  -- the first attendance row or operation-ledger row changes.
  for v_patch in
    select canonical.element
      from pg_catalog.jsonb_array_elements(v_canonical_updates) with ordinality
        as canonical(element, ordinal)
     order by canonical.ordinal
  loop
    select registration.*
      into v_registration
      from public.tournament_registrations as registration
     where registration.tournament_id::text = v_tournament.id::text
       and registration.id = v_patch ->> 'registration_id';

    v_attendee_identity_key := case
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
       and check_in.registration_day_id = pg_catalog.btrim(p_registration_day_id)
       and check_in.registration_id = v_registration.id;
    v_expected_updated_at := case
      when pg_catalog.jsonb_typeof(v_patch -> 'expected_updated_at') = 'null'
        then null
      else (v_patch ->> 'expected_updated_at')::timestamptz
    end;
    if v_existing.id is not null then
      if v_expected_updated_at is null
         or v_existing.updated_at is distinct from v_expected_updated_at then
        raise exception using
          errcode = '40001',
          message = 'JUPR_CHECK_IN_BULK_STALE: at least one selected check-in changed after it was loaded.';
      end if;
      if v_existing.approved_substitute_player_id is not null
         or v_existing.approved_substitute_name is not null
         or v_existing.attendee_identity_key is distinct from v_attendee_identity_key then
        raise exception using
          errcode = '22023',
          message = 'JUPR_CHECK_IN_BULK_ROSTER: bulk check-in cannot change or restore attendee identity; resolve that row separately.';
      end if;
    elsif v_expected_updated_at is not null then
      raise exception using
        errcode = '40001',
        message = 'JUPR_CHECK_IN_BULK_STALE: at least one selected check-in changed after it was loaded.';
    end if;
    v_before_rows := v_before_rows || pg_catalog.jsonb_build_array(
      pg_catalog.jsonb_build_object(
        'registration_id', v_registration.id,
        'patch', v_patch,
        'check_in', case
          when v_existing.id is null then 'null'::jsonb
          else pg_catalog.to_jsonb(v_existing)
        end
      )
    );
  end loop;

  insert into public.tournament_admin_operations (
    operation_key,
    request_fingerprint,
    club_id,
    surface,
    action,
    entity_type,
    entity_id,
    lock_scope,
    expected_state,
    status,
    request_json,
    result_json,
    client_idempotency_key,
    created_by,
    updated_by
  ) values (
    v_internal_operation_key,
    v_request_fingerprint,
    pg_catalog.btrim(p_club_id),
    'tournament_live',
    'tournament_check_in_bulk_update',
    'tournament_registration_day',
    pg_catalog.btrim(p_tournament_id) || ':' || pg_catalog.btrim(p_registration_day_id),
    'tournament:' || pg_catalog.btrim(p_tournament_id) || ':day:' ||
      pg_catalog.btrim(p_registration_day_id) || ':check-in',
    v_request_fingerprint,
    'intent',
    v_canonical_request,
    '{}'::jsonb,
    p_operation_key::text,
    v_actor_email,
    v_actor_email
  );

  for v_patch in
    select canonical.element
      from pg_catalog.jsonb_array_elements(v_canonical_updates) with ordinality
        as canonical(element, ordinal)
     order by canonical.ordinal
  loop
    select registration.*
      into v_registration
      from public.tournament_registrations as registration
     where registration.tournament_id::text = v_tournament.id::text
       and registration.id = v_patch ->> 'registration_id';
    v_attendee_identity_key := case
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
       and check_in.registration_day_id = pg_catalog.btrim(p_registration_day_id)
       and check_in.registration_id = v_registration.id;
    v_attendance_status := case
      when v_patch ? 'attendance_status'
        then v_patch ->> 'attendance_status'
      when v_existing.id is not null
        then v_existing.attendance_status
      else 'EXPECTED'
    end;
    v_waiver_verified := case
      when v_patch ? 'waiver_verified'
        then (v_patch ->> 'waiver_verified')::boolean
      when v_existing.id is not null
        then v_existing.waiver_verified
      else false
    end;
    v_notes := case
      when v_patch ? 'notes'
        then nullif(pg_catalog.btrim(v_patch ->> 'notes'), '')
      when v_existing.id is not null
        then v_existing.notes
      else null
    end;

    if v_existing.id is not null then
      update public.tournament_registration_check_ins as check_in
         set attendance_status = v_attendance_status,
             checked_in = (v_attendance_status = 'CHECKED_IN'),
             waiver_verified = v_waiver_verified,
             attendee_identity_key = v_attendee_identity_key,
             notes = v_notes,
             updated_by = v_actor_email,
             last_operation_key = pg_catalog.md5(
               v_internal_operation_key || ':' || v_registration.id
             )::uuid,
             last_request_fingerprint = v_request_fingerprint,
             updated_at = greatest(
               pg_catalog.clock_timestamp(),
               check_in.updated_at + interval '1 microsecond'
             )
       where check_in.id = v_existing.id
         and check_in.updated_at = v_existing.updated_at
      returning check_in.* into v_after;
    else
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
        v_registration.id,
        pg_catalog.btrim(p_registration_day_id),
        v_attendance_status,
        v_attendance_status = 'CHECKED_IN',
        v_waiver_verified,
        v_attendee_identity_key,
        null,
        null,
        v_notes,
        v_actor_email,
        v_actor_email,
        pg_catalog.md5(v_internal_operation_key || ':' || v_registration.id)::uuid,
        v_request_fingerprint
      )
      returning * into v_after;
    end if;
    if v_after.id is null then
      raise exception using
        errcode = '40001',
        message = 'JUPR_CHECK_IN_BULK_STALE: a selected row changed during the transaction.';
    end if;
    v_after_rows := v_after_rows || pg_catalog.jsonb_build_array(
      pg_catalog.to_jsonb(v_after)
    );
  end loop;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'mode', 'tournament_registration_check_in_bulk_update',
    'operation_key', p_operation_key::text,
    'updated_count', v_requested_count,
    'check_ins', v_after_rows,
    'idempotent_replay', false
  );

  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    pg_catalog.btrim(p_club_id),
    v_actor_email,
    v_actor_role,
    'bulk_update_tournament_registration_check_in_admin',
    'tournament_registration_day',
    pg_catalog.btrim(p_tournament_id) || ':' || pg_catalog.btrim(p_registration_day_id),
    pg_catalog.jsonb_build_object(
      'operation_key', p_operation_key::text,
      'rows', v_before_rows
    ),
    pg_catalog.jsonb_build_object(
      'operation_key', p_operation_key::text,
      'rows', v_after_rows
    ),
    'Atomic bulk tournament check-in for ' || v_requested_count::text || ' registration(s).',
    'next_tournament_check_in',
    true
  );

  update public.tournament_admin_operations as operation
     set status = 'completed',
         result_json = v_result,
         updated_by = v_actor_email,
         completion_audited_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where operation.operation_key = v_internal_operation_key
     and operation.status = 'intent';
  if not found then
    raise exception using
      errcode = '40001',
      message = 'JUPR_CHECK_IN_BULK_STALE: durable operation completion was not recorded.';
  end if;

  return v_result;
end
$function$;

comment on function public.admin_bulk_upsert_tournament_registration_check_ins(
  text, text, text, uuid, jsonb, text, text
) is
  'Service-role-only atomic, day-roster-authoritative bulk tournament check-in with one durable client idempotency key.';

revoke all on function public.admin_bulk_upsert_tournament_registration_check_ins(
  text, text, text, uuid, jsonb, text, text
) from public, anon, authenticated, service_role;
grant execute on function public.admin_bulk_upsert_tournament_registration_check_ins(
  text, text, text, uuid, jsonb, text, text
) to service_role;

notify pgrst, 'reload schema';
