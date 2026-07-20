-- Make token-gated public tournament registration edits atomic, versioned,
-- draw-safe, and server-only. FastAPI validates domain rules before invoking
-- this RPC with the Supabase service role.

do $migration_preflight$
begin
  if to_regclass('public.tournament_registrations') is null
     or to_regclass('public.tournament_registration_selections') is null
     or to_regclass('public.tournament_registration_days') is null
     or to_regclass('public.tournament_event_options') is null
     or to_regclass('public.tournament_teams') is null
     or to_regclass('public.tournament_registration_partner_requests') is null
     or to_regclass('public.tournament_registration_team_links') is null
     or to_regclass('public.tournament_registration_team_members') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament registration and relationship tables must exist before applying public edit transaction guards.';
  end if;

  if to_regprocedure('private.lock_tournament_registration_selection_scope(text[])') is null then
    raise exception using
      errcode = '42883',
      message = 'Selection transaction guards must be applied before public registration edit transaction guards.';
  end if;

  if not exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'tournament_registrations'
      and column_name = 'updated_at'
  ) or not exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'tournament_registration_selections'
      and column_name = 'updated_at'
  ) then
    raise exception using
      errcode = '42703',
      message = 'Registration and selection updated_at versions are required for public edits.';
  end if;
end
$migration_preflight$;

alter table public.tournament_registration_selections
  alter column updated_at set default now();

update public.tournament_registration_selections
set updated_at = coalesce(created_at, now())
where updated_at is null;

alter table public.tournament_registration_selections
  alter column updated_at set not null;

create or replace function public.server_update_public_tournament_registration_edit(
  p_tournament_id text,
  p_registration_id text,
  p_expected_updated_at timestamptz,
  p_expected_selection_versions jsonb,
  p_registration_patch jsonb,
  p_selections jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_registration public.tournament_registrations%rowtype;
  v_existing_ids text[];
  v_bad_key text;
  v_updated_at timestamptz;
  v_selection_count integer;
begin
  if nullif(btrim(p_tournament_id), '') is null
     or nullif(btrim(p_registration_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: tournament and registration IDs are required.';
  end if;

  if p_expected_updated_at is null
     or p_expected_selection_versions is null
     or jsonb_typeof(p_expected_selection_versions) <> 'array'
     or p_registration_patch is null
     or jsonb_typeof(p_registration_patch) <> 'object'
     or p_selections is null
     or jsonb_typeof(p_selections) <> 'array'
     or jsonb_array_length(p_selections) = 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: versions, registration patch, and at least one selection are required.';
  end if;

  select patch_key
  into v_bad_key
  from jsonb_object_keys(p_registration_patch) as patch_keys(patch_key)
  where patch_key not in (
    'first_name',
    'last_name',
    'display_name',
    'phone',
    'dupr_id',
    'doubles_skill',
    'singles_skill',
    'age',
    'age_bracket',
    'gender',
    'notes',
    'wants_partner_board_contact'
  )
  limit 1;

  if v_bad_key is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: unsupported registration field ' || v_bad_key || '.';
  end if;

  select selection_key
  into v_bad_key
  from jsonb_array_elements(p_selections) as desired(selection_json)
  cross join lateral jsonb_object_keys(desired.selection_json) as selection_keys(selection_key)
  where selection_key not in (
    'id',
    'tournament_id',
    'registration_id',
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
    'show_on_partner_board',
    'sort_order',
    'created_at',
    'updated_at'
  )
  limit 1;

  if v_bad_key is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: unsupported selection field ' || v_bad_key || '.';
  end if;

  select array_agg(selection.id order by selection.id)
  into v_existing_ids
  from public.tournament_registration_selections as selection
  where selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id);

  if coalesce(cardinality(v_existing_ids), 0) > 0 then
    perform private.lock_tournament_registration_selection_scope(v_existing_ids);
  else
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'jupr:tournament-registration:' || btrim(p_registration_id),
        0
      )
    );
  end if;

  select registration.*
  into v_registration
  from public.tournament_registrations as registration
  where registration.tournament_id = btrim(p_tournament_id)
    and registration.id = btrim(p_registration_id)
  for update;

  if not found then
    return jsonb_build_object('ok', false, 'code', 'REGISTRATION_NOT_FOUND');
  end if;

  if exists (
    select 1
    from public.tournament_registration_selections as selection
    join public.tournament_teams as team
      on team.tournament_id::text = selection.tournament_id
     and team.registration_day_id::text = selection.registration_day_id
     and team.event_option_id::text = selection.event_option_id
     and upper(coalesce(team.source, '')) = 'REGISTRATION'
    where selection.tournament_id = btrim(p_tournament_id)
      and selection.registration_id = btrim(p_registration_id)
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_IMPORTED_TO_DRAW'
    );
  end if;

  if v_registration.updated_at is distinct from p_expected_updated_at then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_EDIT_CONFLICT',
      'reason', 'STALE_REGISTRATION_VERSION'
    );
  end if;

  if jsonb_array_length(p_expected_selection_versions) <> (
    select count(distinct nullif(btrim(expected.id), ''))
    from jsonb_to_recordset(p_expected_selection_versions)
      as expected(id text, updated_at timestamptz)
  ) or exists (
    select 1
    from public.tournament_registration_selections as selection
    left join jsonb_to_recordset(p_expected_selection_versions)
      as expected(id text, updated_at timestamptz)
      on expected.id = selection.id
     and expected.updated_at = selection.updated_at
    where selection.tournament_id = btrim(p_tournament_id)
      and selection.registration_id = btrim(p_registration_id)
      and expected.id is null
  ) or exists (
    select 1
    from jsonb_to_recordset(p_expected_selection_versions)
      as expected(id text, updated_at timestamptz)
    left join public.tournament_registration_selections as selection
      on selection.id = expected.id
     and selection.tournament_id = btrim(p_tournament_id)
     and selection.registration_id = btrim(p_registration_id)
     and selection.updated_at = expected.updated_at
    where selection.id is null
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_EDIT_CONFLICT',
      'reason', 'STALE_SELECTION_VERSION'
    );
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text,
      partner_mode text
    )
    where nullif(btrim(desired.id), '') is null
       or nullif(btrim(desired.registration_day_id), '') is null
       or nullif(btrim(desired.event_option_id), '') is null
       or upper(coalesce(nullif(btrim(desired.partner_mode), ''), 'NONE'))
         not in ('NONE', 'HAS_PARTNER', 'NEEDS_PARTNER')
  ) or (
    select count(*)
    from jsonb_to_recordset(p_selections) as desired(id text)
  ) <> (
    select count(distinct btrim(desired.id))
    from jsonb_to_recordset(p_selections) as desired(id text)
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: selection identities are missing, duplicated, or invalid.';
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text
    )
    left join public.tournament_registration_selections as conflicting
      on conflicting.id = desired.id
    left join public.tournament_event_options as event_option
      on event_option.id = desired.event_option_id
     and event_option.tournament_id = btrim(p_tournament_id)
    left join public.tournament_registration_days as registration_day
      on registration_day.id = desired.registration_day_id
     and registration_day.tournament_id = btrim(p_tournament_id)
    left join public.tournament_registration_selections as existing
      on existing.id = desired.id
     and existing.tournament_id = btrim(p_tournament_id)
     and existing.registration_id = btrim(p_registration_id)
    where event_option.id is null
       or registration_day.id is null
       or event_option.registration_day_id <> desired.registration_day_id
       or (
         conflicting.id is not null
         and (
           conflicting.tournament_id <> btrim(p_tournament_id)
           or conflicting.registration_id <> btrim(p_registration_id)
         )
       )
       or (
         (existing.id is null or existing.event_option_id <> desired.event_option_id)
         and (
           not coalesce(event_option.enabled, false)
           or not coalesce(registration_day.enabled, false)
           or lower(coalesce(nullif(btrim(event_option.status), ''), 'draft'))
             not in ('open', 'tentative', 'confirmed')
         )
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: a selection target is unavailable or belongs to another registration.';
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text
    )
    join public.tournament_event_options as event_option
      on event_option.id = desired.event_option_id
    group by
      desired.registration_day_id,
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
      message = 'JUPR_REGISTRATION_EDIT_INVALID: choose one division per event family and day.';
  end if;

  if exists (
    select 1
    from public.tournament_registration_selections as existing
    left join jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text,
      partner_mode text
    ) on desired.id = existing.id
    where existing.tournament_id = btrim(p_tournament_id)
      and existing.registration_id = btrim(p_registration_id)
      and (
        (
          desired.id is null
          and (
            exists (
              select 1
              from public.tournament_registration_partner_requests as request
              where request.requester_selection_id = existing.id
                 or request.target_selection_id = existing.id
            )
            or exists (
              select 1
              from public.tournament_registration_team_links as team_link
              where team_link.selection1_id = existing.id
                 or team_link.selection2_id = existing.id
            )
            or exists (
              select 1
              from public.tournament_registration_team_members as team_member
              where team_member.selection_id = existing.id
            )
          )
        )
        or (
          desired.id is not null
          and (
            desired.registration_day_id is distinct from existing.registration_day_id
            or desired.event_option_id is distinct from existing.event_option_id
            or upper(coalesce(desired.partner_mode, 'NONE'))
              is distinct from upper(coalesce(existing.partner_mode, 'NONE'))
          )
          and (
            exists (
              select 1
              from public.tournament_registration_partner_requests as request
              where request.status in ('PENDING', 'ADMIN_CONFIRMED')
                and (
                  request.requester_selection_id = existing.id
                  or request.target_selection_id = existing.id
                )
            )
            or exists (
              select 1
              from public.tournament_registration_team_links as team_link
              where team_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
                and (
                  team_link.selection1_id = existing.id
                  or team_link.selection2_id = existing.id
                )
            )
            or exists (
              select 1
              from public.tournament_registration_team_members as team_member
              where team_member.status = 'ACTIVE'
                and team_member.selection_id = existing.id
            )
          )
        )
      )
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_RELATIONSHIP_LOCKED'
    );
  end if;

  delete from public.tournament_registration_selections as selection
  where selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id)
    and not exists (
      select 1
      from jsonb_to_recordset(p_selections) as desired(id text)
      where desired.id = selection.id
    );

  perform pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true);

  update public.tournament_registration_selections as selection
  set
    registration_day_id = desired.registration_day_id,
    event_option_id = desired.event_option_id,
    partner_mode = upper(coalesce(nullif(btrim(desired.partner_mode), ''), 'NONE')),
    partner_name = nullif(btrim(desired.partner_name), ''),
    partner_email = nullif(lower(btrim(desired.partner_email)), ''),
    partner_phone = nullif(btrim(desired.partner_phone), ''),
    partner_dupr_id = nullif(btrim(desired.partner_dupr_id), ''),
    partner_skill = desired.partner_skill,
    partner_age = desired.partner_age,
    partner_note = nullif(btrim(desired.partner_note), ''),
    show_on_partner_board = coalesce(desired.show_on_partner_board, false),
    sort_order = desired.sort_order
  from jsonb_to_recordset(p_selections) as desired(
    id text,
    registration_day_id text,
    event_option_id text,
    partner_mode text,
    partner_name text,
    partner_email text,
    partner_phone text,
    partner_dupr_id text,
    partner_skill numeric,
    partner_age integer,
    partner_note text,
    show_on_partner_board boolean,
    sort_order integer
  )
  where selection.id = desired.id
    and selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id);

  insert into public.tournament_registration_selections (
    id,
    tournament_id,
    registration_id,
    registration_day_id,
    event_option_id,
    partner_mode,
    partner_name,
    partner_email,
    partner_phone,
    partner_dupr_id,
    partner_skill,
    partner_age,
    partner_note,
    show_on_partner_board,
    sort_order,
    created_at,
    updated_at
  )
  select
    desired.id,
    btrim(p_tournament_id),
    btrim(p_registration_id),
    desired.registration_day_id,
    desired.event_option_id,
    upper(coalesce(nullif(btrim(desired.partner_mode), ''), 'NONE')),
    nullif(btrim(desired.partner_name), ''),
    nullif(lower(btrim(desired.partner_email)), ''),
    nullif(btrim(desired.partner_phone), ''),
    nullif(btrim(desired.partner_dupr_id), ''),
    desired.partner_skill,
    desired.partner_age,
    nullif(btrim(desired.partner_note), ''),
    coalesce(desired.show_on_partner_board, false),
    desired.sort_order,
    coalesce(desired.created_at, pg_catalog.clock_timestamp()),
    pg_catalog.clock_timestamp()
  from jsonb_to_recordset(p_selections) as desired(
    id text,
    registration_day_id text,
    event_option_id text,
    partner_mode text,
    partner_name text,
    partner_email text,
    partner_phone text,
    partner_dupr_id text,
    partner_skill numeric,
    partner_age integer,
    partner_note text,
    show_on_partner_board boolean,
    sort_order integer,
    created_at timestamptz
  )
  where not exists (
    select 1
    from public.tournament_registration_selections as existing
    where existing.id = desired.id
  );

  v_updated_at := greatest(
    pg_catalog.clock_timestamp(),
    v_registration.updated_at + interval '1 microsecond'
  );

  update public.tournament_registrations as registration
  set
    first_name = nullif(btrim(p_registration_patch ->> 'first_name'), ''),
    last_name = nullif(btrim(p_registration_patch ->> 'last_name'), ''),
    display_name = nullif(btrim(p_registration_patch ->> 'display_name'), ''),
    phone = nullif(btrim(p_registration_patch ->> 'phone'), ''),
    dupr_id = nullif(btrim(p_registration_patch ->> 'dupr_id'), ''),
    doubles_skill = (p_registration_patch ->> 'doubles_skill')::numeric,
    singles_skill = (p_registration_patch ->> 'singles_skill')::numeric,
    age = (p_registration_patch ->> 'age')::integer,
    age_bracket = nullif(btrim(p_registration_patch ->> 'age_bracket'), ''),
    gender = nullif(btrim(p_registration_patch ->> 'gender'), ''),
    notes = nullif(btrim(p_registration_patch ->> 'notes'), ''),
    wants_partner_board_contact = coalesce(
      (p_registration_patch ->> 'wants_partner_board_contact')::boolean,
      false
    ),
    updated_at = v_updated_at
  where registration.tournament_id = btrim(p_tournament_id)
    and registration.id = btrim(p_registration_id)
    and registration.updated_at = p_expected_updated_at;

  if not found then
    raise exception using
      errcode = '40001',
      message = 'JUPR_REGISTRATION_EDIT_CONFLICT: registration version changed during the atomic edit.';
  end if;

  select count(*)
  into v_selection_count
  from public.tournament_registration_selections as selection
  where selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id);

  return jsonb_build_object(
    'ok', true,
    'registration_id', btrim(p_registration_id),
    'updated_at', v_updated_at,
    'selection_count', v_selection_count
  );
end
$function$;

revoke all on function public.server_update_public_tournament_registration_edit(
  text,
  text,
  timestamptz,
  jsonb,
  jsonb,
  jsonb
) from public, anon, authenticated;

grant execute on function public.server_update_public_tournament_registration_edit(
  text,
  text,
  timestamptz,
  jsonb,
  jsonb,
  jsonb
) to service_role;

notify pgrst, 'reload schema';
