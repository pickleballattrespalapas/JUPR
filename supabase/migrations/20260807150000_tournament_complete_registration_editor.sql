-- Explicit tournament skill eligibility boundaries and complete partner data.
-- Forward-only. Existing application guards and service-role-only RPC access are
-- preserved while the selection editor gains the missing partner gender field.

do $migration_preflight$
begin
  if to_regclass('public.tournament_event_options') is null
     or to_regclass('public.tournament_registration_selections') is null
     or to_regclass('public.tournament_registration_days') is null
     or to_regclass('public.tournament_registration_partner_requests') is null
     or to_regclass('public.tournament_registration_team_links') is null
     or to_regclass('public.tournament_registration_team_members') is null
     or to_regclass('public.tournament_teams') is null
     or to_regprocedure(
       'public.admin_update_tournament_registration_selection(text,text,timestamp with time zone,jsonb)'
     ) is null
     or to_regprocedure(
       'public.server_update_public_tournament_registration_edit(text,text,timestamp with time zone,jsonb,jsonb,jsonb)'
     ) is null
     or to_regprocedure(
       'public.server_create_public_tournament_registration_with_commerce(text,uuid,jsonb,jsonb,jsonb,uuid,uuid,text,text,text)'
     ) is null
     or to_regprocedure(
       'private.lock_tournament_registration_selection_scope(text[])'
     ) is null then
    raise exception using
      errcode = '55000',
      message = 'JUPR_TOURNAMENT_REGISTRATION_EDITOR_BASE_MISSING';
  end if;
end
$migration_preflight$;

alter table public.tournament_event_options
  add column if not exists skill_min_rating numeric(4,2) null,
  add column if not exists skill_max_rating numeric(4,2) null;

comment on column public.tournament_event_options.skill_min_rating is
  'Inclusive individual-rating floor for MINIMUM and CUSTOM eligibility.';
comment on column public.tournament_event_options.skill_max_rating is
  'Exclusive individual-rating ceiling for CUSTOM eligibility.';

alter table public.tournament_registration_selections
  add column if not exists partner_gender text null;

comment on column public.tournament_registration_selections.partner_gender is
  'Manual partner gender supplied with a tournament registration selection.';

-- The legacy eligibility check does not admit MINIMUM or CUSTOM. Remove the
-- old cross-field checks before backfilling new mode values. The migration is
-- transactional, so a failed backfill restores the original constraints.
alter table public.tournament_event_options
  drop constraint if exists tournament_event_options_eligibility_mode_chk,
  drop constraint if exists tournament_event_options_combined_cap_chk,
  drop constraint if exists tournament_event_options_skill_bounds_chk,
  drop constraint if exists tournament_event_options_team_contract_chk;

-- Preserve legacy intent before the new cross-field checks are installed.
-- A numeric label remains a Standard ceiling even when an old row retained
-- skill_mode=OPEN. Bare N+ labels historically meant open-ended play unless
-- skill_mode explicitly recorded MINIMUM.
do $legacy_skill_policy$
begin
  if exists (
    select 1
    from public.tournament_event_options event
    where event.eligibility_mode = 'STANDARD'
      and upper(coalesce(event.skill_mode, '')) in ('MINIMUM', 'MIN', 'AT_LEAST')
      and btrim(coalesce(event.skill_label, '')) !~* '^(?:skill\s*)?[0-9](?:\.[0-9]{1,2})?\s*\+?$'
  ) then
    raise exception using
      errcode = '23514',
      message = 'JUPR_TOURNAMENT_MINIMUM_SKILL_BACKFILL_INVALID';
  end if;

  update public.tournament_event_options event
  set
    eligibility_mode = 'MINIMUM',
    skill_min_rating = (
      regexp_match(
        btrim(coalesce(event.skill_label, '')),
        '^(?:skill\s*)?([0-9](?:\.[0-9]{1,2})?)\s*\+?$',
        'i'
      )
    )[1]::numeric,
    skill_max_rating = null,
    combined_rating_cap = null
  where event.eligibility_mode = 'STANDARD'
    and upper(coalesce(event.skill_mode, '')) in ('MINIMUM', 'MIN', 'AT_LEAST');

  update public.tournament_event_options event
  set
    eligibility_mode = 'OPEN',
    skill_min_rating = null,
    skill_max_rating = null,
    combined_rating_cap = null
  where event.eligibility_mode = 'STANDARD'
    and (
      lower(btrim(coalesce(event.skill_label, ''))) = 'open'
      or btrim(coalesce(event.skill_label, '')) ~ '\+\s*$'
      or (
        upper(coalesce(event.skill_mode, '')) in ('OPEN', 'NONE')
        and btrim(coalesce(event.skill_label, ''))
          !~* '^(?:skill\s*)?[0-9](?:\.[0-9]{1,2})?$'
      )
    );
end
$legacy_skill_policy$;

-- Install the five-mode checks. NOT VALID keeps the add-constraint lock short;
-- validation below verifies all pre-existing rows before commit.
alter table public.tournament_event_options
  add constraint tournament_event_options_eligibility_mode_chk
    check (
      eligibility_mode in (
        'STANDARD',
        'MINIMUM',
        'OPEN',
        'COMBINED_RATING_CAP',
        'CUSTOM'
      )
    ) not valid,
  add constraint tournament_event_options_combined_cap_chk
    check (
      (
        eligibility_mode = 'COMBINED_RATING_CAP'
        and combined_rating_cap is not null
        and combined_rating_cap > 0
        and combined_rating_cap <= 14
      )
      or
      (
        eligibility_mode <> 'COMBINED_RATING_CAP'
        and combined_rating_cap is null
      )
    ) not valid,
  add constraint tournament_event_options_skill_bounds_chk
    check (
      (
        eligibility_mode in ('STANDARD', 'OPEN', 'COMBINED_RATING_CAP')
        and skill_min_rating is null
        and skill_max_rating is null
      )
      or
      (
        eligibility_mode = 'MINIMUM'
        and skill_min_rating is not null
        and skill_min_rating between 1 and 7
        and skill_max_rating is null
      )
      or
      (
        eligibility_mode = 'CUSTOM'
        and combined_rating_cap is null
        and (skill_min_rating is not null or skill_max_rating is not null)
        and (skill_min_rating is null or skill_min_rating between 1 and 7)
        and (
          skill_max_rating is null
          or (skill_max_rating > 1 and skill_max_rating <= 7.5)
        )
        and (
          skill_min_rating is null
          or skill_max_rating is null
          or skill_min_rating < skill_max_rating
        )
      )
    ) not valid,
  add constraint tournament_event_options_team_contract_chk
    check (
      (
        competition_format = 'STANDARD'
        and team_roster_size = 2
        and team_gender_rule = 'NONE'
        and team_playoff_format = 'NONE'
      )
      or
      (
        competition_format = 'FOUR_PLAYER_TEAM'
        -- Preserve the existing four-player eligibility contract. Its durable
        -- review engine is separate from pair/singles Division eligibility.
        and eligibility_mode = 'STANDARD'
        and team_roster_size = 4
        and team_gender_rule = 'TWO_MEN_TWO_WOMEN'
        and team_tiebreak_mode in ('SINGLES', 'SKINNY_RELAY')
        and team_playoff_format in (
          'NONE',
          'TOP_2_FINAL',
          'TOP_4_SEMIFINALS',
          'TOP_4_SEMIFINALS_WITH_BRONZE'
        )
      )
    ) not valid;

alter table public.tournament_event_options
  validate constraint tournament_event_options_eligibility_mode_chk;
alter table public.tournament_event_options
  validate constraint tournament_event_options_combined_cap_chk;
alter table public.tournament_event_options
  validate constraint tournament_event_options_skill_bounds_chk;
alter table public.tournament_event_options
  validate constraint tournament_event_options_team_contract_chk;

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
    'partner_gender',
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
    or v_candidate.partner_mode is distinct from v_before.partner_mode
    or v_candidate.partner_name is distinct from v_before.partner_name
    or v_candidate.partner_email is distinct from v_before.partner_email
    or v_candidate.partner_phone is distinct from v_before.partner_phone
    or v_candidate.partner_dupr_id is distinct from v_before.partner_dupr_id
    or v_candidate.partner_skill is distinct from v_before.partner_skill
    or v_candidate.partner_age is distinct from v_before.partner_age
    or v_candidate.partner_gender is distinct from v_before.partner_gender;

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
     and not exists (
       select 1
       from public.tournament_registration_team_links as team_link
       where team_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
         and (
           team_link.selection1_id = v_before.id
           or team_link.selection2_id = v_before.id
         )
     )
     and (
       nullif(btrim(v_candidate.partner_name), '') is null
       or nullif(btrim(v_candidate.partner_email), '') is null
       or v_candidate.partner_age is null
       or nullif(btrim(v_candidate.partner_gender), '') is null
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: manual partner name, email, age, and gender are required.';
  end if;

  if v_candidate.partner_mode in ('NONE', 'NEEDS_PARTNER') then
    v_candidate.partner_name := null;
    v_candidate.partner_email := null;
    v_candidate.partner_phone := null;
    v_candidate.partner_dupr_id := null;
    v_candidate.partner_skill := null;
    v_candidate.partner_age := null;
    v_candidate.partner_gender := null;
  end if;
  if v_candidate.partner_mode <> 'NEEDS_PARTNER' then
    v_candidate.show_on_partner_board := false;
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
    partner_gender = v_candidate.partner_gender,
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

create or replace function public.admin_delete_tournament_registration_selection(
  p_tournament_id text,
  p_selection_id text,
  p_expected_updated_at timestamptz
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_before public.tournament_registration_selections%rowtype;
begin
  if nullif(btrim(p_tournament_id), '') is null
     or nullif(btrim(p_selection_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: tournament and selection IDs are required.';
  end if;

  if not exists (
    select 1
    from public.tournament_registration_selections as selection
    where selection.id = btrim(p_selection_id)
      and selection.tournament_id = btrim(p_tournament_id)
  ) then
    return jsonb_build_object('ok', false, 'code', 'SELECTION_NOT_FOUND');
  end if;

  begin
    perform private.lock_tournament_registration_selection_scope(
      array[btrim(p_selection_id)]
    );
  exception
    when others then
      if sqlerrm like 'JUPR_RELATION_SELECTION_NOT_FOUND:%' then
        return jsonb_build_object('ok', false, 'code', 'SELECTION_NOT_FOUND');
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
    return jsonb_build_object('ok', false, 'code', 'SELECTION_NOT_FOUND');
  end if;

  if p_expected_updated_at is null
     or v_before.updated_at is distinct from p_expected_updated_at then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'STALE_VERSION'
    );
  end if;

  if exists (
    select 1
    from public.tournament_registration_partner_requests as request
    where request.status in ('PENDING', 'ACCEPTED', 'ADMIN_CONFIRMED')
      and (
        request.requester_selection_id = v_before.id
        or request.target_selection_id = v_before.id
      )
  ) or exists (
    select 1
    from public.tournament_registration_team_links as team_link
    where team_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
      and (
        team_link.selection1_id = v_before.id
        or team_link.selection2_id = v_before.id
      )
  ) or exists (
    select 1
    from public.tournament_registration_team_members as team_member
    where team_member.status = 'ACTIVE'
      and team_member.selection_id = v_before.id
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_RELATIONSHIP_LOCKED'
    );
  end if;

  if exists (
    select 1
    from public.tournament_teams as team
    where team.tournament_id::text = btrim(p_tournament_id)
      and team.registration_day_id = v_before.registration_day_id
      and team.event_option_id = v_before.event_option_id
      and upper(coalesce(team.source, '')) = 'REGISTRATION'
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_IMPORTED_TO_DRAW'
    );
  end if;

  delete from public.tournament_registration_selections as selection
  where selection.id = v_before.id
    and selection.tournament_id = btrim(p_tournament_id)
    and selection.updated_at = p_expected_updated_at;

  if not found then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'STALE_VERSION'
    );
  end if;

  return jsonb_build_object(
    'ok', true,
    'selection', to_jsonb(v_before)
  );
end
$function$;

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

revoke all on function public.admin_delete_tournament_registration_selection(
  text,
  text,
  timestamptz
) from public, anon, authenticated;
grant execute on function public.admin_delete_tournament_registration_selection(
  text,
  text,
  timestamptz
) to service_role;

-- Keep public edit/create persistence in lockstep with the new durable
-- partner_gender column. These are explicit replacements of the latest
-- guarded functions; their locking, CAS, relationship, idempotency, commerce,
-- and service-role-only contracts remain unchanged.
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
    'partner_gender',
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
    partner_gender = nullif(btrim(desired.partner_gender), ''),
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
    partner_gender text,
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
    partner_gender,
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
    nullif(btrim(desired.partner_gender), ''),
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
    partner_gender text,
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


create or replace function public.server_create_public_tournament_registration_with_commerce(
  p_club_id text,
  p_tournament_id uuid,
  p_registration jsonb,
  p_selections jsonb,
  p_quote_snapshot jsonb,
  p_operation_idempotency_key uuid,
  p_order_idempotency_key uuid,
  p_request_fingerprint text,
  p_actor_label text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.tournament_commerce_operations%rowtype;
  v_tournament public.tournaments%rowtype;
  v_registration_id text;
  v_submitted_at timestamptz;
  v_selection_count integer;
  v_order_result jsonb;
  v_result jsonb;
begin
  select tournament.* into v_tournament
  from public.tournaments tournament
  where tournament.id = p_tournament_id
    and tournament.club_id = p_club_id
  for update;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_TOURNAMENT_NOT_FOUND';
  end if;

  v_operation := private.tournament_commerce_operation_begin(
    p_club_id, p_tournament_id, p_operation_idempotency_key,
    p_request_fingerprint, 'REGISTRATION_CREATE_WITH_COMMERCE',
    'PUBLIC_REGISTRANT', p_actor_label, p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;

  v_registration_id := nullif(btrim(p_registration ->> 'id'), '');
  if v_registration_id is null
     or p_registration is null
     or jsonb_typeof(p_registration) <> 'object'
     or p_selections is null
     or jsonb_typeof(p_selections) <> 'array'
     or jsonb_array_length(p_selections) = 0
     or p_quote_snapshot is null
     or jsonb_typeof(p_quote_snapshot) <> 'object'
     or nullif(btrim(p_registration ->> 'tournament_id'), '') <> p_tournament_id::text
     or nullif(lower(btrim(p_registration ->> 'email')), '') is null
     or nullif(btrim(p_registration ->> 'display_name'), '') is null then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_INVALID';
  end if;

  if exists (
    select 1
    from public.tournament_registrations registration
    where registration.tournament_id = p_tournament_id::text
      and lower(btrim(registration.email))
          = lower(btrim(p_registration ->> 'email'))
  ) then
    raise exception using errcode = '23505', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_DUPLICATE';
  end if;

  if (
    select coalesce(array_agg(distinct event_id order by event_id), array[]::text[])
    from jsonb_array_elements_text(
      coalesce(p_quote_snapshot -> 'request' -> 'event_option_ids', '[]'::jsonb)
    ) as quote_event(event_id)
  ) is distinct from (
    select coalesce(
      array_agg(distinct btrim(selection.event_option_id) order by btrim(selection.event_option_id)),
      array[]::text[]
    )
    from jsonb_to_recordset(p_selections) selection(event_option_id text)
  ) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_EVENT_MISMATCH';
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) selection(
      id text,
      tournament_id text,
      registration_id text,
      registration_day_id text,
      event_option_id text,
      partner_mode text
    )
    left join public.tournament_registration_days day
      on day.id = selection.registration_day_id
     and day.tournament_id = p_tournament_id::text
    left join public.tournament_event_options event
      on event.id = selection.event_option_id
     and event.tournament_id = p_tournament_id::text
     and event.registration_day_id = selection.registration_day_id
    where nullif(btrim(selection.id), '') is null
       or selection.tournament_id <> p_tournament_id::text
       or selection.registration_id <> v_registration_id
       or day.id is null
       or event.id is null
       or not coalesce(day.enabled, false)
       or not coalesce(event.enabled, false)
       or lower(coalesce(nullif(btrim(event.status), ''), 'draft'))
          not in ('open', 'tentative', 'confirmed')
       or upper(coalesce(nullif(btrim(selection.partner_mode), ''), 'NONE'))
          not in ('NONE', 'HAS_PARTNER', 'NEEDS_PARTNER')
  ) or (
    select count(*)
    from jsonb_to_recordset(p_selections) selection(id text)
  ) <> (
    select count(distinct btrim(selection.id))
    from jsonb_to_recordset(p_selections) selection(id text)
  ) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_SELECTION_INVALID';
  end if;

  v_submitted_at := pg_catalog.clock_timestamp();
  insert into public.tournament_registrations (
    id, tournament_id, submitted_at, updated_at, status, payment_status,
    first_name, last_name, display_name, email, phone, player_id, dupr_id,
    doubles_skill, singles_skill, age, age_bracket, gender, notes,
    wants_partner_board_contact
  )
  values (
    v_registration_id,
    p_tournament_id::text,
    v_submitted_at,
    v_submitted_at,
    lower(coalesce(nullif(btrim(p_registration ->> 'status'), ''), 'confirmed')),
    lower(coalesce(nullif(btrim(p_registration ->> 'payment_status'), ''), 'unpaid')),
    nullif(btrim(p_registration ->> 'first_name'), ''),
    nullif(btrim(p_registration ->> 'last_name'), ''),
    btrim(p_registration ->> 'display_name'),
    lower(btrim(p_registration ->> 'email')),
    nullif(btrim(p_registration ->> 'phone'), ''),
    nullif(p_registration ->> 'player_id', '')::integer,
    nullif(btrim(p_registration ->> 'dupr_id'), ''),
    nullif(p_registration ->> 'doubles_skill', '')::numeric,
    nullif(p_registration ->> 'singles_skill', '')::numeric,
    nullif(p_registration ->> 'age', '')::integer,
    nullif(btrim(p_registration ->> 'age_bracket'), ''),
    nullif(btrim(p_registration ->> 'gender'), ''),
    nullif(btrim(p_registration ->> 'notes'), ''),
    coalesce((p_registration ->> 'wants_partner_board_contact')::boolean, false)
  );

  insert into public.tournament_registration_selections (
    id, tournament_id, registration_id, registration_day_id,
    event_option_id, partner_mode, partner_name, partner_email,
    partner_phone, partner_dupr_id, partner_skill, partner_age,
    partner_gender, partner_note, show_on_partner_board, sort_order, created_at, updated_at
  )
  select
    selection.id,
    p_tournament_id::text,
    v_registration_id,
    selection.registration_day_id,
    selection.event_option_id,
    upper(coalesce(nullif(btrim(selection.partner_mode), ''), 'NONE')),
    nullif(btrim(selection.partner_name), ''),
    nullif(lower(btrim(selection.partner_email)), ''),
    nullif(btrim(selection.partner_phone), ''),
    nullif(btrim(selection.partner_dupr_id), ''),
    selection.partner_skill,
    selection.partner_age,
    nullif(btrim(selection.partner_gender), ''),
    nullif(btrim(selection.partner_note), ''),
    coalesce(selection.show_on_partner_board, false),
    selection.sort_order,
    v_submitted_at,
    v_submitted_at
  from jsonb_to_recordset(p_selections) selection(
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
    partner_gender text,
    partner_note text,
    show_on_partner_board boolean,
    sort_order integer
  );
  get diagnostics v_selection_count = row_count;

  v_order_result := public.server_apply_tournament_commerce_order(
    p_club_id,
    p_tournament_id,
    v_registration_id,
    null,
    p_quote_snapshot,
    p_order_idempotency_key,
    p_quote_snapshot ->> 'request_fingerprint',
    'PUBLIC_REGISTRANT',
    p_actor_label,
    p_source
  );

  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'registration_create_with_commerce',
    'registration_id', v_registration_id,
    'submitted_at', v_submitted_at,
    'updated_at', v_submitted_at,
    'selection_count', v_selection_count,
    'commerce_order', v_order_result,
    'operation_id', v_operation.id,
    'idempotent_replay', false
  );
  update public.tournament_commerce_operations
  set order_id = nullif(v_order_result ->> 'order_id', '')::uuid,
      status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  )
  values (
    p_club_id, p_tournament_id,
    nullif(v_order_result ->> 'order_id', '')::uuid, v_operation.id,
    'PUBLIC_REGISTRANT', p_actor_label,
    'REGISTRATION_CREATE_WITH_COMMERCE', null, v_result, p_source
  );
  return v_result;
end
$function$;


revoke all on function public.server_create_public_tournament_registration_with_commerce(
  text,
  uuid,
  jsonb,
  jsonb,
  jsonb,
  uuid,
  uuid,
  text,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.server_create_public_tournament_registration_with_commerce(
  text,
  uuid,
  jsonb,
  jsonb,
  jsonb,
  uuid,
  uuid,
  text,
  text,
  text
) to service_role;

notify pgrst, 'reload schema';
