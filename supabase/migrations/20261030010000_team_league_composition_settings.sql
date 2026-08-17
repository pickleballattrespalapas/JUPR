-- Team-league roster composition policy for the supported two-player model.
-- Larger rosters, alternates, and shared substitute pools require normalized
-- team-membership records and intentionally are not advertised or persisted.

alter table public.team_league_settings
  add column if not exists team_size smallint not null default 2,
  add column if not exists team_category text not null default 'open';

do $constraints$
begin
  if not exists (
    select 1 from pg_catalog.pg_constraint
     where conname = 'team_league_settings_team_size_check'
       and conrelid = 'public.team_league_settings'::regclass
  ) then
    alter table public.team_league_settings
      add constraint team_league_settings_team_size_check
      check (team_size = 2);
  end if;

  if not exists (
    select 1 from pg_catalog.pg_constraint
     where conname = 'team_league_settings_team_category_check'
       and conrelid = 'public.team_league_settings'::regclass
  ) then
    alter table public.team_league_settings
      add constraint team_league_settings_team_category_check
      check (team_category in ('open', 'mens', 'womens', 'mixed'));
  end if;

end
$constraints$;

-- Keep all settings in the existing version-checked, idempotent transaction.
create or replace function public.team_league_save_settings_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_expected_settings_version integer,
  p_settings jsonb,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text := nullif(pg_catalog.left(pg_catalog.btrim(p_league_name), 120), '');
  v_key text := nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_fingerprint text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_request_fingerprint, '')));
  v_actor_email text := nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), 320), '');
  v_actor_role text := coalesce(
    nullif(pg_catalog.left(pg_catalog.lower(pg_catalog.btrim(p_actor_role)), 80), ''),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_settings'
  );
  v_operation public.team_league_operations%rowtype;
  v_before public.team_league_settings%rowtype;
  v_updated public.team_league_settings%rowtype;
  v_current_version integer := 0;
  v_next_status text;
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_league_name is null
     or v_key is null
     or v_key !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$'
     or v_fingerprint !~ '^[0-9a-f]{64}$'
     or p_expected_settings_version is null
     or p_expected_settings_version < 0
     or p_settings is null
     or pg_catalog.jsonb_typeof(p_settings) <> 'object'
     or coalesce(nullif(p_settings ->> 'team_size', '')::smallint, 2) <> 2 then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SETTINGS_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-admin:' || v_club_id || ':' || v_key,
      0
    )
  );

  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.club_id = v_club_id
     and operation.idempotency_key = v_key
   for update;
  if found then
    if v_operation.request_fingerprint <> v_fingerprint then
      raise exception using
        errcode = '23505',
        message = 'TEAM_LEAGUE_IDEMPOTENCY_CONFLICT';
    end if;
    if v_operation.status = 'complete' and v_operation.result_json is not null then
      return v_operation.result_json || '{"idempotent": true}'::jsonb;
    end if;
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_OPERATION_RECOVERY_REQUIRED';
  end if;

  if not exists (
    select 1
      from public.leagues_metadata as league
     where league.club_id = v_club_id
       and league.league_name = v_league_name
  ) then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_LEAGUE_NOT_FOUND';
  end if;

  select settings.*
    into v_before
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if found then
    v_current_version := v_before.settings_version;
  end if;
  if v_current_version <> p_expected_settings_version then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_SETTINGS_VERSION_CONFLICT';
  end if;

  v_next_status := case
    when coalesce((p_settings ->> 'registration_open')::boolean, false)
      then 'registration_open'
    when v_before.status = 'registration_open'
      then 'registration_closed'
    else coalesce(v_before.status, 'draft')
  end;

  insert into public.team_league_operations (
    id,
    club_id,
    league_name,
    idempotency_key,
    request_fingerprint,
    operation_type,
    status,
    request_json,
    actor_email,
    actor_role,
    source
  ) values (
    p_operation_id,
    v_club_id,
    v_league_name,
    v_key,
    v_fingerprint,
    'admin_save_settings',
    'started',
    pg_catalog.jsonb_build_object(
      'settings', p_settings,
      'expected_settings_version', p_expected_settings_version
    ),
    v_actor_email,
    v_actor_role,
    v_source
  );

  insert into public.team_league_settings (
    club_id,
    league_name,
    status,
    registration_open,
    team_size,
    team_category,
    allow_substitutes,
    playoff_format,
    playoff_team_count,
    start_date,
    weekday,
    start_time,
    timezone,
    venue,
    registration_closes_at,
    settings_version,
    created_by,
    updated_by,
    updated_at
  ) values (
    v_club_id,
    v_league_name,
    v_next_status,
    coalesce((p_settings ->> 'registration_open')::boolean, false),
    2,
    coalesce(nullif(p_settings ->> 'team_category', ''), 'open'),
    coalesce((p_settings ->> 'allow_substitutes')::boolean, false),
    coalesce(nullif(p_settings ->> 'playoff_format', ''), 'none'),
    nullif(p_settings ->> 'playoff_team_count', '')::integer,
    nullif(p_settings ->> 'start_date', '')::date,
    coalesce(nullif(p_settings ->> 'weekday', '')::smallint, 0),
    coalesce(nullif(p_settings ->> 'start_time', '')::time, '18:00'::time),
    coalesce(nullif(pg_catalog.left(p_settings ->> 'timezone', 80), ''), 'UTC'),
    nullif(pg_catalog.left(pg_catalog.btrim(p_settings ->> 'venue'), 240), ''),
    nullif(p_settings ->> 'registration_closes_at', '')::timestamptz,
    p_expected_settings_version + 1,
    coalesce(v_before.created_by, v_actor_email),
    v_actor_email,
    pg_catalog.clock_timestamp()
  )
  on conflict (club_id, league_name) do update
    set status = excluded.status,
        registration_open = excluded.registration_open,
        team_size = excluded.team_size,
        team_category = excluded.team_category,
        allow_substitutes = excluded.allow_substitutes,
        playoff_format = excluded.playoff_format,
        playoff_team_count = excluded.playoff_team_count,
        start_date = excluded.start_date,
        weekday = excluded.weekday,
        start_time = excluded.start_time,
        timezone = excluded.timezone,
        venue = excluded.venue,
        registration_closes_at = excluded.registration_closes_at,
        settings_version = excluded.settings_version,
        updated_by = excluded.updated_by,
        updated_at = excluded.updated_at
  returning * into v_updated;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'league_name', v_league_name,
    'settings', pg_catalog.to_jsonb(v_updated),
    'settings_version', v_updated.settings_version,
    'message', 'Team-league setup saved.',
    'idempotent', false
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
    v_club_id,
    coalesce(v_actor_email, 'unknown'),
    v_actor_role,
    'team_league_settings_saved',
    'team_league',
    v_league_name,
    coalesce(pg_catalog.to_jsonb(v_before), '{}'::jsonb),
    pg_catalog.to_jsonb(v_updated),
    null,
    v_source,
    false
  );

  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

create or replace function public.team_league_guard_settings_after_schedule_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if old.schedule_version > 0 and (
    (not old.registration_open and new.registration_open)
    or old.start_date is distinct from new.start_date
    or old.weekday is distinct from new.weekday
    or old.start_time is distinct from new.start_time
    or old.timezone is distinct from new.timezone
    or old.venue is distinct from new.venue
    or old.registration_closes_at is distinct from new.registration_closes_at
    or old.team_size is distinct from new.team_size
    or old.team_category is distinct from new.team_category
    or old.allow_substitutes is distinct from new.allow_substitutes
    or old.playoff_format is distinct from new.playoff_format
    or old.playoff_team_count is distinct from new.playoff_team_count
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_STRUCTURAL_SETTINGS_LOCKED_AFTER_SCHEDULE';
  end if;
  return new;
end
$function$;

revoke all on function public.team_league_save_settings_v1(
  uuid, text, text, text, text, integer, jsonb, text, text, text
) from public, anon, authenticated;
grant execute on function public.team_league_save_settings_v1(
  uuid, text, text, text, text, integer, jsonb, text, text, text
) to service_role;

comment on column public.team_league_settings.team_size is
  'Fixed at two primary players until normalized multi-member team rosters are supported.';
comment on column public.team_league_settings.team_category is
  'Team roster category policy: open, mens, womens, or mixed.';

notify pgrst, 'reload schema';
