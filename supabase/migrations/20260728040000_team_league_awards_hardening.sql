-- Forward-only hardening for fixed-partner team leagues and league awards.
--
-- This migration closes lifecycle, roster, schedule, score, reconciliation,
-- recovery, and award-workflow races discovered during pre-staging review.

create extension if not exists pgcrypto;

alter table public.team_league_settings
  add column if not exists roster_version integer not null default 0;

alter table public.team_league_fixtures
  add column if not exists score_reserved_at timestamptz;

alter table public.leagues_metadata
  add column if not exists awards_config_version integer not null default 0;

do $constraints$
begin
  if not exists (
    select 1
      from pg_catalog.pg_constraint
     where conname = 'team_league_settings_roster_version_check'
       and conrelid = 'public.team_league_settings'::regclass
  ) then
    alter table public.team_league_settings
      add constraint team_league_settings_roster_version_check
      check (roster_version >= 0);
  end if;
  if not exists (
    select 1
      from pg_catalog.pg_constraint
     where conname = 'leagues_metadata_awards_config_version_check'
       and conrelid = 'public.leagues_metadata'::regclass
  ) then
    alter table public.leagues_metadata
      add constraint leagues_metadata_awards_config_version_check
      check (awards_config_version >= 0);
  end if;
end
$constraints$;

create or replace function public.league_awards_save_config_v1(
  p_club_id text,
  p_league_name text,
  p_expected_config_version integer,
  p_awards_config jsonb,
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
  v_league_name text := nullif(
    pg_catalog.left(pg_catalog.btrim(p_league_name), 120),
    ''
  );
  v_actor_email text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
        320
      ),
      ''
    ),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_role)),
        80
      ),
      ''
    ),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_league_manager_awards_config'
  );
  v_before public.leagues_metadata%rowtype;
  v_updated public.leagues_metadata%rowtype;
  v_status text;
  v_workflow_status text;
begin
  if v_club_id is null
     or v_league_name is null
     or p_expected_config_version is null
     or p_expected_config_version < 0
     or p_awards_config is null
     or pg_catalog.jsonb_typeof(p_awards_config) <> 'object'
     or pg_catalog.octet_length(p_awards_config::text) > 50000 then
    raise exception using
      errcode = '22023',
      message = 'LEAGUE_AWARD_CONFIG_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:league-awards:' || v_club_id || ':' || v_league_name,
      0
    )
  );
  select league.*
    into v_before
    from public.leagues_metadata as league
   where league.club_id = v_club_id
     and league.league_name = v_league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'LEAGUE_AWARD_LEAGUE_NOT_FOUND';
  end if;

  v_status := pg_catalog.lower(
    coalesce(
      nullif(pg_catalog.btrim(v_before.status), ''),
      case when v_before.is_active then 'active' else 'draft' end
    )
  );
  v_workflow_status := coalesce(
    nullif(v_before.end_awards::jsonb #>> '{workflow,status}', ''),
    'not_started'
  );
  if v_status not in ('draft', 'active', 'paused')
     or v_workflow_status <> 'not_started'
     or exists (
       select 1
         from public.league_award_result_sets as result_set
        where result_set.club_id = v_club_id
          and result_set.league_name = v_league_name
     ) then
    raise exception using
      errcode = '55000',
      message = 'LEAGUE_AWARD_CONFIG_LOCKED_AFTER_FREEZE';
  end if;

  if v_before.awards_config_version = p_expected_config_version + 1
     and coalesce(v_before.awards_config::jsonb, '{}'::jsonb)
       = p_awards_config then
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'committed', true,
      'idempotent', true,
      'awards_config_version', v_before.awards_config_version,
      'league', pg_catalog.to_jsonb(v_before)
    );
  end if;
  if v_before.awards_config_version <> p_expected_config_version then
    raise exception using
      errcode = '40001',
      message = 'LEAGUE_AWARD_CONFIG_VERSION_CONFLICT';
  end if;

  update public.leagues_metadata
     set awards_config = p_awards_config,
         awards_config_version = awards_config_version + 1,
         updated_at = pg_catalog.clock_timestamp()
   where club_id = v_club_id
     and league_name = v_league_name
  returning * into v_updated;

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
    v_actor_email,
    v_actor_role,
    'league_award_config_saved',
    'league_awards',
    v_league_name,
    pg_catalog.jsonb_build_object(
      'awards_config_version', v_before.awards_config_version,
      'awards_config',
        coalesce(v_before.awards_config::jsonb, '{}'::jsonb)
    ),
    pg_catalog.jsonb_build_object(
      'awards_config_version', v_updated.awards_config_version,
      'awards_config', p_awards_config
    ),
    null,
    v_source,
    false
  );
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'idempotent', false,
    'awards_config_version', v_updated.awards_config_version,
    'league', pg_catalog.to_jsonb(v_updated)
  );
end
$function$;

create or replace function public.team_league_confirmed_roster_fingerprint_v1(
  p_club_id text,
  p_league_name text
)
returns text
language sql
stable
security invoker
set search_path = ''
as $function$
  select pg_catalog.encode(
    extensions.digest(
      coalesce(
        pg_catalog.string_agg(
          team.id::text || ':' ||
          team.captain_player_id::text || ':' ||
          team.partner_player_id::text,
          '|' order by team.id::text
        ),
        ''
      ),
      'sha256'
    ),
    'hex'
  )
  from public.team_league_teams as team
  where team.club_id = pg_catalog.btrim(p_club_id)
    and team.league_name = pg_catalog.btrim(p_league_name)
    and team.status = 'confirmed'
$function$;

create or replace function public.team_league_guard_roster_mutation_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := case when tg_op = 'DELETE' then old.club_id else new.club_id end;
  v_league_name text := case when tg_op = 'DELETE' then old.league_name else new.league_name end;
  v_settings public.team_league_settings%rowtype;
  v_becoming_confirmed boolean := (
    tg_op <> 'DELETE'
    and new.status = 'confirmed'
    and (tg_op = 'INSERT' or old.status is distinct from 'confirmed')
  );
  v_new_registration boolean := (
    tg_op = 'INSERT'
    and new.status in ('pending_partner', 'confirmed')
  );
  v_roster_changed boolean := (
    (tg_op = 'INSERT' and new.status = 'confirmed')
    or (tg_op = 'DELETE' and old.status = 'confirmed')
    or (
      tg_op = 'UPDATE'
      and (old.status = 'confirmed' or new.status = 'confirmed')
      and (
        old.status is distinct from new.status
        or old.captain_player_id is distinct from new.captain_player_id
        or old.partner_player_id is distinct from new.partner_player_id
      )
    )
  );
begin
  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if not found then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;

  if (v_becoming_confirmed or v_new_registration) and (
    not v_settings.registration_open
    or v_settings.status <> 'registration_open'
    or (
      v_settings.registration_closes_at is not null
      and v_settings.registration_closes_at <= pg_catalog.clock_timestamp()
    )
    or v_settings.schedule_version <> 0
    or exists (
      select 1
        from public.team_league_fixtures as fixture
       where fixture.club_id = v_club_id
         and fixture.league_name = v_league_name
    )
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_REGISTRATION_CLOSED';
  end if;

  if v_roster_changed
     and v_settings.schedule_version > 0
     and not (
       tg_op = 'UPDATE'
       and old.status = new.status
       and old.captain_player_id = new.captain_player_id
       and old.partner_player_id = new.partner_player_id
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_ROSTER_LOCKED_AFTER_SCHEDULE';
  end if;
  return case when tg_op = 'DELETE' then old else new end;
end
$function$;

create or replace function public.league_awards_apply_workflow_v2(
  p_club_id text,
  p_league_name text,
  p_expected_workflow_revision integer,
  p_next_workflow jsonb,
  p_lifecycle_patch jsonb,
  p_preview_fingerprint text,
  p_result_fingerprint text,
  p_records jsonb,
  p_source_snapshot jsonb,
  p_finalized boolean,
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
  v_league_name text := nullif(
    pg_catalog.left(pg_catalog.btrim(p_league_name), 120),
    ''
  );
  v_actor_email text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
        320
      ),
      ''
    ),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_role)),
        80
      ),
      ''
    ),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_league_manager_awards'
  );
  v_before public.leagues_metadata%rowtype;
  v_updated public.leagues_metadata%rowtype;
  v_current_revision integer;
  v_next_revision integer;
  v_awards jsonb;
  v_end_awards jsonb;
  v_record_result jsonb;
begin
  if v_club_id is null
     or v_league_name is null
     or p_expected_workflow_revision is null
     or p_expected_workflow_revision < 0
     or p_next_workflow is null
     or pg_catalog.jsonb_typeof(p_next_workflow) <> 'object'
     or p_lifecycle_patch is null
     or pg_catalog.jsonb_typeof(p_lifecycle_patch) <> 'object'
     or exists (
       select 1
         from pg_catalog.jsonb_object_keys(p_lifecycle_patch) as item(key)
        where item.key not in (
          'is_active',
          'status',
          'ended_at',
          'ended_by'
        )
     ) then
    raise exception using
      errcode = '22023',
      message = 'LEAGUE_AWARD_WORKFLOW_INVALID';
  end if;
  begin
    v_next_revision := (p_next_workflow ->> 'revision')::integer;
  exception when others then
    raise exception using
      errcode = '22023',
      message = 'LEAGUE_AWARD_WORKFLOW_REVISION_INVALID';
  end;
  if v_next_revision <> p_expected_workflow_revision + 1 then
    raise exception using
      errcode = '22023',
      message = 'LEAGUE_AWARD_WORKFLOW_REVISION_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:league-awards:' || v_club_id || ':' || v_league_name,
      0
    )
  );
  select league.*
    into v_before
    from public.leagues_metadata as league
   where league.club_id = v_club_id
     and league.league_name = v_league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'LEAGUE_AWARD_LEAGUE_NOT_FOUND';
  end if;
  v_current_revision := coalesce(
    nullif(
      v_before.end_awards::jsonb #>> '{workflow,revision}',
      ''
    )::integer,
    0
  );

  if v_current_revision = v_next_revision then
    if v_before.end_awards::jsonb #> '{workflow}'
       is distinct from p_next_workflow then
      raise exception using
        errcode = '40001',
        message = 'LEAGUE_AWARD_WORKFLOW_REVISION_CONFLICT';
    end if;
    if not exists (
      select 1
        from public.league_award_result_sets as result_set
       where result_set.club_id = v_club_id
         and result_set.league_name = v_league_name
         and result_set.workflow_revision = v_next_revision
         and result_set.preview_fingerprint =
           pg_catalog.lower(pg_catalog.btrim(p_preview_fingerprint))
         and result_set.result_fingerprint =
           pg_catalog.lower(pg_catalog.btrim(p_result_fingerprint))
         and result_set.record_count =
           pg_catalog.jsonb_array_length(p_records)
         and (result_set.finalized_at is not null) = p_finalized
    ) then
      raise exception using
        errcode = '40001',
        message = 'LEAGUE_AWARD_ATOMIC_RESULT_EVIDENCE_MISSING';
    end if;
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'committed', true,
      'idempotent', true,
      'workflow_revision', v_next_revision,
      'record_count', pg_catalog.jsonb_array_length(p_records),
      'league', pg_catalog.to_jsonb(v_before)
    );
  end if;
  if v_current_revision <> p_expected_workflow_revision then
    raise exception using
      errcode = '40001',
      message = 'LEAGUE_AWARD_WORKFLOW_REVISION_CONFLICT';
  end if;

  v_awards := case
    when pg_catalog.jsonb_array_length(
      coalesce(p_next_workflow -> 'final_awards', '[]'::jsonb)
    ) > 0 then p_next_workflow -> 'final_awards'
    when pg_catalog.jsonb_array_length(
      coalesce(p_next_workflow #> '{preview,awards}', '[]'::jsonb)
    ) > 0 then p_next_workflow #> '{preview,awards}'
    else coalesce(
      p_next_workflow #> '{frozen_snapshot,awards}',
      '[]'::jsonb
    )
  end;
  v_end_awards := pg_catalog.jsonb_build_object(
    'schema_version',
      coalesce((p_next_workflow ->> 'version')::integer, 3),
    'top_performers', v_awards,
    'source', v_source,
    'generated_at', p_next_workflow #>> '{preview,generated_at}',
    'workflow', p_next_workflow
  );

  update public.leagues_metadata
     set end_awards = v_end_awards,
         is_active = case
           when p_lifecycle_patch ? 'is_active'
             then (p_lifecycle_patch ->> 'is_active')::boolean
           else is_active
         end,
         status = case
           when p_lifecycle_patch ? 'status'
             then nullif(p_lifecycle_patch ->> 'status', '')
           else status
         end,
         ended_at = case
           when p_lifecycle_patch ? 'ended_at'
             then nullif(p_lifecycle_patch ->> 'ended_at', '')::timestamptz
           else ended_at
         end,
         ended_by = case
           when p_lifecycle_patch ? 'ended_by'
             then nullif(
               pg_catalog.left(p_lifecycle_patch ->> 'ended_by', 320),
               ''
             )
           else ended_by
         end
   where club_id = v_club_id
     and league_name = v_league_name
  returning * into v_updated;

  select public.league_awards_replace_records_v1(
    v_club_id,
    v_league_name,
    v_next_revision,
    p_preview_fingerprint,
    p_result_fingerprint,
    p_records,
    p_source_snapshot,
    p_finalized,
    v_actor_email,
    v_actor_role,
    v_source
  ) into v_record_result;

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
    v_actor_email,
    v_actor_role,
    'league_award_workflow_committed',
    'league_awards',
    v_league_name,
    pg_catalog.jsonb_build_object(
      'workflow_revision', v_current_revision,
      'league_status', v_before.status
    ),
    pg_catalog.jsonb_build_object(
      'workflow_revision', v_next_revision,
      'league_status', v_updated.status,
      'record_count', pg_catalog.jsonb_array_length(p_records),
      'finalized', p_finalized
    ),
    null,
    v_source,
    false
  );
  return v_record_result || pg_catalog.jsonb_build_object(
    'league', pg_catalog.to_jsonb(v_updated),
    'workflow_revision', v_next_revision
  );
end
$function$;

create or replace function public.team_league_reconcile_fixture_v2(
  p_operation_id uuid,
  p_club_id text,
  p_fixture_id uuid,
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
  v_operation public.team_league_operations%rowtype;
  v_before public.team_league_fixtures%rowtype;
  v_updated public.team_league_fixtures%rowtype;
  v_match public.matches%rowtype;
  v_match_found boolean := false;
  v_normal_sides boolean := false;
  v_swapped_sides boolean := false;
  v_new_winner uuid;
  v_dependency_changed boolean := false;
  v_resolved_count integer := 0;
  v_actor_email text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
        320
      ),
      ''
    ),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_role)),
        80
      ),
      ''
    ),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_reconcile'
  );
  v_result jsonb;
begin
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found or v_operation.operation_type <> 'admin_reconcile_fixture' then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_RECONCILE_OPERATION_NOT_FOUND';
  end if;
  if v_operation.status = 'complete'
     and v_operation.result_json is not null then
    return v_operation.result_json || '{"idempotent": true}'::jsonb;
  end if;

  perform 1
    from public.team_league_settings as settings
   where settings.club_id = v_operation.club_id
     and settings.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;

  select fixture.*
    into v_before
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = v_operation.club_id
     and fixture.league_name = v_operation.league_name
   for update;
  if not found or v_before.official_match_id is null then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_CANONICAL_MATCH_NOT_LINKED';
  end if;
  if v_before.phase = 'regular' and exists (
    select 1
      from public.team_league_fixtures as playoff
     where playoff.club_id = v_before.club_id
       and playoff.league_name = v_before.league_name
       and playoff.phase = 'playoff'
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_REGULAR_RESULT_LOCKED_AFTER_PLAYOFF_SEEDING';
  end if;

  select match_row.*
    into v_match
    from public.matches as match_row
   where match_row.club_id = v_before.club_id
     and match_row.id = v_before.official_match_id
   for share;
  v_match_found := found;
  if v_match_found then
    if v_match.league is distinct from v_before.league_name
       or v_match.match_type is distinct from 'Team League' then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_CANONICAL_MATCH_CONTEXT_INVALID';
    end if;
    if v_before.team_a_player_1_id is null
       or v_before.team_a_player_2_id is null
       or v_before.team_b_player_1_id is null
       or v_before.team_b_player_2_id is null then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_FIXTURE_PLAYER_EVIDENCE_MISSING';
    end if;
    v_normal_sides := (
      array[v_match.t1_p1, v_match.t1_p2]
        @> array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
      and array[v_match.t1_p1, v_match.t1_p2]
        <@ array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        @> array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        <@ array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
    );
    v_swapped_sides := (
      array[v_match.t1_p1, v_match.t1_p2]
        @> array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
      and array[v_match.t1_p1, v_match.t1_p2]
        <@ array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        @> array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        <@ array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
    );
    if not v_normal_sides and not v_swapped_sides then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_CANONICAL_MATCH_PLAYER_SET_INVALID';
    end if;
  end if;

  if not v_match_found
     or v_match.deleted_at is not null
     or coalesce(v_match.excluded_from_ratings, false) then
    v_new_winner := null;
  else
    if v_match.score_t1 is null
       or v_match.score_t2 is null
       or v_match.score_t1 = v_match.score_t2 then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_CANONICAL_MATCH_SCORE_INVALID';
    end if;
    v_new_winner := case
      when v_normal_sides and v_match.score_t1 > v_match.score_t2
        then v_before.team_a_id
      when v_normal_sides
        then v_before.team_b_id
      when v_match.score_t1 > v_match.score_t2
        then v_before.team_b_id
      else v_before.team_a_id
    end;
  end if;
  v_dependency_changed :=
    v_before.phase = 'playoff'
    and v_before.winner_team_id is distinct from v_new_winner;

  if v_dependency_changed and exists (
    with recursive affected as (
      select
        source.round_number,
        source.bracket_slot,
        0 as depth
      from public.team_league_fixtures as source
      where source.id = p_fixture_id
      union all
      select
        target.round_number,
        target.bracket_slot,
        affected.depth + 1
      from affected
      join public.team_league_fixtures as target
        on target.club_id = v_before.club_id
       and target.league_name = v_before.league_name
       and target.phase = 'playoff'
       and (
         target.team_a_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
         or target.team_b_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
       )
    )
    select 1
      from affected
      join public.team_league_fixtures as fixture
        on fixture.club_id = v_before.club_id
       and fixture.league_name = v_before.league_name
       and fixture.phase = 'playoff'
       and fixture.round_number = affected.round_number
       and fixture.bracket_slot = affected.bracket_slot
     where affected.depth > 0
       and (
         fixture.status in ('complete', 'forfeit')
         or fixture.official_match_id is not null
         or fixture.score_operation_id is not null
       )
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_PLAYOFF_DEPENDENT_RESULT_LOCKED';
  end if;

  if v_dependency_changed then
    with recursive affected as (
      select
        source.round_number,
        source.bracket_slot,
        0 as depth
      from public.team_league_fixtures as source
      where source.id = p_fixture_id
      union all
      select
        target.round_number,
        target.bracket_slot,
        affected.depth + 1
      from affected
      join public.team_league_fixtures as target
        on target.club_id = v_before.club_id
       and target.league_name = v_before.league_name
       and target.phase = 'playoff'
       and (
         target.team_a_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
         or target.team_b_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
       )
    ),
    descendant as (
      select distinct round_number, bracket_slot
        from affected
       where depth > 0
    ),
    source_key as (
      select distinct
        'winner:' || round_number::text || ':' ||
        bracket_slot::text as value
      from affected
    )
    update public.team_league_fixtures as fixture
       set team_a_id = case
             when fixture.team_a_source in (
               select value from source_key
             ) then null
             else fixture.team_a_id
           end,
           team_b_id = case
             when fixture.team_b_source in (
               select value from source_key
             ) then null
             else fixture.team_b_id
           end,
           status = 'scheduled',
           resolution = null,
           team_a_score = null,
           team_b_score = null,
           winner_team_id = null,
           official_match_id = null,
           team_a_player_1_id = null,
           team_a_player_2_id = null,
           team_b_player_1_id = null,
           team_b_player_2_id = null,
           substitutions_json = '[]'::jsonb,
           score_note = null,
           score_operation_id = null,
           score_reserved_at = null,
           scored_by = null,
           scored_at = null,
           updated_at = pg_catalog.clock_timestamp()
      from descendant
     where fixture.club_id = v_before.club_id
       and fixture.league_name = v_before.league_name
       and fixture.phase = 'playoff'
       and fixture.round_number = descendant.round_number
       and fixture.bracket_slot = descendant.bracket_slot;
  end if;

  if not v_match_found
     or v_match.deleted_at is not null
     or coalesce(v_match.excluded_from_ratings, false) then
    update public.team_league_fixtures
       set status = 'cancelled',
           resolution = 'cancelled',
           winner_team_id = null,
           team_a_score = null,
           team_b_score = null,
           score_note =
             'Canonical match was excluded or removed through Match Log.',
           updated_at = pg_catalog.clock_timestamp()
     where id = p_fixture_id
    returning * into v_updated;
  else
    update public.team_league_fixtures
       set status = 'complete',
           resolution = 'played',
           team_a_score = case
             when v_normal_sides then v_match.score_t1
             else v_match.score_t2
           end,
           team_b_score = case
             when v_normal_sides then v_match.score_t2
             else v_match.score_t1
           end,
           winner_team_id = v_new_winner,
           score_note = null,
           updated_at = pg_catalog.clock_timestamp()
     where id = p_fixture_id
    returning * into v_updated;
  end if;

  if v_dependency_changed and v_updated.winner_team_id is not null then
    loop
      update public.team_league_fixtures as target
         set team_a_id = case
               when target.team_a_id is null
                and target.team_a_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_a_id
             end,
             team_b_id = case
               when target.team_b_id is null
                and target.team_b_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_b_id
             end,
             updated_at = pg_catalog.clock_timestamp()
        from public.team_league_fixtures as source
       where target.club_id = v_before.club_id
         and target.league_name = v_before.league_name
         and target.phase = 'playoff'
         and source.club_id = target.club_id
         and source.league_name = target.league_name
         and source.phase = 'playoff'
         and source.winner_team_id is not null
         and (
           (
             target.team_a_id is null
             and target.team_a_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
           or (
             target.team_b_id is null
             and target.team_b_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
         );
      get diagnostics v_resolved_count = row_count;
      exit when v_resolved_count = 0;
    end loop;
  end if;

  update public.team_league_settings
     set standings_version = standings_version + 1,
         updated_at = pg_catalog.clock_timestamp()
   where club_id = v_before.club_id
     and league_name = v_before.league_name;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'fixture_id', p_fixture_id,
    'status', v_updated.status,
    'official_match_id', v_updated.official_match_id,
    'sides_swapped', v_swapped_sides,
    'downstream_invalidated', v_dependency_changed,
    'message', 'Fixture refreshed from the canonical match.',
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
    v_updated.club_id,
    v_actor_email,
    v_actor_role,
    'team_league_fixture_reconciled',
    'team_league_fixture',
    p_fixture_id::text,
    pg_catalog.to_jsonb(v_before),
    pg_catalog.to_jsonb(v_updated),
    case
      when v_dependency_changed
        then 'Unplayed dependent playoff slots were rebuilt.'
      else null
    end,
    v_source,
    false
  );
  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         recovery_note = null,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

create or replace function public.team_league_finalize_fixture_v2(
  p_operation_id uuid,
  p_club_id text,
  p_fixture_id uuid,
  p_status text,
  p_team_a_score integer,
  p_team_b_score integer,
  p_winner_team_id uuid,
  p_official_match_id bigint,
  p_team_a_player_1_id bigint,
  p_team_a_player_2_id bigint,
  p_team_b_player_1_id bigint,
  p_team_b_player_2_id bigint,
  p_substitutions jsonb,
  p_score_note text,
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
  v_operation public.team_league_operations%rowtype;
  v_fixture public.team_league_fixtures%rowtype;
  v_settings public.team_league_settings%rowtype;
  v_result jsonb;
begin
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found or v_operation.operation_type <> 'admin_score_fixture' then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SCORE_OPERATION_NOT_FOUND';
  end if;
  if nullif(v_operation.request_json ->> 'fixture_id', '') is distinct from
     p_fixture_id::text then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SCORE_OPERATION_FIXTURE_MISMATCH';
  end if;
  if v_operation.status = 'complete'
     and v_operation.result_json is not null then
    return v_operation.result_json || '{"idempotent": true}'::jsonb;
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_operation.club_id
     and settings.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;

  select fixture.*
    into v_fixture
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = v_operation.club_id
     and fixture.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_FIXTURE_NOT_FOUND';
  end if;
  if v_fixture.score_operation_id = p_operation_id
     and v_fixture.status in ('complete', 'forfeit') then
    null;
  elsif v_fixture.status <> 'scheduled'
        or v_fixture.score_operation_id is distinct from p_operation_id
        or v_fixture.score_reserved_at is null then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_FIXTURE_SCORE_RESERVATION_CONFLICT';
  else
    if p_status = 'complete' then
      if not exists (
        select 1
          from public.matches as match_row
         where match_row.club_id = v_fixture.club_id
           and match_row.id = p_official_match_id
           and match_row.league = v_fixture.league_name
           and match_row.match_type = 'Team League'
           and match_row.deleted_at is null
           and not coalesce(match_row.excluded_from_ratings, false)
           and match_row.score_t1 = p_team_a_score
           and match_row.score_t2 = p_team_b_score
           and (
             case
               when match_row.score_t1 > match_row.score_t2
                 then v_fixture.team_a_id
               else v_fixture.team_b_id
             end
           ) = p_winner_team_id
           and (
             (
               array[
                 match_row.t1_p1,
                 match_row.t1_p2
               ] @> array[
                 p_team_a_player_1_id,
                 p_team_a_player_2_id
               ]
               and array[
                 match_row.t1_p1,
                 match_row.t1_p2
               ] <@ array[
                 p_team_a_player_1_id,
                 p_team_a_player_2_id
               ]
               and array[
                 match_row.t2_p1,
                 match_row.t2_p2
               ] @> array[
                 p_team_b_player_1_id,
                 p_team_b_player_2_id
               ]
               and array[
                 match_row.t2_p1,
                 match_row.t2_p2
               ] <@ array[
                 p_team_b_player_1_id,
                 p_team_b_player_2_id
               ]
             )
           )
        for share
      ) then
        raise exception using
          errcode = '22023',
          message = 'TEAM_LEAGUE_CANONICAL_MATCH_INVALID';
      end if;
      if not v_settings.allow_substitutes and (
        not exists (
          select 1
            from public.team_league_teams as team
           where team.id = v_fixture.team_a_id
             and array[
               team.captain_player_id,
               team.partner_player_id
             ] @> array[
               p_team_a_player_1_id,
               p_team_a_player_2_id
             ]
             and array[
               team.captain_player_id,
               team.partner_player_id
             ] <@ array[
               p_team_a_player_1_id,
               p_team_a_player_2_id
             ]
        )
        or not exists (
          select 1
            from public.team_league_teams as team
           where team.id = v_fixture.team_b_id
             and array[
               team.captain_player_id,
               team.partner_player_id
             ] @> array[
               p_team_b_player_1_id,
               p_team_b_player_2_id
             ]
             and array[
               team.captain_player_id,
               team.partner_player_id
             ] <@ array[
               p_team_b_player_1_id,
               p_team_b_player_2_id
             ]
        )
      ) then
        raise exception using
          errcode = '55000',
          message = 'TEAM_LEAGUE_SUBSTITUTES_DISABLED';
      end if;
    end if;

    -- The row remains locked throughout the nested call. Clearing the
    -- reservation only adapts the original finalizer's precondition; any
    -- failure rolls the reservation back with this transaction.
    update public.team_league_fixtures
       set score_operation_id = null
     where id = p_fixture_id;
  end if;

  select public.team_league_finalize_fixture_v1(
    p_operation_id,
    p_club_id,
    p_fixture_id,
    p_status,
    p_team_a_score,
    p_team_b_score,
    p_winner_team_id,
    p_official_match_id,
    p_team_a_player_1_id,
    p_team_a_player_2_id,
    p_team_b_player_1_id,
    p_team_b_player_2_id,
    p_substitutions,
    p_score_note,
    p_actor_email,
    p_actor_role,
    p_source
  ) into v_result;
  return v_result;
end
$function$;

create or replace function public.team_league_resolve_operation_v2(
  p_operation_id uuid,
  p_club_id text,
  p_resolution text,
  p_result jsonb,
  p_recovery_note text,
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
  v_operation public.team_league_operations%rowtype;
  v_result jsonb;
begin
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_OPERATION_NOT_FOUND';
  end if;
  if pg_catalog.lower(pg_catalog.btrim(coalesce(p_resolution, '')))
     = 'compensate' then
    if exists (
      select 1
        from public.team_league_fixtures as fixture
        join public.admin_direct_match_entry_operations as direct_operation
          on direct_operation.club_id = fixture.club_id
         and direct_operation.idempotency_key =
           'teamfx:' || fixture.id::text
       where fixture.club_id = v_operation.club_id
         and fixture.league_name = v_operation.league_name
         and fixture.score_operation_id = p_operation_id
         and coalesce(
           (direct_operation.result_json ->> 'committed')::boolean,
           false
         )
    ) then
      raise exception using
        errcode = '55000',
        message = 'TEAM_LEAGUE_DIRECT_MATCH_ALREADY_COMMITTED';
    end if;
    update public.team_league_fixtures
       set score_operation_id = null,
           score_reserved_at = null,
           updated_at = pg_catalog.clock_timestamp()
     where club_id = v_operation.club_id
       and league_name = v_operation.league_name
       and score_operation_id = p_operation_id
       and status = 'scheduled'
       and official_match_id is null;
  end if;
  select public.team_league_resolve_operation_v1(
    p_operation_id,
    p_club_id,
    p_resolution,
    p_result,
    p_recovery_note,
    p_actor_email,
    p_actor_role,
    p_source
  ) into v_result;
  return v_result;
end
$function$;

create or replace function public.team_league_bump_roster_version_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := case when tg_op = 'DELETE' then old.club_id else new.club_id end;
  v_league_name text := case when tg_op = 'DELETE' then old.league_name else new.league_name end;
  v_changed boolean := (
    tg_op in ('INSERT', 'DELETE')
    or old.status is distinct from new.status
    or old.captain_player_id is distinct from new.captain_player_id
    or old.partner_player_id is distinct from new.partner_player_id
  );
begin
  if v_changed then
    update public.team_league_settings
       set roster_version = roster_version + 1,
           updated_at = pg_catalog.clock_timestamp()
     where club_id = v_club_id
       and league_name = v_league_name;
  end if;
  return null;
end
$function$;

drop trigger if exists team_league_guard_roster_mutation
  on public.team_league_teams;
create trigger team_league_guard_roster_mutation
before insert or update or delete on public.team_league_teams
for each row execute function public.team_league_guard_roster_mutation_v1();

drop trigger if exists team_league_bump_roster_version
  on public.team_league_teams;
create trigger team_league_bump_roster_version
after insert or update or delete on public.team_league_teams
for each row execute function public.team_league_bump_roster_version_v1();

create or replace function public.team_league_guard_waitlist_registration_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_settings public.team_league_settings%rowtype;
  v_becoming_waiting boolean := (
    tg_op = 'INSERT'
    or (
      tg_op = 'UPDATE'
      and new.status = 'waiting'
      and old.status is distinct from 'waiting'
    )
  );
begin
  if not v_becoming_waiting then
    return new;
  end if;
  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = new.club_id
     and settings.league_name = new.league_name
   for update;
  if not found
     or not v_settings.registration_open
     or v_settings.status <> 'registration_open'
     or (
       v_settings.registration_closes_at is not null
       and v_settings.registration_closes_at <=
         pg_catalog.clock_timestamp()
     )
     or v_settings.schedule_version <> 0
     or exists (
       select 1
         from public.team_league_fixtures as fixture
        where fixture.club_id = new.club_id
          and fixture.league_name = new.league_name
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_REGISTRATION_CLOSED';
  end if;
  return new;
end
$function$;

drop trigger if exists team_league_guard_waitlist_registration
  on public.team_league_solo_waitlist;
create trigger team_league_guard_waitlist_registration
before insert or update on public.team_league_solo_waitlist
for each row execute function public.team_league_guard_waitlist_registration_v1();

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

drop trigger if exists team_league_guard_settings_after_schedule
  on public.team_league_settings;
create trigger team_league_guard_settings_after_schedule
before update on public.team_league_settings
for each row execute function public.team_league_guard_settings_after_schedule_v1();

create or replace function public.team_league_reserve_fixture_score_v1(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_fixture_id uuid,
  p_team_a_id uuid,
  p_team_b_id uuid
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.team_league_operations%rowtype;
  v_fixture public.team_league_fixtures%rowtype;
begin
  if p_operation_id is null
     or nullif(pg_catalog.btrim(p_club_id), '') is null
     or nullif(pg_catalog.btrim(p_league_name), '') is null
     or p_fixture_id is null
     or p_team_a_id is null
     or p_team_b_id is null
     or p_team_a_id = p_team_b_id then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SCORE_RESERVATION_INVALID';
  end if;

  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
     and operation.league_name = pg_catalog.btrim(p_league_name)
   for update;
  if not found or v_operation.operation_type <> 'admin_score_fixture' then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SCORE_OPERATION_NOT_FOUND';
  end if;

  perform 1
    from public.team_league_settings as settings
   where settings.club_id = pg_catalog.btrim(p_club_id)
     and settings.league_name = pg_catalog.btrim(p_league_name)
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;

  select fixture.*
    into v_fixture
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = pg_catalog.btrim(p_club_id)
     and fixture.league_name = pg_catalog.btrim(p_league_name)
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_FIXTURE_NOT_FOUND';
  end if;
  if v_fixture.score_operation_id = p_operation_id
     and v_fixture.status = 'scheduled'
     and v_fixture.team_a_id = p_team_a_id
     and v_fixture.team_b_id = p_team_b_id then
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'reserved', true,
      'idempotent', true,
      'operation_id', p_operation_id,
      'fixture_id', p_fixture_id
    );
  end if;
  if v_fixture.status <> 'scheduled'
     or v_fixture.score_operation_id is not null
     or v_fixture.team_a_id is distinct from p_team_a_id
     or v_fixture.team_b_id is distinct from p_team_b_id then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_FIXTURE_SCORE_RESERVATION_CONFLICT';
  end if;

  update public.team_league_fixtures
     set score_operation_id = p_operation_id,
         score_reserved_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_fixture_id;
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'reserved', true,
    'idempotent', false,
    'operation_id', p_operation_id,
    'fixture_id', p_fixture_id
  );
end
$function$;

create or replace function public.team_league_replace_schedule_v2(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_phase text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_expected_schedule_version integer,
  p_expected_standings_version integer,
  p_expected_roster_version integer,
  p_confirmed_roster_fingerprint text,
  p_fixtures jsonb,
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
  v_phase text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_phase, '')));
  v_settings public.team_league_settings%rowtype;
  v_roster_fingerprint text;
  v_team_count integer;
  v_expected_rounds integer;
  v_expected_fixture_count integer;
  v_bye_count integer;
  v_pair_count integer;
  v_result jsonb;
begin
  if p_expected_roster_version is null
     or p_expected_roster_version < 0
     or p_confirmed_roster_fingerprint !~ '^[0-9a-f]{64}$'
     or p_fixtures is null
     or pg_catalog.jsonb_typeof(p_fixtures) <> 'array' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_ROSTER_PRECONDITION_INVALID';
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;
  v_roster_fingerprint :=
    public.team_league_confirmed_roster_fingerprint_v1(
      v_club_id,
      v_league_name
    );
  if v_settings.roster_version <> p_expected_roster_version
     or v_roster_fingerprint <> p_confirmed_roster_fingerprint then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_ROSTER_VERSION_CONFLICT';
  end if;
  if exists (
    select 1
      from public.team_league_fixtures as fixture
     where fixture.club_id = v_club_id
       and fixture.league_name = v_league_name
       and fixture.phase = v_phase
       and fixture.score_operation_id is not null
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_SCORE_IN_FLIGHT';
  end if;

  if v_phase = 'regular' then
    select pg_catalog.count(*)::integer
      into v_team_count
      from public.team_league_teams as team
     where team.club_id = v_club_id
       and team.league_name = v_league_name
       and team.status = 'confirmed';
    if v_team_count < 2 then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_ROUND_ROBIN_TEAM_COUNT_INVALID';
    end if;
    v_expected_rounds := case
      when v_team_count % 2 = 0 then v_team_count - 1
      else v_team_count
    end;
    v_expected_fixture_count :=
      v_expected_rounds * ((v_team_count + 1) / 2);
    if pg_catalog.jsonb_array_length(p_fixtures)
       <> v_expected_fixture_count then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_ROUND_ROBIN_FIXTURE_COUNT_INVALID';
    end if;

    if exists (
      select 1
        from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
          round_number integer,
          week_number integer,
          bracket_slot integer,
          team_a_id uuid,
          team_b_id uuid,
          status text
        )
       where item.round_number not between 1 and v_expected_rounds
          or item.week_number is distinct from item.round_number
          or item.bracket_slot is null
          or item.bracket_slot < 1
          or item.team_a_id is null
          or item.team_a_id = item.team_b_id
          or (
            item.status = 'scheduled'
            and item.team_b_id is null
          )
          or (
            item.status = 'bye'
            and item.team_b_id is not null
          )
          or item.status not in ('scheduled', 'bye')
    ) or exists (
      select 1
        from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
          team_a_id uuid,
          team_b_id uuid
        )
        left join public.team_league_teams as team_a
          on team_a.id = item.team_a_id
         and team_a.club_id = v_club_id
         and team_a.league_name = v_league_name
         and team_a.status = 'confirmed'
        left join public.team_league_teams as team_b
          on team_b.id = item.team_b_id
         and team_b.club_id = v_club_id
         and team_b.league_name = v_league_name
         and team_b.status = 'confirmed'
       where team_a.id is null
          or (item.team_b_id is not null and team_b.id is null)
    ) then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_ROUND_ROBIN_TEAM_INVALID';
    end if;

    if (
      select pg_catalog.count(*)
        from (
          select item.round_number, item.bracket_slot
            from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
              round_number integer,
              bracket_slot integer
            )
           group by item.round_number, item.bracket_slot
          having pg_catalog.count(*) <> 1
        ) as duplicate_slot
    ) > 0 then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_ROUND_ROBIN_SLOT_DUPLICATE';
    end if;

    if exists (
      select 1
        from public.team_league_teams as team
        cross join pg_catalog.generate_series(
          1,
          v_expected_rounds
        ) as expected(round_number)
        left join (
          select item.round_number, item.team_a_id as team_id
            from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
              round_number integer,
              team_a_id uuid
            )
          union all
          select item.round_number, item.team_b_id as team_id
            from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
              round_number integer,
              team_b_id uuid
            )
           where item.team_b_id is not null
        ) as appearance
          on appearance.round_number = expected.round_number
         and appearance.team_id = team.id
       where team.club_id = v_club_id
         and team.league_name = v_league_name
         and team.status = 'confirmed'
       group by team.id, expected.round_number
      having pg_catalog.count(appearance.team_id) <> 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_ROUND_ROBIN_TEAM_APPEARANCE_INVALID';
    end if;

    select pg_catalog.count(*)::integer
      into v_pair_count
      from (
        select
          least(item.team_a_id::text, item.team_b_id::text),
          greatest(item.team_a_id::text, item.team_b_id::text)
        from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
          team_a_id uuid,
          team_b_id uuid
        )
        where item.team_b_id is not null
        group by
          least(item.team_a_id::text, item.team_b_id::text),
          greatest(item.team_a_id::text, item.team_b_id::text)
        having pg_catalog.count(*) = 1
      ) as unique_pair;
    if v_pair_count <> (v_team_count * (v_team_count - 1)) / 2 then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_ROUND_ROBIN_PAIR_SET_INVALID';
    end if;

    select pg_catalog.count(*)::integer
      into v_bye_count
      from pg_catalog.jsonb_to_recordset(p_fixtures) as item(
        team_b_id uuid,
        status text
      )
     where item.status = 'bye';
    if (v_team_count % 2 = 0 and v_bye_count <> 0)
       or (v_team_count % 2 = 1 and v_bye_count <> v_team_count) then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_ROUND_ROBIN_BYE_SET_INVALID';
    end if;
  end if;

  select public.team_league_replace_schedule_v1(
    p_operation_id,
    p_club_id,
    p_league_name,
    p_phase,
    p_idempotency_key,
    p_request_fingerprint,
    p_expected_schedule_version,
    p_expected_standings_version,
    p_fixtures,
    p_actor_email,
    p_actor_role,
    p_source
  ) into v_result;
  return v_result || pg_catalog.jsonb_build_object(
    'roster_version', p_expected_roster_version,
    'confirmed_roster_fingerprint', p_confirmed_roster_fingerprint
  );
end
$function$;

revoke all on function public.team_league_guard_roster_mutation_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_bump_roster_version_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_guard_waitlist_registration_v1()
  from public, anon, authenticated;
revoke all on function public.team_league_guard_settings_after_schedule_v1()
  from public, anon, authenticated;
revoke all on function public.league_awards_save_config_v1(
  text,
  text,
  integer,
  jsonb,
  text,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.team_league_confirmed_roster_fingerprint_v1(
  text,
  text
) from public, anon, authenticated;
revoke all on function public.team_league_reserve_fixture_score_v1(
  uuid,
  text,
  text,
  uuid,
  uuid,
  uuid
) from public, anon, authenticated;
revoke all on function public.team_league_finalize_fixture_v2(
  uuid,
  text,
  uuid,
  text,
  integer,
  integer,
  uuid,
  bigint,
  bigint,
  bigint,
  bigint,
  bigint,
  jsonb,
  text,
  text,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.team_league_reconcile_fixture_v2(
  uuid,
  text,
  uuid,
  text,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.team_league_resolve_operation_v2(
  uuid,
  text,
  text,
  jsonb,
  text,
  text,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.team_league_replace_schedule_v2(
  uuid,
  text,
  text,
  text,
  text,
  text,
  integer,
  integer,
  integer,
  text,
  jsonb,
  text,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.league_awards_apply_workflow_v2(
  text,
  text,
  integer,
  jsonb,
  jsonb,
  text,
  text,
  jsonb,
  jsonb,
  boolean,
  text,
  text,
  text
) from public, anon, authenticated;

grant execute on function public.team_league_confirmed_roster_fingerprint_v1(
  text,
  text
) to service_role;
grant execute on function public.league_awards_save_config_v1(
  text,
  text,
  integer,
  jsonb,
  text,
  text,
  text
) to service_role;
grant execute on function public.team_league_reserve_fixture_score_v1(
  uuid,
  text,
  text,
  uuid,
  uuid,
  uuid
) to service_role;
grant execute on function public.team_league_finalize_fixture_v2(
  uuid,
  text,
  uuid,
  text,
  integer,
  integer,
  uuid,
  bigint,
  bigint,
  bigint,
  bigint,
  bigint,
  jsonb,
  text,
  text,
  text,
  text
) to service_role;
grant execute on function public.team_league_reconcile_fixture_v2(
  uuid,
  text,
  uuid,
  text,
  text,
  text
) to service_role;
grant execute on function public.team_league_resolve_operation_v2(
  uuid,
  text,
  text,
  jsonb,
  text,
  text,
  text,
  text
) to service_role;
grant execute on function public.team_league_replace_schedule_v2(
  uuid,
  text,
  text,
  text,
  text,
  text,
  integer,
  integer,
  integer,
  text,
  jsonb,
  text,
  text,
  text
) to service_role;
grant execute on function public.league_awards_apply_workflow_v2(
  text,
  text,
  integer,
  jsonb,
  jsonb,
  text,
  text,
  jsonb,
  jsonb,
  boolean,
  text,
  text,
  text
) to service_role;

comment on function public.team_league_replace_schedule_v2(
  uuid,
  text,
  text,
  text,
  text,
  text,
  integer,
  integer,
  integer,
  text,
  jsonb,
  text,
  text,
  text
) is
  'Atomically replaces a roster-bound schedule after SQL verifies the exact round-robin fixture set.';

comment on function public.league_awards_apply_workflow_v2(
  text,
  text,
  integer,
  jsonb,
  jsonb,
  text,
  text,
  jsonb,
  jsonb,
  boolean,
  text,
  text,
  text
) is
  'Commits one version-checked award workflow revision, result set, records, lifecycle patch, and audit evidence in one transaction.';
