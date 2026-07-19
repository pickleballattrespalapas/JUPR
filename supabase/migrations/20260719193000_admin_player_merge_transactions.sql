-- Make Player Editor account merges atomic, stale-preview guarded, auditable,
-- recoverable, and callable only by the FastAPI service-role client.

do $migration_preflight$
begin
  if to_regclass('public.players') is null
     or to_regclass('public.matches') is null
     or to_regclass('public.league_ratings') is null
     or to_regclass('public.club_people') is null
     or to_regclass('public.admin_activity_log') is null
     or to_regclass('public.replay_jobs') is null then
    raise exception using
      errcode = '42P01',
      message = 'Player, match, social identity, audit, and replay tables must exist before applying transactional player merges.';
  end if;
end
$migration_preflight$;

create table if not exists public.admin_player_merge_operations (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  source_player_id bigint not null,
  target_player_id bigint not null,
  status text not null default 'merged_pending_replay'
    check (status in ('merged_pending_replay', 'replay_verified', 'compensated')),
  preview_fingerprint text not null,
  before_json jsonb not null default '{}'::jsonb,
  result_json jsonb not null default '{}'::jsonb,
  actor_email text not null,
  actor_role text not null,
  source_page text,
  replay_job_id uuid,
  replay_verified_at timestamptz,
  compensated_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (source_player_id <> target_player_id)
);

create index if not exists admin_player_merge_operations_club_created_idx
  on public.admin_player_merge_operations (club_id, created_at desc);

alter table public.admin_player_merge_operations enable row level security;
alter table public.admin_player_merge_operations force row level security;

revoke all on table public.admin_player_merge_operations from public, anon, authenticated;
grant select, insert, update on table public.admin_player_merge_operations to service_role;

create or replace function public.server_merge_player_accounts(
  p_operation_id uuid,
  p_club_id text,
  p_source_player_id bigint,
  p_target_player_id bigint,
  p_preview_fingerprint text,
  p_expected_state jsonb,
  p_actor_email text,
  p_actor_role text,
  p_source_page text default 'next_player_editor_merge'
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_source public.players%rowtype;
  v_target public.players%rowtype;
  v_actual_state jsonb;
  v_before jsonb;
  v_result jsonb;
  v_match_t1_p1 integer := 0;
  v_match_t1_p2 integer := 0;
  v_match_t2_p1 integer := 0;
  v_match_t2_p2 integer := 0;
  v_deleted_league integer := 0;
  v_moved_league integer := 0;
  v_social integer := 0;
  v_source_name text;
  v_target_name text;
  v_inactive_name text;
begin
  if p_operation_id is null
     or nullif(btrim(p_club_id), '') is null
     or p_source_player_id is null
     or p_target_player_id is null
     or p_source_player_id = p_target_player_id
     or nullif(btrim(p_preview_fingerprint), '') is null
     or p_expected_state is null
     or jsonb_typeof(p_expected_state) <> 'object' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_PLAYER_MERGE_INVALID: operation, club, distinct players, fingerprint, and expected state are required.';
  end if;

  if exists (
    select 1
    from public.admin_player_merge_operations as operation
    where operation.id = p_operation_id
  ) then
    select operation.result_json
    into v_result
    from public.admin_player_merge_operations as operation
    where operation.id = p_operation_id
      and operation.club_id = btrim(p_club_id);
    if found then
      return v_result || jsonb_build_object('idempotent_replay', true);
    end if;
    raise exception using
      errcode = '23505',
      message = 'JUPR_PLAYER_MERGE_INVALID: operation id already belongs to another club.';
  end if;

  -- Serialize every merge involving either player and lock every row whose
  -- state participates in the stale-preview check.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:player-merge:' || btrim(p_club_id) || ':' || least(p_source_player_id, p_target_player_id)::text,
      0
    )
  );
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:player-merge:' || btrim(p_club_id) || ':' || greatest(p_source_player_id, p_target_player_id)::text,
      0
    )
  );

  -- A concurrent retry can pass the optimistic check above while the first
  -- transaction is still open. Re-check after serialization so one operation
  -- ID can never apply the merge twice.
  select operation.result_json
  into v_result
  from public.admin_player_merge_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = btrim(p_club_id);
  if found then
    return v_result || jsonb_build_object('idempotent_replay', true);
  end if;

  -- Replay job creation/status writes use ROW EXCLUSIVE table locks. Holding
  -- SHARE here makes a new replay wait until this merge commits; an already
  -- pending/running replay is detected before any player state changes.
  lock table public.replay_jobs in share mode;
  if exists (
    select 1
    from public.replay_jobs as replay_job
    where replay_job.club_id = btrim(p_club_id)
      and lower(coalesce(replay_job.status, '')) in ('pending', 'running')
  ) then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_REPLAY_IN_PROGRESS');
  end if;

  select player.*
  into v_source
  from public.players as player
  where player.club_id::text = btrim(p_club_id)
    and player.id = p_source_player_id
  for update;
  if not found then
    return jsonb_build_object('ok', false, 'code', 'SOURCE_PLAYER_NOT_FOUND');
  end if;

  select player.*
  into v_target
  from public.players as player
  where player.club_id::text = btrim(p_club_id)
    and player.id = p_target_player_id
  for update;
  if not found then
    return jsonb_build_object('ok', false, 'code', 'TARGET_PLAYER_NOT_FOUND');
  end if;

  perform 1
  from public.matches as match_row
  where match_row.club_id::text = btrim(p_club_id)
    and (
      p_source_player_id in (match_row.t1_p1, match_row.t1_p2, match_row.t2_p1, match_row.t2_p2)
      or p_target_player_id in (match_row.t1_p1, match_row.t1_p2, match_row.t2_p1, match_row.t2_p2)
    )
  for update;
  perform 1
  from public.league_ratings as league_rating
  where league_rating.club_id::text = btrim(p_club_id)
    and league_rating.player_id in (p_source_player_id, p_target_player_id)
  for update;
  perform 1
  from public.club_people as person
  where person.club_id::text = btrim(p_club_id)
    and person.linked_player_id in (p_source_player_id, p_target_player_id)
  for update;

  v_source_name := coalesce(nullif(btrim(v_source.name), ''), '#' || p_source_player_id::text);
  v_target_name := coalesce(nullif(btrim(v_target.name), ''), '#' || p_target_player_id::text);

  v_actual_state := jsonb_build_object(
    'source_player', jsonb_build_object(
      'id', p_source_player_id,
      'name', v_source.name,
      'active', coalesce(v_source.active, true),
      'inactive_at', v_source.inactive_at
    ),
    'target_player', jsonb_build_object(
      'id', p_target_player_id,
      'name', v_target.name,
      'active', coalesce(v_target.active, true),
      'inactive_at', v_target.inactive_at
    ),
    'match_reference_ids', jsonb_build_object(
      't1_p1', coalesce((select jsonb_agg(match_row.id order by match_row.id) from public.matches as match_row where match_row.club_id::text = btrim(p_club_id) and match_row.t1_p1 = p_source_player_id), '[]'::jsonb),
      't1_p2', coalesce((select jsonb_agg(match_row.id order by match_row.id) from public.matches as match_row where match_row.club_id::text = btrim(p_club_id) and match_row.t1_p2 = p_source_player_id), '[]'::jsonb),
      't2_p1', coalesce((select jsonb_agg(match_row.id order by match_row.id) from public.matches as match_row where match_row.club_id::text = btrim(p_club_id) and match_row.t2_p1 = p_source_player_id), '[]'::jsonb),
      't2_p2', coalesce((select jsonb_agg(match_row.id order by match_row.id) from public.matches as match_row where match_row.club_id::text = btrim(p_club_id) and match_row.t2_p2 = p_source_player_id), '[]'::jsonb)
    ),
    'source_league_rows', coalesce((
      select jsonb_agg(to_jsonb(source_rating) order by source_rating.id)
      from public.league_ratings as source_rating
      where source_rating.club_id::text = btrim(p_club_id)
        and source_rating.player_id = p_source_player_id
    ), '[]'::jsonb),
    'league_rating_plan', jsonb_build_object(
      'move_ids', coalesce((
        select jsonb_agg(source_rating.id order by source_rating.id)
        from public.league_ratings as source_rating
        where source_rating.club_id::text = btrim(p_club_id)
          and source_rating.player_id = p_source_player_id
          and not exists (
            select 1
            from public.league_ratings as target_rating
            where target_rating.club_id::text = btrim(p_club_id)
              and target_rating.player_id = p_target_player_id
              and coalesce(target_rating.league_name, '') = coalesce(source_rating.league_name, '')
          )
      ), '[]'::jsonb),
      'delete_ids', coalesce((
        select jsonb_agg(source_rating.id order by source_rating.id)
        from public.league_ratings as source_rating
        where source_rating.club_id::text = btrim(p_club_id)
          and source_rating.player_id = p_source_player_id
          and exists (
            select 1
            from public.league_ratings as target_rating
            where target_rating.club_id::text = btrim(p_club_id)
              and target_rating.player_id = p_target_player_id
              and coalesce(target_rating.league_name, '') = coalesce(source_rating.league_name, '')
          )
      ), '[]'::jsonb)
    ),
    'source_social_ids', coalesce((select jsonb_agg(person.id::text order by person.id::text) from public.club_people as person where person.club_id::text = btrim(p_club_id) and person.linked_player_id = p_source_player_id), '[]'::jsonb),
    'target_social_ids', coalesce((select jsonb_agg(person.id::text order by person.id::text) from public.club_people as person where person.club_id::text = btrim(p_club_id) and person.linked_player_id = p_target_player_id), '[]'::jsonb)
  );

  if v_actual_state is distinct from p_expected_state then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_STALE_PREVIEW');
  end if;

  if not coalesce(v_source.active, true) or v_source.inactive_at is not null then
    return jsonb_build_object('ok', false, 'code', 'SOURCE_PLAYER_INACTIVE');
  end if;
  if not coalesce(v_target.active, true) or v_target.inactive_at is not null then
    return jsonb_build_object('ok', false, 'code', 'TARGET_PLAYER_INACTIVE');
  end if;

  if exists (
    select 1
    from public.matches as match_row
    where match_row.club_id::text = btrim(p_club_id)
      and p_source_player_id in (match_row.t1_p1, match_row.t1_p2, match_row.t2_p1, match_row.t2_p2)
      and p_target_player_id in (match_row.t1_p1, match_row.t1_p2, match_row.t2_p1, match_row.t2_p2)
  ) then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_MATCH_COLLISION');
  end if;

  v_before := jsonb_build_object(
    'source_player', to_jsonb(v_source),
    'target_player', to_jsonb(v_target),
    'match_reference_ids', v_actual_state -> 'match_reference_ids',
    'league_rating_plan', v_actual_state -> 'league_rating_plan',
    'source_league_rows', v_actual_state -> 'source_league_rows',
    'source_social_rows', coalesce((select jsonb_agg(to_jsonb(person) order by person.id) from public.club_people as person where person.club_id::text = btrim(p_club_id) and person.linked_player_id = p_source_player_id), '[]'::jsonb),
    'target_social_ids', v_actual_state -> 'target_social_ids'
  );

  update public.matches set t1_p1 = p_target_player_id where club_id::text = btrim(p_club_id) and t1_p1 = p_source_player_id;
  get diagnostics v_match_t1_p1 = row_count;
  update public.matches set t1_p2 = p_target_player_id where club_id::text = btrim(p_club_id) and t1_p2 = p_source_player_id;
  get diagnostics v_match_t1_p2 = row_count;
  update public.matches set t2_p1 = p_target_player_id where club_id::text = btrim(p_club_id) and t2_p1 = p_source_player_id;
  get diagnostics v_match_t2_p1 = row_count;
  update public.matches set t2_p2 = p_target_player_id where club_id::text = btrim(p_club_id) and t2_p2 = p_source_player_id;
  get diagnostics v_match_t2_p2 = row_count;

  delete from public.league_ratings as source_rating
  using public.league_ratings as target_rating
  where source_rating.club_id::text = btrim(p_club_id)
    and source_rating.player_id = p_source_player_id
    and target_rating.club_id::text = btrim(p_club_id)
    and target_rating.player_id = p_target_player_id
    and coalesce(target_rating.league_name, '') = coalesce(source_rating.league_name, '');
  get diagnostics v_deleted_league = row_count;

  update public.league_ratings
  set player_id = p_target_player_id
  where club_id::text = btrim(p_club_id)
    and player_id = p_source_player_id;
  get diagnostics v_moved_league = row_count;

  update public.club_people
  set linked_player_id = case
    when jsonb_array_length(v_actual_state -> 'target_social_ids') > 0 then null
    else p_target_player_id
  end
  where club_id::text = btrim(p_club_id)
    and linked_player_id = p_source_player_id;
  get diagnostics v_social = row_count;

  v_inactive_name := left(v_source_name || ' (MERGED into ' || v_target_name || ' #' || p_target_player_id::text || ')', 160);
  update public.players
  set active = false,
      inactive_at = now(),
      name = v_inactive_name
  where club_id::text = btrim(p_club_id)
    and id = p_source_player_id;

  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'player_merge_execute',
    'transaction_mode', 'postgres_atomic_rpc',
    'operation_id', p_operation_id,
    'operation_status', 'merged_pending_replay',
    'source_player_id', p_source_player_id,
    'target_player_id', p_target_player_id,
    'preview_fingerprint', btrim(p_preview_fingerprint),
    'match_updates', jsonb_build_object('t1_p1', v_match_t1_p1, 't1_p2', v_match_t1_p2, 't2_p1', v_match_t2_p1, 't2_p2', v_match_t2_p2),
    'moved_league_rating_count', v_moved_league,
    'deleted_conflicting_league_rating_count', v_deleted_league,
    'social_identity_rows_updated', v_social,
    'requires_replay', true
  );

  insert into public.admin_player_merge_operations (
    id, club_id, source_player_id, target_player_id, status,
    preview_fingerprint, before_json, result_json,
    actor_email, actor_role, source_page
  ) values (
    p_operation_id, btrim(p_club_id), p_source_player_id, p_target_player_id,
    'merged_pending_replay', btrim(p_preview_fingerprint), v_before, v_result,
    lower(coalesce(nullif(btrim(p_actor_email), ''), 'unknown')),
    coalesce(nullif(btrim(p_actor_role), ''), 'unknown'),
    nullif(btrim(p_source_page), '')
  );

  insert into public.admin_activity_log (
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
  ) values (
    btrim(p_club_id),
    lower(coalesce(nullif(btrim(p_actor_email), ''), 'unknown')),
    coalesce(nullif(btrim(p_actor_role), ''), 'unknown'),
    'merge_player_editor_players_admin',
    'players',
    p_source_player_id::text || '->' || p_target_player_id::text,
    v_before,
    v_result,
    'Atomic player merge; full Replay History recovery is required.',
    nullif(btrim(p_source_page), ''),
    true
  );

  return v_result;
end
$function$;

create or replace function public.server_compensate_player_merge(
  p_operation_id uuid,
  p_club_id text,
  p_actor_email text,
  p_actor_role text,
  p_source_page text default 'next_player_editor_merge_compensation'
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.admin_player_merge_operations%rowtype;
  v_before jsonb;
  v_source bigint;
  v_target bigint;
  v_player_name text;
  v_player_active boolean;
  v_player_inactive_at timestamptz;
  v_expected_merged_name text;
  v_result jsonb;
begin
  select operation.*
  into v_operation
  from public.admin_player_merge_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = btrim(p_club_id)
  for update;
  if not found then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_OPERATION_NOT_FOUND');
  end if;
  if v_operation.status <> 'merged_pending_replay' then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_COMPENSATION_NOT_ALLOWED', 'status', v_operation.status);
  end if;

  v_before := v_operation.before_json;
  v_source := v_operation.source_player_id;
  v_target := v_operation.target_player_id;
  v_expected_merged_name := left(
    coalesce(nullif(btrim(v_before #>> '{source_player,name}'), ''), '#' || v_source::text)
    || ' (MERGED into '
    || coalesce(nullif(btrim(v_before #>> '{target_player,name}'), ''), '#' || v_target::text)
    || ' #' || v_target::text || ')',
    160
  );

  -- Serialize against replay job creation/status writes. A replay job that
  -- committed first is detected below; one requested concurrently starts only
  -- after compensation commits and therefore replays the restored state.
  lock table public.replay_jobs in share mode;

  if exists (
    select 1
    from public.replay_jobs as replay_job
    where replay_job.club_id = btrim(p_club_id)
      and lower(coalesce(replay_job.status, '')) in ('pending', 'running')
  ) then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_COMPENSATION_REPLAY_STARTED');
  end if;

  -- Freeze every surviving row whose post-merge state is checked below.
  -- Concurrent editors then wait until compensation commits and cannot have
  -- their work overwritten between the stale predicate and restoration.
  perform 1
  from public.players as source_player
  where source_player.club_id::text = btrim(p_club_id)
    and source_player.id = v_source
  for update;

  perform 1
  from public.matches as match_row
  where match_row.club_id::text = btrim(p_club_id)
    and match_row.id in (
      select item.id::bigint
      from jsonb_array_elements_text(
        (v_before #> '{match_reference_ids,t1_p1}')
        || (v_before #> '{match_reference_ids,t1_p2}')
        || (v_before #> '{match_reference_ids,t2_p1}')
        || (v_before #> '{match_reference_ids,t2_p2}')
      ) as item(id)
    )
  order by match_row.id
  for update;

  perform 1
  from public.league_ratings as current_rating
  where current_rating.club_id::text = btrim(p_club_id)
    and current_rating.id in (
      select (rows.row_json ->> 'id')::bigint
      from jsonb_array_elements(v_before -> 'source_league_rows') as rows(row_json)
    )
  order by current_rating.id
  for update;

  perform 1
  from public.club_people as current_person
  where current_person.club_id::text = btrim(p_club_id)
    and current_person.id::text in (
      select rows.row_json ->> 'id'
      from jsonb_array_elements(v_before -> 'source_social_rows') as rows(row_json)
    )
  order by current_person.id
  for update;

  if exists (
    select 1
    from public.replay_jobs as replay_job
    where replay_job.club_id = btrim(p_club_id)
      and replay_job.target_reset = 'ALL (Full System Reset)'
      and replay_job.created_at >= v_operation.created_at
      and lower(coalesce(replay_job.status, '')) not in ('failed', 'cancelled', 'canceled')
  ) then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_COMPENSATION_REPLAY_STARTED');
  end if;

  -- Refuse compensation if any row originally changed by the merge has since
  -- moved away from the target. The operator must use Match Log/manual recovery
  -- instead of overwriting newer work.
  if exists (
    select 1 from jsonb_array_elements_text(v_before #> '{match_reference_ids,t1_p1}') as item(id)
    left join public.matches as match_row on match_row.id = item.id::bigint and match_row.club_id::text = btrim(p_club_id) and match_row.t1_p1 = v_target
    where match_row.id is null
  ) or exists (
    select 1 from jsonb_array_elements_text(v_before #> '{match_reference_ids,t1_p2}') as item(id)
    left join public.matches as match_row on match_row.id = item.id::bigint and match_row.club_id::text = btrim(p_club_id) and match_row.t1_p2 = v_target
    where match_row.id is null
  ) or exists (
    select 1 from jsonb_array_elements_text(v_before #> '{match_reference_ids,t2_p1}') as item(id)
    left join public.matches as match_row on match_row.id = item.id::bigint and match_row.club_id::text = btrim(p_club_id) and match_row.t2_p1 = v_target
    where match_row.id is null
  ) or exists (
    select 1 from jsonb_array_elements_text(v_before #> '{match_reference_ids,t2_p2}') as item(id)
    left join public.matches as match_row on match_row.id = item.id::bigint and match_row.club_id::text = btrim(p_club_id) and match_row.t2_p2 = v_target
    where match_row.id is null
  ) or not exists (
    select 1
    from public.players as source_player
    where source_player.club_id::text = btrim(p_club_id)
      and source_player.id = v_source
      and source_player.active is false
      and source_player.inactive_at is not null
      and source_player.name = v_expected_merged_name
  ) or exists (
    select 1
    from jsonb_array_elements(v_before -> 'source_league_rows') as rows(row_json)
    where (rows.row_json ->> 'id')::bigint in (
      select item.id::bigint
      from jsonb_array_elements_text(v_before #> '{league_rating_plan,move_ids}') as item(id)
    )
      and not exists (
        select 1
        from public.league_ratings as current_rating
        where current_rating.club_id::text = btrim(p_club_id)
          and current_rating.id = (rows.row_json ->> 'id')::bigint
          and current_rating.player_id = v_target
          and (to_jsonb(current_rating) - 'player_id') = (rows.row_json - 'player_id')
      )
  ) or exists (
    select 1
    from public.league_ratings as current_rating
    where current_rating.club_id::text = btrim(p_club_id)
      and current_rating.id in (
        select item.id::bigint
        from jsonb_array_elements_text(v_before #> '{league_rating_plan,delete_ids}') as item(id)
      )
  ) or exists (
    select 1
    from jsonb_array_elements(v_before -> 'source_social_rows') as rows(row_json)
    where not exists (
      select 1
      from public.club_people as current_person
      where current_person.club_id::text = btrim(p_club_id)
        and current_person.id::text = rows.row_json ->> 'id'
        and current_person.linked_player_id is not distinct from case
          when jsonb_array_length(v_before -> 'target_social_ids') > 0 then null
          else v_target
        end
        -- linked_player_id and trigger-maintained updated_at are the only
        -- fields the merge itself may change on a social identity.
        and (to_jsonb(current_person) - 'linked_player_id' - 'updated_at')
            = (rows.row_json - 'linked_player_id' - 'updated_at')
    )
  ) then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_COMPENSATION_STALE');
  end if;

  update public.matches set t1_p1 = v_source where club_id::text = btrim(p_club_id) and id in (select item.id::bigint from jsonb_array_elements_text(v_before #> '{match_reference_ids,t1_p1}') as item(id));
  update public.matches set t1_p2 = v_source where club_id::text = btrim(p_club_id) and id in (select item.id::bigint from jsonb_array_elements_text(v_before #> '{match_reference_ids,t1_p2}') as item(id));
  update public.matches set t2_p1 = v_source where club_id::text = btrim(p_club_id) and id in (select item.id::bigint from jsonb_array_elements_text(v_before #> '{match_reference_ids,t2_p1}') as item(id));
  update public.matches set t2_p2 = v_source where club_id::text = btrim(p_club_id) and id in (select item.id::bigint from jsonb_array_elements_text(v_before #> '{match_reference_ids,t2_p2}') as item(id));

  delete from public.league_ratings
  where club_id::text = btrim(p_club_id)
    and id in (select (row_json ->> 'id')::bigint from jsonb_array_elements(v_before -> 'source_league_rows') as rows(row_json));
  insert into public.league_ratings overriding system value
  select (jsonb_populate_record(null::public.league_ratings, rows.row_json)).*
  from jsonb_array_elements(v_before -> 'source_league_rows') as rows(row_json);

  update public.club_people as person
  set linked_player_id = (rows.row_json ->> 'linked_player_id')::bigint
  from jsonb_array_elements(v_before -> 'source_social_rows') as rows(row_json)
  where person.club_id::text = btrim(p_club_id)
    and person.id::text = rows.row_json ->> 'id';

  v_player_name := v_before #>> '{source_player,name}';
  v_player_active := coalesce((v_before #>> '{source_player,active}')::boolean, true);
  v_player_inactive_at := (v_before #>> '{source_player,inactive_at}')::timestamptz;
  update public.players
  set name = v_player_name,
      active = v_player_active,
      inactive_at = v_player_inactive_at
  where club_id::text = btrim(p_club_id)
    and id = v_source;

  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'player_merge_compensated',
    'operation_id', p_operation_id,
    'operation_status', 'compensated',
    'source_player_id', v_source,
    'target_player_id', v_target,
    'requires_replay', false
  );

  update public.admin_player_merge_operations
  set status = 'compensated', compensated_at = now(), updated_at = now(), result_json = result_json || v_result
  where id = p_operation_id;

  insert into public.admin_activity_log (
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
  ) values (
    btrim(p_club_id), lower(coalesce(nullif(btrim(p_actor_email), ''), 'unknown')),
    coalesce(nullif(btrim(p_actor_role), ''), 'unknown'),
    'compensate_player_editor_merge_admin', 'players', v_source::text || '->' || v_target::text,
    v_operation.result_json, v_result,
    'Restored pre-merge match references, league business fields, social links, and source player state; trigger-maintained timestamps may advance.',
    nullif(btrim(p_source_page), ''), true
  );

  return v_result;
end
$function$;

create or replace function public.server_verify_player_merge_replay(
  p_operation_id uuid,
  p_club_id text,
  p_replay_job_id uuid,
  p_actor_email text,
  p_actor_role text,
  p_source_page text default 'next_player_editor_merge_replay_evidence'
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.admin_player_merge_operations%rowtype;
  v_job public.replay_jobs%rowtype;
  v_result jsonb;
begin
  select operation.* into v_operation
  from public.admin_player_merge_operations as operation
  where operation.id = p_operation_id and operation.club_id = btrim(p_club_id)
  for update;
  if not found then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_OPERATION_NOT_FOUND');
  end if;
  if v_operation.status <> 'merged_pending_replay' then
    return jsonb_build_object('ok', false, 'code', 'PLAYER_MERGE_REPLAY_ALREADY_RESOLVED', 'status', v_operation.status);
  end if;

  select replay_job.* into v_job
  from public.replay_jobs as replay_job
  where replay_job.id = p_replay_job_id
    and replay_job.club_id = btrim(p_club_id)
  limit 1;
  if not found then
    return jsonb_build_object('ok', false, 'code', 'REPLAY_JOB_NOT_FOUND');
  end if;
  if lower(coalesce(v_job.status, '')) <> 'succeeded'
     or v_job.target_reset <> 'ALL (Full System Reset)'
     or v_job.created_at < v_operation.created_at then
    return jsonb_build_object(
      'ok', false,
      'code', 'REPLAY_JOB_NOT_VALID_RECOVERY_EVIDENCE',
      'job_status', v_job.status,
      'target_reset', v_job.target_reset
    );
  end if;

  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'player_merge_replay_verified',
    'operation_id', p_operation_id,
    'operation_status', 'replay_verified',
    'replay_job_id', p_replay_job_id,
    'requires_replay', false
  );

  update public.admin_player_merge_operations
  set status = 'replay_verified', replay_job_id = p_replay_job_id,
      replay_verified_at = now(), updated_at = now(), result_json = result_json || v_result
  where id = p_operation_id;

  insert into public.admin_activity_log (
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
  ) values (
    btrim(p_club_id), lower(coalesce(nullif(btrim(p_actor_email), ''), 'unknown')),
    coalesce(nullif(btrim(p_actor_role), ''), 'unknown'),
    'verify_player_editor_merge_replay_admin', 'players',
    v_operation.source_player_id::text || '->' || v_operation.target_player_id::text,
    v_operation.result_json, v_result,
    'Linked a succeeded full Replay History job as merge recovery evidence.',
    nullif(btrim(p_source_page), ''), true
  );

  return v_result;
end
$function$;

revoke execute on function public.server_merge_player_accounts(uuid, text, bigint, bigint, text, jsonb, text, text, text) from public, anon, authenticated;
revoke execute on function public.server_compensate_player_merge(uuid, text, text, text, text) from public, anon, authenticated;
revoke execute on function public.server_verify_player_merge_replay(uuid, text, uuid, text, text, text) from public, anon, authenticated;

grant execute on function public.server_merge_player_accounts(uuid, text, bigint, bigint, text, jsonb, text, text, text) to service_role;
grant execute on function public.server_compensate_player_merge(uuid, text, text, text, text) to service_role;
grant execute on function public.server_verify_player_merge_replay(uuid, text, uuid, text, text, text) to service_role;

comment on table public.admin_player_merge_operations is
  'Server-only recovery ledger for atomic Player Editor merges, compensation, and verified full-replay evidence.';
