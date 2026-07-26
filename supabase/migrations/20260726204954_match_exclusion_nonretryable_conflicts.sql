-- Expected match row-version conflicts are durable application outcomes, not
-- serialization failures. A retryable SQLSTATE invites transaction retries
-- and can prevent PostgREST from returning the conflict to the API for
-- classification.

create or replace function public.bump_match_row_version()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if new.row_version is distinct from old.row_version then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_MATCH_ROW_VERSION_IMMUTABLE: use an expected row_version predicate instead of assigning row_version.';
  end if;

  new.row_version := old.row_version + 1;
  return new;
end
$function$;

create or replace function public.apply_match_exclusions_atomic(
  p_operation_id uuid,
  p_club_id text,
  p_mode text,
  p_targets jsonb,
  p_badge_ids text[],
  p_badge_contract_version text,
  p_actor_email text,
  p_actor_role text,
  p_source text,
  p_delete_note text,
  p_idempotency_key text,
  p_replay_target text default 'ALL (Full System Reset)'
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_mode text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_mode, '')));
  v_actor_email text := nullif(
    pg_catalog.left(
      pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
      320
    ),
    ''
  );
  v_actor_role text := nullif(pg_catalog.btrim(p_actor_role), '');
  v_source text := nullif(pg_catalog.left(pg_catalog.btrim(p_source), 120), '');
  v_delete_note text := nullif(pg_catalog.left(pg_catalog.btrim(p_delete_note), 2000), '');
  v_idempotency_key text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_replay_target text :=
    coalesce(
      nullif(pg_catalog.left(pg_catalog.btrim(p_replay_target), 160), ''),
      'ALL (Full System Reset)'
    );
  v_badge_contract_version text :=
    nullif(
      pg_catalog.left(pg_catalog.btrim(p_badge_contract_version), 120),
      ''
    );
  v_badge_ids text[];
  v_targets jsonb;
  v_existing public.match_exclusion_operations%rowtype;
  v_match public.matches%rowtype;
  v_keeper public.matches%rowtype;
  v_target record;
  v_before_json jsonb := '[]'::jsonb;
  v_after_json jsonb := '[]'::jsonb;
  v_duplicate_keeper_json jsonb := '[]'::jsonb;
  v_target_ids bigint[];
  v_excluded_ids bigint[] := '{}'::bigint[];
  v_affected_player_ids bigint[] := '{}'::bigint[];
  v_replay_job_id uuid := gen_random_uuid();
  v_replay_key text;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_mode is null
     or v_mode not in ('exclude', 'duplicate_cleanup')
     or v_actor_email is null
     or v_actor_role is null
     or v_source is null
     or v_delete_note is null
     or v_idempotency_key is null
     or v_badge_contract_version is null
     or v_replay_target <> 'ALL (Full System Reset)' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_INVALID: operation, club, mode, actor, source, note, idempotency key, badge contract version, and the full-system replay target are required.';
  end if;

  if p_targets is null
     or pg_catalog.jsonb_typeof(p_targets) <> 'array'
     or pg_catalog.jsonb_array_length(p_targets) not between 1 and 100 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_TARGETS_INVALID: targets must contain 1 to 100 exact match versions.';
  end if;

  if exists (
    select 1
    from pg_catalog.jsonb_array_elements(p_targets) as target(item)
    where pg_catalog.jsonb_typeof(target.item) <> 'object'
       or not (target.item ?& array['match_id', 'expected_row_version'])
       or exists (
         select 1
         from pg_catalog.jsonb_object_keys(target.item) as target_key(key_name)
         where target_key.key_name not in ('match_id', 'expected_row_version')
       )
       or coalesce(target.item ->> 'match_id', '') !~ '^[1-9][0-9]*$'
       or coalesce(
         target.item ->> 'expected_row_version',
         ''
       ) !~ '^[1-9][0-9]*$'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_TARGET_INVALID: every target must contain only positive match_id and expected_row_version integers.';
  end if;

  if exists (
    select 1
    from (
      select (target.item ->> 'match_id')::bigint as match_id
      from pg_catalog.jsonb_array_elements(p_targets) as target(item)
    ) as parsed_target
    group by parsed_target.match_id
    having pg_catalog.count(*) > 1
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_TARGET_DUPLICATE: match IDs must be unique within one operation.';
  end if;

  select pg_catalog.jsonb_agg(
    pg_catalog.jsonb_build_object(
      'match_id',
      parsed_target.match_id,
      'expected_row_version',
      parsed_target.expected_row_version
    )
    order by parsed_target.match_id
  )
  into v_targets
  from (
    select
      (target.item ->> 'match_id')::bigint as match_id,
      (target.item ->> 'expected_row_version')::bigint
        as expected_row_version
    from pg_catalog.jsonb_array_elements(p_targets) as target(item)
  ) as parsed_target;

  select pg_catalog.array_agg(
    (target.item ->> 'match_id')::bigint
    order by (target.item ->> 'match_id')::bigint
  )
  into v_target_ids
  from pg_catalog.jsonb_array_elements(v_targets) as target(item);

  select pg_catalog.array_agg(
    normalized.badge_id order by normalized.badge_id
  )
  into v_badge_ids
  from (
    select distinct nullif(pg_catalog.btrim(badge.badge_id), '') as badge_id
    from pg_catalog.unnest(p_badge_ids) as badge(badge_id)
  ) as normalized
  where normalized.badge_id is not null;

  if v_badge_ids is null
     or pg_catalog.cardinality(v_badge_ids) not between 1 and 128
     or pg_catalog.cardinality(v_badge_ids)
        <> pg_catalog.cardinality(p_badge_ids) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_BADGE_ALLOWLIST_INVALID: provide 1 to 128 distinct, nonblank badge IDs.';
  end if;

  if exists (
    select 1
    from pg_catalog.unnest(v_badge_ids) as requested_badge(badge_id)
    where not exists (
      select 1
      from public.badges as badge
      where badge.badge_id = requested_badge.badge_id
        and pg_catalog.lower(coalesce(badge.state::text, 'live'))
          = 'live'
        and coalesce(badge.is_active, true)
    )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_BADGE_ALLOWLIST_UNKNOWN: every frozen badge ID must be an active live badge definition.';
  end if;

  -- Serialize both idempotency and club-wide replay ownership. Replay claims
  -- use the same club lock, so an exclusion cannot race an active reset.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'jupr:match-exclusion:' || v_club_id || ':' || v_idempotency_key,
      0
    )
  );
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select operation.*
  into v_existing
  from public.match_exclusion_operations as operation
  where operation.id = p_operation_id
  for update;

  if found then
    if v_existing.club_id <> v_club_id
       or v_existing.idempotency_key <> v_idempotency_key
       or v_existing.mode <> v_mode
       or v_existing.targets_json is distinct from v_targets
       or v_existing.badge_ids is distinct from v_badge_ids
       or v_existing.badge_contract_version <> v_badge_contract_version
       or v_existing.replay_target <> v_replay_target
       or v_existing.actor_email <> v_actor_email
       or v_existing.actor_role <> v_actor_role
       or v_existing.source <> v_source
       or v_existing.delete_note <> v_delete_note then
      raise exception using
        errcode = '23505',
        message = 'JUPR_MATCH_EXCLUSION_IDEMPOTENCY_CONFLICT: operation ID is already bound to another request.';
    end if;

    return pg_catalog.jsonb_build_object(
      'ok', true,
      'operation_id', v_existing.id,
      'operation_status', v_existing.status,
      'mode', v_existing.mode,
      'duplicate_keepers', v_existing.duplicate_keeper_json,
      'excluded_ids', pg_catalog.to_jsonb(v_existing.excluded_match_ids),
      'excluded_count', pg_catalog.cardinality(v_existing.excluded_match_ids),
      'affected_player_ids',
        pg_catalog.to_jsonb(v_existing.affected_player_ids),
      'replay_job_id', v_existing.replay_job_id,
      'replay_target', v_existing.replay_target,
      'badge_ids', pg_catalog.to_jsonb(v_existing.badge_ids),
      'badge_contract_version', v_existing.badge_contract_version,
      'result_json', v_existing.result_json,
      'error_text', v_existing.error_text,
      'idempotent', true
    );
  end if;

  select operation.*
  into v_existing
  from public.match_exclusion_operations as operation
  where operation.club_id = v_club_id
    and operation.idempotency_key = v_idempotency_key
  for update;

  if found then
    if v_existing.mode <> v_mode
       or v_existing.targets_json is distinct from v_targets
       or v_existing.badge_ids is distinct from v_badge_ids
       or v_existing.badge_contract_version <> v_badge_contract_version
       or v_existing.replay_target <> v_replay_target
       or v_existing.actor_email <> v_actor_email
       or v_existing.actor_role <> v_actor_role
       or v_existing.source <> v_source
       or v_existing.delete_note <> v_delete_note then
      raise exception using
        errcode = '23505',
        message = 'JUPR_MATCH_EXCLUSION_IDEMPOTENCY_CONFLICT: idempotency key is already bound to a different request body.';
    end if;

    return pg_catalog.jsonb_build_object(
      'ok', true,
      'operation_id', v_existing.id,
      'operation_status', v_existing.status,
      'mode', v_existing.mode,
      'duplicate_keepers', v_existing.duplicate_keeper_json,
      'excluded_ids', pg_catalog.to_jsonb(v_existing.excluded_match_ids),
      'excluded_count', pg_catalog.cardinality(v_existing.excluded_match_ids),
      'affected_player_ids',
        pg_catalog.to_jsonb(v_existing.affected_player_ids),
      'replay_job_id', v_existing.replay_job_id,
      'replay_target', v_existing.replay_target,
      'badge_ids', pg_catalog.to_jsonb(v_existing.badge_ids),
      'badge_contract_version', v_existing.badge_contract_version,
      'result_json', v_existing.result_json,
      'error_text', v_existing.error_text,
      'idempotent', true
    );
  end if;

  -- Exact-operation and exact-idempotency retries have already returned
  -- above. Any other nonterminal operation owns this club's destructive
  -- recovery lane until it succeeds.
  select operation.*
  into v_existing
  from public.match_exclusion_operations as operation
  where operation.club_id = v_club_id
    and operation.status <> 'succeeded'
  order by operation.created_at, operation.id
  limit 1
  for update;

  if found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_OPERATION_IN_PROGRESS',
      'message',
        'Another match exclusion operation still requires replay or badge recovery.',
      'operation_id', v_existing.id,
      'operation_status', v_existing.status,
      'recovery_stage', case
        when v_existing.status = 'pending_replay' then 'replay'
        when v_existing.status = 'pending_badge_reconcile'
          then 'badge_reconcile'
        else v_existing.recovery_stage
      end,
      'replay_job_id', v_existing.replay_job_id
    );
  end if;

  if exists (
    select 1
    from public.replay_jobs as replay_job
    where replay_job.club_id = v_club_id
      and replay_job.status in ('pending', 'running')
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_REPLAY_IN_PROGRESS'
    );
  end if;

  -- Lock in deterministic ID order. The row-version comparison is repeated in
  -- the UPDATE so the trigger and predicate together form an exact CAS.
  for v_target in
    select
      (target.item ->> 'match_id')::bigint as match_id,
      (target.item ->> 'expected_row_version')::bigint
        as expected_row_version
    from pg_catalog.jsonb_array_elements(v_targets) as target(item)
    order by (target.item ->> 'match_id')::bigint
  loop
    select match_row.*
    into v_match
    from public.matches as match_row
    where match_row.club_id::text = v_club_id
      and match_row.id = v_target.match_id
    for update;

    if not found then
      raise exception using
        errcode = 'P0002',
        message = pg_catalog.format(
          'JUPR_MATCH_EXCLUSION_NOT_FOUND: match %s is not in club %s.',
          v_target.match_id,
          v_club_id
        );
    end if;

    if v_match.deleted_at is not null then
      raise exception using
        errcode = '22023',
        message = pg_catalog.format(
          'JUPR_MATCH_EXCLUSION_ALREADY_EXCLUDED: match %s is already excluded.',
          v_target.match_id
        );
    end if;

    if v_match.row_version <> v_target.expected_row_version then
      raise exception using
        errcode = 'P0001',
        message = pg_catalog.format(
          'JUPR_MATCH_EXCLUSION_STALE: match %s expected row_version %s but is %s.',
          v_target.match_id,
          v_target.expected_row_version,
          v_match.row_version
        );
    end if;

    if v_mode = 'duplicate_cleanup' then
      select keeper_row.*
      into v_keeper
      from public.matches as keeper_row
      where keeper_row.club_id::text = v_club_id
        and keeper_row.id < v_target.match_id
        and not (keeper_row.id = any(v_target_ids))
        and keeper_row.deleted_at is null
        and public.match_exclusion_canonical_duplicate_key(
          v_club_id,
          keeper_row.league,
          keeper_row.week_tag,
          keeper_row.match_type,
          keeper_row.t1_p1,
          keeper_row.t1_p2,
          keeper_row.t2_p1,
          keeper_row.t2_p2,
          keeper_row.score_t1,
          keeper_row.score_t2
        ) = public.match_exclusion_canonical_duplicate_key(
          v_club_id,
          v_match.league,
          v_match.week_tag,
          v_match.match_type,
          v_match.t1_p1,
          v_match.t1_p2,
          v_match.t2_p1,
          v_match.t2_p2,
          v_match.score_t1,
          v_match.score_t2
        )
      order by keeper_row.id
      limit 1
      for update;

      if not found then
        raise exception using
          errcode = 'P0001',
          message = pg_catalog.format(
            'JUPR_MATCH_EXCLUSION_DUPLICATE_KEEPER_STALE: match %s no longer has an active lower-ID canonical duplicate outside the exclusion set.',
            v_target.match_id
          );
      end if;

      v_duplicate_keeper_json :=
        v_duplicate_keeper_json
        || pg_catalog.jsonb_build_array(
          pg_catalog.jsonb_build_object(
            'match_id', v_target.match_id,
            'keeper_match_id', v_keeper.id,
            'keeper_row_version', v_keeper.row_version,
            'canonical_key',
              public.match_exclusion_canonical_duplicate_key(
                v_club_id,
                v_keeper.league,
                v_keeper.week_tag,
                v_keeper.match_type,
                v_keeper.t1_p1,
                v_keeper.t1_p2,
                v_keeper.t2_p1,
                v_keeper.t2_p2,
                v_keeper.score_t1,
                v_keeper.score_t2
              )
          )
        );
    end if;

    v_before_json :=
      v_before_json || pg_catalog.jsonb_build_array(pg_catalog.to_jsonb(v_match));
    v_excluded_ids :=
      pg_catalog.array_append(v_excluded_ids, v_target.match_id);
    v_affected_player_ids :=
      v_affected_player_ids
      || pg_catalog.array_remove(
        array[v_match.t1_p1, v_match.t1_p2, v_match.t2_p1, v_match.t2_p2]
          ::bigint[],
        null
      );

    update public.matches as match_row
    set
      deleted_at = v_now,
      deleted_by = v_actor_email,
      deleted_source = v_source,
      delete_note = v_delete_note,
      updated_at = v_now,
      updated_by = v_actor_email
    where match_row.club_id::text = v_club_id
      and match_row.id = v_target.match_id
      and match_row.row_version = v_target.expected_row_version
      and match_row.deleted_at is null;

    if not found then
      raise exception using
        errcode = 'P0001',
        message = pg_catalog.format(
          'JUPR_MATCH_EXCLUSION_STALE: match %s changed before exclusion.',
          v_target.match_id
        );
    end if;

    select match_row.*
    into v_match
    from public.matches as match_row
    where match_row.club_id::text = v_club_id
      and match_row.id = v_target.match_id;

    v_after_json :=
      v_after_json || pg_catalog.jsonb_build_array(pg_catalog.to_jsonb(v_match));
  end loop;

  -- A full replay is transitive: removing one early result can change ratings
  -- and rating-dependent badges for players in later matches who never played
  -- the removed row. Freeze every club player as the durable reconciliation
  -- scope, not merely the direct participants in the excluded targets.
  select pg_catalog.array_agg(player_row.id order by player_row.id)
  into v_affected_player_ids
  from public.players as player_row
  where player_row.club_id::text = v_club_id;

  if v_affected_player_ids is null
     or pg_catalog.cardinality(v_affected_player_ids) not between 1 and 10000 then
    raise exception using
      errcode = '54000',
      message = 'JUPR_MATCH_EXCLUSION_BADGE_SCOPE_INVALID: full replay badge recovery requires 1 to 10000 exact club player IDs.';
  end if;

  v_replay_key := 'match-exclusion:' || p_operation_id::text;

  insert into public.replay_jobs (
    id,
    club_id,
    target_reset,
    status,
    actor_email,
    actor_role,
    idempotency_key,
    source,
    attempt_count,
    result_json,
    created_at,
    updated_at
  ) values (
    v_replay_job_id,
    v_club_id,
    v_replay_target,
    'pending',
    v_actor_email,
    v_actor_role,
    v_replay_key,
    v_source,
    0,
    '{}'::jsonb,
    v_now,
    v_now
  );

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'operation_id', p_operation_id,
    'operation_status', 'pending_replay',
    'mode', v_mode,
    'duplicate_keepers', v_duplicate_keeper_json,
    'excluded_ids', pg_catalog.to_jsonb(v_excluded_ids),
    'excluded_count', pg_catalog.cardinality(v_excluded_ids),
    'affected_player_ids', pg_catalog.to_jsonb(v_affected_player_ids),
    'replay_job_id', v_replay_job_id,
    'replay_target', v_replay_target,
    'badge_ids', pg_catalog.to_jsonb(v_badge_ids),
    'badge_contract_version', v_badge_contract_version,
    'idempotent', false
  );

  insert into public.match_exclusion_operations (
    id,
    club_id,
    mode,
    idempotency_key,
    status,
    targets_json,
    duplicate_keeper_json,
    before_json,
    after_json,
    excluded_match_ids,
    affected_player_ids,
    badge_ids,
    badge_contract_version,
    replay_target,
    replay_job_id,
    result_json,
    actor_email,
    actor_role,
    source,
    delete_note,
    created_at,
    updated_at
  ) values (
    p_operation_id,
    v_club_id,
    v_mode,
    v_idempotency_key,
    'pending_replay',
    v_targets,
    v_duplicate_keeper_json,
    v_before_json,
    v_after_json,
    v_excluded_ids,
    v_affected_player_ids,
    v_badge_ids,
    v_badge_contract_version,
    v_replay_target,
    v_replay_job_id,
    v_result,
    v_actor_email,
    v_actor_role,
    v_source,
    v_delete_note,
    v_now,
    v_now
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
    v_actor_email,
    v_actor_role,
    case
      when v_mode = 'duplicate_cleanup'
        then 'match_duplicate_cleanup_atomic'
      else 'match_bulk_exclusion_atomic'
    end,
    'match_exclusion_operation',
    p_operation_id::text,
    v_before_json,
    pg_catalog.jsonb_build_object(
      'matches', v_after_json,
      'operation', v_result,
      'duplicate_keepers', v_duplicate_keeper_json,
      'badge_contract_version', v_badge_contract_version
    ),
    v_delete_note,
    v_source,
    true
  );

  return v_result;
end
$function$;

revoke all on function public.bump_match_row_version()
  from public, anon, authenticated;
grant execute on function public.bump_match_row_version()
  to service_role;

revoke all on function public.apply_match_exclusions_atomic(
  uuid,
  text,
  text,
  jsonb,
  text[],
  text,
  text,
  text,
  text,
  text,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.apply_match_exclusions_atomic(
  uuid,
  text,
  text,
  jsonb,
  text[],
  text,
  text,
  text,
  text,
  text,
  text,
  text
) to service_role;

notify pgrst, 'reload schema';
