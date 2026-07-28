-- Serialize every Match Log mutation and replay claim for a club behind one
-- transaction-scoped advisory lock. Route preflight remains useful feedback,
-- while these guards are the authoritative race-free boundary.

-- Fail atomically before installing the mutex if legacy state already has
-- more than one recovery owner for a club. Picking a winner would make the
-- other operation unrecoverable, so an operator must first complete or repair
-- all but one exact operation and then reapply this migration.
do $migration_preflight$
declare
  v_conflict jsonb;
begin
  with open_operation as (
    select
      operation.club_id,
      'match_edit'::text as operation_kind,
      operation.id::text as operation_id,
      operation.status as operation_status,
      operation.created_at
    from public.match_edit_operations as operation
    where operation.status in ('pending_replay', 'recovery_required')

    union all

    select
      operation.club_id,
      'match_exclusion'::text as operation_kind,
      operation.id::text as operation_id,
      operation.status as operation_status,
      operation.created_at
    from public.match_exclusion_operations as operation
    where operation.status in (
      'pending_replay',
      'pending_badge_reconcile',
      'recovery_required'
    )
  ),
  conflicting_club as (
    select open_operation.club_id
    from open_operation
    group by open_operation.club_id
    having pg_catalog.count(*) > 1
    order by open_operation.club_id
    limit 1
  )
  select pg_catalog.jsonb_build_object(
    'club_id',
    open_operation.club_id,
    'open_operations',
    pg_catalog.jsonb_agg(
      pg_catalog.jsonb_build_object(
        'operation_kind',
        open_operation.operation_kind,
        'operation_id',
        open_operation.operation_id,
        'operation_status',
        open_operation.operation_status
      )
      order by
        open_operation.created_at,
        open_operation.operation_kind,
        open_operation.operation_id
    )
  )
  into v_conflict
  from open_operation
  join conflicting_club
    on conflicting_club.club_id = open_operation.club_id
  group by open_operation.club_id;

  if v_conflict is not null then
    raise exception using
      errcode = '55000',
      message =
        'JUPR_MATCH_LOG_RECOVERY_MIGRATION_CONFLICT: multiple recovery operations already exist for one club.',
      detail = v_conflict::text,
      hint =
        'Complete or repair all but one exact recovery operation before reapplying this migration.';
  end if;
end
$migration_preflight$;

create or replace function public.assert_match_log_recovery_guard_atomic(
  p_club_id text,
  p_exact_operation_kind text default null,
  p_exact_operation_id uuid default null,
  p_exact_replay_job_id uuid default null,
  p_check_replay_jobs boolean default true
)
returns void
language plpgsql
security definer
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_exact_operation_kind text :=
    nullif(pg_catalog.lower(pg_catalog.btrim(p_exact_operation_kind)), '');
  v_open_operations jsonb;
  v_open_operation jsonb;
  v_open_count integer;
  v_conflicting_replay_job_id uuid;
begin
  if v_club_id is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_LOG_RECOVERY_GUARD_INVALID: club is required.';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select coalesce(
    pg_catalog.jsonb_agg(
      pg_catalog.jsonb_build_object(
        'operation_kind',
        open_operation.operation_kind,
        'operation_id',
        open_operation.operation_id,
        'operation_status',
        open_operation.operation_status,
        'replay_job_id',
        open_operation.replay_job_id,
        'recovery_stage',
        open_operation.recovery_stage
      )
      order by
        open_operation.created_at,
        open_operation.operation_kind,
        open_operation.operation_id
    ),
    '[]'::jsonb
  )
  into v_open_operations
  from (
    select
      'match_edit'::text as operation_kind,
      operation.id::text as operation_id,
      operation.status as operation_status,
      operation.replay_job_id::text as replay_job_id,
      null::text as recovery_stage,
      operation.created_at
    from public.match_edit_operations as operation
    where operation.club_id = v_club_id
      and operation.status in ('pending_replay', 'recovery_required')

    union all

    select
      'match_exclusion'::text as operation_kind,
      operation.id::text as operation_id,
      operation.status as operation_status,
      operation.replay_job_id::text as replay_job_id,
      operation.recovery_stage,
      operation.created_at
    from public.match_exclusion_operations as operation
    where operation.club_id = v_club_id
      and operation.status in (
        'pending_replay',
        'pending_badge_reconcile',
        'recovery_required'
      )
  ) as open_operation;

  v_open_count := pg_catalog.jsonb_array_length(v_open_operations);
  if v_open_count > 1 then
    raise exception using
      errcode = '55000',
      message =
        'JUPR_MATCH_LOG_RECOVERY_LOCK_AMBIGUOUS: multiple recovery operations are open for this club.',
      detail = v_open_operations::text,
      hint =
        'Repair the recovery ledger before starting another Match Log write or replay.';
  end if;

  if v_open_count = 1 then
    v_open_operation := v_open_operations -> 0;
    if not (
      (
        v_exact_operation_kind =
          v_open_operation ->> 'operation_kind'
        and p_exact_operation_id is not null
        and p_exact_operation_id::text =
          v_open_operation ->> 'operation_id'
      )
      or
      (
        p_exact_replay_job_id is not null
        and p_exact_replay_job_id::text =
          v_open_operation ->> 'replay_job_id'
      )
    ) then
      raise exception using
        errcode = '55000',
        message =
          'JUPR_MATCH_LOG_RECOVERY_LOCKED: an exact Match Log recovery operation owns this club.',
        detail = v_open_operation::text,
        hint =
          'Complete the exact recovery operation before starting another Match Log write or replay.';
    end if;
  end if;

  if coalesce(p_check_replay_jobs, true) then
    select replay_job.id
    into v_conflicting_replay_job_id
    from public.replay_jobs as replay_job
    where replay_job.club_id = v_club_id
      -- A pending row has no lease and cannot write. Treating it as an owner
      -- would let an interruption between durable creation and claim wedge
      -- every Match Log write forever. The claim transition is guarded below,
      -- and any pending replay frozen into an open Match Log operation is
      -- still protected by the exact-operation check above.
      and replay_job.status = 'running'
      and (
        p_exact_replay_job_id is null
        or replay_job.id <> p_exact_replay_job_id
      )
    order by replay_job.created_at, replay_job.id
    limit 1;

    if found then
      raise exception using
        errcode = '55000',
        message =
          'JUPR_MATCH_LOG_RECOVERY_LOCKED: a replay job already owns this club.',
        detail = pg_catalog.jsonb_build_object(
          'operation_kind',
          'replay_job',
          'replay_job_id',
          v_conflicting_replay_job_id
        )::text,
        hint =
          'Complete the active replay before starting another Match Log write.';
    end if;
  end if;
end
$function$;

revoke all on function public.assert_match_log_recovery_guard_atomic(
  text,
  text,
  uuid,
  uuid,
  boolean
) from public, anon, authenticated, service_role;

-- Keep the original RPC contract, but make the public entry point hold the
-- club lock from its authoritative guard through the complete mutation.
alter function public.apply_match_log_patches_atomic(
  text,
  jsonb,
  text,
  text,
  text,
  text,
  text,
  text
) rename to apply_match_log_patches_atomic_unguarded;

revoke all on function public.apply_match_log_patches_atomic_unguarded(
  text,
  jsonb,
  text,
  text,
  text,
  text,
  text,
  text
) from public, anon, authenticated, service_role;

create or replace function public.apply_match_log_patches_atomic(
  p_club_id text,
  p_patches jsonb,
  p_actor_email text,
  p_actor_role text,
  p_source text,
  p_correction_note text,
  p_idempotency_key text,
  p_replay_target text default 'ALL (Full System Reset)'
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_idempotency_key text :=
    nullif(
      pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160),
      ''
    );
  v_is_retry boolean;
begin
  if v_club_id is null or v_idempotency_key is null then
    raise exception using
      errcode = '22023',
      message =
        'JUPR_MATCH_LOG_EDIT_INVALID: club and a stable idempotency key are required.';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select exists (
    select 1
    from public.match_edit_operations as operation
    where operation.club_id = v_club_id
      and operation.idempotency_key = v_idempotency_key
  )
  into v_is_retry;

  if not v_is_retry then
    perform public.assert_match_log_recovery_guard_atomic(
      v_club_id,
      null,
      null,
      null,
      true
    );
  end if;

  return public.apply_match_log_patches_atomic_unguarded(
    v_club_id,
    p_patches,
    p_actor_email,
    p_actor_role,
    p_source,
    p_correction_note,
    v_idempotency_key,
    p_replay_target
  );
end
$function$;

revoke all on function public.apply_match_log_patches_atomic(
  text,
  jsonb,
  text,
  text,
  text,
  text,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.apply_match_log_patches_atomic(
  text,
  jsonb,
  text,
  text,
  text,
  text,
  text,
  text
) to service_role;

alter function public.apply_match_exclusions_atomic(
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
) rename to apply_match_exclusions_atomic_unguarded;

revoke all on function public.apply_match_exclusions_atomic_unguarded(
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
) from public, anon, authenticated, service_role;

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
security definer
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_idempotency_key text :=
    nullif(
      pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160),
      ''
    );
  v_is_retry boolean;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_idempotency_key is null then
    raise exception using
      errcode = '22023',
      message =
        'JUPR_MATCH_EXCLUSION_INVALID: operation, club, and idempotency key are required.';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select exists (
    select 1
    from public.match_exclusion_operations as operation
    where operation.id = p_operation_id
       or (
         operation.club_id = v_club_id
         and operation.idempotency_key = v_idempotency_key
       )
  )
  into v_is_retry;

  if not v_is_retry then
    perform public.assert_match_log_recovery_guard_atomic(
      v_club_id,
      null,
      null,
      null,
      true
    );
  end if;

  return public.apply_match_exclusions_atomic_unguarded(
    p_operation_id,
    v_club_id,
    p_mode,
    p_targets,
    p_badge_ids,
    p_badge_contract_version,
    p_actor_email,
    p_actor_role,
    p_source,
    p_delete_note,
    v_idempotency_key,
    p_replay_target
  );
end
$function$;

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

-- These triggers make bypassing the guarded RPCs fail closed. They also close
-- the insert-after-preflight race if a pending replay appears mid-request.
create or replace function public.guard_match_log_operation_insert()
returns trigger
language plpgsql
security definer
set search_path = ''
as $function$
declare
  v_operation_kind text;
begin
  v_operation_kind := case pg_catalog.lower(tg_table_name)
    when 'match_edit_operations' then 'match_edit'
    when 'match_exclusion_operations' then 'match_exclusion'
    else null
  end;

  if v_operation_kind is null then
    raise exception using
      errcode = '55000',
      message =
        'JUPR_MATCH_LOG_RECOVERY_GUARD_INVALID: unsupported operation ledger.';
  end if;

  perform public.assert_match_log_recovery_guard_atomic(
    new.club_id,
    v_operation_kind,
    new.id,
    new.replay_job_id,
    true
  );
  return new;
end
$function$;

revoke all on function public.guard_match_log_operation_insert()
  from public, anon, authenticated, service_role;

drop trigger if exists trg_match_edit_operation_recovery_guard
  on public.match_edit_operations;
create trigger trg_match_edit_operation_recovery_guard
before insert on public.match_edit_operations
for each row
execute function public.guard_match_log_operation_insert();

drop trigger if exists trg_match_exclusion_operation_recovery_guard
  on public.match_exclusion_operations;
create trigger trg_match_exclusion_operation_recovery_guard
before insert on public.match_exclusion_operations
for each row
execute function public.guard_match_log_operation_insert();

create or replace function public.guard_match_log_duplicate_resolution()
returns trigger
language plpgsql
security definer
set search_path = ''
as $function$
begin
  perform public.assert_match_log_recovery_guard_atomic(
    new.club_id,
    null,
    null,
    null,
    true
  );
  return new;
end
$function$;

revoke all on function public.guard_match_log_duplicate_resolution()
  from public, anon, authenticated, service_role;

drop trigger if exists trg_match_log_duplicate_resolution_recovery_guard
  on public.admin_match_log_duplicate_resolutions;
create trigger trg_match_log_duplicate_resolution_recovery_guard
before insert or update on public.admin_match_log_duplicate_resolutions
for each row
execute function public.guard_match_log_duplicate_resolution();

-- Replay claims are the authoritative point where a generic pending request
-- becomes capable of writing ratings. Only the replay job frozen into the one
-- open recovery operation may cross this boundary.
create or replace function public.guard_match_log_replay_claim()
returns trigger
language plpgsql
security definer
set search_path = ''
as $function$
begin
  if new.status = 'running'
     and old.status is distinct from new.status then
    perform public.assert_match_log_recovery_guard_atomic(
      new.club_id,
      null,
      null,
      new.id,
      false
    );
  end if;
  return new;
end
$function$;

revoke all on function public.guard_match_log_replay_claim()
  from public, anon, authenticated, service_role;

drop trigger if exists trg_match_log_replay_claim_recovery_guard
  on public.replay_jobs;
create trigger trg_match_log_replay_claim_recovery_guard
before update of status on public.replay_jobs
for each row
execute function public.guard_match_log_replay_claim();

notify pgrst, 'reload schema';
