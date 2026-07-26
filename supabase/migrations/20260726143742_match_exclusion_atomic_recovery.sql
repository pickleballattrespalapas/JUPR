-- Atomic, exact-ID Match Log exclusions with leased Replay History recovery
-- and narrow, per-player live-badge reconciliation.
--
-- This migration is intentionally server-only. FastAPI authenticates and
-- authorizes the operator, then calls these RPCs through the service-role
-- client. No browser role can read either recovery ledger or execute an RPC.

do $migration_preflight$
declare
  v_missing_columns text[];
  v_revoked_by_type text;
begin
  if to_regclass('public.matches') is null
     or to_regclass('public.players') is null
     or to_regclass('public.replay_jobs') is null
     or to_regclass('public.badges') is null
     or to_regclass('public.player_badges') is null
     or to_regclass('public.admin_activity_log') is null then
    raise exception using
      errcode = '42P01',
      message = 'JUPR_MATCH_EXCLUSION_SCHEMA_MISSING: matches, players, replay_jobs, badges, player_badges, and admin_activity_log must exist first.';
  end if;

  select pg_catalog.array_agg(required.column_name order by required.column_name)
  into v_missing_columns
  from (
    values
      ('club_id'),
      ('player_id'),
      ('badge_id'),
      ('context_type'),
      ('context_id'),
      ('match_id'),
      ('value_num'),
      ('value_json'),
      ('awarded_by'),
      ('rule_version'),
      ('revoked_at'),
      ('revoked_by'),
      ('revoke_reason')
  ) as required(column_name)
  where not exists (
    select 1
    from information_schema.columns as column_info
    where column_info.table_schema = 'public'
      and column_info.table_name = 'player_badges'
      and column_info.column_name = required.column_name
  );

  if pg_catalog.cardinality(v_missing_columns) > 0 then
    raise exception using
      errcode = '42703',
      message = 'JUPR_MATCH_EXCLUSION_BADGE_SCHEMA_MISSING: player_badges lacks required provenance/revocation columns.',
      detail = pg_catalog.array_to_string(v_missing_columns, ', ');
  end if;

  select column_info.udt_name
  into v_revoked_by_type
  from information_schema.columns as column_info
  where column_info.table_schema = 'public'
    and column_info.table_name = 'player_badges'
    and column_info.column_name = 'revoked_by';

  -- Staging currently uses auth-user UUIDs while the reproducible legacy
  -- chain used text. Reconciliation deliberately leaves revoked_by NULL and
  -- writes the sanitized operator email into revoke_reason/audit evidence, so
  -- both known shapes are safe. Refuse any unreviewed third shape.
  if v_revoked_by_type not in ('uuid', 'text') then
    raise exception using
      errcode = '42804',
      message = 'JUPR_MATCH_EXCLUSION_REVOKED_BY_TYPE_UNSUPPORTED: player_badges.revoked_by must be uuid or legacy text.';
  end if;

  -- Adding a strict lease-shape constraint while a legacy replay is running
  -- could strand that worker. Apply this migration only during a quiescent
  -- replay window, then deploy the leased worker before re-enabling writes.
  if exists (
    select 1
    from public.replay_jobs as replay_job
    where pg_catalog.lower(coalesce(replay_job.status, '')) = 'running'
  ) then
    raise exception using
      errcode = '55006',
      message = 'JUPR_REPLAY_MIGRATION_BUSY: finish or recover every running Replay History job before applying this migration.';
  end if;

  if exists (
    select 1
    from public.replay_jobs as replay_job
    where pg_catalog.lower(coalesce(replay_job.status, ''))
      not in ('pending', 'succeeded', 'failed', 'cancelled', 'canceled')
  ) then
    raise exception using
      errcode = '23514',
      message = 'JUPR_REPLAY_STATUS_INVALID: replay_jobs contains an unsupported legacy status.';
  end if;
end
$migration_preflight$;

-- Re-establish the live five-column badge identity on databases created from
-- the older reproducible chain, which briefly omitted context_type. The
-- five-column key is also the exact key used by narrow reconciliation below.
do $player_badges_unique_contract$
declare
  v_constraint_name text;
begin
  for v_constraint_name in
    select constraint_row.conname
    from pg_catalog.pg_constraint as constraint_row
    where constraint_row.conrelid = 'public.player_badges'::regclass
      and constraint_row.contype = 'u'
      and (
        select pg_catalog.array_agg(
          attribute_row.attname::text
          order by constraint_column.ordinality
        )
        from pg_catalog.unnest(constraint_row.conkey)
          with ordinality as constraint_column(attnum, ordinality)
        join pg_catalog.pg_attribute as attribute_row
          on attribute_row.attrelid = constraint_row.conrelid
         and attribute_row.attnum = constraint_column.attnum
      ) = array[
        'club_id',
        'player_id',
        'badge_id',
        'context_id'
      ]::text[]
  loop
    execute pg_catalog.format(
      'alter table public.player_badges drop constraint %I',
      v_constraint_name
    );
  end loop;

  if not exists (
    select 1
    from pg_catalog.pg_constraint as constraint_row
    where constraint_row.conrelid = 'public.player_badges'::regclass
      and constraint_row.contype = 'u'
      and (
        select pg_catalog.array_agg(
          attribute_row.attname::text
          order by constraint_column.ordinality
        )
        from pg_catalog.unnest(constraint_row.conkey)
          with ordinality as constraint_column(attnum, ordinality)
        join pg_catalog.pg_attribute as attribute_row
          on attribute_row.attrelid = constraint_row.conrelid
         and attribute_row.attnum = constraint_column.attnum
      ) = array[
        'club_id',
        'player_id',
        'badge_id',
        'context_type',
        'context_id'
      ]::text[]
  ) then
    alter table public.player_badges
      add constraint player_badges_unique_context
      unique (club_id, player_id, badge_id, context_type, context_id);
  end if;
end
$player_badges_unique_contract$;

-- Every match mutation advances one immutable version. Callers perform CAS in
-- the UPDATE predicate; the trigger prevents them from forging the next value.
alter table public.matches
  add column if not exists row_version bigint not null default 1;

do $match_row_version_constraint$
begin
  if not exists (
    select 1
    from pg_catalog.pg_constraint as constraint_row
    where constraint_row.conname = 'matches_row_version_positive_check'
      and constraint_row.conrelid = 'public.matches'::regclass
  ) then
    alter table public.matches
      add constraint matches_row_version_positive_check
      check (row_version > 0);
  end if;
end
$match_row_version_constraint$;

create or replace function public.bump_match_row_version()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if new.row_version is distinct from old.row_version then
    raise exception using
      errcode = '40001',
      message = 'JUPR_MATCH_ROW_VERSION_IMMUTABLE: use an expected row_version predicate instead of assigning row_version.';
  end if;

  new.row_version := old.row_version + 1;
  return new;
end
$function$;

drop trigger if exists trg_matches_bump_row_version on public.matches;
create trigger trg_matches_bump_row_version
before update on public.matches
for each row
execute function public.bump_match_row_version();

revoke all on function public.bump_match_row_version()
  from public, anon, authenticated;
grant execute on function public.bump_match_row_version()
  to service_role;

-- Keep duplicate-cleanup identity byte-for-byte aligned with
-- jupr_app.domain.dupes.canonical_dup_key. Players are sorted within teams;
-- teams are sorted against each other with scores swapped alongside them.
create or replace function public.match_exclusion_canonical_duplicate_key(
  p_club_id text,
  p_league text,
  p_week_tag text,
  p_match_type text,
  p_t1_p1 bigint,
  p_t1_p2 bigint,
  p_t2_p1 bigint,
  p_t2_p2 bigint,
  p_score_t1 integer,
  p_score_t2 integer
)
returns text
language plpgsql
immutable
security invoker
set search_path = ''
as $function$
declare
  v_team_a_low bigint := case
    when coalesce(p_t1_p1, -1) <= coalesce(p_t1_p2, -1)
      then coalesce(p_t1_p1, -1)
    else coalesce(p_t1_p2, -1)
  end;
  v_team_a_high bigint := case
    when coalesce(p_t1_p1, -1) <= coalesce(p_t1_p2, -1)
      then coalesce(p_t1_p2, -1)
    else coalesce(p_t1_p1, -1)
  end;
  v_team_b_low bigint := case
    when coalesce(p_t2_p1, -1) <= coalesce(p_t2_p2, -1)
      then coalesce(p_t2_p1, -1)
    else coalesce(p_t2_p2, -1)
  end;
  v_team_b_high bigint := case
    when coalesce(p_t2_p1, -1) <= coalesce(p_t2_p2, -1)
      then coalesce(p_t2_p2, -1)
    else coalesce(p_t2_p1, -1)
  end;
  v_score_a integer := coalesce(p_score_t1, -1);
  v_score_b integer := coalesce(p_score_t2, -1);
  v_swap bigint;
  v_score_swap integer;
begin
  if array[v_team_b_low, v_team_b_high]
     < array[v_team_a_low, v_team_a_high] then
    v_swap := v_team_a_low;
    v_team_a_low := v_team_b_low;
    v_team_b_low := v_swap;

    v_swap := v_team_a_high;
    v_team_a_high := v_team_b_high;
    v_team_b_high := v_swap;

    v_score_swap := v_score_a;
    v_score_a := v_score_b;
    v_score_b := v_score_swap;
  end if;

  return
    coalesce(p_club_id, '')
    || '|' || pg_catalog.btrim(coalesce(p_league, ''))
    || '|' || pg_catalog.btrim(coalesce(p_week_tag, ''))
    || '|' || pg_catalog.btrim(coalesce(p_match_type, ''))
    || '|' || v_team_a_low::text || '-' || v_team_a_high::text
    || '|' || v_team_b_low::text || '-' || v_team_b_high::text
    || '|' || v_score_a::text || '-' || v_score_b::text;
end
$function$;

revoke all on function public.match_exclusion_canonical_duplicate_key(
  text,
  text,
  text,
  text,
  bigint,
  bigint,
  bigint,
  bigint,
  integer,
  integer
) from public, anon, authenticated;
grant execute on function public.match_exclusion_canonical_duplicate_key(
  text,
  text,
  text,
  text,
  bigint,
  bigint,
  bigint,
  bigint,
  integer,
  integer
) to service_role;

-- Replay jobs become leased work items. A token is capability-like: only the
-- worker holding the current, unexpired token may heartbeat or finish the job.
alter table public.replay_jobs
  add column if not exists lease_token uuid,
  add column if not exists leased_by text,
  add column if not exists lease_expires_at timestamptz,
  add column if not exists heartbeat_at timestamptz,
  add column if not exists last_claimed_at timestamptz;

do $replay_job_constraints$
begin
  if not exists (
    select 1
    from pg_catalog.pg_constraint as constraint_row
    where constraint_row.conname = 'replay_jobs_status_check'
      and constraint_row.conrelid = 'public.replay_jobs'::regclass
  ) then
    alter table public.replay_jobs
      add constraint replay_jobs_status_check
      check (
        status in (
          'pending',
          'running',
          'succeeded',
          'failed',
          'cancelled',
          'canceled'
        )
      );
  end if;

  if not exists (
    select 1
    from pg_catalog.pg_constraint as constraint_row
    where constraint_row.conname = 'replay_jobs_attempt_count_check'
      and constraint_row.conrelid = 'public.replay_jobs'::regclass
  ) then
    alter table public.replay_jobs
      add constraint replay_jobs_attempt_count_check
      check (attempt_count >= 0);
  end if;

  if not exists (
    select 1
    from pg_catalog.pg_constraint as constraint_row
    where constraint_row.conname = 'replay_jobs_lease_shape_check'
      and constraint_row.conrelid = 'public.replay_jobs'::regclass
  ) then
    alter table public.replay_jobs
      add constraint replay_jobs_lease_shape_check
      check (
        (
          status = 'running'
          and lease_token is not null
          and nullif(pg_catalog.btrim(leased_by), '') is not null
          and lease_expires_at is not null
          and heartbeat_at is not null
          and finished_at is null
        )
        or
        (
          status <> 'running'
          and lease_token is null
          and leased_by is null
          and lease_expires_at is null
          and heartbeat_at is null
        )
      );
  end if;
end
$replay_job_constraints$;

create unique index if not exists replay_jobs_one_running_per_club_uidx
  on public.replay_jobs (club_id)
  where status = 'running';

create index if not exists replay_jobs_claim_idx
  on public.replay_jobs (club_id, status, lease_expires_at, created_at, id);

-- One immutable operation owns the exact target versions, snapshots, replay
-- job, badge contract, and terminal result.
create table if not exists public.match_exclusion_operations (
  id uuid primary key,
  club_id text not null,
  mode text not null,
  idempotency_key text not null,
  status text not null default 'pending_replay',
  targets_json jsonb not null,
  duplicate_keeper_json jsonb not null default '[]'::jsonb,
  before_json jsonb not null,
  after_json jsonb not null,
  excluded_match_ids bigint[] not null,
  affected_player_ids bigint[] not null,
  badge_ids text[] not null,
  badge_contract_version text not null,
  replay_target text not null,
  replay_job_id uuid not null references public.replay_jobs(id) on delete restrict,
  replay_result_json jsonb not null default '{}'::jsonb,
  recovery_stage text,
  result_json jsonb not null default '{}'::jsonb,
  error_text text,
  actor_email text not null,
  actor_role text not null,
  source text not null,
  delete_note text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  replay_finished_at timestamptz,
  finished_at timestamptz,
  constraint match_exclusion_operations_mode_check
    check (mode in ('exclude', 'duplicate_cleanup')),
  constraint match_exclusion_operations_status_check
    check (
      status in (
        'pending_replay',
        'pending_badge_reconcile',
        'recovery_required',
        'succeeded'
      )
    ),
  constraint match_exclusion_operations_targets_shape_check
    check (
      pg_catalog.jsonb_typeof(targets_json) = 'array'
      and pg_catalog.jsonb_array_length(targets_json) between 1 and 100
    ),
  constraint match_exclusion_operations_duplicate_keeper_shape_check
    check (
      pg_catalog.jsonb_typeof(duplicate_keeper_json) = 'array'
      and (
        (
          mode = 'duplicate_cleanup'
          and pg_catalog.jsonb_array_length(duplicate_keeper_json)
            = pg_catalog.cardinality(excluded_match_ids)
        )
        or
        (
          mode = 'exclude'
          and pg_catalog.jsonb_array_length(duplicate_keeper_json) = 0
        )
      )
    ),
  constraint match_exclusion_operations_before_shape_check
    check (
      pg_catalog.jsonb_typeof(before_json) = 'array'
      and pg_catalog.jsonb_array_length(before_json)
        = pg_catalog.cardinality(excluded_match_ids)
    ),
  constraint match_exclusion_operations_after_shape_check
    check (
      pg_catalog.jsonb_typeof(after_json) = 'array'
      and pg_catalog.jsonb_array_length(after_json)
        = pg_catalog.cardinality(excluded_match_ids)
    ),
  constraint match_exclusion_operations_exact_ids_check
    check (
      pg_catalog.cardinality(excluded_match_ids) between 1 and 100
      and pg_catalog.cardinality(affected_player_ids) between 1 and 10000
    ),
  constraint match_exclusion_operations_badge_contract_check
    check (
      pg_catalog.cardinality(badge_ids) between 1 and 128
      and nullif(pg_catalog.btrim(badge_contract_version), '') is not null
    ),
  constraint match_exclusion_operations_recovery_stage_check
    check (
      (
        status = 'recovery_required'
        and recovery_stage in ('replay', 'badge_reconcile')
      )
      or
      (
        status <> 'recovery_required'
        and recovery_stage is null
      )
    ),
  constraint match_exclusion_operations_club_key_unique
    unique (club_id, idempotency_key)
);

create index if not exists match_exclusion_operations_club_created_idx
  on public.match_exclusion_operations (club_id, created_at desc, id);

create index if not exists match_exclusion_operations_open_idx
  on public.match_exclusion_operations (club_id, status, updated_at, id)
  where status <> 'succeeded';

-- The club replay lock serializes the normal path; this partial unique index
-- is the durable backstop that prevents two independently recoverable
-- exclusion operations from ever being open for one club.
create unique index if not exists
  match_exclusion_operations_one_open_per_club_uidx
  on public.match_exclusion_operations (club_id)
  where status <> 'succeeded';

-- The badge worker leases one affected player at a time. Its before/after
-- evidence includes only engine-owned rows in the frozen live-badge allowlist.
create table if not exists public.match_exclusion_badge_progress (
  id uuid primary key default gen_random_uuid(),
  operation_id uuid not null
    references public.match_exclusion_operations(id) on delete restrict,
  club_id text not null,
  player_id bigint not null,
  status text not null default 'pending',
  badge_ids text[] not null,
  badge_contract_version text not null,
  desired_badges_json jsonb not null default '[]'::jsonb,
  before_json jsonb not null default '[]'::jsonb,
  after_json jsonb not null default '[]'::jsonb,
  result_json jsonb not null default '{}'::jsonb,
  error_text text,
  attempt_count integer not null default 0,
  lease_token uuid,
  leased_by text,
  lease_expires_at timestamptz,
  heartbeat_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz,
  constraint match_exclusion_badge_progress_status_check
    check (status in ('pending', 'running', 'failed', 'succeeded')),
  constraint match_exclusion_badge_progress_attempt_count_check
    check (attempt_count >= 0),
  constraint match_exclusion_badge_progress_badge_contract_check
    check (
      pg_catalog.cardinality(badge_ids) between 1 and 128
      and nullif(pg_catalog.btrim(badge_contract_version), '') is not null
    ),
  constraint match_exclusion_badge_progress_json_shape_check
    check (
      pg_catalog.jsonb_typeof(desired_badges_json) = 'array'
      and pg_catalog.jsonb_typeof(before_json) = 'array'
      and pg_catalog.jsonb_typeof(after_json) = 'array'
      and pg_catalog.jsonb_typeof(result_json) = 'object'
    ),
  constraint match_exclusion_badge_progress_lease_shape_check
    check (
      (
        status = 'running'
        and lease_token is not null
        and nullif(pg_catalog.btrim(leased_by), '') is not null
        and lease_expires_at is not null
        and heartbeat_at is not null
        and finished_at is null
      )
      or
      (
        status <> 'running'
        and lease_token is null
        and leased_by is null
        and lease_expires_at is null
        and heartbeat_at is null
      )
    ),
  constraint match_exclusion_badge_progress_operation_player_unique
    unique (operation_id, player_id)
);

create index if not exists match_exclusion_badge_progress_claim_idx
  on public.match_exclusion_badge_progress (
    operation_id,
    status,
    lease_expires_at,
    created_at,
    id
  );

alter table public.match_exclusion_operations enable row level security;
alter table public.match_exclusion_operations force row level security;
alter table public.match_exclusion_badge_progress enable row level security;
alter table public.match_exclusion_badge_progress force row level security;

revoke all on table public.match_exclusion_operations
  from public, anon, authenticated;
revoke all on table public.match_exclusion_badge_progress
  from public, anon, authenticated;
grant select, insert, update on table public.match_exclusion_operations
  to service_role;
grant select, insert, update on table public.match_exclusion_badge_progress
  to service_role;

comment on column public.matches.row_version is
  'Immutable compare-and-swap version advanced by every match UPDATE.';
comment on table public.match_exclusion_operations is
  'Server-only exact-ID Match Log exclusion and replay/badge recovery ledger.';
comment on table public.match_exclusion_badge_progress is
  'Server-only leased per-player progress for narrow engine badge reconciliation.';

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
        errcode = '40001',
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
          errcode = '40001',
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
        errcode = '40001',
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

create or replace function public.claim_replay_job_atomic(
  p_job_id uuid,
  p_club_id text,
  p_worker_id text,
  p_lease_seconds integer default 120,
  p_retry_failed boolean default false
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_worker_id text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_worker_id), 160), '');
  v_lease_seconds integer := p_lease_seconds;
  v_open_operation public.match_exclusion_operations%rowtype;
  v_job public.replay_jobs%rowtype;
  v_lease_token uuid;
  v_now timestamptz;
begin
  if p_job_id is null
     or v_club_id is null
     or v_worker_id is null
     or v_lease_seconds is null
     or v_lease_seconds not between 30 and 3600 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_CLAIM_INVALID: job, club, worker, and a 30-to-3600-second lease are required.';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  -- Keep broad/manual replays out of a club whose exclusion recovery is
  -- nonterminal. The exact replay job frozen into that operation is the sole
  -- exception. Lock operation before job to match transition/finalize order.
  select operation.*
  into v_open_operation
  from public.match_exclusion_operations as operation
  where operation.club_id = v_club_id
    and operation.status <> 'succeeded'
  order by operation.created_at, operation.id
  limit 1
  for update;

  if found and v_open_operation.replay_job_id <> p_job_id then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'claimed', false,
      'code', 'MATCH_EXCLUSION_OPERATION_IN_PROGRESS',
      'message',
        'Another replay is blocked while match exclusion recovery is open.',
      'job_id', p_job_id,
      'club_id', v_club_id,
      'operation_id', v_open_operation.id,
      'operation_status', v_open_operation.status,
      'replay_job_id', v_open_operation.replay_job_id
    );
  end if;

  select replay_job.*
  into v_job
  from public.replay_jobs as replay_job
  where replay_job.id = p_job_id
    and replay_job.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'claimed', false,
      'code', 'REPLAY_JOB_NOT_FOUND',
      'job_id', p_job_id
    );
  end if;

  -- Capture time only after lock acquisition. A waiter must not make lease
  -- decisions using a timestamp sampled before the prior owner committed.
  v_now := pg_catalog.clock_timestamp();

  -- Expired leases for other jobs are terminalized before a new club-wide
  -- replay starts. The stale worker's token can no longer finish that job.
  update public.replay_jobs as replay_job
  set
    status = 'failed',
    finished_at = v_now,
    updated_at = v_now,
    error_text = coalesce(
      nullif(replay_job.error_text, ''),
      'Replay lease expired before completion.'
    ),
    lease_token = null,
    leased_by = null,
    lease_expires_at = null,
    heartbeat_at = null
  where replay_job.club_id = v_club_id
    and replay_job.id <> p_job_id
    and replay_job.status = 'running'
    and replay_job.lease_expires_at <= v_now;

  if exists (
    select 1
    from public.replay_jobs as replay_job
    where replay_job.club_id = v_club_id
      and replay_job.id <> p_job_id
      and replay_job.status = 'running'
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'claimed', false,
      'code', 'CLUB_REPLAY_IN_PROGRESS',
      'job_id', p_job_id,
      'club_id', v_club_id,
      'status', v_job.status
    );
  end if;

  if v_job.status = 'succeeded' then
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'claimed', false,
      'job_id', v_job.id,
      'club_id', v_job.club_id,
      'status', v_job.status,
      'target_reset', v_job.target_reset,
      'lease_token', null,
      'lease_expires_at', null,
      'attempt_count', v_job.attempt_count,
      'result_json', v_job.result_json,
      'error_text', v_job.error_text,
      'idempotent_replay', true
    );
  end if;

  if v_job.status in ('cancelled', 'canceled')
     or (v_job.status = 'failed' and not coalesce(p_retry_failed, false))
     or (
       v_job.status = 'running'
       and v_job.lease_expires_at > v_now
     ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'claimed', false,
      'code', case
        when v_job.status = 'failed' then 'REPLAY_JOB_FAILED'
        when v_job.status = 'running' then 'REPLAY_JOB_ALREADY_LEASED'
        else 'REPLAY_JOB_CANCELLED'
      end,
      'job_id', v_job.id,
      'club_id', v_job.club_id,
      'status', v_job.status,
      'target_reset', v_job.target_reset,
      'lease_token', null,
      'lease_expires_at', v_job.lease_expires_at,
      'attempt_count', v_job.attempt_count,
      'result_json', v_job.result_json,
      'error_text', v_job.error_text,
      'idempotent_replay', true
    );
  end if;

  if v_job.status not in ('pending', 'failed', 'running') then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'claimed', false,
      'code', 'REPLAY_JOB_NOT_CLAIMABLE',
      'job_id', v_job.id,
      'club_id', v_job.club_id,
      'status', v_job.status
    );
  end if;

  v_lease_token := gen_random_uuid();

  update public.replay_jobs as replay_job
  set
    status = 'running',
    started_at = coalesce(replay_job.started_at, v_now),
    finished_at = null,
    result_json = '{}'::jsonb,
    error_text = null,
    attempt_count = replay_job.attempt_count + 1,
    lease_token = v_lease_token,
    leased_by = v_worker_id,
    lease_expires_at =
      v_now + pg_catalog.make_interval(secs => v_lease_seconds),
    heartbeat_at = v_now,
    last_claimed_at = v_now,
    updated_at = v_now
  where replay_job.id = p_job_id
    and replay_job.club_id = v_club_id
  returning replay_job.* into v_job;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'claimed', true,
    'job_id', v_job.id,
    'club_id', v_job.club_id,
    'status', v_job.status,
    'target_reset', v_job.target_reset,
    'lease_token', v_job.lease_token,
    'lease_expires_at', v_job.lease_expires_at,
    'attempt_count', v_job.attempt_count,
    'result_json', v_job.result_json,
    'error_text', v_job.error_text,
    'idempotent_replay', false
  );
end
$function$;

create or replace function public.heartbeat_replay_job_atomic(
  p_job_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_lease_seconds integer default 120
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_worker_id text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_worker_id), 160), '');
  v_lease_seconds integer := p_lease_seconds;
  v_job public.replay_jobs%rowtype;
  v_now timestamptz;
begin
  if p_job_id is null
     or v_club_id is null
     or p_lease_token is null
     or v_worker_id is null
     or v_lease_seconds is null
     or v_lease_seconds not between 30 and 3600 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_HEARTBEAT_INVALID: exact job, club, worker, token, and lease duration are required.';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select replay_job.*
  into v_job
  from public.replay_jobs as replay_job
  where replay_job.id = p_job_id
    and replay_job.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'renewed', false,
      'code', 'REPLAY_JOB_NOT_FOUND',
      'job_id', p_job_id,
      'status', null,
      'lease_expires_at', null
    );
  end if;

  v_now := pg_catalog.clock_timestamp();

  if v_job.status <> 'running'
     or v_job.lease_token is distinct from p_lease_token
     or v_job.leased_by is distinct from v_worker_id
     or v_job.lease_expires_at <= v_now then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'renewed', false,
      'code', 'REPLAY_LEASE_LOST',
      'job_id', v_job.id,
      'status', v_job.status,
      'lease_expires_at', v_job.lease_expires_at
    );
  end if;

  update public.replay_jobs as replay_job
  set
    lease_expires_at =
      v_now + pg_catalog.make_interval(secs => v_lease_seconds),
    heartbeat_at = v_now,
    updated_at = v_now
  where replay_job.id = p_job_id
    and replay_job.club_id = v_club_id
  returning replay_job.* into v_job;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'renewed', true,
    'job_id', v_job.id,
    'status', v_job.status,
    'lease_expires_at', v_job.lease_expires_at
  );
end
$function$;

create or replace function public.finish_replay_job_atomic(
  p_job_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_status text,
  p_result_json jsonb default '{}'::jsonb,
  p_error_text text default null
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_worker_id text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_worker_id), 160), '');
  v_status text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_status, '')));
  v_error_text text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_error_text), 4000), '');
  v_result_json jsonb := coalesce(p_result_json, '{}'::jsonb);
  v_job public.replay_jobs%rowtype;
  v_now timestamptz;
begin
  if p_job_id is null
     or v_club_id is null
     or p_lease_token is null
     or v_worker_id is null
     or v_status is null
     or v_status not in ('succeeded', 'failed')
     or pg_catalog.jsonb_typeof(v_result_json) <> 'object'
     or (v_status = 'failed' and v_error_text is null) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_FINISH_INVALID: exact lease and succeeded/object-result or failed/error outcome are required.';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select replay_job.*
  into v_job
  from public.replay_jobs as replay_job
  where replay_job.id = p_job_id
    and replay_job.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'finished', false,
      'code', 'REPLAY_JOB_NOT_FOUND',
      'job_id', p_job_id
    );
  end if;

  v_now := pg_catalog.clock_timestamp();

  if v_job.status = v_status then
    if (
      v_status = 'succeeded'
      and v_job.result_json is distinct from v_result_json
    ) or (
      v_status = 'failed'
      and v_job.error_text is distinct from v_error_text
    ) then
      return pg_catalog.jsonb_build_object(
        'ok', false,
        'finished', false,
        'code', 'REPLAY_FINISH_CONFLICT',
        'job_id', v_job.id,
        'status', v_job.status,
        'result_json', v_job.result_json,
        'error_text', v_job.error_text
      );
    end if;

    return pg_catalog.jsonb_build_object(
      'ok', true,
      'finished', true,
      'job_id', v_job.id,
      'status', v_job.status,
      'result_json', v_job.result_json,
      'error_text', v_job.error_text,
      'idempotent', true
    );
  end if;

  if v_job.status <> 'running'
     or v_job.lease_token is distinct from p_lease_token
     or v_job.leased_by is distinct from v_worker_id
     or v_job.lease_expires_at <= v_now then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'finished', false,
      'code', 'REPLAY_LEASE_LOST',
      'job_id', v_job.id,
      'status', v_job.status,
      'lease_expires_at', v_job.lease_expires_at
    );
  end if;

  if v_status = 'succeeded'
     and v_job.target_reset = 'ALL (Full System Reset)'
     and (v_result_json ->> 'singles_replay_supported') is distinct from 'true' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_RESULT_INCOMPLETE: full replay must attest singles_replay_supported=true.';
  end if;

  update public.replay_jobs as replay_job
  set
    status = v_status,
    result_json = case
      when v_status = 'succeeded' then v_result_json
      else replay_job.result_json
    end,
    error_text = case
      when v_status = 'failed' then v_error_text
      else null
    end,
    finished_at = v_now,
    updated_at = v_now,
    lease_token = null,
    leased_by = null,
    lease_expires_at = null,
    heartbeat_at = null
  where replay_job.id = p_job_id
    and replay_job.club_id = v_club_id
  returning replay_job.* into v_job;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'finished', true,
    'job_id', v_job.id,
    'status', v_job.status,
    'result_json', v_job.result_json,
    'error_text', v_job.error_text,
    'idempotent', false
  );
end
$function$;

-- Every replay mutation validates ownership while holding both the same
-- per-club advisory transaction lock used by claims and the exact replay-job
-- row lock. The locks remain held by the caller's transaction, so a lease
-- cannot be reclaimed between this check and the mutation it fences.
create or replace function public.assert_replay_write_fence_atomic(
  p_job_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_target_reset text
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_worker_id text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_worker_id), 160), '');
  v_target_reset text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_target_reset), 160), '');
  v_job public.replay_jobs%rowtype;
  v_now timestamptz;
begin
  if p_job_id is null
     or v_club_id is null
     or p_lease_token is null
     or v_worker_id is null
     or v_target_reset is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_WRITE_FENCE_INVALID: exact job, club, token, worker, and target are required.';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('jupr:replay-club:' || v_club_id, 0)
  );

  select replay_job.*
  into v_job
  from public.replay_jobs as replay_job
  where replay_job.id = p_job_id
    and replay_job.club_id = v_club_id
  for update;

  if not found then
    raise exception using
      errcode = '55000',
      message = 'JUPR_REPLAY_WRITE_FENCE_LOST: replay job is unavailable for this club.';
  end if;

  -- Sample after both locks. A statement that waited behind the prior owner
  -- must use current time when deciding whether this token is still valid.
  v_now := pg_catalog.clock_timestamp();

  if v_job.status <> 'running'
     or v_job.target_reset is distinct from v_target_reset
     or v_job.lease_token is distinct from p_lease_token
     or v_job.leased_by is distinct from v_worker_id
     or v_job.lease_expires_at is null
     or v_job.lease_expires_at <= v_now then
    raise exception using
      errcode = '55000',
      message = 'JUPR_REPLAY_WRITE_FENCE_LOST: replay mutation rejected for a stale or mismatched lease.',
      detail = pg_catalog.jsonb_build_object(
        'job_id', v_job.id,
        'status', v_job.status,
        'target_reset', v_job.target_reset,
        'lease_expires_at', v_job.lease_expires_at
      )::text;
  end if;
end
$function$;

-- The worker never writes replay projections directly when it owns a durable
-- job. Each batch enters this single fenced RPC, which validates the lease and
-- applies one idempotent mutation in the same transaction. Repeating a batch
-- after a lost HTTP response leaves values (and match row_version) unchanged.
create or replace function public.apply_replay_write_batch_atomic(
  p_job_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_target_reset text,
  p_write_kind text,
  p_rows jsonb default '[]'::jsonb,
  p_delete_all boolean default false,
  p_league_names text[] default '{}'::text[]
)
returns integer
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_target_reset text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_target_reset), 160), '');
  v_write_kind text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_write_kind, '')));
  v_rows jsonb := coalesce(p_rows, '[]'::jsonb);
  v_league_names text[] := coalesce(p_league_names, '{}'::text[]);
  v_expected integer;
  v_verified integer;
  v_changed integer;
begin
  if v_club_id is null
     or v_target_reset is null
     or v_write_kind not in (
       'players_stats',
       'player_singles_stats',
       'delete_league_ratings',
       'insert_league_ratings',
       'match_snapshots'
     )
     or pg_catalog.jsonb_typeof(v_rows) <> 'array' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_WRITE_BATCH_INVALID: exact club, target, write kind, and rows array are required.';
  end if;

  v_expected := pg_catalog.jsonb_array_length(v_rows);

  if v_write_kind = 'delete_league_ratings' then
    if v_expected <> 0
       or (
         coalesce(p_delete_all, false)
         and v_target_reset <> 'ALL (Full System Reset)'
       )
       or (
         not coalesce(p_delete_all, false)
         and (
           v_target_reset = 'ALL (Full System Reset)'
           or pg_catalog.cardinality(v_league_names) not between 1 and 2
           or not (v_target_reset = any(v_league_names))
           or exists (
             select 1
             from pg_catalog.unnest(v_league_names)
               as league_name(value)
             where nullif(pg_catalog.btrim(league_name.value), '') is null
           )
         )
       ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_RATING_DELETE_INVALID: full reset must delete all ratings; league reset must name its one exact scope.';
    end if;
  elsif v_expected not between 1 and 500
        or coalesce(p_delete_all, false)
        or pg_catalog.cardinality(v_league_names) <> 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REPLAY_WRITE_ROWS_INVALID: mutation batches require 1 to 500 rows and no delete-only arguments.';
  end if;

  perform public.assert_replay_write_fence_atomic(
    p_job_id,
    v_club_id,
    p_lease_token,
    p_worker_id,
    v_target_reset
  );

  if v_write_kind = 'players_stats' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text,
        rating numeric,
        wins integer,
        losses integer,
        matches_played integer,
        last_game_at timestamptz
      )
      where data.id is null
         or data.id <= 0
         or data.club_id is distinct from v_club_id
         or data.rating is null
         or data.wins is null
         or data.losses is null
         or data.matches_played is null
         or data.wins < 0
         or data.losses < 0
         or data.matches_played < 0
    ) or exists (
      select 1
      from pg_catalog.jsonb_array_elements(v_rows) as row_item(item)
      where not (row_item.item ? 'last_game_at')
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      group by data.id
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_PLAYER_BATCH_INVALID: exact unique club player projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        id bigint,
        club_id text,
        rating numeric,
        wins integer,
        losses integer,
        matches_played integer,
        last_game_at timestamptz
      )
    )
    update public.players as player
    set
      rating = data.rating,
      wins = data.wins,
      losses = data.losses,
      matches_played = data.matches_played,
      last_game_at = data.last_game_at
    from data
    where player.club_id::text = v_club_id
      and player.id = data.id
      and row(
        player.rating,
        player.wins,
        player.losses,
        player.matches_played,
        player.last_game_at
      ) is distinct from row(
        data.rating,
        data.wins,
        data.losses,
        data.matches_played,
        data.last_game_at
      );

    select pg_catalog.count(*)
    into v_verified
    from public.players as player
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      id bigint,
      club_id text,
      rating numeric,
      wins integer,
      losses integer,
      matches_played integer,
      last_game_at timestamptz
    ) on data.id = player.id
      and data.club_id = player.club_id::text
    where player.club_id::text = v_club_id
      and row(
        player.rating,
        player.wins,
        player.losses,
        player.matches_played,
        player.last_game_at
      ) is not distinct from row(
        data.rating,
        data.wins,
        data.losses,
        data.matches_played,
        data.last_game_at
      );

  elsif v_write_kind = 'player_singles_stats' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text,
        singles_rating double precision,
        singles_wins integer,
        singles_losses integer,
        singles_matches_played integer,
        singles_last_game_at timestamptz
      )
      where data.id is null
         or data.id <= 0
         or data.club_id is distinct from v_club_id
         or data.singles_rating is null
         or data.singles_wins is null
         or data.singles_losses is null
         or data.singles_matches_played is null
         or data.singles_wins < 0
         or data.singles_losses < 0
         or data.singles_matches_played < 0
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      group by data.id
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_SINGLES_PLAYER_BATCH_INVALID: exact unique club singles projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        id bigint,
        club_id text,
        singles_rating double precision,
        singles_wins integer,
        singles_losses integer,
        singles_matches_played integer,
        singles_last_game_at timestamptz
      )
    )
    update public.players as player
    set
      singles_rating = data.singles_rating,
      singles_wins = data.singles_wins,
      singles_losses = data.singles_losses,
      singles_matches_played = data.singles_matches_played,
      singles_last_game_at = data.singles_last_game_at
    from data
    where player.club_id::text = v_club_id
      and player.id = data.id
      and row(
        player.singles_rating,
        player.singles_wins,
        player.singles_losses,
        player.singles_matches_played,
        player.singles_last_game_at
      ) is distinct from row(
        data.singles_rating,
        data.singles_wins,
        data.singles_losses,
        data.singles_matches_played,
        data.singles_last_game_at
      );

    select pg_catalog.count(*)
    into v_verified
    from public.players as player
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      id bigint,
      club_id text,
      singles_rating double precision,
      singles_wins integer,
      singles_losses integer,
      singles_matches_played integer,
      singles_last_game_at timestamptz
    ) on data.id = player.id
      and data.club_id = player.club_id::text
    where player.club_id::text = v_club_id
      and row(
        player.singles_rating,
        player.singles_wins,
        player.singles_losses,
        player.singles_matches_played,
        player.singles_last_game_at
      ) is not distinct from row(
        data.singles_rating,
        data.singles_wins,
        data.singles_losses,
        data.singles_matches_played,
        data.singles_last_game_at
      );

  elsif v_write_kind = 'delete_league_ratings' then
    if coalesce(p_delete_all, false) then
      delete from public.league_ratings as league_rating
      where league_rating.club_id::text = v_club_id;
    else
      delete from public.league_ratings as league_rating
      where league_rating.club_id::text = v_club_id
        and league_rating.league_name = any(v_league_names);
    end if;
    get diagnostics v_changed = row_count;
    return v_changed;

  elsif v_write_kind = 'insert_league_ratings' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        club_id text,
        player_id bigint,
        league_name text,
        rating numeric,
        wins integer,
        losses integer,
        matches_played integer,
        starting_rating numeric
      )
      where data.club_id is distinct from v_club_id
         or data.player_id is null
         or data.player_id <= 0
         or nullif(pg_catalog.btrim(data.league_name), '') is null
         or data.rating is null
         or data.starting_rating is null
         or data.wins is null
         or data.losses is null
         or data.matches_played is null
         or data.wins < 0
         or data.losses < 0
         or data.matches_played < 0
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        club_id text,
        player_id bigint,
        league_name text
      )
      group by data.club_id, data.player_id, data.league_name
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_LEAGUE_RATING_BATCH_INVALID: exact unique club/player/league projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        club_id text,
        player_id bigint,
        league_name text,
        rating numeric,
        wins integer,
        losses integer,
        matches_played integer,
        starting_rating numeric
      )
    )
    update public.league_ratings as league_rating
    set
      rating = data.rating,
      wins = data.wins,
      losses = data.losses,
      matches_played = data.matches_played,
      starting_rating = data.starting_rating,
      is_active = true,
      inactive_at = null
    from data
    where league_rating.club_id::text = data.club_id
      and league_rating.player_id = data.player_id
      and league_rating.league_name = data.league_name
      and row(
        league_rating.rating,
        league_rating.wins,
        league_rating.losses,
        league_rating.matches_played,
        league_rating.starting_rating,
        league_rating.is_active,
        league_rating.inactive_at
      ) is distinct from row(
        data.rating,
        data.wins,
        data.losses,
        data.matches_played,
        data.starting_rating,
        true,
        null::timestamptz
      );

    insert into public.league_ratings (
      club_id,
      player_id,
      league_name,
      rating,
      wins,
      losses,
      matches_played,
      starting_rating,
      is_active,
      inactive_at
    )
    select
      data.club_id,
      data.player_id,
      data.league_name,
      data.rating,
      data.wins,
      data.losses,
      data.matches_played,
      data.starting_rating,
      true,
      null
    from pg_catalog.jsonb_to_recordset(v_rows) as data(
      club_id text,
      player_id bigint,
      league_name text,
      rating numeric,
      wins integer,
      losses integer,
      matches_played integer,
      starting_rating numeric
    )
    where not exists (
      select 1
      from public.league_ratings as existing
      where existing.club_id::text = data.club_id
        and existing.player_id = data.player_id
        and existing.league_name = data.league_name
    )
    on conflict (club_id, player_id, league_name) do update
    set
      rating = excluded.rating,
      wins = excluded.wins,
      losses = excluded.losses,
      matches_played = excluded.matches_played,
      starting_rating = excluded.starting_rating,
      is_active = true,
      inactive_at = null
    where row(
      league_ratings.rating,
      league_ratings.wins,
      league_ratings.losses,
      league_ratings.matches_played,
      league_ratings.starting_rating,
      league_ratings.is_active,
      league_ratings.inactive_at
    ) is distinct from row(
      excluded.rating,
      excluded.wins,
      excluded.losses,
      excluded.matches_played,
      excluded.starting_rating,
      true,
      null::timestamptz
    );

    select pg_catalog.count(*)
    into v_verified
    from public.league_ratings as league_rating
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      club_id text,
      player_id bigint,
      league_name text,
      rating numeric,
      wins integer,
      losses integer,
      matches_played integer,
      starting_rating numeric
    ) on data.club_id = league_rating.club_id::text
      and data.player_id = league_rating.player_id
      and data.league_name = league_rating.league_name
    where row(
      league_rating.rating,
      league_rating.wins,
      league_rating.losses,
      league_rating.matches_played,
      league_rating.starting_rating,
      league_rating.is_active,
      league_rating.inactive_at
    ) is not distinct from row(
      data.rating,
      data.wins,
      data.losses,
      data.matches_played,
      data.starting_rating,
      true,
      null::timestamptz
    );

  elsif v_write_kind = 'match_snapshots' then
    if exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      where data.id is null
         or data.id <= 0
         or data.club_id is distinct from v_club_id
    ) or exists (
      select 1
      from pg_catalog.jsonb_to_recordset(v_rows) as data(
        id bigint,
        club_id text
      )
      group by data.id
      having pg_catalog.count(*) > 1
    ) then
      raise exception using
        errcode = '22023',
        message = 'JUPR_REPLAY_MATCH_SNAPSHOT_BATCH_INVALID: exact unique club match projections are required.';
    end if;

    with data as (
      select *
      from pg_catalog.jsonb_to_recordset(v_rows) as row_data(
        id bigint,
        club_id text,
        elo_delta numeric,
        t1_p1_r numeric,
        t1_p2_r numeric,
        t2_p1_r numeric,
        t2_p2_r numeric,
        t1_p1_r_end numeric,
        t1_p2_r_end numeric,
        t2_p1_r_end numeric,
        t2_p2_r_end numeric
      )
    )
    update public.matches as match_row
    set
      elo_delta = data.elo_delta,
      t1_p1_r = data.t1_p1_r,
      t1_p2_r = data.t1_p2_r,
      t2_p1_r = data.t2_p1_r,
      t2_p2_r = data.t2_p2_r,
      t1_p1_r_end = data.t1_p1_r_end,
      t1_p2_r_end = data.t1_p2_r_end,
      t2_p1_r_end = data.t2_p1_r_end,
      t2_p2_r_end = data.t2_p2_r_end
    from data
    where match_row.club_id::text = v_club_id
      and match_row.id = data.id
      and row(
        match_row.elo_delta,
        match_row.t1_p1_r,
        match_row.t1_p2_r,
        match_row.t2_p1_r,
        match_row.t2_p2_r,
        match_row.t1_p1_r_end,
        match_row.t1_p2_r_end,
        match_row.t2_p1_r_end,
        match_row.t2_p2_r_end
      ) is distinct from row(
        data.elo_delta,
        data.t1_p1_r,
        data.t1_p2_r,
        data.t2_p1_r,
        data.t2_p2_r,
        data.t1_p1_r_end,
        data.t1_p2_r_end,
        data.t2_p1_r_end,
        data.t2_p2_r_end
      );

    select pg_catalog.count(*)
    into v_verified
    from public.matches as match_row
    join pg_catalog.jsonb_to_recordset(v_rows) as data(
      id bigint,
      club_id text,
      elo_delta numeric,
      t1_p1_r numeric,
      t1_p2_r numeric,
      t2_p1_r numeric,
      t2_p2_r numeric,
      t1_p1_r_end numeric,
      t1_p2_r_end numeric,
      t2_p1_r_end numeric,
      t2_p2_r_end numeric
    ) on data.id = match_row.id
      and data.club_id = match_row.club_id::text
    where row(
      match_row.elo_delta,
      match_row.t1_p1_r,
      match_row.t1_p2_r,
      match_row.t2_p1_r,
      match_row.t2_p2_r,
      match_row.t1_p1_r_end,
      match_row.t1_p2_r_end,
      match_row.t2_p1_r_end,
      match_row.t2_p2_r_end
    ) is not distinct from row(
      data.elo_delta,
      data.t1_p1_r,
      data.t1_p2_r,
      data.t2_p1_r,
      data.t2_p2_r,
      data.t1_p1_r_end,
      data.t1_p2_r_end,
      data.t2_p1_r_end,
      data.t2_p2_r_end
    );
  end if;

  if v_verified is distinct from v_expected then
    raise exception using
      errcode = '40001',
      message = pg_catalog.format(
        'JUPR_REPLAY_WRITE_BATCH_INCOMPLETE: %s verified %s of %s exact rows.',
        v_write_kind,
        coalesce(v_verified, 0),
        v_expected
      );
  end if;

  return v_expected;
end
$function$;

create or replace function public.transition_match_exclusion_after_replay(
  p_operation_id uuid,
  p_club_id text,
  p_replay_job_id uuid,
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
  v_actor_email text := nullif(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), '');
  v_actor_role text := nullif(pg_catalog.btrim(p_actor_role), '');
  v_source text := nullif(pg_catalog.left(pg_catalog.btrim(p_source), 120), '');
  v_operation public.match_exclusion_operations%rowtype;
  v_job public.replay_jobs%rowtype;
  v_progress_count integer;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or p_replay_job_id is null
     or v_actor_email is null
     or v_actor_role is null
     or v_source is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_TRANSITION_INVALID: operation, club, replay job, actor, role, and source are required.';
  end if;

  select operation.*
  into v_operation
  from public.match_exclusion_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_OPERATION_NOT_FOUND',
      'operation_id', p_operation_id
    );
  end if;

  if v_operation.replay_job_id <> p_replay_job_id then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_REPLAY_JOB_MISMATCH',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'replay_job_id', v_operation.replay_job_id
    );
  end if;

  if v_operation.status in ('pending_badge_reconcile', 'succeeded') then
    select pg_catalog.count(*)::integer
    into v_progress_count
    from public.match_exclusion_badge_progress as progress
    where progress.operation_id = v_operation.id;

    return pg_catalog.jsonb_build_object(
      'ok', true,
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'replay_job_id', v_operation.replay_job_id,
      'affected_player_ids',
        pg_catalog.to_jsonb(v_operation.affected_player_ids),
      'badge_ids', pg_catalog.to_jsonb(v_operation.badge_ids),
      'badge_allowlist', pg_catalog.to_jsonb(v_operation.badge_ids),
      'badge_contract_version', v_operation.badge_contract_version,
      'badge_progress_total', v_progress_count,
      'idempotent', true
    );
  end if;

  if v_operation.status not in ('pending_replay', 'recovery_required')
     or (
       v_operation.status = 'recovery_required'
       and v_operation.recovery_stage <> 'replay'
     ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_REPLAY_TRANSITION_NOT_ALLOWED',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage
    );
  end if;

  select replay_job.*
  into v_job
  from public.replay_jobs as replay_job
  where replay_job.id = p_replay_job_id
    and replay_job.club_id = v_club_id
  for share;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'REPLAY_JOB_NOT_FOUND',
      'operation_id', v_operation.id,
      'replay_job_id', p_replay_job_id
    );
  end if;

  if v_job.status <> 'succeeded'
     or v_job.target_reset <> v_operation.replay_target
     or v_job.target_reset <> 'ALL (Full System Reset)'
     or (v_job.result_json ->> 'singles_replay_supported') is distinct from 'true'
     or v_job.finished_at is null then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_REPLAY_NOT_COMPLETE',
      'operation_id', v_operation.id,
      'replay_job_id', v_job.id,
      'replay_status', v_job.status,
      'replay_target', v_job.target_reset
    );
  end if;

  insert into public.match_exclusion_badge_progress (
    operation_id,
    club_id,
    player_id,
    status,
    badge_ids,
    badge_contract_version,
    created_at,
    updated_at
  )
  select
    v_operation.id,
    v_club_id,
    affected.player_id,
    'pending',
    v_operation.badge_ids,
    v_operation.badge_contract_version,
    v_now,
    v_now
  from pg_catalog.unnest(v_operation.affected_player_ids)
    as affected(player_id)
  on conflict (operation_id, player_id) do nothing;

  select pg_catalog.count(*)::integer
  into v_progress_count
  from public.match_exclusion_badge_progress as progress
  where progress.operation_id = v_operation.id;

  if v_progress_count <> pg_catalog.cardinality(v_operation.affected_player_ids) then
    raise exception using
      errcode = '23514',
      message = 'JUPR_MATCH_EXCLUSION_BADGE_PROGRESS_MISMATCH: progress rows do not match the frozen affected-player set.';
  end if;

  v_result := v_operation.result_json || pg_catalog.jsonb_build_object(
    'ok', true,
    'operation_id', v_operation.id,
    'operation_status', 'pending_badge_reconcile',
    'replay_job_id', v_job.id,
    'replay_status', v_job.status,
    'affected_player_ids',
      pg_catalog.to_jsonb(v_operation.affected_player_ids),
    'badge_ids', pg_catalog.to_jsonb(v_operation.badge_ids),
    'badge_allowlist', pg_catalog.to_jsonb(v_operation.badge_ids),
    'badge_contract_version', v_operation.badge_contract_version,
    'badge_progress_total', v_progress_count,
    'idempotent', false
  );

  update public.match_exclusion_operations as operation
  set
    status = 'pending_badge_reconcile',
    replay_result_json = v_job.result_json,
    result_json = v_result,
    recovery_stage = null,
    error_text = null,
    replay_finished_at = v_job.finished_at,
    updated_at = v_now
  where operation.id = v_operation.id;

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
    'match_exclusion_replay_verified',
    'match_exclusion_operation',
    v_operation.id::text,
    pg_catalog.jsonb_build_object(
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage
    ),
    v_result,
    'Verified the exact succeeded full replay and opened narrow badge reconciliation.',
    v_source,
    true
  );

  return v_result;
end
$function$;

create or replace function public.mark_match_exclusion_recovery_required(
  p_operation_id uuid,
  p_club_id text,
  p_recovery_stage text,
  p_error_text text,
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
  v_recovery_stage text :=
    pg_catalog.lower(
      pg_catalog.btrim(coalesce(p_recovery_stage, ''))
    );
  v_error_text text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_error_text), 4000), '');
  v_actor_email text := nullif(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), '');
  v_actor_role text := nullif(pg_catalog.btrim(p_actor_role), '');
  v_source text := nullif(pg_catalog.left(pg_catalog.btrim(p_source), 120), '');
  v_operation public.match_exclusion_operations%rowtype;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_recovery_stage is null
     or v_recovery_stage not in ('replay', 'badge_reconcile')
     or v_error_text is null
     or v_actor_email is null
     or v_actor_role is null
     or v_source is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_RECOVERY_INVALID: exact operation, stage, error, actor, role, and source are required.';
  end if;

  select operation.*
  into v_operation
  from public.match_exclusion_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_OPERATION_NOT_FOUND',
      'operation_id', p_operation_id
    );
  end if;

  if v_operation.status = 'succeeded' then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_ALREADY_SUCCEEDED',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status
    );
  end if;

  if (
    v_recovery_stage = 'replay'
    and v_operation.status not in ('pending_replay', 'recovery_required')
  ) or (
    v_recovery_stage = 'badge_reconcile'
    and v_operation.status not in (
      'pending_badge_reconcile',
      'recovery_required'
    )
  ) or (
    v_operation.status = 'recovery_required'
    and v_operation.recovery_stage <> v_recovery_stage
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_RECOVERY_STAGE_MISMATCH',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage
    );
  end if;

  if v_operation.status = 'recovery_required'
     and v_operation.recovery_stage = v_recovery_stage
     and v_operation.error_text = v_error_text then
    return pg_catalog.jsonb_build_object(
      'ok', true,
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage,
      'error_text', v_operation.error_text,
      'idempotent', true
    );
  end if;

  v_result := v_operation.result_json || pg_catalog.jsonb_build_object(
    'ok', false,
    'operation_id', v_operation.id,
    'operation_status', 'recovery_required',
    'recovery_stage', v_recovery_stage,
    'error_text', v_error_text,
    'idempotent', false
  );

  update public.match_exclusion_operations as operation
  set
    status = 'recovery_required',
    recovery_stage = v_recovery_stage,
    error_text = v_error_text,
    result_json = v_result,
    updated_at = v_now
  where operation.id = v_operation.id;

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
    'match_exclusion_recovery_required',
    'match_exclusion_operation',
    v_operation.id::text,
    pg_catalog.jsonb_build_object(
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage,
      'error_text', v_operation.error_text
    ),
    v_result,
    v_error_text,
    v_source,
    true
  );

  return v_result;
end
$function$;

create or replace function public.claim_match_exclusion_badge_progress(
  p_operation_id uuid,
  p_club_id text,
  p_worker_id text,
  p_lease_seconds integer default 120,
  p_retry_failed boolean default false
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_worker_id text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_worker_id), 160), '');
  v_lease_seconds integer := p_lease_seconds;
  v_operation public.match_exclusion_operations%rowtype;
  v_progress public.match_exclusion_badge_progress%rowtype;
  v_lease_token uuid;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_pending_count integer;
  v_running_count integer;
  v_failed_count integer;
  v_succeeded_count integer;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_worker_id is null
     or v_lease_seconds is null
     or v_lease_seconds not between 30 and 3600 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_BADGE_RECONCILE_CLAIM_INVALID: operation, club, worker, and a 30-to-3600-second lease are required.';
  end if;

  select operation.*
  into v_operation
  from public.match_exclusion_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'claimed', false,
      'code', 'MATCH_EXCLUSION_OPERATION_NOT_FOUND',
      'operation_id', p_operation_id
    );
  end if;

  if v_operation.status not in (
    'pending_badge_reconcile',
    'recovery_required'
  ) or (
    v_operation.status = 'recovery_required'
    and v_operation.recovery_stage <> 'badge_reconcile'
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'claimed', false,
      'code', 'BADGE_RECONCILE_NOT_AVAILABLE',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage,
      'badge_ids', pg_catalog.to_jsonb(v_operation.badge_ids),
      'badge_allowlist', pg_catalog.to_jsonb(v_operation.badge_ids),
      'badge_contract_version', v_operation.badge_contract_version
    );
  end if;

  select progress.*
  into v_progress
  from public.match_exclusion_badge_progress as progress
  where progress.operation_id = v_operation.id
    and progress.club_id = v_club_id
    and (
      progress.status = 'pending'
      or (
        progress.status = 'failed'
        and coalesce(p_retry_failed, false)
      )
      or (
        progress.status = 'running'
        and progress.lease_expires_at <= v_now
      )
    )
  order by
    case
      when progress.status = 'running' then 0
      when progress.status = 'failed' then 1
      else 2
    end,
    progress.created_at,
    progress.id
  for update skip locked
  limit 1;

  if not found then
    select
      pg_catalog.count(*) filter (
        where progress.status = 'pending'
      )::integer,
      pg_catalog.count(*) filter (
        where progress.status = 'running'
      )::integer,
      pg_catalog.count(*) filter (
        where progress.status = 'failed'
      )::integer,
      pg_catalog.count(*) filter (
        where progress.status = 'succeeded'
      )::integer
    into
      v_pending_count,
      v_running_count,
      v_failed_count,
      v_succeeded_count
    from public.match_exclusion_badge_progress as progress
    where progress.operation_id = v_operation.id
      and progress.club_id = v_club_id;

    return pg_catalog.jsonb_build_object(
      'ok', v_pending_count = 0
        and v_running_count = 0
        and v_failed_count = 0,
      'claimed', false,
      'code', case
        when v_failed_count > 0 then 'BADGE_RECONCILE_FAILED'
        when v_running_count > 0 then 'BADGE_RECONCILE_IN_PROGRESS'
        when v_pending_count > 0 then 'BADGE_RECONCILE_CLAIM_CONTENDED'
        else 'BADGE_RECONCILE_COMPLETE'
      end,
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'badge_ids', pg_catalog.to_jsonb(v_operation.badge_ids),
      'badge_allowlist', pg_catalog.to_jsonb(v_operation.badge_ids),
      'badge_contract_version', v_operation.badge_contract_version,
      'pending_count', v_pending_count,
      'running_count', v_running_count,
      'failed_count', v_failed_count,
      'succeeded_count', v_succeeded_count,
      'idempotent', true
    );
  end if;

  v_lease_token := gen_random_uuid();

  update public.match_exclusion_badge_progress as progress
  set
    status = 'running',
    attempt_count = progress.attempt_count + 1,
    error_text = null,
    lease_token = v_lease_token,
    leased_by = v_worker_id,
    lease_expires_at =
      v_now + pg_catalog.make_interval(secs => v_lease_seconds),
    heartbeat_at = v_now,
    finished_at = null,
    updated_at = v_now
  where progress.id = v_progress.id
  returning progress.* into v_progress;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'claimed', true,
    'operation_id', v_operation.id,
    'operation_status', v_operation.status,
    'progress_id', v_progress.id,
    'club_id', v_progress.club_id,
    'player_id', v_progress.player_id,
    'status', v_progress.status,
    'badge_ids', pg_catalog.to_jsonb(v_progress.badge_ids),
    'badge_allowlist', pg_catalog.to_jsonb(v_progress.badge_ids),
    'badge_contract_version', v_progress.badge_contract_version,
    'lease_token', v_progress.lease_token,
    'lease_expires_at', v_progress.lease_expires_at,
    'attempt_count', v_progress.attempt_count,
    'idempotent', false
  );
end
$function$;

create or replace function public.apply_match_exclusion_badge_reconciliation(
  p_operation_id uuid,
  p_progress_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_desired_badges jsonb,
  p_actor_email text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_worker_id text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_worker_id), 160), '');
  v_actor_email text := nullif(
    pg_catalog.left(
      pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
      320
    ),
    ''
  );
  v_operation public.match_exclusion_operations%rowtype;
  v_progress public.match_exclusion_badge_progress%rowtype;
  v_existing_badge public.player_badges%rowtype;
  v_desired_badges jsonb := '[]'::jsonb;
  v_desired_badge jsonb;
  v_before_json jsonb := '[]'::jsonb;
  v_after_json jsonb := '[]'::jsonb;
  v_inserted_count integer := 0;
  v_updated_count integer := 0;
  v_revoked_count integer := 0;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_result jsonb;
begin
  if p_operation_id is null
     or p_progress_id is null
     or v_club_id is null
     or p_lease_token is null
     or v_worker_id is null
     or v_actor_email is null
     or p_desired_badges is null
     or pg_catalog.jsonb_typeof(p_desired_badges) <> 'array'
     or pg_catalog.jsonb_array_length(p_desired_badges) > 512 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_BADGE_RECONCILE_APPLY_INVALID: exact operation/progress lease and an array of at most 512 desired badges are required.';
  end if;

  select operation.*
  into v_operation
  from public.match_exclusion_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_OPERATION_NOT_FOUND',
      'operation_id', p_operation_id
    );
  end if;

  if v_operation.status not in (
    'pending_badge_reconcile',
    'recovery_required'
  ) or (
    v_operation.status = 'recovery_required'
    and v_operation.recovery_stage <> 'badge_reconcile'
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'BADGE_RECONCILE_NOT_AVAILABLE',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage
    );
  end if;

  select progress.*
  into v_progress
  from public.match_exclusion_badge_progress as progress
  where progress.id = p_progress_id
    and progress.operation_id = p_operation_id
    and progress.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'BADGE_RECONCILE_PROGRESS_NOT_FOUND',
      'operation_id', v_operation.id,
      'progress_id', p_progress_id
    );
  end if;

  if v_progress.badge_ids is distinct from v_operation.badge_ids
     or v_progress.badge_contract_version
        <> v_operation.badge_contract_version then
    raise exception using
      errcode = '23514',
      message = 'JUPR_BADGE_RECONCILE_CONTRACT_MISMATCH: progress does not match the operation badge contract.';
  end if;

  if v_progress.status = 'succeeded' then
    return v_progress.result_json || pg_catalog.jsonb_build_object(
      'ok', true,
      'operation_id', v_operation.id,
      'progress_id', v_progress.id,
      'player_id', v_progress.player_id,
      'status', v_progress.status,
      'badge_ids', pg_catalog.to_jsonb(v_progress.badge_ids),
      'badge_contract_version', v_progress.badge_contract_version,
      'idempotent', true
    );
  end if;

  if v_progress.status <> 'running'
     or v_progress.lease_token is distinct from p_lease_token
     or v_progress.leased_by is distinct from v_worker_id
     or v_progress.lease_expires_at <= v_now then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'BADGE_RECONCILE_LEASE_LOST',
      'operation_id', v_operation.id,
      'progress_id', v_progress.id,
      'player_id', v_progress.player_id,
      'status', v_progress.status,
      'lease_expires_at', v_progress.lease_expires_at
    );
  end if;

  if exists (
    select 1
    from pg_catalog.jsonb_array_elements(p_desired_badges)
      as desired(item)
    where pg_catalog.jsonb_typeof(desired.item) <> 'object'
       or not (desired.item ?& array[
         'badge_id',
         'context_type',
         'context_id',
         'rule_version'
       ])
       or exists (
         select 1
         from pg_catalog.jsonb_object_keys(desired.item)
           as desired_key(key_name)
         where desired_key.key_name not in (
           'badge_id',
           'context_type',
           'context_id',
           'match_id',
           'value_num',
           'value_json',
           'rule_version'
         )
       )
       or nullif(pg_catalog.btrim(desired.item ->> 'badge_id'), '') is null
       or nullif(pg_catalog.btrim(desired.item ->> 'context_type'), '') is null
       or nullif(pg_catalog.btrim(desired.item ->> 'context_id'), '') is null
       or pg_catalog.length(pg_catalog.btrim(desired.item ->> 'badge_id')) > 160
       or pg_catalog.length(pg_catalog.btrim(desired.item ->> 'context_type')) > 80
       or pg_catalog.length(pg_catalog.btrim(desired.item ->> 'context_id')) > 240
       or (
         desired.item ? 'match_id'
         and pg_catalog.jsonb_typeof(desired.item -> 'match_id')
           not in ('null', 'string')
       )
       or (
         pg_catalog.jsonb_typeof(desired.item -> 'match_id') = 'string'
         and pg_catalog.length(
           pg_catalog.btrim(desired.item ->> 'match_id')
         ) > 240
       )
       or (
         desired.item ? 'value_num'
         and pg_catalog.jsonb_typeof(desired.item -> 'value_num')
           not in ('null', 'number')
       )
       or (
         desired.item ? 'value_json'
         and pg_catalog.jsonb_typeof(desired.item -> 'value_json')
           not in ('null', 'object')
       )
       or (
         desired.item ? 'rule_version'
         and pg_catalog.jsonb_typeof(desired.item -> 'rule_version')
           not in ('null', 'string')
       )
       or nullif(
         pg_catalog.btrim(desired.item ->> 'rule_version'),
         ''
       ) is distinct from v_operation.badge_contract_version
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_BADGE_RECONCILE_DESIRED_INVALID: desired badges contain an unsupported or malformed row.';
  end if;

  if exists (
    select 1
    from pg_catalog.jsonb_array_elements(p_desired_badges)
      as desired(item)
    where not (
      pg_catalog.btrim(desired.item ->> 'badge_id')
        = any(v_operation.badge_ids)
    )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_BADGE_RECONCILE_BADGE_NOT_ALLOWED: desired badge is outside the frozen live-badge contract.';
  end if;

  if exists (
    select 1
    from (
      select
        pg_catalog.btrim(desired.item ->> 'badge_id') as badge_id,
        pg_catalog.btrim(desired.item ->> 'context_type') as context_type,
        pg_catalog.btrim(desired.item ->> 'context_id') as context_id
      from pg_catalog.jsonb_array_elements(p_desired_badges)
        as desired(item)
    ) as desired_key
    group by
      desired_key.badge_id,
      desired_key.context_type,
      desired_key.context_id
    having pg_catalog.count(*) > 1
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_BADGE_RECONCILE_DESIRED_DUPLICATE: badge_id/context_type/context_id keys must be unique.';
  end if;

  select coalesce(
    pg_catalog.jsonb_agg(
      pg_catalog.jsonb_build_object(
        'badge_id', pg_catalog.btrim(desired.item ->> 'badge_id'),
        'context_type',
          pg_catalog.btrim(desired.item ->> 'context_type'),
        'context_id', pg_catalog.btrim(desired.item ->> 'context_id'),
        'match_id',
          nullif(pg_catalog.left(
            pg_catalog.btrim(desired.item ->> 'match_id'),
            240
          ), ''),
        'value_num', case
          when pg_catalog.jsonb_typeof(desired.item -> 'value_num') = 'number'
            then desired.item -> 'value_num'
          else 'null'::jsonb
        end,
        'value_json', case
          when pg_catalog.jsonb_typeof(desired.item -> 'value_json') = 'object'
            then desired.item -> 'value_json'
          else '{}'::jsonb
        end,
        'rule_version',
          nullif(pg_catalog.left(
            pg_catalog.btrim(desired.item ->> 'rule_version'),
            120
          ), '')
      )
      order by
        pg_catalog.btrim(desired.item ->> 'badge_id'),
        pg_catalog.btrim(desired.item ->> 'context_type'),
        pg_catalog.btrim(desired.item ->> 'context_id')
    ),
    '[]'::jsonb
  )
  into v_desired_badges
  from pg_catalog.jsonb_array_elements(p_desired_badges)
    as desired(item);

  -- Lock every potentially mutable engine row and snapshot it before deciding
  -- which exact keys remain desired.
  perform 1
  from public.player_badges as player_badge
  where player_badge.club_id = v_club_id
    and player_badge.player_id = v_progress.player_id
    and player_badge.badge_id = any(v_operation.badge_ids)
  order by
    player_badge.badge_id,
    player_badge.context_type,
    player_badge.context_id,
    player_badge.id
  for update;

  if exists (
    select 1
    from pg_catalog.jsonb_array_elements(v_desired_badges) as desired(item)
    join public.player_badges as player_badge
      on player_badge.club_id = v_club_id
     and player_badge.player_id = v_progress.player_id
     and player_badge.badge_id = desired.item ->> 'badge_id'
     and player_badge.context_type = desired.item ->> 'context_type'
     and player_badge.context_id = desired.item ->> 'context_id'
    where player_badge.awarded_by is distinct from 'engine'
  ) then
    raise exception using
      errcode = '23505',
      message = 'JUPR_BADGE_RECONCILE_PROTECTED_KEY: a desired badge key is owned by a non-engine award.';
  end if;

  select coalesce(
    pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(player_badge)
      order by
        player_badge.badge_id,
        player_badge.context_type,
        player_badge.context_id,
        player_badge.id
    ),
    '[]'::jsonb
  )
  into v_before_json
  from public.player_badges as player_badge
  where player_badge.club_id = v_club_id
    and player_badge.player_id = v_progress.player_id
    and player_badge.badge_id = any(v_operation.badge_ids)
    and player_badge.awarded_by = 'engine';

  update public.player_badges as player_badge
  set
    revoked_at = v_now,
    -- revoked_by is an auth-user UUID in staging. This worker authenticates
    -- through service_role with a verified operator email, not a user UUID.
    revoked_by = null,
    revoke_reason =
      'Match exclusion operation '
      || v_operation.id::text
      || ' by '
      || v_actor_email
  where player_badge.club_id = v_club_id
    and player_badge.player_id = v_progress.player_id
    and player_badge.badge_id = any(v_operation.badge_ids)
    and player_badge.awarded_by = 'engine'
    and player_badge.revoked_at is null
    and not exists (
      select 1
      from pg_catalog.jsonb_array_elements(v_desired_badges) as desired(item)
      where desired.item ->> 'badge_id' = player_badge.badge_id
        and desired.item ->> 'context_type' = player_badge.context_type
        and desired.item ->> 'context_id' = player_badge.context_id
    );
  get diagnostics v_revoked_count = row_count;

  for v_desired_badge in
    select desired.item
    from pg_catalog.jsonb_array_elements(v_desired_badges) as desired(item)
    order by
      desired.item ->> 'badge_id',
      desired.item ->> 'context_type',
      desired.item ->> 'context_id'
  loop
    select player_badge.*
    into v_existing_badge
    from public.player_badges as player_badge
    where player_badge.club_id = v_club_id
      and player_badge.player_id = v_progress.player_id
      and player_badge.badge_id = v_desired_badge ->> 'badge_id'
      and player_badge.context_type =
        v_desired_badge ->> 'context_type'
      and player_badge.context_id = v_desired_badge ->> 'context_id'
    for update;

    if found then
      if v_existing_badge.awarded_by is distinct from 'engine' then
        raise exception using
          errcode = '23505',
          message = 'JUPR_BADGE_RECONCILE_PROTECTED_KEY: existing desired badge is not engine-owned.';
      end if;

      update public.player_badges as player_badge
      set
        context_type = v_desired_badge ->> 'context_type',
        match_id = nullif(v_desired_badge ->> 'match_id', ''),
        value_num = (v_desired_badge ->> 'value_num')::numeric,
        value_json = v_desired_badge -> 'value_json',
        rule_version = nullif(v_desired_badge ->> 'rule_version', ''),
        revoked_at = null,
        revoked_by = null,
        revoke_reason = null
      where player_badge.id = v_existing_badge.id;
      v_updated_count := v_updated_count + 1;
    else
      insert into public.player_badges (
        club_id,
        player_id,
        badge_id,
        earned_at,
        context_type,
        context_id,
        match_id,
        value_num,
        value_json,
        awarded_by,
        rule_version,
        revoked_at,
        revoked_by,
        revoke_reason
      ) values (
        v_club_id,
        v_progress.player_id,
        v_desired_badge ->> 'badge_id',
        v_now,
        v_desired_badge ->> 'context_type',
        v_desired_badge ->> 'context_id',
        nullif(v_desired_badge ->> 'match_id', ''),
        (v_desired_badge ->> 'value_num')::numeric,
        v_desired_badge -> 'value_json',
        'engine',
        nullif(v_desired_badge ->> 'rule_version', ''),
        null,
        null,
        null
      );
      v_inserted_count := v_inserted_count + 1;
    end if;
  end loop;

  select coalesce(
    pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(player_badge)
      order by
        player_badge.badge_id,
        player_badge.context_type,
        player_badge.context_id,
        player_badge.id
    ),
    '[]'::jsonb
  )
  into v_after_json
  from public.player_badges as player_badge
  where player_badge.club_id = v_club_id
    and player_badge.player_id = v_progress.player_id
    and player_badge.badge_id = any(v_operation.badge_ids)
    and player_badge.awarded_by = 'engine';

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'operation_id', v_operation.id,
    'progress_id', v_progress.id,
    'player_id', v_progress.player_id,
    'status', 'succeeded',
    'badge_ids', pg_catalog.to_jsonb(v_operation.badge_ids),
    'badge_contract_version', v_operation.badge_contract_version,
    'desired_count', pg_catalog.jsonb_array_length(v_desired_badges),
    'inserted_count', v_inserted_count,
    'updated_count', v_updated_count,
    'revoked_count', v_revoked_count,
    'idempotent', false
  );

  update public.match_exclusion_badge_progress as progress
  set
    status = 'succeeded',
    desired_badges_json = v_desired_badges,
    before_json = v_before_json,
    after_json = v_after_json,
    result_json = v_result,
    error_text = null,
    lease_token = null,
    leased_by = null,
    lease_expires_at = null,
    heartbeat_at = null,
    finished_at = v_now,
    updated_at = v_now
  where progress.id = v_progress.id;

  return v_result;
end
$function$;

create or replace function public.fail_match_exclusion_badge_progress(
  p_operation_id uuid,
  p_progress_id uuid,
  p_club_id text,
  p_lease_token uuid,
  p_worker_id text,
  p_error_text text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_worker_id text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_worker_id), 160), '');
  v_error_text text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_error_text), 4000), '');
  v_operation public.match_exclusion_operations%rowtype;
  v_progress public.match_exclusion_badge_progress%rowtype;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_result jsonb;
begin
  if p_operation_id is null
     or p_progress_id is null
     or v_club_id is null
     or p_lease_token is null
     or v_worker_id is null
     or v_error_text is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_BADGE_RECONCILE_FAIL_INVALID: exact operation/progress lease and error are required.';
  end if;

  select operation.*
  into v_operation
  from public.match_exclusion_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_OPERATION_NOT_FOUND',
      'operation_id', p_operation_id
    );
  end if;

  select progress.*
  into v_progress
  from public.match_exclusion_badge_progress as progress
  where progress.id = p_progress_id
    and progress.operation_id = p_operation_id
    and progress.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'BADGE_RECONCILE_PROGRESS_NOT_FOUND',
      'operation_id', v_operation.id,
      'progress_id', p_progress_id
    );
  end if;

  if v_progress.status = 'failed'
     and v_progress.error_text = v_error_text then
    return v_progress.result_json || pg_catalog.jsonb_build_object(
      'ok', false,
      'operation_id', v_operation.id,
      'operation_status', 'recovery_required',
      'recovery_stage', 'badge_reconcile',
      'progress_id', v_progress.id,
      'player_id', v_progress.player_id,
      'status', v_progress.status,
      'error_text', v_progress.error_text,
      'idempotent', true
    );
  end if;

  -- An expired lease may still report failure only if no new worker has
  -- replaced its token. Row locking makes that exact-token test atomic.
  if v_progress.status <> 'running'
     or v_progress.lease_token is distinct from p_lease_token
     or v_progress.leased_by is distinct from v_worker_id then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'BADGE_RECONCILE_LEASE_LOST',
      'operation_id', v_operation.id,
      'progress_id', v_progress.id,
      'player_id', v_progress.player_id,
      'status', v_progress.status
    );
  end if;

  v_result := pg_catalog.jsonb_build_object(
    'ok', false,
    'operation_id', v_operation.id,
    'operation_status', 'recovery_required',
    'recovery_stage', 'badge_reconcile',
    'progress_id', v_progress.id,
    'player_id', v_progress.player_id,
    'status', 'failed',
    'error_text', v_error_text,
    'idempotent', false
  );

  update public.match_exclusion_badge_progress as progress
  set
    status = 'failed',
    result_json = v_result,
    error_text = v_error_text,
    lease_token = null,
    leased_by = null,
    lease_expires_at = null,
    heartbeat_at = null,
    finished_at = v_now,
    updated_at = v_now
  where progress.id = v_progress.id;

  update public.match_exclusion_operations as operation
  set
    status = 'recovery_required',
    recovery_stage = 'badge_reconcile',
    error_text = v_error_text,
    result_json = operation.result_json || v_result,
    updated_at = v_now
  where operation.id = v_operation.id;

  return v_result;
end
$function$;

create or replace function public.finalize_match_exclusion_operation(
  p_operation_id uuid,
  p_club_id text,
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
  v_actor_email text := nullif(pg_catalog.lower(pg_catalog.btrim(p_actor_email)), '');
  v_actor_role text := nullif(pg_catalog.btrim(p_actor_role), '');
  v_source text := nullif(pg_catalog.left(pg_catalog.btrim(p_source), 120), '');
  v_operation public.match_exclusion_operations%rowtype;
  v_total_count integer;
  v_pending_count integer;
  v_running_count integer;
  v_failed_count integer;
  v_succeeded_count integer;
  v_inserted_count integer;
  v_updated_count integer;
  v_revoked_count integer;
  v_badge_results jsonb;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_result jsonb;
begin
  if p_operation_id is null
     or v_club_id is null
     or v_actor_email is null
     or v_actor_role is null
     or v_source is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_MATCH_EXCLUSION_FINALIZE_INVALID: operation, club, actor, role, and source are required.';
  end if;

  select operation.*
  into v_operation
  from public.match_exclusion_operations as operation
  where operation.id = p_operation_id
    and operation.club_id = v_club_id
  for update;

  if not found then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_OPERATION_NOT_FOUND',
      'operation_id', p_operation_id
    );
  end if;

  if v_operation.status = 'succeeded' then
    return v_operation.result_json || pg_catalog.jsonb_build_object(
      'ok', true,
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'idempotent', true
    );
  end if;

  if v_operation.status not in (
    'pending_badge_reconcile',
    'recovery_required'
  ) or (
    v_operation.status = 'recovery_required'
    and v_operation.recovery_stage <> 'badge_reconcile'
  ) then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_FINALIZE_NOT_ALLOWED',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage
    );
  end if;

  perform 1
  from public.match_exclusion_badge_progress as progress
  where progress.operation_id = v_operation.id
  order by progress.player_id, progress.id
  for update;

  select
    pg_catalog.count(*)::integer,
    pg_catalog.count(*) filter (
      where progress.status = 'pending'
    )::integer,
    pg_catalog.count(*) filter (
      where progress.status = 'running'
    )::integer,
    pg_catalog.count(*) filter (
      where progress.status = 'failed'
    )::integer,
    pg_catalog.count(*) filter (
      where progress.status = 'succeeded'
    )::integer,
    coalesce(pg_catalog.sum(
      case
        when progress.status = 'succeeded'
          then coalesce(
            (progress.result_json ->> 'inserted_count')::integer,
            0
          )
        else 0
      end
    ), 0)::integer,
    coalesce(pg_catalog.sum(
      case
        when progress.status = 'succeeded'
          then coalesce(
            (progress.result_json ->> 'updated_count')::integer,
            0
          )
        else 0
      end
    ), 0)::integer,
    coalesce(pg_catalog.sum(
      case
        when progress.status = 'succeeded'
          then coalesce(
            (progress.result_json ->> 'revoked_count')::integer,
            0
          )
        else 0
      end
    ), 0)::integer,
    coalesce(
      pg_catalog.jsonb_agg(
        progress.result_json order by progress.player_id, progress.id
      ),
      '[]'::jsonb
    )
  into
    v_total_count,
    v_pending_count,
    v_running_count,
    v_failed_count,
    v_succeeded_count,
    v_inserted_count,
    v_updated_count,
    v_revoked_count,
    v_badge_results
  from public.match_exclusion_badge_progress as progress
  where progress.operation_id = v_operation.id;

  if v_total_count <> pg_catalog.cardinality(v_operation.affected_player_ids)
     or v_pending_count > 0
     or v_running_count > 0
     or v_failed_count > 0
     or v_succeeded_count <> v_total_count then
    return pg_catalog.jsonb_build_object(
      'ok', false,
      'code', 'MATCH_EXCLUSION_BADGE_RECONCILE_INCOMPLETE',
      'operation_id', v_operation.id,
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage,
      'progress_total', v_total_count,
      'pending_count', v_pending_count,
      'running_count', v_running_count,
      'failed_count', v_failed_count,
      'succeeded_count', v_succeeded_count
    );
  end if;

  v_result := v_operation.result_json || pg_catalog.jsonb_build_object(
    'ok', true,
    'operation_id', v_operation.id,
    'operation_status', 'succeeded',
    'mode', v_operation.mode,
    'excluded_ids', pg_catalog.to_jsonb(v_operation.excluded_match_ids),
    'excluded_count',
      pg_catalog.cardinality(v_operation.excluded_match_ids),
    'affected_player_ids',
      pg_catalog.to_jsonb(v_operation.affected_player_ids),
    'replay_job_id', v_operation.replay_job_id,
    'replay_target', v_operation.replay_target,
    'badge_ids', pg_catalog.to_jsonb(v_operation.badge_ids),
    'badge_contract_version', v_operation.badge_contract_version,
    'badge_progress_total', v_total_count,
    'inserted_count', v_inserted_count,
    'updated_count', v_updated_count,
    'revoked_count', v_revoked_count,
    'badge_results', v_badge_results,
    'idempotent', false
  );

  update public.match_exclusion_operations as operation
  set
    status = 'succeeded',
    recovery_stage = null,
    error_text = null,
    result_json = v_result,
    finished_at = v_now,
    updated_at = v_now
  where operation.id = v_operation.id;

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
    'match_exclusion_recovery_completed',
    'match_exclusion_operation',
    v_operation.id::text,
    pg_catalog.jsonb_build_object(
      'operation_status', v_operation.status,
      'recovery_stage', v_operation.recovery_stage
    ),
    v_result,
    'Exact exclusion recovery completed after full replay and narrow badge reconciliation.',
    v_source,
    true
  );

  return v_result;
end
$function$;

alter table public.replay_jobs enable row level security;
alter table public.replay_jobs force row level security;
revoke all on table public.replay_jobs from public, anon, authenticated;
grant select, insert, update on table public.replay_jobs to service_role;

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
revoke all on function public.claim_replay_job_atomic(
  uuid,
  text,
  text,
  integer,
  boolean
) from public, anon, authenticated;
revoke all on function public.heartbeat_replay_job_atomic(
  uuid,
  text,
  uuid,
  text,
  integer
) from public, anon, authenticated;
revoke all on function public.finish_replay_job_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  jsonb,
  text
) from public, anon, authenticated;
revoke all on function public.assert_replay_write_fence_atomic(
  uuid,
  text,
  uuid,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.apply_replay_write_batch_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  text,
  jsonb,
  boolean,
  text[]
) from public, anon, authenticated;
revoke all on function public.transition_match_exclusion_after_replay(
  uuid,
  text,
  uuid,
  text,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.mark_match_exclusion_recovery_required(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.claim_match_exclusion_badge_progress(
  uuid,
  text,
  text,
  integer,
  boolean
) from public, anon, authenticated;
revoke all on function public.apply_match_exclusion_badge_reconciliation(
  uuid,
  uuid,
  text,
  uuid,
  text,
  jsonb,
  text
) from public, anon, authenticated;
revoke all on function public.fail_match_exclusion_badge_progress(
  uuid,
  uuid,
  text,
  uuid,
  text,
  text
) from public, anon, authenticated;
revoke all on function public.finalize_match_exclusion_operation(
  uuid,
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
grant execute on function public.claim_replay_job_atomic(
  uuid,
  text,
  text,
  integer,
  boolean
) to service_role;
grant execute on function public.heartbeat_replay_job_atomic(
  uuid,
  text,
  uuid,
  text,
  integer
) to service_role;
grant execute on function public.finish_replay_job_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  jsonb,
  text
) to service_role;
grant execute on function public.assert_replay_write_fence_atomic(
  uuid,
  text,
  uuid,
  text,
  text
) to service_role;
grant execute on function public.apply_replay_write_batch_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  text,
  jsonb,
  boolean,
  text[]
) to service_role;
grant execute on function public.transition_match_exclusion_after_replay(
  uuid,
  text,
  uuid,
  text,
  text,
  text
) to service_role;
grant execute on function public.mark_match_exclusion_recovery_required(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text
) to service_role;
grant execute on function public.claim_match_exclusion_badge_progress(
  uuid,
  text,
  text,
  integer,
  boolean
) to service_role;
grant execute on function public.apply_match_exclusion_badge_reconciliation(
  uuid,
  uuid,
  text,
  uuid,
  text,
  jsonb,
  text
) to service_role;
grant execute on function public.fail_match_exclusion_badge_progress(
  uuid,
  uuid,
  text,
  uuid,
  text,
  text
) to service_role;
grant execute on function public.finalize_match_exclusion_operation(
  uuid,
  text,
  text,
  text,
  text
) to service_role;

comment on function public.apply_match_exclusions_atomic(
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
) is
  'Service-only atomic exact-ID match soft exclusion with row-version CAS and one durable replay job.';
comment on function public.apply_replay_write_batch_atomic(
  uuid,
  text,
  uuid,
  text,
  text,
  text,
  jsonb,
  boolean,
  text[]
) is
  'Service-only idempotent replay projection batch fenced by the exact active job lease in the same transaction.';
comment on function public.apply_match_exclusion_badge_reconciliation(
  uuid,
  uuid,
  text,
  uuid,
  text,
  jsonb,
  text
) is
  'Service-only exact-player reconciliation restricted to frozen, engine-owned live badge IDs.';

notify pgrst, 'reload schema';
