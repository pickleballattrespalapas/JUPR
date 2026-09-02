-- Durable, day-scoped Tournament Live operations.
--
-- The browser never accesses these rows or RPCs directly. FastAPI authenticates
-- the operator, persists a tournament_admin_operations intent, and invokes the
-- service-role-only RPCs below. PostgreSQL is the final authority for the shared
-- court and player claims that span multiple draws.

do $migration_preflight$
begin
  if to_regclass('public.tournaments') is null
     or to_regclass('public.tournament_registration_days') is null
     or to_regclass('public.tournament_registration_settings') is null
     or to_regclass('public.tournament_event_options') is null
     or to_regclass('public.tournament_event_draws') is null
     or to_regclass('public.tournament_teams') is null
     or to_regclass('public.tournament_games') is null
     or to_regclass('public.tournament_podium') is null
     or to_regclass('public.matches') is null
     or to_regclass('public.tournament_registrations') is null
     or to_regclass('public.tournament_registration_check_ins') is null
     or to_regclass('public.tournament_commerce_orders') is null
     or to_regclass('public.tournament_admin_operations') is null
     or to_regclass('public.admin_activity_log') is null
     or to_regclass('public.player_badges') is null
     or to_regclass('public.players') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament day live dependencies must exist before the day-live schema is installed.';
  end if;
end
$migration_preflight$;

create table if not exists public.tournament_day_live_runs (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  registration_day_id text not null references public.tournament_registration_days(id) on delete restrict,
  state text not null default 'DRAFT',
  version bigint not null default 1,
  queue_version bigint not null default 1,
  activation_fingerprint text not null,
  activation_evidence jsonb not null default '{}'::jsonb,
  last_operation_key text not null references public.tournament_admin_operations(operation_key) on delete restrict,
  activated_by text not null,
  activated_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_by text not null,
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  closed_at timestamptz null,
  constraint tournament_day_live_runs_day_unique unique (tournament_id, registration_day_id),
  constraint tournament_day_live_runs_scope_unique unique (id, tournament_id, registration_day_id),
  constraint tournament_day_live_runs_state_chk
    check (state in ('DRAFT', 'ACTIVE', 'PAUSED', 'CLOSED')),
  constraint tournament_day_live_runs_version_chk check (version >= 1 and queue_version >= 1),
  constraint tournament_day_live_runs_fingerprint_chk
    check (activation_fingerprint ~ '^[0-9a-f]{64}$')
);

create table if not exists public.tournament_day_live_draws (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null,
  tournament_id uuid not null,
  registration_day_id text not null,
  draw_id uuid not null references public.tournament_event_draws(id) on delete restrict,
  state text not null default 'ACTIVE',
  priority integer not null default 0,
  source_draw_updated_at timestamptz not null,
  version bigint not null default 1,
  last_operation_key text not null references public.tournament_admin_operations(operation_key) on delete restrict,
  activated_by text not null,
  activated_at timestamptz not null default pg_catalog.clock_timestamp(),
  last_assigned_at timestamptz null,
  updated_by text not null,
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint tournament_day_live_draws_run_fk
    foreign key (run_id, tournament_id, registration_day_id)
    references public.tournament_day_live_runs(id, tournament_id, registration_day_id)
    on delete cascade,
  constraint tournament_day_live_draws_membership_unique unique (run_id, draw_id),
  constraint tournament_day_live_draws_state_chk
    check (state in ('ACTIVE', 'PAUSED', 'REMOVED')),
  constraint tournament_day_live_draws_version_chk check (version >= 1)
);

create unique index if not exists uq_tournament_day_live_runs_active_tournament
  on public.tournament_day_live_runs (tournament_id)
  where state in ('ACTIVE', 'PAUSED');

create index if not exists idx_tournament_day_live_draws_draw
  on public.tournament_day_live_draws (draw_id, state);

create table if not exists public.tournament_day_live_courts (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null,
  tournament_id uuid not null,
  registration_day_id text not null,
  court_key text not null,
  label text not null,
  position integer not null,
  state text not null default 'OPEN',
  version bigint not null default 1,
  last_operation_key text not null references public.tournament_admin_operations(operation_key) on delete restrict,
  updated_by text not null,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint tournament_day_live_courts_run_fk
    foreign key (run_id, tournament_id, registration_day_id)
    references public.tournament_day_live_runs(id, tournament_id, registration_day_id)
    on delete cascade,
  constraint tournament_day_live_courts_key_unique unique (run_id, court_key),
  constraint tournament_day_live_courts_position_unique unique (run_id, position),
  constraint tournament_day_live_courts_key_chk
    check (nullif(pg_catalog.btrim(court_key), '') is not null),
  constraint tournament_day_live_courts_state_chk
    check (state in ('OPEN', 'OUT_OF_SERVICE', 'CLOSED')),
  constraint tournament_day_live_courts_version_chk check (version >= 1)
);

create table if not exists public.tournament_day_live_queue (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null,
  tournament_id uuid not null,
  registration_day_id text not null,
  day_draw_id uuid not null references public.tournament_day_live_draws(id) on delete cascade,
  draw_id uuid not null references public.tournament_event_draws(id) on delete restrict,
  game_id uuid not null references public.tournament_games(id) on delete restrict,
  team_a_id uuid null references public.tournament_teams(id) on delete restrict,
  team_b_id uuid null references public.tournament_teams(id) on delete restrict,
  state text not null,
  priority bigint not null,
  court_id uuid null references public.tournament_day_live_courts(id) on delete restrict,
  blocker_code text null,
  blocker_detail text null,
  version bigint not null default 1,
  last_operation_key text not null references public.tournament_admin_operations(operation_key) on delete restrict,
  eligible_since timestamptz null,
  held_at timestamptz null,
  called_at timestamptz null,
  started_at timestamptz null,
  released_at timestamptz null,
  completed_at timestamptz null,
  updated_by text not null,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint tournament_day_live_queue_run_fk
    foreign key (run_id, tournament_id, registration_day_id)
    references public.tournament_day_live_runs(id, tournament_id, registration_day_id)
    on delete cascade,
  constraint tournament_day_live_queue_game_unique unique (run_id, game_id),
  constraint tournament_day_live_queue_game_day_authority_unique unique (game_id),
  constraint tournament_day_live_queue_state_chk
    check (state in ('WAITING', 'HELD', 'CALLED', 'ON_COURT', 'COMPLETED', 'BLOCKED', 'WITHDRAWN')),
  constraint tournament_day_live_queue_version_chk check (version >= 1),
  constraint tournament_day_live_queue_court_state_chk check (
    (state in ('HELD', 'CALLED', 'ON_COURT') and court_id is not null and released_at is null)
    or
    (state not in ('HELD', 'CALLED', 'ON_COURT') and court_id is null)
  )
);

create index if not exists idx_tournament_day_live_queue_order
  on public.tournament_day_live_queue (run_id, state, priority, id);

create unique index if not exists uq_tournament_day_live_queue_active_court
  on public.tournament_day_live_queue (run_id, court_id)
  where court_id is not null and released_at is null;

create table if not exists public.tournament_day_live_participant_claims (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null,
  tournament_id uuid not null,
  registration_day_id text not null,
  queue_id uuid not null references public.tournament_day_live_queue(id) on delete restrict,
  game_id uuid not null references public.tournament_games(id) on delete restrict,
  player_id integer not null references public.players(id) on delete restrict,
  state text not null,
  version bigint not null default 1,
  last_operation_key text not null references public.tournament_admin_operations(operation_key) on delete restrict,
  claimed_at timestamptz not null default pg_catalog.clock_timestamp(),
  released_at timestamptz null,
  released_by text null,
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint tournament_day_live_claims_run_fk
    foreign key (run_id, tournament_id, registration_day_id)
    references public.tournament_day_live_runs(id, tournament_id, registration_day_id)
    on delete cascade,
  constraint tournament_day_live_claims_queue_player_unique unique (queue_id, player_id),
  constraint tournament_day_live_claims_state_chk
    check (state in ('HELD', 'CALLED', 'ON_COURT', 'RELEASED')),
  constraint tournament_day_live_claims_release_chk check (
    (state = 'RELEASED' and released_at is not null)
    or
    (state <> 'RELEASED' and released_at is null)
  ),
  constraint tournament_day_live_claims_version_chk check (version >= 1)
);

create unique index if not exists uq_tournament_day_live_claims_active_player
  on public.tournament_day_live_participant_claims (run_id, player_id)
  where released_at is null;

create unique index if not exists uq_tournament_day_live_claims_active_tournament_player
  on public.tournament_day_live_participant_claims (tournament_id, player_id)
  where released_at is null;

create index if not exists idx_tournament_day_live_claims_queue
  on public.tournament_day_live_participant_claims (queue_id, released_at);

alter table public.tournament_day_live_runs enable row level security;
alter table public.tournament_day_live_runs force row level security;
alter table public.tournament_day_live_draws enable row level security;
alter table public.tournament_day_live_draws force row level security;
alter table public.tournament_day_live_courts enable row level security;
alter table public.tournament_day_live_courts force row level security;
alter table public.tournament_day_live_queue enable row level security;
alter table public.tournament_day_live_queue force row level security;
alter table public.tournament_day_live_participant_claims enable row level security;
alter table public.tournament_day_live_participant_claims force row level security;

revoke all on table public.tournament_day_live_runs from public, anon, authenticated, service_role;
revoke all on table public.tournament_day_live_draws from public, anon, authenticated, service_role;
revoke all on table public.tournament_day_live_courts from public, anon, authenticated, service_role;
revoke all on table public.tournament_day_live_queue from public, anon, authenticated, service_role;
revoke all on table public.tournament_day_live_participant_claims from public, anon, authenticated, service_role;
grant select, insert, update on table public.tournament_day_live_runs to service_role;
grant select, insert, update on table public.tournament_day_live_draws to service_role;
grant select, insert, update on table public.tournament_day_live_courts to service_role;
grant select, insert, update on table public.tournament_day_live_queue to service_role;
grant select, insert, update on table public.tournament_day_live_participant_claims to service_role;

create or replace function public.assert_tournament_day_live_operation(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_operation_key text,
  p_request_fingerprint text,
  p_action text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_intent jsonb;
begin
  select pg_catalog.jsonb_build_object(
           'expected_state', operation.expected_state,
           'actor', operation.created_by,
           'payload', operation.request_json->'payload'
         )
    into v_intent
    from public.tournament_admin_operations as operation
    join public.tournaments as tournament
      on tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   where operation.operation_key = p_operation_key
     and operation.request_fingerprint = p_request_fingerprint
     and operation.club_id = p_club_id
     and operation.surface = 'tournament_live'
     and operation.action = p_action
     and operation.entity_type = 'tournament_registration_day'
     and operation.entity_id = pg_catalog.concat_ws(':', p_tournament_id, p_registration_day_id)
     and operation.lock_scope = pg_catalog.concat_ws(':', 'tournament', p_tournament_id, 'day', p_registration_day_id)
     and operation.status = 'intent'
   for update of operation;

  if not found then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: exact durable day operation intent is required.';
  end if;
  if v_intent->>'expected_state' is distinct from (
       select operation.request_json->>'expected_state'
         from public.tournament_admin_operations as operation
        where operation.operation_key = p_operation_key
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: durable expected-state evidence is inconsistent.';
  end if;
  return v_intent;
end;
$function$;

create or replace function public.guard_tournament_game_day_live_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw_id uuid := case when tg_op = 'DELETE' then old.draw_id else new.draw_id end;
  v_game_id uuid := case when tg_op = 'INSERT' then new.id else old.id end;
  v_operation_key text := nullif(
    pg_catalog.current_setting('jupr.day_live_operation_key', true),
    ''
  );
  v_fenced boolean := false;
  v_authorized boolean := false;
begin
  if v_draw_id is null then
    return case when tg_op = 'DELETE' then old else new end;
  end if;

  if tg_op = 'INSERT' then
    select exists (
      select 1
        from public.tournament_day_live_draws as day_draw
        join public.tournament_day_live_runs as run on run.id = day_draw.run_id
       where day_draw.draw_id = v_draw_id
         and day_draw.state in ('ACTIVE', 'PAUSED')
         and run.state in ('ACTIVE', 'PAUSED')
    ) into v_fenced;
  else
    select exists (
      select 1
        from public.tournament_day_live_queue as queue
        join public.tournament_day_live_runs as run on run.id = queue.run_id
       where queue.game_id = v_game_id
         and run.state in ('ACTIVE', 'PAUSED')
    ) into v_fenced;
  end if;

  if not v_fenced then
    return case when tg_op = 'DELETE' then old else new end;
  end if;

  if v_operation_key is not null then
    select exists (
      select 1
        from public.tournament_admin_operations as operation
        join public.tournament_day_live_runs as run
          on run.club_id = operation.club_id
         and operation.entity_type = 'tournament_registration_day'
         and operation.entity_id = pg_catalog.concat_ws(
           ':', run.tournament_id::text, run.registration_day_id
         )
         and operation.lock_scope = pg_catalog.concat_ws(
           ':', 'tournament', run.tournament_id::text,
           'day', run.registration_day_id
         )
        join public.tournament_day_live_draws as day_draw
          on day_draw.run_id = run.id and day_draw.draw_id = v_draw_id
       where operation.operation_key = v_operation_key
         and operation.surface = 'tournament_live'
         and operation.status = 'intent'
         and operation.action in (
           'tournament_day_live_score_and_release',
           'tournament_day_live_correct_completed_score',
           'tournament_day_live_generate_playoffs'
         )
    ) into v_authorized;
  end if;

  if not v_authorized then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_SCORE_PATH_REQUIRED: use score_and_release for an actively queued game; day-fenced game generation must use its day command.';
  end if;
  return case when tg_op = 'DELETE' then old else new end;
end;
$function$;

drop trigger if exists trg_00_tournament_games_day_live_fence on public.tournament_games;
create trigger trg_00_tournament_games_day_live_fence
before insert or update or delete on public.tournament_games
for each row execute function public.guard_tournament_game_day_live_mutation();

create or replace function public.guard_tournament_check_in_during_player_claim()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_old_player_id integer;
  v_new_player_id integer;
begin
  if old.attendance_status is not distinct from new.attendance_status
     and old.waiver_verified is not distinct from new.waiver_verified
     and old.attendee_identity_key is not distinct from new.attendee_identity_key
     and old.approved_substitute_player_id is not distinct from new.approved_substitute_player_id
     and old.tournament_id is not distinct from new.tournament_id
     and old.registration_day_id is not distinct from new.registration_day_id
     and old.registration_id is not distinct from new.registration_id then
    return new;
  end if;
  if old.attendee_identity_key ~ '^player:[0-9]+$' then
    v_old_player_id := pg_catalog.split_part(old.attendee_identity_key, ':', 2)::integer;
  end if;
  if new.attendee_identity_key ~ '^player:[0-9]+$' then
    v_new_player_id := pg_catalog.split_part(new.attendee_identity_key, ':', 2)::integer;
  end if;
  if exists (
    select 1
      from public.tournament_day_live_participant_claims as claim
      join public.tournament_day_live_runs as run on run.id = claim.run_id
     where (
         (run.tournament_id = old.tournament_id and run.registration_day_id = old.registration_day_id)
         or
         (run.tournament_id = new.tournament_id and run.registration_day_id = new.registration_day_id)
       )
       and run.state in ('ACTIVE', 'PAUSED')
       and claim.released_at is null
       and claim.player_id in (v_old_player_id, v_new_player_id)
  ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYER_CLAIM: release or score the player current court before changing check-in identity, attendance, or waiver.';
  end if;
  return new;
end;
$function$;

drop trigger if exists trg_00_tournament_check_in_day_live_claim on public.tournament_registration_check_ins;
create trigger trg_00_tournament_check_in_day_live_claim
before update on public.tournament_registration_check_ins
for each row execute function public.guard_tournament_check_in_during_player_claim();

create or replace function public.guard_tournament_team_during_player_claim()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if old.player1_id is not distinct from new.player1_id
     and old.player2_id is not distinct from new.player2_id
     and old.tournament_id is not distinct from new.tournament_id
     and old.draw_id is not distinct from new.draw_id
     and old.registration_day_id is not distinct from new.registration_day_id
     and old.event_option_id is not distinct from new.event_option_id then
    return new;
  end if;
  if exists (
    select 1
      from public.tournament_day_live_queue as queue
      join public.tournament_day_live_runs as run on run.id = queue.run_id
     where run.tournament_id = old.tournament_id
       and run.state in ('ACTIVE', 'PAUSED')
       and (queue.team_a_id = old.id or queue.team_b_id = old.id)
  ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYER_CLAIM: release or score the team current court before changing its player roster.';
  end if;
  return new;
end;
$function$;

drop trigger if exists trg_00_tournament_team_day_live_claim on public.tournament_teams;
create trigger trg_00_tournament_team_day_live_claim
before update of player1_id, player2_id, tournament_id, draw_id, registration_day_id, event_option_id
on public.tournament_teams
for each row execute function public.guard_tournament_team_during_player_claim();

create or replace function public.admin_score_release_tournament_day_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_game_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_court_version bigint,
  p_expected_game_updated_at timestamptz,
  p_expected_draw_updated_at timestamptz,
  p_game_patch jsonb,
  p_dependency_updates jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_queue public.tournament_day_live_queue%rowtype;
  v_game public.tournament_games%rowtype;
  v_intent jsonb;
  v_court_id uuid;
  v_score_result jsonb;
  v_dependency_ids uuid[] := '{}'::uuid[];
  v_expected_dependency_ids uuid[] := '{}'::uuid[];
  v_locked_game_ids uuid[] := '{}'::uuid[];
  v_locked_game_count integer := 0;
  v_dependency_count integer := 0;
  v_assignments jsonb := '[]'::jsonb;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_score_and_release'
  );
  if v_intent #>> '{payload,action}' is distinct from 'score_and_release'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,payload,game_id}' is distinct from p_game_id
     or v_intent #> '{payload,payload}' is distinct from pg_catalog.jsonb_build_object(
          'game_id', p_game_id,
          'score_a', (p_game_patch->>'score_a')::integer,
          'score_b', (p_game_patch->>'score_b')::integer
        )
     or (v_intent #>> '{payload,payload,score_a}')::integer
          is distinct from (p_game_patch->>'score_a')::integer
     or (v_intent #>> '{payload,payload,score_b}')::integer
          is distinct from (p_game_patch->>'score_b')::integer
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,court_version}')::bigint
          is distinct from p_expected_court_version
     or nullif(v_intent #>> '{payload,expected,game_version}', '')::timestamptz
          is distinct from p_expected_game_updated_at
     or nullif(v_intent #>> '{payload,score_evidence,source_draw_updated_at}', '')::timestamptz
          is distinct from p_expected_draw_updated_at
     or v_intent #> '{payload,score_evidence,game_patch}' is distinct from p_game_patch
     or v_intent #> '{payload,score_evidence,dependency_updates}'
          is distinct from p_dependency_updates then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: score arguments do not match durable intent.';
  end if;
  if (p_game_patch->>'score_a') is null
     or (p_game_patch->>'score_b') is null
     or (p_game_patch->>'score_a')::integer < 0
     or (p_game_patch->>'score_b')::integer < 0
     or (p_game_patch->>'score_a')::integer = (p_game_patch->>'score_b')::integer
     or nullif(p_game_patch->>'winner_team_id', '') is null
     or nullif(p_game_patch->>'loser_team_id', '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_SCORE: a finalized non-tied score is required.';
  end if;
  if p_dependency_updates is null
     or pg_catalog.jsonb_typeof(p_dependency_updates) is distinct from 'array' then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: dependency updates must be an exact array.';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_array_elements(p_dependency_updates) as dependency(value)
     where pg_catalog.jsonb_typeof(dependency.value) is distinct from 'object'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: every dependency update must be an object.';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_array_elements(p_dependency_updates) as dependency(value)
      cross join lateral pg_catalog.jsonb_object_keys(dependency.value) as field(key)
     where field.key not in (
             'id', 'expected_updated_at', 'team_a_id', 'team_b_id',
             'score_a', 'score_b', 'winner_team_id', 'loser_team_id',
             'finalized_at'
           )
        or (
          field.key in (
            'score_a', 'score_b', 'winner_team_id', 'loser_team_id',
            'finalized_at'
          )
          and dependency.value->field.key is distinct from 'null'::jsonb
        )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: downstream updates may resolve teams only.';
  end if;
  select coalesce(
           pg_catalog.array_agg(nullif(dependency.value->>'id', '')::uuid
             order by nullif(dependency.value->>'id', '')::uuid),
           '{}'::uuid[]
         )
    into v_dependency_ids
    from pg_catalog.jsonb_array_elements(p_dependency_updates) as dependency(value);
  v_dependency_count := pg_catalog.jsonb_array_length(p_dependency_updates);
  if pg_catalog.cardinality(v_dependency_ids) is distinct from v_dependency_count
     or pg_catalog.cardinality(v_dependency_ids) is distinct from pg_catalog.cardinality(
       array(select distinct dependency_id from pg_catalog.unnest(v_dependency_ids) as dependency(dependency_id))
     )
     or nullif(p_game_id, '')::uuid = any(v_dependency_ids) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: dependency game identities must be present, unique, and exclude the scored game.';
  end if;

  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state = 'ACTIVE'
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: day run changed or is not active.';
  end if;

  select queue.* into v_queue
    from public.tournament_day_live_queue as queue
    join public.tournament_day_live_draws as day_draw
      on day_draw.id = queue.day_draw_id
     and day_draw.state in ('ACTIVE', 'PAUSED')
   where queue.run_id = v_run.id
     and queue.game_id::text = p_game_id
     and queue.state in ('HELD', 'CALLED', 'ON_COURT')
     and queue.court_id is not null
     and queue.released_at is null
   for update of queue;
  if v_queue.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: game is no longer on a day court.';
  end if;
  v_court_id := v_queue.court_id;

  -- The legacy score CAS locks target and dependency games in UUID order.
  -- Acquire that exact set in the same order so this wrapper cannot create a
  -- target-first/dependency-first deadlock with a concurrent legacy caller.
  select coalesce(
           pg_catalog.array_agg(game_id order by game_id),
           '{}'::uuid[]
         )
    into v_locked_game_ids
    from pg_catalog.unnest(
      pg_catalog.array_append(v_dependency_ids, v_queue.game_id)
    ) as locked(game_id);
  perform game.id
    from public.tournament_games as game
   where game.id = any(v_locked_game_ids)
     and game.tournament_id = v_queue.tournament_id
     and game.draw_id = v_queue.draw_id
   order by game.id
   for update;
  get diagnostics v_locked_game_count = row_count;
  if v_locked_game_count is distinct from pg_catalog.cardinality(v_locked_game_ids) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: a scored or downstream game changed.';
  end if;

  select game.* into v_game
    from public.tournament_games as game
   where game.id = v_queue.game_id
     and game.tournament_id = v_queue.tournament_id
     and game.draw_id = v_queue.draw_id
     and game.team_a_id = v_queue.team_a_id
     and game.team_b_id = v_queue.team_b_id
     and game.updated_at = p_expected_game_updated_at
  ;
  if v_game.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: queued game or team assignment changed.';
  end if;
  select coalesce(
           pg_catalog.array_agg(distinct downstream.id order by downstream.id),
           '{}'::uuid[]
         )
    into v_expected_dependency_ids
    from public.tournament_games as downstream
   where downstream.tournament_id = v_game.tournament_id
     and downstream.draw_id = v_game.draw_id
     and downstream.id <> v_game.id
     and pg_catalog.upper(coalesce(downstream.stage, '')) = 'PLAYOFF'
     and nullif(v_game.playoff_game_code, '') is not null
     and (
       downstream.team_a_source->>'winnerOf' = v_game.playoff_game_code
       or downstream.team_a_source->>'loserOf' = v_game.playoff_game_code
       or downstream.team_b_source->>'winnerOf' = v_game.playoff_game_code
       or downstream.team_b_source->>'loserOf' = v_game.playoff_game_code
     );
  if v_dependency_ids is distinct from v_expected_dependency_ids then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: exact direct downstream dependency updates are required.';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_array_elements(p_dependency_updates) as dependency(value)
      join public.tournament_games as downstream
        on downstream.id = nullif(dependency.value->>'id', '')::uuid
       and downstream.tournament_id = v_game.tournament_id
       and downstream.draw_id = v_game.draw_id
     where not (dependency.value ? 'team_a_id' or dependency.value ? 'team_b_id')
        or (
          dependency.value ? 'team_a_id'
          and (
            nullif(dependency.value->>'team_a_id', '') is null
            or case
              when downstream.team_a_source->>'winnerOf' = v_game.playoff_game_code
                then dependency.value->>'team_a_id' is distinct from p_game_patch->>'winner_team_id'
              when downstream.team_a_source->>'loserOf' = v_game.playoff_game_code
                then dependency.value->>'team_a_id' is distinct from p_game_patch->>'loser_team_id'
              else true
            end
          )
        )
        or (
          dependency.value ? 'team_b_id'
          and (
            nullif(dependency.value->>'team_b_id', '') is null
            or case
              when downstream.team_b_source->>'winnerOf' = v_game.playoff_game_code
                then dependency.value->>'team_b_id' is distinct from p_game_patch->>'winner_team_id'
              when downstream.team_b_source->>'loserOf' = v_game.playoff_game_code
                then dependency.value->>'team_b_id' is distinct from p_game_patch->>'loser_team_id'
              else true
            end
          )
        )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: downstream team resolution must exactly follow the scored bracket source.';
  end if;
  if (
    select pg_catalog.count(*)
      from pg_catalog.jsonb_to_recordset(p_dependency_updates)
        as dependency(id text, expected_updated_at timestamptz, team_a_id text, team_b_id text)
      join public.tournament_day_live_queue as dependency_queue
        on dependency_queue.game_id = nullif(dependency.id, '')::uuid
       and dependency_queue.run_id = v_run.id
       and dependency_queue.day_draw_id = v_queue.day_draw_id
       and dependency_queue.draw_id = v_queue.draw_id
       and dependency_queue.registration_day_id = v_run.registration_day_id
       and dependency_queue.state = 'BLOCKED'
       and dependency_queue.court_id is null
       and dependency_queue.released_at is null
      join public.tournament_games as dependency_game
        on dependency_game.id = dependency_queue.game_id
       and dependency_game.tournament_id = v_queue.tournament_id
       and dependency_game.draw_id = v_queue.draw_id
       and dependency_game.registration_day_id = v_run.registration_day_id
       and dependency_game.updated_at = dependency.expected_updated_at
     where dependency.expected_updated_at is not null
       and dependency_game.finalized_at is null
       and dependency_game.score_a is null
       and dependency_game.score_b is null
       and dependency_game.winner_team_id is null
       and dependency_game.loser_team_id is null
       and not exists (
         select 1
           from public.tournament_day_live_participant_claims as claim
          where claim.queue_id = dependency_queue.id and claim.released_at is null
       )
       and (
         nullif(dependency.team_a_id, '') is null
         or exists (
           select 1 from public.tournament_teams as team
            where team.id = nullif(dependency.team_a_id, '')::uuid
              and team.tournament_id = v_queue.tournament_id
              and team.draw_id = v_queue.draw_id
              and team.registration_day_id = v_run.registration_day_id
              and team.event_option_id = dependency_game.event_option_id
         )
       )
       and (
         nullif(dependency.team_b_id, '') is null
         or exists (
           select 1 from public.tournament_teams as team
           where team.id = nullif(dependency.team_b_id, '')::uuid
              and team.tournament_id = v_queue.tournament_id
              and team.draw_id = v_queue.draw_id
              and team.registration_day_id = v_run.registration_day_id
              and team.event_option_id = dependency_game.event_option_id
         )
       )
       and (
         nullif(dependency.team_a_id, '') is null
         or nullif(dependency.team_b_id, '') is null
         or dependency.team_a_id is distinct from dependency.team_b_id
       )
  ) is distinct from v_dependency_count then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DEPENDENCY: downstream games are not exact blocked, unassigned, unscored games in this day draw.';
  end if;
  if nullif(p_game_patch->>'finalized_at', '') is null
     or nullif(p_game_patch->>'winner_team_id', '') is null
     or nullif(p_game_patch->>'loser_team_id', '') is null
     or (p_game_patch->>'winner_team_id') = (p_game_patch->>'loser_team_id')
     or p_game_patch->>'winner_team_id' not in (
       v_queue.team_a_id::text, v_queue.team_b_id::text
     )
     or p_game_patch->>'loser_team_id' not in (
       v_queue.team_a_id::text, v_queue.team_b_id::text
     )
     or (
       (p_game_patch->>'score_a')::integer > (p_game_patch->>'score_b')::integer
       and p_game_patch->>'winner_team_id' <> v_queue.team_a_id::text
     )
     or (
       (p_game_patch->>'score_b')::integer > (p_game_patch->>'score_a')::integer
       and p_game_patch->>'winner_team_id' <> v_queue.team_b_id::text
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_SCORE: result identity must exactly match the queued teams and winning score.';
  end if;

  perform court.id
    from public.tournament_day_live_courts as court
   where court.id = v_court_id
     and court.run_id = v_run.id
     and court.version = p_expected_court_version
   for update;
  if not found then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: assigned court changed.';
  end if;
  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.queue_id = v_queue.id and claim.released_at is null
   order by claim.player_id
   for update;
  if coalesce(
    (
      select pg_catalog.array_agg(claim.player_id order by claim.player_id)
        from public.tournament_day_live_participant_claims as claim
       where claim.queue_id = v_queue.id and claim.released_at is null
    ),
    '{}'::integer[]
  ) <> public.tournament_day_live_game_player_ids(
    p_tournament_id, v_queue.draw_id, v_queue.game_id
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYER_CLAIM: exact active player claims are required before scoring.';
  end if;

  perform pg_catalog.set_config('jupr.day_live_operation_key', p_operation_key, true);
  v_score_result := public.admin_score_tournament_game_cas(
    p_club_id,
    p_tournament_id,
    p_game_id,
    p_expected_game_updated_at,
    p_expected_draw_updated_at,
    p_game_patch,
    p_dependency_updates
  );

  update public.tournament_day_live_participant_claims as claim
     set state = 'RELEASED',
         released_at = pg_catalog.clock_timestamp(),
         released_by = p_actor,
         version = claim.version + 1,
         last_operation_key = p_operation_key,
         updated_at = pg_catalog.clock_timestamp()
   where claim.queue_id = v_queue.id and claim.released_at is null;
  update public.tournament_day_live_queue as queue
     set state = 'COMPLETED',
         court_id = null,
         released_at = pg_catalog.clock_timestamp(),
         completed_at = pg_catalog.clock_timestamp(),
         version = queue.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where queue.id = v_queue.id;
  update public.tournament_day_live_courts as court
     set version = court.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where court.id = v_court_id;

  -- The score CAS resolves playoff dependency teams. Promote only exact,
  -- concrete downstream games; the shared scheduler rechecks attendance and
  -- player claims before putting one on a court.
  update public.tournament_day_live_queue as queue
     set team_a_id = game.team_a_id,
         team_b_id = game.team_b_id,
         state = 'WAITING',
         blocker_code = null,
         blocker_detail = null,
         eligible_since = pg_catalog.clock_timestamp(),
         version = queue.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
    from public.tournament_games as game
   where queue.run_id = v_run.id
     and queue.draw_id = v_queue.draw_id
     and queue.game_id = game.id
     and queue.game_id = any(v_dependency_ids)
     and queue.state = 'BLOCKED'
     and game.finalized_at is null
     and game.team_a_id is not null
     and game.team_b_id is not null;

  update public.tournament_day_live_draws as day_draw
     set source_draw_updated_at = draw.updated_at,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
    from public.tournament_event_draws as draw
   where day_draw.run_id = v_run.id
     and day_draw.draw_id = v_queue.draw_id
     and draw.id = day_draw.draw_id;

  v_assignments := public.fill_tournament_day_live_courts(v_run.id, p_operation_key, p_actor);
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return v_score_result || pg_catalog.jsonb_build_object(
    'run', pg_catalog.to_jsonb(v_run),
    'released_court_id', v_court_id,
    'assignments', v_assignments,
    'score_and_release', true
  );
end;
$function$;

create or replace function public.admin_correct_completed_tournament_day_game_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_game_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_day_draw_version bigint,
  p_expected_game_updated_at timestamptz,
  p_expected_draw_updated_at timestamptz,
  p_game_patch jsonb,
  p_dependency_updates jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_queue public.tournament_day_live_queue%rowtype;
  v_day_draw public.tournament_day_live_draws%rowtype;
  v_game public.tournament_games%rowtype;
  v_intent jsonb;
  v_score_result jsonb;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_correct_completed_score'
  );
  if v_intent #>> '{payload,action}' is distinct from 'correct_completed_score'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #> '{payload,payload}' is distinct from pg_catalog.jsonb_build_object(
          'game_id', p_game_id,
          'score_a', (p_game_patch->>'score_a')::integer,
          'score_b', (p_game_patch->>'score_b')::integer
        )
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,draw_version}')::bigint
          is distinct from p_expected_day_draw_version
     or nullif(v_intent #>> '{payload,expected,game_version}', '')::timestamptz
          is distinct from p_expected_game_updated_at
     or nullif(v_intent #>> '{payload,score_evidence,source_draw_updated_at}', '')::timestamptz
          is distinct from p_expected_draw_updated_at
     or v_intent #> '{payload,score_evidence,game_patch}' is distinct from p_game_patch
     or v_intent #> '{payload,score_evidence,dependency_updates}'
          is distinct from p_dependency_updates then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: correction arguments do not match durable intent.';
  end if;
  if p_dependency_updates is distinct from '[]'::jsonb then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_DOWNSTREAM: playoff corrections require an explicit bracket-reset operation.';
  end if;
  if (p_game_patch->>'score_a') is null
     or (p_game_patch->>'score_b') is null
     or (p_game_patch->>'score_a')::integer < 0
     or (p_game_patch->>'score_b')::integer < 0
     or (p_game_patch->>'score_a')::integer = (p_game_patch->>'score_b')::integer
     or nullif(p_game_patch->>'finalized_at', '') is null
     or nullif(p_game_patch->>'winner_team_id', '') is null
     or nullif(p_game_patch->>'loser_team_id', '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_SCORE: a finalized non-tied score is required.';
  end if;

  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state in ('ACTIVE', 'PAUSED')
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_STALE: day run changed or is not open.';
  end if;
  select queue.* into v_queue
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
     and queue.game_id::text = p_game_id
     and queue.state = 'COMPLETED'
     and queue.court_id is null
     and queue.released_at is not null
   for update;
  if v_queue.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_STALE: game is not an exact released completed day result.';
  end if;
  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.id = v_queue.day_draw_id
     and day_draw.run_id = v_run.id
     and day_draw.draw_id = v_queue.draw_id
     and day_draw.state in ('ACTIVE', 'PAUSED')
     and day_draw.version = p_expected_day_draw_version
   for update;
  if v_day_draw.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_STALE: day draw changed.';
  end if;
  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.queue_id = v_queue.id
   order by claim.player_id, claim.id
   for update;
  if exists (
    select 1 from public.tournament_day_live_participant_claims as claim
     where claim.queue_id = v_queue.id and claim.released_at is null
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_CLAIM: all participant claims must remain released.';
  end if;
  select game.* into v_game
    from public.tournament_games as game
    join public.tournament_event_draws as draw
      on draw.id = game.draw_id
     and draw.tournament_id = game.tournament_id
     and draw.event_option_id is not distinct from game.event_option_id
    join public.tournament_teams as team_a
      on team_a.id = game.team_a_id
     and team_a.tournament_id = game.tournament_id
     and team_a.draw_id = game.draw_id
    join public.tournament_teams as team_b
      on team_b.id = game.team_b_id
     and team_b.tournament_id = game.tournament_id
     and team_b.draw_id = game.draw_id
   where game.id = v_queue.game_id
     and game.tournament_id = v_run.tournament_id
     and game.draw_id = v_queue.draw_id
     and game.registration_day_id = v_run.registration_day_id
     and game.event_option_id is not distinct from draw.event_option_id
     and game.stage = 'ROUND_ROBIN'
     and game.team_a_id = v_queue.team_a_id
     and game.team_b_id = v_queue.team_b_id
     and game.updated_at = p_expected_game_updated_at
     and game.finalized_at is not null
     and game.score_a is not null and game.score_b is not null
     and game.score_a <> game.score_b
     and game.winner_team_id in (game.team_a_id, game.team_b_id)
     and game.loser_team_id in (game.team_a_id, game.team_b_id)
     and game.winner_team_id <> game.loser_team_id
     and (
       (game.score_a > game.score_b
        and game.winner_team_id = game.team_a_id
        and game.loser_team_id = game.team_b_id)
       or
       (game.score_b > game.score_a
        and game.winner_team_id = game.team_b_id
        and game.loser_team_id = game.team_a_id)
     )
   for update of game;
  if v_game.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_STALE: finalized game identity or result evidence changed.';
  end if;
  if p_game_patch->>'winner_team_id' not in (
       v_game.team_a_id::text, v_game.team_b_id::text
     )
     or p_game_patch->>'loser_team_id' not in (
       v_game.team_a_id::text, v_game.team_b_id::text
     )
     or p_game_patch->>'winner_team_id' = p_game_patch->>'loser_team_id'
     or (
       (p_game_patch->>'score_a')::integer > (p_game_patch->>'score_b')::integer
       and p_game_patch->>'winner_team_id' <> v_game.team_a_id::text
     )
     or (
       (p_game_patch->>'score_b')::integer > (p_game_patch->>'score_a')::integer
       and p_game_patch->>'winner_team_id' <> v_game.team_b_id::text
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_SCORE: result identity must exactly match the game sides and winning score.';
  end if;
  if exists (
    select 1 from public.tournament_games as playoff
     where playoff.tournament_id = v_game.tournament_id
       and playoff.draw_id = v_game.draw_id
       and playoff.stage = 'PLAYOFF'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_RESET_REQUIRED: round-robin correction is blocked after playoff generation.';
  end if;
  if exists (
    select 1 from public.tournament_podium as podium
     where podium.tournament_id = v_game.tournament_id
       and podium.draw_id = v_game.draw_id
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_PODIUM: correction is blocked after podium creation.';
  end if;
  if exists (
    select 1
      from public.matches as official_match
      join public.tournament_games as published_game
        on published_game.id = official_match.tournament_game_id
     where published_game.tournament_id = v_game.tournament_id
       and published_game.draw_id = v_game.draw_id
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_PUBLISHED: correction is blocked after official match publication.';
  end if;
  if exists (
    select 1 from public.tournament_admin_operations as operation
     where operation.club_id = p_club_id
       and operation.operation_key <> p_operation_key
       and operation.status in ('intent', 'mutated', 'recovery_required')
       and (
         operation.lock_scope = pg_catalog.concat_ws(
           ':', 'tournament', p_tournament_id, 'day', p_registration_day_id
         )
         or (
           operation.entity_type = 'tournament_event_draw'
           and operation.entity_id = v_game.draw_id::text
         )
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CORRECTION_OPERATION: reconcile unsettled day or draw operations first.';
  end if;

  perform pg_catalog.set_config('jupr.day_live_operation_key', p_operation_key, true);
  v_score_result := public.admin_score_tournament_game_cas(
    p_club_id,
    p_tournament_id,
    p_game_id,
    p_expected_game_updated_at,
    p_expected_draw_updated_at,
    p_game_patch,
    p_dependency_updates
  );
  update public.tournament_day_live_queue as queue
     set version = queue.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where queue.id = v_queue.id;
  update public.tournament_day_live_draws as day_draw
     set source_draw_updated_at = draw.updated_at,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
    from public.tournament_event_draws as draw
   where day_draw.id = v_day_draw.id and draw.id = day_draw.draw_id;
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return v_score_result || pg_catalog.jsonb_build_object(
    'run', pg_catalog.to_jsonb(v_run),
    'corrected_game_id', v_queue.game_id,
    'corrected_completed_score', true
  );
end;
$function$;

create or replace function public.admin_generate_tournament_day_playoffs_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_draw_id text,
  p_advance_count integer,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_day_draw_version bigint,
  p_expected_draw_version timestamptz,
  p_expected_team_versions jsonb,
  p_expected_source_game_versions jsonb,
  p_games jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_day_draw public.tournament_day_live_draws%rowtype;
  v_day public.tournament_registration_days%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_event_option_id text;
  v_intent jsonb;
  v_insert_result jsonb;
  v_priority_base bigint;
  v_requested_game_ids uuid[] := '{}'::uuid[];
  v_inserted_game_ids uuid[] := '{}'::uuid[];
  v_queued_game_ids uuid[] := '{}'::uuid[];
  v_queue_insert_count integer := 0;
  v_assignments jsonb := '[]'::jsonb;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_generate_playoffs'
  );
  if v_intent #>> '{payload,action}' is distinct from 'generate_playoffs'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,payload,draw_id}' is distinct from p_draw_id
     or (v_intent #>> '{payload,payload,advance_count}')::integer
          is distinct from p_advance_count
     or v_intent #> '{payload,payload}' is distinct from
          pg_catalog.jsonb_build_object(
            'draw_id', p_draw_id,
            'advance_count', p_advance_count
          )
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (v_intent #>> '{payload,expected,draw_version}')::bigint
          is distinct from p_expected_day_draw_version
     or nullif(v_intent #>> '{payload,playoff_evidence,source_draw_updated_at}', '')::timestamptz
          is distinct from p_expected_draw_version
     or v_intent #> '{payload,playoff_evidence,team_versions}'
          is distinct from p_expected_team_versions
     or v_intent #> '{payload,playoff_evidence,source_game_versions}'
          is distinct from p_expected_source_game_versions
     or v_intent #> '{payload,playoff_games}' is distinct from p_games then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: playoff arguments do not match durable intent.';
  end if;
  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state = 'ACTIVE'
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_STALE: day run changed or is not active.';
  end if;
  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.run_id = v_run.id
     and day_draw.draw_id::text = p_draw_id
     and day_draw.state = 'ACTIVE'
     and day_draw.version = p_expected_day_draw_version
   for update;
  if v_day_draw.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_STALE: active draw membership changed.';
  end if;

  select source_draw.event_option_id
    into v_event_option_id
    from public.tournament_event_draws as source_draw
   where source_draw.id = v_day_draw.draw_id
     and source_draw.tournament_id::text = p_tournament_id;
  if v_event_option_id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_SCOPE: reviewed source draw identity is no longer available.';
  end if;
  select source_day.* into v_day
    from public.tournament_registration_days as source_day
   where source_day.id = p_registration_day_id
     and source_day.tournament_id::text = p_tournament_id
     and source_day.enabled is true
   for share;
  select source_event.* into v_event
    from public.tournament_event_options as source_event
   where source_event.id = v_event_option_id
     and source_event.tournament_id::text = p_tournament_id
   for share;
  if v_day.id is null or v_event.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_SCOPE: selected day or event changed before playoff generation.';
  end if;

  perform locked_team.id
    from public.tournament_teams as locked_team
   where locked_team.tournament_id = v_run.tournament_id
     and locked_team.draw_id = v_day_draw.draw_id
   order by locked_team.id
   for share;
  perform locked_game.id
    from public.tournament_games as locked_game
   where locked_game.tournament_id = v_run.tournament_id
     and locked_game.draw_id = v_day_draw.draw_id
   order by locked_game.id
   for share;
  select source_draw.* into v_draw
    from public.tournament_event_draws as source_draw
   where source_draw.id = v_day_draw.draw_id
     and source_draw.tournament_id = v_run.tournament_id
   for share;
  if v_draw.id is null
     or v_draw.updated_at is distinct from p_expected_draw_version
     or v_draw.event_option_id is distinct from v_event.id
     or pg_catalog.upper(coalesce(v_draw.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or coalesce(v_draw.hidden_from_primary_ops, false) is true
     or pg_catalog.upper(coalesce(v_draw.draw_kind, 'STANDARD')) <> 'STANDARD'
     or v_event.enabled is not true
     or pg_catalog.upper(coalesce(v_event.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or (case
       when v_draw.registration_day_id is not null
         then v_draw.registration_day_id = p_registration_day_id
          and case
            when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
              then case
                when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) > 0
                  then v_event.scheduled_day_ids ? v_draw.registration_day_id
                else v_event.registration_day_id is null
                  or v_event.registration_day_id = v_draw.registration_day_id
              end
            else false
          end
       else case
         when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
           then case
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 1
               then v_event.scheduled_day_ids ? p_registration_day_id
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
               then v_event.registration_day_id = p_registration_day_id
             else false
           end
         else false
       end
     end) is not true then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_SCOPE: draw is inactive, unsupported, or no longer scheduled on this enabled day.';
  end if;
  if exists (
    select 1
      from public.tournament_teams as team
     where team.tournament_id = v_run.tournament_id
       and team.draw_id = v_day_draw.draw_id
       and (
         team.registration_day_id is distinct from p_registration_day_id
         or team.event_option_id is distinct from v_draw.event_option_id
       )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_TEAM_SCOPE: every draw team must remain on this exact day and event.';
  end if;
  if exists (
    select participant.player_id
      from public.tournament_teams as team
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where team.tournament_id = v_run.tournament_id
       and team.draw_id = v_day_draw.draw_id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
       and participant.player_id is not null
     group by participant.player_id
    having pg_catalog.count(*) > 1
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ROSTER_PLAYER_DUPLICATE: each player may belong to only one exact team in this draw.';
  end if;
  if (
    select coalesce(
             pg_catalog.array_agg(team.id order by team.id),
             '{}'::uuid[]
           )
      from public.tournament_teams as team
     where team.tournament_id = v_run.tournament_id
       and team.draw_id = v_day_draw.draw_id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) is distinct from (
    select coalesce(
             pg_catalog.array_agg(distinct side.team_id order by side.team_id)
               filter (where side.team_id is not null),
             '{}'::uuid[]
           )
      from public.tournament_games as game
      cross join lateral (values (game.team_a_id), (game.team_b_id)) as side(team_id)
     where game.tournament_id = v_run.tournament_id
       and game.draw_id = v_day_draw.draw_id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ROSTER: every exact draw team must have reviewed round-robin evidence before playoff generation.';
  end if;
  if p_advance_count not in (4, 5, 6)
     or p_advance_count > (
       select pg_catalog.count(*)
        from public.tournament_teams as team
       where team.tournament_id = v_run.tournament_id
         and team.draw_id = v_day_draw.draw_id
          and team.registration_day_id = p_registration_day_id
          and team.event_option_id = v_draw.event_option_id
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ADVANCE_COUNT: choose a supported advancing-team count available for this draw.';
  end if;
  if p_games is null
     or pg_catalog.jsonb_typeof(p_games) is distinct from 'array'
     or pg_catalog.jsonb_array_length(p_games) = 0 then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_GAMES: an exact nonempty playoff plan is required.';
  end if;
  if exists (
    select 1
      from pg_catalog.jsonb_to_recordset(p_games) as planned(
        id text, tournament_id text, draw_id text,
        registration_day_id text, event_option_id text, stage text
      )
     where nullif(planned.id, '') is null
        or planned.tournament_id is distinct from p_tournament_id
        or planned.draw_id is distinct from p_draw_id
        or planned.registration_day_id is distinct from p_registration_day_id
        or planned.event_option_id is distinct from v_draw.event_option_id::text
        or pg_catalog.upper(coalesce(planned.stage, '')) is distinct from 'PLAYOFF'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_GAMES: every planned game must belong to this exact draw, event, and day.';
  end if;
  select coalesce(
           pg_catalog.array_agg(nullif(planned.id, '')::uuid
             order by nullif(planned.id, '')::uuid),
           '{}'::uuid[]
         )
    into v_requested_game_ids
    from pg_catalog.jsonb_to_recordset(p_games) as planned(id text);
  if pg_catalog.cardinality(v_requested_game_ids)
       is distinct from pg_catalog.jsonb_array_length(p_games)
     or pg_catalog.cardinality(v_requested_game_ids) is distinct from pg_catalog.cardinality(
       array(select distinct game_id from pg_catalog.unnest(v_requested_game_ids) as planned(game_id))
     ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_GAMES: planned game identities must be present and unique.';
  end if;
  if exists (
    select 1 from public.tournament_day_live_queue as queue
     where queue.day_draw_id = v_day_draw.id
       and queue.state in ('HELD', 'CALLED', 'ON_COURT')
       and queue.released_at is null
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ASSIGNMENT: finish the draw current court assignments first.';
  end if;
  if not exists (
    select 1 from public.tournament_games as game
     where game.tournament_id::text = p_tournament_id
       and game.draw_id::text = p_draw_id
       and game.stage = 'ROUND_ROBIN'
  ) or exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
     where game.tournament_id::text = p_tournament_id
       and game.draw_id::text = p_draw_id
       and game.stage = 'ROUND_ROBIN'
       and (
         game.registration_day_id is distinct from p_registration_day_id
         or game.event_option_id is distinct from v_draw.event_option_id
         or game.finalized_at is null
         or game.score_a is null or game.score_b is null
         or game.score_a = game.score_b
         or game.team_a_id is null or game.team_b_id is null
         or game.team_a_id = game.team_b_id
         or team_a.id is null or team_b.id is null
         or game.winner_team_id is null or game.loser_team_id is null
         or game.winner_team_id = game.loser_team_id
         or game.winner_team_id not in (game.team_a_id, game.team_b_id)
         or game.loser_team_id not in (game.team_a_id, game.team_b_id)
         or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
         or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id)
       )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_ROUND_ROBIN: every round-robin game must be finalized and non-tied.';
  end if;

  if exists (
    select 1 from public.tournament_admin_operations as operation
     where operation.club_id = p_club_id
       and operation.operation_key <> p_operation_key
       and operation.status in ('intent', 'mutated', 'recovery_required')
       and (
         operation.lock_scope = pg_catalog.concat_ws(':', 'tournament', p_tournament_id, 'day', p_registration_day_id)
         or (operation.entity_type = 'tournament_event_draw' and operation.entity_id = p_draw_id)
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_OPERATION: reconcile the unsettled day or draw operation first.';
  end if;

  perform pg_catalog.set_config('jupr.day_live_operation_key', p_operation_key, true);
  v_insert_result := public.admin_insert_tournament_draw_games_cas(
    p_club_id,
    p_tournament_id,
    p_draw_id,
    p_expected_draw_version,
    'PLAYOFF',
    p_expected_team_versions,
    p_expected_source_game_versions,
    p_games
  );
  if pg_catalog.jsonb_typeof(v_insert_result->'games') is distinct from 'array' then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_INSERT: atomic game insertion returned incomplete evidence.';
  end if;
  select coalesce(
           pg_catalog.array_agg(nullif(inserted.id, '')::uuid
             order by nullif(inserted.id, '')::uuid),
           '{}'::uuid[]
         )
    into v_inserted_game_ids
    from pg_catalog.jsonb_to_recordset(v_insert_result->'games') as inserted(id text);
  if v_inserted_game_ids is distinct from v_requested_game_ids then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_INSERT: inserted games do not exactly match the reviewed plan.';
  end if;

  select coalesce(pg_catalog.max(queue.priority), 0)
    into v_priority_base
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id;
  insert into public.tournament_day_live_queue (
    run_id, tournament_id, registration_day_id, day_draw_id, draw_id, game_id,
    team_a_id, team_b_id, state, priority, court_id,
    blocker_code, blocker_detail, version, last_operation_key,
    eligible_since, updated_by
  )
  select
    v_run.id, game.tournament_id, v_run.registration_day_id,
    v_day_draw.id, game.draw_id, game.id, game.team_a_id, game.team_b_id,
    case when game.team_a_id is not null and game.team_b_id is not null then 'WAITING' else 'BLOCKED' end,
    v_priority_base + pg_catalog.row_number() over (
      order by game.playoff_round nulls last, game.playoff_game_code nulls last, game.id
    ),
    null,
    case when game.team_a_id is null or game.team_b_id is null then 'DEPENDENCY_PENDING' else null end,
    case when game.team_a_id is null or game.team_b_id is null then 'Playoff source teams are not resolved yet.' else null end,
    1, p_operation_key,
    case when game.team_a_id is not null and game.team_b_id is not null then pg_catalog.clock_timestamp() else null end,
    p_actor
  from public.tournament_games as game
  where game.tournament_id::text = p_tournament_id
    and game.draw_id::text = p_draw_id
    and game.id = any(v_requested_game_ids)
    and game.stage = 'PLAYOFF'
    and game.registration_day_id = p_registration_day_id
    and game.event_option_id is not distinct from (
      select draw.event_option_id
        from public.tournament_event_draws as draw
       where draw.id = v_day_draw.draw_id
    )
    and not exists (
      select 1 from public.tournament_day_live_queue as existing
       where existing.run_id = v_run.id and existing.game_id = game.id
    )
  order by game.playoff_round nulls last, game.playoff_game_code nulls last, game.id;
  get diagnostics v_queue_insert_count = row_count;
  if v_queue_insert_count is distinct from pg_catalog.cardinality(v_requested_game_ids) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_QUEUE: every inserted playoff game must enter this day queue.';
  end if;
  select coalesce(
           pg_catalog.array_agg(queue.game_id order by queue.game_id),
           '{}'::uuid[]
         )
    into v_queued_game_ids
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
     and queue.day_draw_id = v_day_draw.id
     and queue.game_id = any(v_requested_game_ids)
     and queue.last_operation_key = p_operation_key;
  if v_queued_game_ids is distinct from v_requested_game_ids then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_QUEUE: queued games do not exactly match the inserted playoff plan.';
  end if;

  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.id = v_day_draw.id
   for update;
  update public.tournament_day_live_draws as day_draw
     set source_draw_updated_at = draw.updated_at,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
    from public.tournament_event_draws as draw
   where day_draw.id = v_day_draw.id and draw.id = day_draw.draw_id;
  v_assignments := public.fill_tournament_day_live_courts(v_run.id, p_operation_key, p_actor);
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return v_insert_result || pg_catalog.jsonb_build_object(
    'run', pg_catalog.to_jsonb(v_run),
    'assignments', v_assignments,
    'playoff_generation', true
  );
end;
$function$;

create or replace function public.admin_activate_tournament_day_live_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_activation_fingerprint text,
  p_activation_evidence jsonb,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_day public.tournament_registration_days%rowtype;
  v_settings public.tournament_registration_settings%rowtype;
  v_intent jsonb;
  v_current_activation_evidence jsonb;
  v_court_count integer;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_activate_day'
  );
  if v_intent #>> '{payload,action}' is distinct from 'activate_day'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent->>'expected_state' is distinct from p_activation_fingerprint
     or v_intent #>> '{payload,expected,state_fingerprint}' is distinct from p_activation_fingerprint
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or v_intent #> '{payload,activation_evidence}'
          is distinct from p_activation_evidence
     or v_intent #> '{payload,payload}' is distinct from '{}'::jsonb then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: activation arguments do not match durable intent.';
  end if;
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('jupr:tournament-day-live'),
    pg_catalog.hashtext(p_tournament_id)
  );
  if coalesce(p_expected_run_version, -1) <> 0
     or coalesce(p_expected_queue_version, -1) <> 0
     or p_activation_fingerprint !~ '^[0-9a-f]{64}$'
     or pg_catalog.jsonb_typeof(p_activation_evidence) <> 'object'
     or pg_catalog.jsonb_typeof(p_activation_evidence->'courts') <> 'array'
     or nullif(pg_catalog.btrim(p_actor), '') is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: activation requires a reviewed inactive day and authenticated actor.';
  end if;
  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
   for update;
  if v_run.id is not null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: this day already has a durable run.';
  end if;
  if exists (
    select 1
      from public.tournament_day_live_runs as other_run
     where other_run.tournament_id::text = p_tournament_id
       and other_run.registration_day_id <> p_registration_day_id
       and other_run.state in ('ACTIVE', 'PAUSED')
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: close the current active tournament day before activating another.';
  end if;

  perform tournament.id
    from public.tournaments as tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id::text = p_club_id
   for share;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_SCOPE: tournament is not part of this club.';
  end if;

  select day.* into v_day
    from public.tournament_registration_days as day
   where day.id = p_registration_day_id
     and day.tournament_id::text = p_tournament_id
     and day.enabled is true
   for share;
  if v_day.id is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DAY: select an enabled day in this tournament.';
  end if;

  perform source_draw.id
    from public.tournament_event_draws as source_draw
    join public.tournament_event_options as source_event
      on source_event.id = source_draw.event_option_id
     and source_event.tournament_id::text = p_tournament_id
     and source_event.enabled is true
     and pg_catalog.upper(coalesce(source_event.status, 'DRAFT')) not in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
   where source_draw.tournament_id::text = p_tournament_id
     and pg_catalog.upper(coalesce(source_draw.status, 'DRAFT')) not in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     and coalesce(source_draw.hidden_from_primary_ops, false) is false
     and pg_catalog.upper(coalesce(source_draw.draw_kind, 'STANDARD')) = 'STANDARD'
     and (case
       when source_draw.registration_day_id is not null
         then source_draw.registration_day_id = p_registration_day_id
          and case
            when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
              then case
                when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) > 0
                  then source_event.scheduled_day_ids ? source_draw.registration_day_id
                else source_event.registration_day_id is null
                  or source_event.registration_day_id = source_draw.registration_day_id
              end
            else false
          end
       else case
         when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
           then case
             when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 1
               then source_event.scheduled_day_ids ? p_registration_day_id
             when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 0
               then source_event.registration_day_id = p_registration_day_id
             else false
           end
         else false
       end
     end) is true
   order by source_draw.id
   for share of source_event, source_draw;
  if not found then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAWS: configure at least one supported draw on this day before activation.';
  end if;

  select settings.* into v_settings
    from public.tournament_registration_settings as settings
   where settings.tournament_id::text = p_tournament_id
   for share;
  if v_settings.id is null
     or pg_catalog.jsonb_typeof(v_settings.venue_courts_json) <> 'array'
     or pg_catalog.jsonb_typeof(v_day.available_court_ids) <> 'array'
     or pg_catalog.jsonb_array_length(v_day.available_court_ids) = 0
     or (
       select pg_catalog.count(distinct court_id.value)
         from pg_catalog.jsonb_array_elements_text(v_day.available_court_ids) as court_id(value)
     ) <> pg_catalog.jsonb_array_length(v_day.available_court_ids) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_COURTS: a unique relationalizable venue court set is required.';
  end if;

  if exists (
    select 1
      from pg_catalog.jsonb_array_elements_text(v_day.available_court_ids) as available(court_key)
     where (
       select pg_catalog.count(*)
         from pg_catalog.jsonb_array_elements(v_settings.venue_courts_json) as inventory(court)
        where nullif(pg_catalog.btrim(inventory.court->>'id'), '') = available.court_key
     ) <> 1
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_COURTS: every selected day court must resolve exactly once in venue inventory.';
  end if;

  select pg_catalog.jsonb_build_object(
           'courts',
           coalesce(
             pg_catalog.jsonb_agg(
               pg_catalog.jsonb_build_object(
                 'court_key', available.court_key,
                 'label', coalesce(
                   nullif(pg_catalog.btrim(inventory.court->>'title'), ''),
                   nullif(pg_catalog.btrim(inventory.court->>'label'), ''),
                   nullif(
                     pg_catalog.btrim(
                       v_day.court_labels ->> (available.position::integer - 1)
                     ),
                     ''
                   ),
                   pg_catalog.concat('Court ', available.position::text)
                 ),
                 'position', available.position
               )
               order by available.position
             ),
             '[]'::jsonb
           )
         )
    into v_current_activation_evidence
    from pg_catalog.jsonb_array_elements_text(v_day.available_court_ids)
         with ordinality as available(court_key, position)
    join lateral (
      select court
        from pg_catalog.jsonb_array_elements(v_settings.venue_courts_json)
          as candidate(court)
       where nullif(pg_catalog.btrim(candidate.court->>'id'), '') = available.court_key
       limit 1
    ) as inventory on true;
  if v_current_activation_evidence is distinct from p_activation_evidence then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed tournament day court plan changed before activation.';
  end if;

  insert into public.tournament_day_live_runs (
    club_id, tournament_id, registration_day_id, state, version, queue_version,
    activation_fingerprint, activation_evidence, last_operation_key,
    activated_by, updated_by
  ) values (
    p_club_id, p_tournament_id::uuid, p_registration_day_id, 'ACTIVE', 1, 1,
    p_activation_fingerprint,
    p_activation_evidence ||
      pg_catalog.jsonb_build_object('contract', 'individual_draw_activation'),
    p_operation_key, p_actor, p_actor
  ) returning * into v_run;

  insert into public.tournament_day_live_courts (
    run_id, tournament_id, registration_day_id, court_key, label, position,
    state, version, last_operation_key, updated_by
  )
  select
    v_run.id, v_run.tournament_id, v_run.registration_day_id,
    available.court_key,
    coalesce(
      nullif(pg_catalog.btrim(inventory.court->>'title'), ''),
      nullif(pg_catalog.btrim(inventory.court->>'label'), ''),
      nullif(
        pg_catalog.btrim(
          v_day.court_labels ->> (available.position::integer - 1)
        ),
        ''
      ),
      pg_catalog.concat('Court ', available.position::text)
    ),
    available.position::integer,
    'OPEN', 1, p_operation_key, p_actor
  from pg_catalog.jsonb_array_elements_text(v_day.available_court_ids)
       with ordinality as available(court_key, position)
  join lateral (
    select court
      from pg_catalog.jsonb_array_elements(v_settings.venue_courts_json) as candidate(court)
     where nullif(pg_catalog.btrim(candidate.court->>'id'), '') = available.court_key
     limit 1
  ) as inventory on true
  order by available.position;
  get diagnostics v_court_count = row_count;
  if v_court_count <> pg_catalog.jsonb_array_length(v_day.available_court_ids) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_COURTS: exact day court materialization failed.';
  end if;

  select run.* into v_run from public.tournament_day_live_runs as run where run.id = v_run.id for update;
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'run', pg_catalog.to_jsonb(v_run),
    'assignments', '[]'::jsonb
  );
end;
$function$;

create or replace function public.admin_transition_tournament_day_draw_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_action text,
  p_draw_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_expected_day_draw_version bigint,
  p_expected_draw_updated_at timestamptz,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_action text := pg_catalog.upper(coalesce(p_action, ''));
  v_run public.tournament_day_live_runs%rowtype;
  v_day_draw public.tournament_day_live_draws%rowtype;
  v_day public.tournament_registration_days%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_event_option_id text;
  v_intent jsonb;
  v_assignments jsonb := '[]'::jsonb;
  v_priority integer;
begin
  if v_action not in ('ACTIVATE', 'PAUSE', 'RESUME') then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_ACTION: activate, pause, or resume is required.';
  end if;
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_' || pg_catalog.lower(v_action) || '_draw'
  );
  if v_intent #>> '{payload,action}' is distinct from
       (case v_action
          when 'ACTIVATE' then 'activate_draw'
          when 'PAUSE' then 'pause_draw'
          else 'resume_draw'
        end)
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #>> '{payload,payload,draw_id}' is distinct from p_draw_id
     or v_intent #> '{payload,payload}' is distinct from
          pg_catalog.jsonb_build_object('draw_id', p_draw_id)
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version
     or (
       v_action = 'ACTIVATE'
       and (
         p_expected_day_draw_version is distinct from 0
         or nullif(v_intent #>> '{payload,expected,draw_version}', '')::timestamptz
              is distinct from p_expected_draw_updated_at
       )
     )
     or (
       v_action <> 'ACTIVATE'
       and (v_intent #>> '{payload,expected,draw_version}')::bigint
             is distinct from p_expected_day_draw_version
     )
     or nullif(v_intent #>> '{payload,draw_evidence,source_draw_updated_at}', '')::timestamptz
          is distinct from p_expected_draw_updated_at then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: draw transition arguments do not match durable intent.';
  end if;

  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.club_id = p_club_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state = 'ACTIVE'
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: day run changed or is not active.';
  end if;

  select day_draw.* into v_day_draw
    from public.tournament_day_live_draws as day_draw
   where day_draw.run_id = v_run.id and day_draw.draw_id::text = p_draw_id
   for update;

  if v_action <> 'ACTIVATE' then
    if v_action = 'RESUME' then
      -- Read the event identity without a row lock, then lock the independent
      -- setup rows before the source children and draw.  ACTIVATE delegates
      -- the same child -> draw order to the seed helper.
      select draw.event_option_id
        into v_event_option_id
        from public.tournament_event_draws as draw
        join public.tournaments as tournament
          on tournament.id = draw.tournament_id
         and tournament.club_id::text = p_club_id
       where draw.id::text = p_draw_id
         and draw.tournament_id::text = p_tournament_id;
      if v_event_option_id is null then
        raise exception using errcode = '40001',
          message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: reviewed draw identity is no longer available.';
      end if;

      select day.* into v_day
        from public.tournament_registration_days as day
       where day.id = p_registration_day_id
         and day.tournament_id::text = p_tournament_id
         and day.enabled is true
       for share;
      if v_day.id is null then
        raise exception using errcode = '40001',
          message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: selected tournament day changed or is disabled.';
      end if;

      select event.* into v_event
        from public.tournament_event_options as event
       where event.id = v_event_option_id
         and event.tournament_id::text = p_tournament_id
       for share;
      if v_event.id is null then
        raise exception using errcode = '40001',
          message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: reviewed event identity changed.';
      end if;

      -- RESUME can immediately fill courts, and the allocator locks effective
      -- team rows.  Prelock every source child in the same child -> draw order
      -- as activation so a concurrent child trigger cannot deadlock on draw.
      perform team.id
        from public.tournament_teams as team
       where team.tournament_id::text = p_tournament_id
         and team.draw_id::text = p_draw_id
       order by team.id
       for share;
      perform game.id
        from public.tournament_games as game
       where game.tournament_id::text = p_tournament_id
         and game.draw_id::text = p_draw_id
       order by game.id
       for share;
    end if;

    select draw.* into v_draw
      from public.tournament_event_draws as draw
      join public.tournaments as tournament
        on tournament.id = draw.tournament_id
       and tournament.club_id::text = p_club_id
     where draw.id::text = p_draw_id
       and draw.tournament_id::text = p_tournament_id
     for share;
    if v_draw.id is null
       or v_draw.updated_at is distinct from p_expected_draw_updated_at then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed source draw changed before transition.';
    end if;

    if v_action = 'RESUME' and (
      v_draw.event_option_id is distinct from v_event.id
      or pg_catalog.upper(coalesce(v_draw.status, 'DRAFT')) in
        ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
      or coalesce(v_draw.hidden_from_primary_ops, false) is true
      or pg_catalog.upper(coalesce(v_draw.draw_kind, 'STANDARD')) <> 'STANDARD'
      or v_event.enabled is not true
      or pg_catalog.upper(coalesce(v_event.status, 'DRAFT')) in
        ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
      or (case
        when v_draw.registration_day_id is not null
          then v_draw.registration_day_id = p_registration_day_id
           and case
             when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
               then case
                 when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) > 0
                   then v_event.scheduled_day_ids ? v_draw.registration_day_id
                 else v_event.registration_day_id is null
                   or v_event.registration_day_id = v_draw.registration_day_id
               end
             else false
           end
        else case
          when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
            then case
              when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 1
                then v_event.scheduled_day_ids ? p_registration_day_id
              when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
                then v_event.registration_day_id = p_registration_day_id
              else false
            end
          else false
        end
      end) is not true
    ) then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: draw is inactive, unsupported, or no longer scheduled on this enabled day.';
    end if;
  end if;

  if v_action = 'ACTIVATE' then
    if v_day_draw.id is not null then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: draw is already part of this day run.';
    end if;
    if p_expected_draw_updated_at is null then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed source draw version is required.';
    end if;
    select coalesce(pg_catalog.max(day_draw.priority), 0) + 1
      into v_priority
      from public.tournament_day_live_draws as day_draw
     where day_draw.run_id = v_run.id;
    perform public.seed_tournament_day_live_draw(
      v_run.id, p_club_id, p_tournament_id, p_registration_day_id,
      p_draw_id::uuid, p_expected_draw_updated_at, v_priority,
      p_operation_key, p_actor
    );
    v_assignments := public.fill_tournament_day_live_courts(v_run.id, p_operation_key, p_actor);
  else
    if v_day_draw.id is null
       or v_day_draw.version <> p_expected_day_draw_version
       or (v_action = 'PAUSE' and v_day_draw.state <> 'ACTIVE')
       or (v_action = 'RESUME' and v_day_draw.state <> 'PAUSED') then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: activated draw state changed after review.';
    end if;
    update public.tournament_day_live_draws as day_draw
       set state = case when v_action = 'PAUSE' then 'PAUSED' else 'ACTIVE' end,
           source_draw_updated_at = v_draw.updated_at,
           version = day_draw.version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = pg_catalog.clock_timestamp()
     where day_draw.id = v_day_draw.id
     returning * into v_day_draw;
    if v_action = 'RESUME' then
      v_assignments := public.fill_tournament_day_live_courts(v_run.id, p_operation_key, p_actor);
    end if;
  end if;

  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return pg_catalog.jsonb_build_object(
    'ok', true, 'run', pg_catalog.to_jsonb(v_run),
    'draw_id', p_draw_id, 'action', v_action,
    'assignments', v_assignments
  );
end;
$function$;

create or replace function public.admin_fill_tournament_day_courts_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_intent jsonb;
  v_assignments jsonb;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_auto_fill_courts'
  );
  if v_intent #>> '{payload,action}' is distinct from 'auto_fill_courts'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #> '{payload,payload}' is distinct from '{}'::jsonb
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: court-fill arguments do not match durable intent.';
  end if;
  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state = 'ACTIVE'
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: day run changed or is not active.';
  end if;
  v_assignments := public.fill_tournament_day_live_courts(v_run.id, p_operation_key, p_actor);
  update public.tournament_day_live_runs as run
     set version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return pg_catalog.jsonb_build_object(
    'ok', true, 'run', pg_catalog.to_jsonb(v_run),
    'assignments', v_assignments
  );
end;
$function$;

create or replace function public.admin_close_tournament_day_live_cas(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_expected_run_version bigint,
  p_expected_queue_version bigint,
  p_operation_key text,
  p_request_fingerprint text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_intent jsonb;
  v_close_draw record;
  v_reviewed boolean;
begin
  v_intent := public.assert_tournament_day_live_operation(
    p_club_id, p_tournament_id, p_registration_day_id,
    p_operation_key, p_request_fingerprint,
    'tournament_day_live_close_day'
  );
  if v_intent #>> '{payload,action}' is distinct from 'close_day'
     or v_intent->>'actor' is distinct from p_actor
     or v_intent #> '{payload,payload}' is distinct from '{}'::jsonb
     or (v_intent #>> '{payload,expected,day_run_version}')::bigint
          is distinct from p_expected_run_version
     or (v_intent #>> '{payload,expected,queue_version}')::bigint
          is distinct from p_expected_queue_version then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_OPERATION: close-day arguments do not match durable intent.';
  end if;
  if exists (
    select 1
      from public.tournament_admin_operations as operation
     where operation.club_id = p_club_id
       and operation.operation_key <> p_operation_key
       and operation.status in ('intent', 'mutated', 'recovery_required')
       and (
         operation.entity_id = pg_catalog.concat_ws(':', p_tournament_id, p_registration_day_id)
         or operation.lock_scope = pg_catalog.concat_ws(
           ':', 'tournament', p_tournament_id, 'day', p_registration_day_id
         )
         or (
           operation.entity_type = 'tournament_event_draw'
           and exists (
             select 1
               from public.tournament_day_live_draws as day_draw
              where day_draw.run_id = (
                select run.id
                  from public.tournament_day_live_runs as run
                 where run.club_id = p_club_id
                   and run.tournament_id::text = p_tournament_id
                   and run.registration_day_id = p_registration_day_id
              )
                and day_draw.draw_id::text = operation.entity_id
           )
         )
       )
  ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_OPERATION: reconcile unsettled day operations before close.';
  end if;
  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.club_id = p_club_id
     and run.tournament_id::text = p_tournament_id
     and run.registration_day_id = p_registration_day_id
     and run.version = p_expected_run_version
     and run.queue_version = p_expected_queue_version
     and run.state in ('ACTIVE', 'PAUSED')
   for update;
  if v_run.id is null then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: day run changed or is not open.';
  end if;

  if not exists (
    select 1
      from public.tournament_day_live_draws as day_draw
     where day_draw.run_id = v_run.id
       and day_draw.state <> 'REMOVED'
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_DRAWS: activate and complete at least one scheduled draw before closing the day.';
  end if;
  -- Close is rare and terminal.  A single deterministic relation-lock request
  -- waits out every in-flight setup writer and blocks day/event/draw phantoms
  -- until the exact supported-draw set has been rechecked and the run closes.
  -- A lock/deadlock victim fails closed and the guarded operation is retryable.
  lock table
    public.tournament_registration_days,
    public.tournament_event_options,
    public.tournament_event_draws
  in share mode;
  if not exists (
    select 1
      from public.tournament_registration_days as source_day
     where source_day.id = v_run.registration_day_id
       and source_day.tournament_id::text = p_tournament_id
  ) then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: selected day scope changed before close.';
  end if;

  perform queue.id
    from public.tournament_day_live_queue as queue
   where queue.run_id = v_run.id
   order by queue.id
   for update;
  perform claim.id
    from public.tournament_day_live_participant_claims as claim
   where claim.run_id = v_run.id
   order by claim.player_id, claim.id
   for update;
  if exists (
    select 1
      from public.tournament_day_live_queue as queue
     where queue.run_id = v_run.id
       and queue.state not in ('COMPLETED', 'WITHDRAWN')
  ) or exists (
    select 1
      from public.tournament_day_live_participant_claims as claim
     where claim.run_id = v_run.id and claim.released_at is null
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_BLOCKED: finish every owned game and release every player claim before closing the day.';
  end if;

  -- Close is the terminal day lifecycle step. Lock the exact draw-owned
  -- progression evidence before taking the draw rows so concurrent legacy
  -- podium/award operations cannot race the close decision.
  perform podium.id
    from public.tournament_podium as podium
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = podium.draw_id
   where podium.tournament_id = v_run.tournament_id
   order by podium.id
   for share;
  perform team.id
    from public.tournament_teams as team
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = team.draw_id
   where team.tournament_id = v_run.tournament_id
   order by team.id
   for share;
  perform game.id
    from public.tournament_games as game
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = game.draw_id
   where game.tournament_id = v_run.tournament_id
   order by game.id
   for share;
  perform badge.id
    from public.player_badges as badge
   where badge.club_id = p_club_id
     and badge.context_type = 'tournament'
     and badge.context_id::text like p_tournament_id || ':draw:%:podium:%'
   order by badge.id
   for share;
  perform draw.id
    from public.tournament_event_draws as draw
    join public.tournament_day_live_draws as day_draw
      on day_draw.run_id = v_run.id and day_draw.draw_id = draw.id
   where draw.tournament_id = v_run.tournament_id
   order by draw.id
   for share;
  perform day_draw.id
    from public.tournament_day_live_draws as day_draw
   where day_draw.run_id = v_run.id
   order by day_draw.draw_id
   for update;

  if exists (
    select 1
      from public.tournament_event_draws as source_draw
      join public.tournament_event_options as source_event
        on source_event.id = source_draw.event_option_id
       and source_event.tournament_id = v_run.tournament_id::text
       and source_event.enabled is true
       and pg_catalog.upper(coalesce(source_event.status, 'DRAFT')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     where source_draw.tournament_id = v_run.tournament_id
       and pg_catalog.upper(coalesce(source_draw.status, 'DRAFT')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
       and coalesce(source_draw.hidden_from_primary_ops, false) is false
       and pg_catalog.upper(coalesce(source_draw.draw_kind, 'STANDARD')) = 'STANDARD'
       and (case
         when source_draw.registration_day_id is not null
           then source_draw.registration_day_id = v_run.registration_day_id
            and case
              when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
                then case
                  when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) > 0
                    then source_event.scheduled_day_ids ? source_draw.registration_day_id
                  else source_event.registration_day_id is null
                    or source_event.registration_day_id = source_draw.registration_day_id
                end
              else false
            end
         else case
           when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
             then case
               when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 1
                 then source_event.scheduled_day_ids ? v_run.registration_day_id
               when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 0
                 then source_event.registration_day_id = v_run.registration_day_id
               else false
             end
           else false
         end
       end) is true
       and not exists (
         select 1
           from public.tournament_day_live_draws as day_draw
          where day_draw.run_id = v_run.id
            and day_draw.draw_id = source_draw.id
            and day_draw.state <> 'REMOVED'
       )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_DRAWS: every currently supported scheduled draw must be activated and completed before close.';
  end if;

  for v_close_draw in
    select
      day_draw.draw_id,
      draw.tournament_id,
      draw.registration_day_id,
      draw.event_option_id,
      draw.name,
      draw.status,
      draw.draw_kind,
      draw.hidden_from_primary_ops
      from public.tournament_day_live_draws as day_draw
      join public.tournament_event_draws as draw on draw.id = day_draw.draw_id
     where day_draw.run_id = v_run.id and day_draw.state <> 'REMOVED'
     order by day_draw.draw_id
  loop
    if not exists (
      select 1
        from public.tournament_games as game
       where game.tournament_id = v_run.tournament_id
         and game.draw_id = v_close_draw.draw_id
         and game.stage = 'PLAYOFF'
    ) or exists (
      select 1
        from public.tournament_games as game
        left join public.tournament_teams as team_a
          on team_a.id = game.team_a_id
         and team_a.tournament_id = game.tournament_id
         and team_a.draw_id = game.draw_id
         and team_a.registration_day_id = v_run.registration_day_id
         and team_a.event_option_id = v_close_draw.event_option_id
        left join public.tournament_teams as team_b
          on team_b.id = game.team_b_id
         and team_b.tournament_id = game.tournament_id
         and team_b.draw_id = game.draw_id
         and team_b.registration_day_id = v_run.registration_day_id
         and team_b.event_option_id = v_close_draw.event_option_id
       where game.tournament_id = v_run.tournament_id
         and game.draw_id = v_close_draw.draw_id
         and (
           game.registration_day_id is distinct from v_run.registration_day_id
           or game.finalized_at is null
           or game.score_a is null or game.score_b is null
           or game.score_a = game.score_b
           or game.team_a_id is null or game.team_b_id is null
           or game.team_a_id = game.team_b_id
           or team_a.id is null or team_b.id is null
           or game.winner_team_id is null or game.loser_team_id is null
           or game.winner_team_id = game.loser_team_id
           or game.winner_team_id not in (game.team_a_id, game.team_b_id)
           or game.loser_team_id not in (game.team_a_id, game.team_b_id)
           or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
           or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id)
         )
    ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_PROGRESSION: every activated draw must finish its generated playoffs before close.';
    end if;

    if (
      select pg_catalog.count(*)
        from public.tournament_games as game
       where game.tournament_id = v_run.tournament_id
         and game.draw_id = v_close_draw.draw_id
         and game.stage = 'PLAYOFF'
         and pg_catalog.upper(coalesce(game.playoff_round, '')) = 'FINAL'
    ) <> 1 or (
      select pg_catalog.count(*)
        from public.tournament_games as game
       where game.tournament_id = v_run.tournament_id
         and game.draw_id = v_close_draw.draw_id
         and game.stage = 'PLAYOFF'
         and pg_catalog.upper(coalesce(game.playoff_round, '')) = 'BRONZE'
    ) <> 1 or not exists (
      select 1
        from public.tournament_games as final_game
        join public.tournament_games as bronze_game
          on bronze_game.tournament_id = final_game.tournament_id
         and bronze_game.draw_id = final_game.draw_id
         and bronze_game.stage = 'PLAYOFF'
         and pg_catalog.upper(coalesce(bronze_game.playoff_round, '')) = 'BRONZE'
        join public.tournament_podium as first_place
          on first_place.tournament_id = final_game.tournament_id
         and first_place.draw_id = final_game.draw_id
         and first_place.placement = 1
         and first_place.team_id = final_game.winner_team_id
        join public.tournament_podium as second_place
          on second_place.tournament_id = final_game.tournament_id
         and second_place.draw_id = final_game.draw_id
         and second_place.placement = 2
         and second_place.team_id = final_game.loser_team_id
        join public.tournament_podium as third_place
          on third_place.tournament_id = bronze_game.tournament_id
         and third_place.draw_id = bronze_game.draw_id
         and third_place.placement = 3
         and third_place.team_id = bronze_game.winner_team_id
       where final_game.tournament_id = v_run.tournament_id
         and final_game.draw_id = v_close_draw.draw_id
         and final_game.stage = 'PLAYOFF'
         and pg_catalog.upper(coalesce(final_game.playoff_round, '')) = 'FINAL'
         and final_game.registration_day_id = v_run.registration_day_id
         and bronze_game.registration_day_id = v_run.registration_day_id
         and final_game.event_option_id = v_close_draw.event_option_id
         and bronze_game.event_option_id = v_close_draw.event_option_id
    ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_PODIUM_RESULT: podium placements must exactly match the finalized Final and Bronze results.';
    end if;

    if (
      select pg_catalog.count(*) <> 3
          or pg_catalog.count(distinct podium.placement) <> 3
          or pg_catalog.count(distinct podium.team_id) <> 3
          or pg_catalog.min(podium.placement) <> 1
          or pg_catalog.max(podium.placement) <> 3
        from public.tournament_podium as podium
        left join public.tournament_teams as team
          on team.id = podium.team_id
         and team.tournament_id = podium.tournament_id
         and team.draw_id = podium.draw_id
       where podium.tournament_id = v_run.tournament_id
         and podium.draw_id = v_close_draw.draw_id
         and team.id is not null
    ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_PODIUM: exact first, second, and third place evidence is required before close.';
    end if;

    select exists (
      select 1
        from public.admin_activity_log as activity
        cross join lateral (
          select activity.after_json #> '{podium_review_evidence}' as evidence
        ) as reviewed
       where activity.club_id = p_club_id
         and activity.entity_type = 'tournament_event_draw'
         and activity.entity_id = v_close_draw.draw_id::text
         and activity.action_type = 'review_tournament_draw_podium_admin'
         and reviewed.evidence->>'contract' = 'jupr:tournament-podium-review:v1'
         and reviewed.evidence->>'tournament_id' = p_tournament_id
         and reviewed.evidence->>'draw_id' = v_close_draw.draw_id::text
         and pg_catalog.length(coalesce(reviewed.evidence->>'review_fingerprint', '')) = 64
         and reviewed.evidence #>> '{draw,id}' = v_close_draw.draw_id::text
         and reviewed.evidence #>> '{draw,tournament_id}' = v_close_draw.tournament_id::text
         and nullif(reviewed.evidence #>> '{draw,registration_day_id}', '')
               is not distinct from v_close_draw.registration_day_id
         and nullif(reviewed.evidence #>> '{draw,event_option_id}', '')
               is not distinct from v_close_draw.event_option_id::text
         and nullif(reviewed.evidence #>> '{draw,name}', '')
               is not distinct from nullif(v_close_draw.name, '')
         and nullif(reviewed.evidence #>> '{draw,status}', '')
               is not distinct from nullif(v_close_draw.status, '')
         and nullif(reviewed.evidence #>> '{draw,draw_kind}', '')
               is not distinct from nullif(v_close_draw.draw_kind, '')
         and coalesce(
               (reviewed.evidence #>> '{draw,hidden_from_primary_ops}')::boolean,
               false
             ) is not distinct from coalesce(v_close_draw.hidden_from_primary_ops, false)
         and pg_catalog.jsonb_typeof(reviewed.evidence->'team_versions') = 'array'
         and pg_catalog.jsonb_array_length(reviewed.evidence->'team_versions') = (
           select pg_catalog.count(*)
             from public.tournament_teams as team
            where team.tournament_id = v_run.tournament_id
              and team.draw_id = v_close_draw.draw_id
         )
         and not exists (
           select 1
             from public.tournament_teams as team
            where team.tournament_id = v_run.tournament_id
              and team.draw_id = v_close_draw.draw_id
              and not exists (
                select 1
                  from pg_catalog.jsonb_to_recordset(reviewed.evidence->'team_versions')
                    as version(id text, updated_at timestamptz)
                 where version.id = team.id::text
                   and version.updated_at is not distinct from team.updated_at
              )
         )
         and pg_catalog.jsonb_typeof(reviewed.evidence->'game_versions') = 'array'
         and pg_catalog.jsonb_array_length(reviewed.evidence->'game_versions') = (
           select pg_catalog.count(*)
             from public.tournament_games as game
            where game.tournament_id = v_run.tournament_id
              and game.draw_id = v_close_draw.draw_id
         )
         and not exists (
           select 1
             from public.tournament_games as game
            where game.tournament_id = v_run.tournament_id
              and game.draw_id = v_close_draw.draw_id
              and not exists (
                select 1
                  from pg_catalog.jsonb_to_recordset(reviewed.evidence->'game_versions')
                    as version(id text, updated_at timestamptz)
                 where version.id = game.id::text
                   and version.updated_at is not distinct from game.updated_at
              )
         )
         and pg_catalog.jsonb_typeof(reviewed.evidence->'podium') = 'array'
         and pg_catalog.jsonb_array_length(reviewed.evidence->'podium') = 3
         and not exists (
           select 1
             from public.tournament_podium as podium
            where podium.tournament_id = v_run.tournament_id
              and podium.draw_id = v_close_draw.draw_id
              and not exists (
                select 1
                  from pg_catalog.jsonb_to_recordset(reviewed.evidence->'podium')
                    as evidence_row(id text, draw_id text, placement integer, team_id text, source text)
                 where evidence_row.id = podium.id::text
                   and evidence_row.draw_id = podium.draw_id::text
                   and evidence_row.placement = podium.placement
                   and evidence_row.team_id = podium.team_id::text
                   and pg_catalog.upper(evidence_row.source) = pg_catalog.upper(podium.source)
              )
         )
    ) into v_reviewed;
    if not v_reviewed then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_REVIEW: a current explicit podium review is required before close.';
    end if;

    if exists (
      with expected_awards as (
        select distinct
          participant.player_id,
          case podium.placement
            when 1 then 'tournament_champion'
            when 2 then 'tournament_runner_up'
            when 3 then 'tournament_third_place'
          end as badge_id,
          pg_catalog.concat_ws(
            ':', p_tournament_id, 'draw', v_close_draw.draw_id::text,
            'podium', podium.placement::text
          ) as context_id
          from public.tournament_podium as podium
          join public.tournament_teams as team
            on team.id = podium.team_id
           and team.tournament_id = podium.tournament_id
           and team.draw_id = podium.draw_id
          cross join lateral (values (team.player1_id), (team.player2_id))
            as participant(player_id)
         where podium.tournament_id = v_run.tournament_id
           and podium.draw_id = v_close_draw.draw_id
           and participant.player_id is not null
      ), actual_awards as (
        select badge.player_id, badge.badge_id, badge.context_id::text as context_id
          from public.player_badges as badge
         where badge.club_id = p_club_id
           and badge.context_type = 'tournament'
           and badge.revoked_at is null
           and badge.context_id::text like
             p_tournament_id || ':draw:' || v_close_draw.draw_id::text || ':podium:%'
      )
      select 1
       where exists (select * from expected_awards except select * from actual_awards)
          or exists (select * from actual_awards except select * from expected_awards)
    ) then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CLOSE_AWARDS: every expected podium medal must exist exactly before close.';
    end if;
  end loop;

  update public.tournament_day_live_draws as day_draw
     set state = case when day_draw.state = 'REMOVED' then 'REMOVED' else 'PAUSED' end,
         version = day_draw.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where day_draw.run_id = v_run.id;
  update public.tournament_day_live_courts as court
     set state = 'CLOSED',
         version = court.version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp()
   where court.run_id = v_run.id;
  update public.tournament_day_live_runs as run
     set state = 'CLOSED',
         version = run.version + 1,
         queue_version = run.queue_version + 1,
         last_operation_key = p_operation_key,
         updated_by = p_actor,
         updated_at = pg_catalog.clock_timestamp(),
         closed_at = pg_catalog.clock_timestamp()
   where run.id = v_run.id
   returning * into v_run;
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'run', pg_catalog.to_jsonb(v_run),
    'closed', true
  );
end;
$function$;

create or replace function public.tournament_day_live_game_player_ids(
  p_tournament_id text,
  p_draw_id uuid,
  p_game_id uuid
)
returns integer[]
language sql
stable
security invoker
set search_path = ''
as $function$
  select coalesce(
    pg_catalog.array_agg(distinct participant.player_id order by participant.player_id)
      filter (where participant.player_id is not null),
    '{}'::integer[]
  )
  from public.tournament_games as game
  left join public.tournament_teams as team_a
    on team_a.id = game.team_a_id
   and team_a.tournament_id = game.tournament_id
   and team_a.draw_id = game.draw_id
   and team_a.registration_day_id = game.registration_day_id
   and team_a.event_option_id = game.event_option_id
  left join public.tournament_teams as team_b
    on team_b.id = game.team_b_id
   and team_b.tournament_id = game.tournament_id
   and team_b.draw_id = game.draw_id
   and team_b.registration_day_id = game.registration_day_id
   and team_b.event_option_id = game.event_option_id
  cross join lateral (
    values (team_a.player1_id), (team_a.player2_id),
           (team_b.player1_id), (team_b.player2_id)
  ) as participant(player_id)
  where game.id = p_game_id
    and game.tournament_id::text = p_tournament_id
    and game.draw_id = p_draw_id;
$function$;

create or replace function public.assert_tournament_day_live_draw_ready(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_draw_id uuid,
  p_expected_draw_updated_at timestamptz
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_day public.tournament_registration_days%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_event_option_id text;
  v_player_id integer;
  v_registration_id text;
  v_registration_count integer;
  v_payment_status text;
begin
  -- Read only the setup identity first.  The final draw row is deliberately
  -- locked after its team/game children because their version triggers touch
  -- the draw and use that same child -> parent order.
  select draw.event_option_id
    into v_event_option_id
    from public.tournament_event_draws as draw
    join public.tournaments as tournament
      on tournament.id = draw.tournament_id
     and tournament.club_id::text = p_club_id
   where draw.id = p_draw_id
     and draw.tournament_id::text = p_tournament_id;
  if v_event_option_id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed draw identity is no longer available.';
  end if;

  select day.* into v_day
    from public.tournament_registration_days as day
   where day.id = p_registration_day_id
     and day.tournament_id::text = p_tournament_id
     and day.enabled is true
   for share;
  if v_day.id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: selected tournament day changed or is disabled.';
  end if;

  select event.* into v_event
    from public.tournament_event_options as event
   where event.id = v_event_option_id
     and event.tournament_id::text = p_tournament_id
   for share;
  if v_event.id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: reviewed event identity changed.';
  end if;

  perform team.id
    from public.tournament_teams as team
   where team.tournament_id::text = p_tournament_id
     and team.draw_id = p_draw_id
   order by team.id
   for share;
  perform game.id
    from public.tournament_games as game
   where game.tournament_id::text = p_tournament_id
     and game.draw_id = p_draw_id
   order by game.id
   for share;

  select draw.*
    into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament
      on tournament.id = draw.tournament_id
     and tournament.club_id::text = p_club_id
   where draw.id = p_draw_id
     and draw.tournament_id::text = p_tournament_id
   for share;

  if v_draw.id is null
     or v_draw.updated_at is distinct from p_expected_draw_updated_at
     or v_draw.event_option_id is distinct from v_event.id
     or pg_catalog.upper(coalesce(v_draw.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or coalesce(v_draw.hidden_from_primary_ops, false) is true
     or pg_catalog.upper(coalesce(v_draw.draw_kind, 'STANDARD')) <> 'STANDARD' then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: draw changed, is inactive, or is not supported by the day runner.';
  end if;

  if v_event.enabled is not true
     or pg_catalog.upper(coalesce(v_event.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or (case
       when v_draw.registration_day_id is not null
         then v_draw.registration_day_id = p_registration_day_id
          and case
            when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
              then case
                when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) > 0
                  then v_event.scheduled_day_ids ? v_draw.registration_day_id
                else v_event.registration_day_id is null
                  or v_event.registration_day_id = v_draw.registration_day_id
              end
            else false
          end
       else case
         when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
           then case
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 1
               then v_event.scheduled_day_ids ? p_registration_day_id
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
               then v_event.registration_day_id = p_registration_day_id
             else false
           end
         else false
       end
     end) is not true then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_UNSCHEDULED: draw is not scheduled on this enabled tournament day.';
  end if;

  if not exists (
    select 1 from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAMES_REQUIRED: generate games before activating this draw.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         game.registration_day_id is distinct from p_registration_day_id
         or game.event_option_id is distinct from v_draw.event_option_id
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_DAY: every draw game must belong to the selected tournament day and event.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) not in ('ROUND_ROBIN', 'PLAYOFF')
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_STAGE: every draw game must use a supported ROUND_ROBIN or PLAYOFF stage.';
  end if;

  if (
    select pg_catalog.count(*)
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and (
         team.registration_day_id is distinct from p_registration_day_id
         or team.event_option_id is distinct from v_draw.event_option_id
       )
  ) > 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_TEAM_DAY_SCOPE: every draw team must belong to the selected tournament day and draw event.';
  end if;

  if exists (
    select participant.player_id
      from public.tournament_teams as team
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
       and participant.player_id is not null
     group by participant.player_id
    having pg_catalog.count(*) > 1
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROSTER_PLAYER_DUPLICATE: each player may belong to only one exact team in this draw.';
  end if;

  if (
    select coalesce(
             pg_catalog.array_agg(team.id order by team.id),
             '{}'::uuid[]
           )
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) is distinct from (
    select coalesce(
             pg_catalog.array_agg(distinct side.team_id order by side.team_id)
               filter (where side.team_id is not null),
             '{}'::uuid[]
           )
      from public.tournament_games as game
      cross join lateral (values (game.team_a_id), (game.team_b_id)) as side(team_id)
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROUND_ROBIN_ROSTER: every exact draw team must appear in the reviewed round-robin schedule.';
  end if;

  if (
    select pg_catalog.count(*)
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) < 4 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_FORMAT: activate at least four exact in-draw teams so a supported playoff format can complete day closeout.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
      cross join lateral (
        select
          pg_catalog.count(participant.player_id) as player_count,
          pg_catalog.count(distinct participant.player_id) as distinct_player_count
        from (values
          (team_a.player1_id), (team_a.player2_id),
          (team_b.player1_id), (team_b.player2_id)
        ) as participant(player_id)
        where participant.player_id is not null
      ) as participant_counts
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
       and (
         game.team_a_id is null
         or game.team_b_id is null
         or game.team_a_id = game.team_b_id
         or team_a.id is null
         or team_b.id is null
         or team_a.player1_id is null
         or team_b.player1_id is null
         or participant_counts.player_count not in (2, 4)
         or participant_counts.player_count <> participant_counts.distinct_player_count
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROUND_ROBIN: every round-robin game requires two distinct in-draw teams with two or four distinct effective players.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'PLAYOFF'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFFS_ALREADY_GENERATED: activate reviewed round-robin games first, then generate playoffs through the guarded day operation.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         (game.finalized_at is null and (
           game.score_a is not null or game.score_b is not null
           or game.winner_team_id is not null or game.loser_team_id is not null
         ))
         or
         (game.finalized_at is not null and (
           game.score_a is null or game.score_b is null
           or game.score_a = game.score_b
           or game.winner_team_id is null or game.loser_team_id is null
           or game.team_a_id is null or game.team_b_id is null
           or game.team_a_id = game.team_b_id
           or game.winner_team_id = game.loser_team_id
           or game.winner_team_id not in (game.team_a_id, game.team_b_id)
           or game.loser_team_id not in (game.team_a_id, game.team_b_id)
           or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
           or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id)
         ))
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_STATE: partially scored or tied games require reconciliation before activation.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      cross join lateral (
        select
          pg_catalog.count(participant.player_id) as player_count,
          pg_catalog.count(distinct participant.player_id) as distinct_player_count
        from public.tournament_teams as team
        cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
        where team.id in (game.team_a_id, game.team_b_id)
          and team.tournament_id = game.tournament_id
          and team.draw_id = game.draw_id
          and team.registration_day_id = p_registration_day_id
          and team.event_option_id = v_draw.event_option_id
          and participant.player_id is not null
      ) as participant_counts
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and game.team_a_id is not null
       and game.team_b_id is not null
       and (
         participant_counts.player_count not in (2, 4)
         or participant_counts.player_count <> participant_counts.distinct_player_count
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PARTICIPANTS: every playable game needs two or four distinct effective players.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         (game.team_a_id is not null and (
           team_a.id is null or team_a.player1_id is null
         ))
         or (game.team_b_id is not null and (
           team_b.id is null or team_b.player1_id is null
         ))
         or (
           game.team_a_id is not null and game.team_b_id is not null
           and team_a.id = team_b.id
         )
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PARTICIPANTS: a playable game has incomplete or foreign team evidence.';
  end if;

  for v_player_id in
    select distinct participant.player_id
      from public.tournament_games as game
      join public.tournament_teams as team
        on team.id in (game.team_a_id, game.team_b_id)
       and team.tournament_id = game.tournament_id
       and team.draw_id = game.draw_id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and participant.player_id is not null
     order by participant.player_id
  loop
    perform player.id
      from public.players as player
     where player.id = v_player_id
       and player.club_id::text = p_club_id
     for share;
    if not found then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYER_SCOPE: every participant must belong to the tournament club.';
    end if;
    perform registration.id
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED')
     order by registration.id
     for share;
    select pg_catalog.count(*), pg_catalog.min(registration.id)
      into v_registration_count, v_registration_id
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED');

    if v_registration_count <> 1 then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CHECK_IN: every player must resolve to exactly one active tournament registration.';
    end if;

    perform commerce.id
      from public.tournament_commerce_orders as commerce
     where commerce.club_id::text = p_club_id
       and commerce.tournament_id::text = p_tournament_id
       and commerce.registration_id = v_registration_id
     order by commerce.updated_at desc, commerce.id desc
     limit 1
     for share;

    select coalesce(
      (
        select pg_catalog.upper(coalesce(commerce.payment_status, 'UNPAID'))
          from public.tournament_commerce_orders as commerce
         where commerce.club_id::text = p_club_id
           and commerce.tournament_id::text = p_tournament_id
           and commerce.registration_id = v_registration_id
         order by commerce.updated_at desc, commerce.id desc
         limit 1
      ),
      (
        select pg_catalog.upper(coalesce(registration.payment_status, 'UNPAID'))
          from public.tournament_registrations as registration
         where registration.id = v_registration_id
      ),
      'UNPAID'
    ) into v_payment_status;

    if v_payment_status not in ('PAID', 'WAIVED') then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_PAYMENT: every attending player must have resolved paid or waived status.';
    end if;

    perform check_in.id
      from public.tournament_registration_check_ins as check_in
     where check_in.tournament_id::text = p_tournament_id
       and check_in.registration_day_id = p_registration_day_id
       and check_in.registration_id = v_registration_id
       and check_in.attendance_status = 'CHECKED_IN'
       and check_in.waiver_verified is true
       and check_in.attendee_identity_key = pg_catalog.concat_ws(':', 'player', v_player_id::text)
     for share;
    if not found then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_CHECK_IN: every attending player must be checked in with a current waiver and canonical identity.';
    end if;
  end loop;
end;
$function$;

create or replace function public.tournament_day_live_players_ready(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_player_ids integer[]
)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_player_id integer;
  v_registration_id text;
  v_registration_count integer;
  v_payment_status text;
begin
  if pg_catalog.cardinality(coalesce(p_player_ids, '{}'::integer[])) not in (2, 4)
     or pg_catalog.cardinality(p_player_ids) <>
       pg_catalog.cardinality(array(
         select distinct participant.player_id
           from pg_catalog.unnest(p_player_ids) as participant(player_id)
       )) then
    return false;
  end if;

  foreach v_player_id in array p_player_ids
  loop
    perform player.id
      from public.players as player
     where player.id = v_player_id
       and player.club_id::text = p_club_id
     for share;
    if not found then
      return false;
    end if;
    perform registration.id
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED')
     order by registration.id
     for share;
    select pg_catalog.count(*), pg_catalog.min(registration.id)
      into v_registration_count, v_registration_id
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED');
    if v_registration_count <> 1 then
      return false;
    end if;

    perform check_in.id
      from public.tournament_registration_check_ins as check_in
     where check_in.tournament_id::text = p_tournament_id
       and check_in.registration_day_id = p_registration_day_id
       and check_in.registration_id = v_registration_id
       and check_in.attendance_status = 'CHECKED_IN'
       and check_in.waiver_verified is true
       and check_in.attendee_identity_key = pg_catalog.concat_ws(':', 'player', v_player_id::text)
     for share;
    if not found then
      return false;
    end if;

    perform commerce.id
      from public.tournament_commerce_orders as commerce
     where commerce.club_id::text = p_club_id
       and commerce.tournament_id::text = p_tournament_id
       and commerce.registration_id = v_registration_id
     order by commerce.updated_at desc, commerce.id desc
     limit 1
     for share;

    select coalesce(
      (
        select pg_catalog.upper(coalesce(commerce.payment_status, 'UNPAID'))
          from public.tournament_commerce_orders as commerce
         where commerce.club_id::text = p_club_id
           and commerce.tournament_id::text = p_tournament_id
           and commerce.registration_id = v_registration_id
         order by commerce.updated_at desc, commerce.id desc
         limit 1
      ),
      (
        select pg_catalog.upper(coalesce(registration.payment_status, 'UNPAID'))
          from public.tournament_registrations as registration
         where registration.id = v_registration_id
      ),
      'UNPAID'
    ) into v_payment_status;
    if v_payment_status not in ('PAID', 'WAIVED') then
      return false;
    end if;
  end loop;
  return true;
end;
$function$;

create or replace function public.seed_tournament_day_live_draw(
  p_run_id uuid,
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_draw_id uuid,
  p_expected_draw_updated_at timestamptz,
  p_priority integer,
  p_operation_key text,
  p_actor text
)
returns uuid
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_day_draw_id uuid;
  v_priority_base bigint;
  v_source_game_count integer;
  v_inserted_game_count integer;
begin
  perform public.assert_tournament_day_live_draw_ready(
    p_club_id,
    p_tournament_id,
    p_registration_day_id,
    p_draw_id,
    p_expected_draw_updated_at
  );

  insert into public.tournament_day_live_draws (
    run_id, tournament_id, registration_day_id, draw_id, state, priority,
    source_draw_updated_at, version, last_operation_key,
    activated_by, updated_by
  ) values (
    p_run_id, p_tournament_id::uuid, p_registration_day_id, p_draw_id,
    'ACTIVE', p_priority, p_expected_draw_updated_at, 1, p_operation_key,
    p_actor, p_actor
  )
  returning id into v_day_draw_id;

  select coalesce(pg_catalog.max(queue.priority), 0)
    into v_priority_base
    from public.tournament_day_live_queue as queue
   where queue.run_id = p_run_id;

  select pg_catalog.count(*)
    into v_source_game_count
    from public.tournament_games as game
   where game.tournament_id::text = p_tournament_id
     and game.draw_id = p_draw_id;

  insert into public.tournament_day_live_queue (
    run_id, tournament_id, registration_day_id, day_draw_id, draw_id, game_id,
    team_a_id, team_b_id, state, priority, court_id,
    blocker_code, blocker_detail, version, last_operation_key,
    eligible_since, released_at, completed_at, updated_by
  )
  select
    p_run_id,
    game.tournament_id,
    p_registration_day_id,
    v_day_draw_id,
    game.draw_id,
    game.id,
    game.team_a_id,
    game.team_b_id,
    case
      when game.finalized_at is not null then 'COMPLETED'
      when game.team_a_id is not null and game.team_b_id is not null then 'WAITING'
      else 'BLOCKED'
    end,
    v_priority_base + pg_catalog.row_number() over (
      order by
        case when game.stage = 'ROUND_ROBIN' then 0 else 1 end,
        game.rr_round_number nulls last,
        game.rr_slot_number nulls last,
        game.playoff_round nulls last,
        game.playoff_game_code nulls last,
        game.id
    ),
    null,
    case
      when game.finalized_at is not null then null
      when game.team_a_id is null or game.team_b_id is null then 'DEPENDENCY_PENDING'
      else null
    end,
    case
      when game.team_a_id is null or game.team_b_id is null
        then 'Playoff source teams are not resolved yet.'
      else null
    end,
    1,
    p_operation_key,
    case
      when game.finalized_at is null and game.team_a_id is not null and game.team_b_id is not null
        then pg_catalog.clock_timestamp()
      else null
    end,
    case when game.finalized_at is not null then game.finalized_at else null end,
    game.finalized_at,
    p_actor
  from public.tournament_games as game
  where game.tournament_id::text = p_tournament_id
    and game.draw_id = p_draw_id
    and game.registration_day_id = p_registration_day_id
    and game.event_option_id is not distinct from (
      select draw.event_option_id
        from public.tournament_event_draws as draw
       where draw.id = p_draw_id
    )
  order by
    case when game.stage = 'ROUND_ROBIN' then 0 else 1 end,
    game.rr_round_number nulls last,
    game.rr_slot_number nulls last,
    game.playoff_round nulls last,
    game.playoff_game_code nulls last,
    game.id
  on conflict (run_id, game_id) do nothing;
  get diagnostics v_inserted_game_count = row_count;
  if v_inserted_game_count <> v_source_game_count then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_DAY: exact draw game materialization failed.';
  end if;

  return v_day_draw_id;
end;
$function$;

create or replace function public.fill_tournament_day_live_courts(
  p_run_id uuid,
  p_operation_key text,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_run public.tournament_day_live_runs%rowtype;
  v_court public.tournament_day_live_courts%rowtype;
  v_queue public.tournament_day_live_queue%rowtype;
  v_day_draw public.tournament_day_live_draws%rowtype;
  v_day public.tournament_registration_days%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_game public.tournament_games%rowtype;
  v_player_ids integer[];
  v_locked_team_count integer;
  v_assigned jsonb := '[]'::jsonb;
begin
  select run.* into v_run
    from public.tournament_day_live_runs as run
   where run.id = p_run_id
   for update;
  if v_run.id is null or v_run.state <> 'ACTIVE' then
    return v_assigned;
  end if;

  for v_court in
    select court.*
      from public.tournament_day_live_courts as court
     where court.run_id = p_run_id
       and court.state = 'OPEN'
       and not exists (
         select 1
           from public.tournament_day_live_queue as occupied
          where occupied.run_id = p_run_id
            and occupied.court_id = court.id
            and occupied.released_at is null
       )
     order by court.position, court.id
     for update
  loop
    select queue.* into v_queue
      from public.tournament_day_live_queue as queue
      join public.tournament_day_live_draws as day_draw
        on day_draw.id = queue.day_draw_id
       and day_draw.run_id = queue.run_id
       and day_draw.state = 'ACTIVE'
      join public.tournament_games as game
        on game.id = queue.game_id
       and game.tournament_id = queue.tournament_id
       and game.draw_id = queue.draw_id
      join public.tournament_registration_days as source_day
        on source_day.id = queue.registration_day_id
       and source_day.tournament_id = queue.tournament_id::text
       and source_day.enabled is true
      join public.tournament_event_draws as source_draw
        on source_draw.id = queue.draw_id
       and source_draw.tournament_id = queue.tournament_id
       and source_draw.updated_at = day_draw.source_draw_updated_at
       and pg_catalog.upper(coalesce(source_draw.status, 'DRAFT')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
       and coalesce(source_draw.hidden_from_primary_ops, false) is false
       and pg_catalog.upper(coalesce(source_draw.draw_kind, 'STANDARD')) = 'STANDARD'
      join public.tournament_event_options as source_event
        on source_event.id = source_draw.event_option_id
       and source_event.id = game.event_option_id
       and source_event.tournament_id = queue.tournament_id::text
       and source_event.enabled is true
       and pg_catalog.upper(coalesce(source_event.status, 'DRAFT')) not in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     where queue.run_id = p_run_id
       and queue.tournament_id = v_run.tournament_id
       and queue.registration_day_id = v_run.registration_day_id
       and queue.state = 'WAITING'
       and queue.court_id is null
       and queue.released_at is null
       and game.finalized_at is null
       and game.registration_day_id = queue.registration_day_id
       and game.team_a_id = queue.team_a_id
       and game.team_b_id = queue.team_b_id
       and (case
         when source_draw.registration_day_id is not null
           then source_draw.registration_day_id = queue.registration_day_id
            and case
              when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
                then case
                  when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) > 0
                    then source_event.scheduled_day_ids ? source_draw.registration_day_id
                  else source_event.registration_day_id is null
                    or source_event.registration_day_id = source_draw.registration_day_id
                end
              else false
            end
         else case
           when pg_catalog.jsonb_typeof(source_event.scheduled_day_ids) = 'array'
             then case
               when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 1
                 then source_event.scheduled_day_ids ? queue.registration_day_id
               when pg_catalog.jsonb_array_length(source_event.scheduled_day_ids) = 0
                 then source_event.registration_day_id = queue.registration_day_id
               else false
             end
           else false
         end
       end) is true
       and (
         game.stage <> 'PLAYOFF'
         or not exists (
           select 1
             from public.tournament_day_live_queue as rr_queue
             join public.tournament_games as rr_game
               on rr_game.id = rr_queue.game_id
              and rr_game.tournament_id = rr_queue.tournament_id
              and rr_game.draw_id = rr_queue.draw_id
            where rr_queue.run_id = queue.run_id
              and rr_queue.draw_id = queue.draw_id
              and rr_game.stage = 'ROUND_ROBIN'
              and rr_queue.state not in ('COMPLETED', 'WITHDRAWN')
         )
       )
       and public.tournament_day_live_players_ready(
         v_run.club_id,
         v_run.tournament_id::text,
         v_run.registration_day_id,
         public.tournament_day_live_game_player_ids(
           v_run.tournament_id::text,
           queue.draw_id,
           queue.game_id
         )
       )
       and not exists (
         select 1
           from public.tournament_day_live_participant_claims as claim
          where claim.run_id = p_run_id
            and claim.released_at is null
            and claim.player_id = any(
              public.tournament_day_live_game_player_ids(
                v_run.tournament_id::text,
                queue.draw_id,
                queue.game_id
              )
            )
       )
       and not exists (
         select 1
           from public.tournament_day_live_queue as earlier
          where earlier.run_id = queue.run_id
            and earlier.draw_id = queue.draw_id
            and earlier.priority < queue.priority
            and earlier.state not in ('COMPLETED', 'WITHDRAWN')
            and (
              earlier.team_a_id in (queue.team_a_id, queue.team_b_id)
              or earlier.team_b_id in (queue.team_a_id, queue.team_b_id)
            )
       )
     order by
       (
         select pg_catalog.count(*)
           from public.tournament_day_live_queue as current_assignment
          where current_assignment.run_id = p_run_id
            and current_assignment.day_draw_id = queue.day_draw_id
            and current_assignment.state in ('HELD', 'CALLED', 'ON_COURT')
            and current_assignment.released_at is null
       ),
       day_draw.last_assigned_at nulls first,
       day_draw.activated_at,
       day_draw.draw_id,
       case when game.stage = 'ROUND_ROBIN' then 0 else 1 end,
       game.rr_round_number nulls last,
       game.rr_slot_number nulls last,
       game.playoff_round nulls last,
       game.playoff_game_code nulls last,
       game.id
     for update of queue skip locked
     limit 1;

    if v_queue.id is null then
      continue;
    end if;

    perform locked_team.id
      from public.tournament_teams as locked_team
     where locked_team.tournament_id = v_queue.tournament_id
       and locked_team.draw_id = v_queue.draw_id
       and locked_team.registration_day_id = v_queue.registration_day_id
       and locked_team.event_option_id = (
         select source_game.event_option_id
           from public.tournament_games as source_game
          where source_game.id = v_queue.game_id
            and source_game.tournament_id = v_queue.tournament_id
            and source_game.draw_id = v_queue.draw_id
       )
       and locked_team.id in (v_queue.team_a_id, v_queue.team_b_id)
     order by locked_team.id
     for share;
    get diagnostics v_locked_team_count = row_count;
    select locked_game.* into v_game
      from public.tournament_games as locked_game
     where locked_game.id = v_queue.game_id
       and locked_game.tournament_id = v_queue.tournament_id
       and locked_game.draw_id = v_queue.draw_id
       and locked_game.registration_day_id = v_queue.registration_day_id
       and locked_game.team_a_id = v_queue.team_a_id
       and locked_game.team_b_id = v_queue.team_b_id
       and locked_game.finalized_at is null
     for share;
    if v_locked_team_count <> 2 or v_game.id is null then
      continue;
    end if;

    select locked_day_draw.* into v_day_draw
      from public.tournament_day_live_draws as locked_day_draw
     where locked_day_draw.id = v_queue.day_draw_id
       and locked_day_draw.run_id = v_run.id
       and locked_day_draw.state = 'ACTIVE'
     for share;
    select source_day.* into v_day
      from public.tournament_registration_days as source_day
     where source_day.id = v_run.registration_day_id
       and source_day.tournament_id = v_run.tournament_id::text
       and source_day.enabled is true
     for share;
    select source_event.* into v_event
      from public.tournament_event_options as source_event
     where source_event.id = v_game.event_option_id
       and source_event.tournament_id = v_run.tournament_id::text
     for share;
    select source_draw.* into v_draw
      from public.tournament_event_draws as source_draw
     where source_draw.id = v_queue.draw_id
       and source_draw.tournament_id = v_run.tournament_id
     for share;

    if v_day_draw.id is null
       or v_day.id is null
       or v_event.id is null
       or v_draw.id is null
       or v_draw.updated_at is distinct from v_day_draw.source_draw_updated_at
       or v_draw.event_option_id is distinct from v_event.id
       or v_game.event_option_id is distinct from v_event.id
       or pg_catalog.upper(coalesce(v_draw.status, 'DRAFT')) in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
       or coalesce(v_draw.hidden_from_primary_ops, false) is true
       or pg_catalog.upper(coalesce(v_draw.draw_kind, 'STANDARD')) <> 'STANDARD'
       or v_event.enabled is not true
       or pg_catalog.upper(coalesce(v_event.status, 'DRAFT')) in
         ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
       or (case
         when v_draw.registration_day_id is not null
           then v_draw.registration_day_id = v_run.registration_day_id
            and case
              when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
                then case
                  when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) > 0
                    then v_event.scheduled_day_ids ? v_draw.registration_day_id
                  else v_event.registration_day_id is null
                    or v_event.registration_day_id = v_draw.registration_day_id
                end
              else false
            end
         else case
           when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
             then case
               when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 1
                 then v_event.scheduled_day_ids ? v_run.registration_day_id
               when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
                 then v_event.registration_day_id = v_run.registration_day_id
               else false
             end
           else false
         end
       end) is not true then
      -- Setup drift never prevents a current result from being saved and its
      -- court from being released; it only suppresses the next assignment.
      continue;
    end if;

    v_player_ids := public.tournament_day_live_game_player_ids(
      v_run.tournament_id::text,
      v_queue.draw_id,
      v_queue.game_id
    );

    update public.tournament_day_live_queue as queue
       set state = 'ON_COURT',
           court_id = v_court.id,
           blocker_code = null,
           blocker_detail = null,
           started_at = pg_catalog.clock_timestamp(),
           version = queue.version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = pg_catalog.clock_timestamp()
     where queue.id = v_queue.id
       and queue.state = 'WAITING';

    insert into public.tournament_day_live_participant_claims (
      run_id, tournament_id, registration_day_id, queue_id, game_id,
      player_id, state, version, last_operation_key
    )
    select
      v_run.id, v_run.tournament_id, v_run.registration_day_id,
      v_queue.id, v_queue.game_id, player_id,
      'ON_COURT', 1, p_operation_key
    from pg_catalog.unnest(v_player_ids) as participant(player_id)
    on conflict (run_id, player_id) where released_at is null do nothing;

    if (
      select pg_catalog.count(*)
        from public.tournament_day_live_participant_claims as claim
       where claim.queue_id = v_queue.id and claim.released_at is null
    ) <> pg_catalog.cardinality(v_player_ids) then
      update public.tournament_day_live_participant_claims
         set state = 'RELEASED',
             released_at = pg_catalog.clock_timestamp(),
             released_by = p_actor,
             version = version + 1,
             updated_at = pg_catalog.clock_timestamp()
       where queue_id = v_queue.id and released_at is null;
      update public.tournament_day_live_queue
         set state = 'WAITING', court_id = null, started_at = null,
             version = version + 1, updated_at = pg_catalog.clock_timestamp()
       where id = v_queue.id;
      continue;
    end if;

    update public.tournament_day_live_courts
       set version = version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = pg_catalog.clock_timestamp()
     where id = v_court.id;
    update public.tournament_day_live_draws
       set last_assigned_at = pg_catalog.clock_timestamp(),
           version = version + 1,
           last_operation_key = p_operation_key,
           updated_by = p_actor,
           updated_at = pg_catalog.clock_timestamp()
     where id = v_queue.day_draw_id;

    v_assigned := v_assigned || pg_catalog.jsonb_build_array(
      pg_catalog.jsonb_build_object(
        'queue_id', v_queue.id,
        'game_id', v_queue.game_id,
        'draw_id', v_queue.draw_id,
        'court_id', v_court.id,
        'court_key', v_court.court_key
      )
    );
  end loop;
  return v_assigned;
end;
$function$;

revoke execute on function public.assert_tournament_day_live_operation(text, text, text, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.tournament_day_live_game_player_ids(text, uuid, uuid)
  from public, anon, authenticated, service_role;
revoke execute on function public.assert_tournament_day_live_draw_ready(text, text, text, uuid, timestamptz)
  from public, anon, authenticated, service_role;
revoke execute on function public.tournament_day_live_players_ready(text, text, text, integer[])
  from public, anon, authenticated, service_role;
revoke execute on function public.seed_tournament_day_live_draw(uuid, text, text, text, uuid, timestamptz, integer, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.fill_tournament_day_live_courts(uuid, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.admin_activate_tournament_day_live_cas(text, text, text, bigint, bigint, text, jsonb, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.admin_transition_tournament_day_draw_cas(text, text, text, text, text, bigint, bigint, bigint, timestamptz, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.admin_fill_tournament_day_courts_cas(text, text, text, bigint, bigint, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.admin_close_tournament_day_live_cas(text, text, text, bigint, bigint, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.admin_score_release_tournament_day_game_cas(text, text, text, text, bigint, bigint, bigint, timestamptz, timestamptz, jsonb, jsonb, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.admin_correct_completed_tournament_day_game_cas(text, text, text, text, bigint, bigint, bigint, timestamptz, timestamptz, jsonb, jsonb, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.admin_generate_tournament_day_playoffs_cas(text, text, text, text, integer, bigint, bigint, bigint, timestamptz, jsonb, jsonb, jsonb, text, text, text)
  from public, anon, authenticated, service_role;
revoke execute on function public.guard_tournament_game_day_live_mutation()
  from public, anon, authenticated, service_role;
revoke execute on function public.guard_tournament_check_in_during_player_claim()
  from public, anon, authenticated, service_role;
revoke execute on function public.guard_tournament_team_during_player_claim()
  from public, anon, authenticated, service_role;

grant execute on function public.assert_tournament_day_live_operation(text, text, text, text, text, text) to service_role;
grant execute on function public.tournament_day_live_game_player_ids(text, uuid, uuid) to service_role;
grant execute on function public.assert_tournament_day_live_draw_ready(text, text, text, uuid, timestamptz) to service_role;
grant execute on function public.tournament_day_live_players_ready(text, text, text, integer[]) to service_role;
grant execute on function public.seed_tournament_day_live_draw(uuid, text, text, text, uuid, timestamptz, integer, text, text) to service_role;
grant execute on function public.fill_tournament_day_live_courts(uuid, text, text) to service_role;
grant execute on function public.admin_activate_tournament_day_live_cas(text, text, text, bigint, bigint, text, jsonb, text, text, text) to service_role;
grant execute on function public.admin_transition_tournament_day_draw_cas(text, text, text, text, text, bigint, bigint, bigint, timestamptz, text, text, text) to service_role;
grant execute on function public.admin_fill_tournament_day_courts_cas(text, text, text, bigint, bigint, text, text, text) to service_role;
grant execute on function public.admin_close_tournament_day_live_cas(text, text, text, bigint, bigint, text, text, text) to service_role;
grant execute on function public.admin_score_release_tournament_day_game_cas(text, text, text, text, bigint, bigint, bigint, timestamptz, timestamptz, jsonb, jsonb, text, text, text) to service_role;
grant execute on function public.admin_correct_completed_tournament_day_game_cas(text, text, text, text, bigint, bigint, bigint, timestamptz, timestamptz, jsonb, jsonb, text, text, text) to service_role;
grant execute on function public.admin_generate_tournament_day_playoffs_cas(text, text, text, text, integer, bigint, bigint, bigint, timestamptz, jsonb, jsonb, jsonb, text, text, text) to service_role;
grant execute on function public.guard_tournament_game_day_live_mutation() to service_role;
grant execute on function public.guard_tournament_check_in_during_player_claim() to service_role;
grant execute on function public.guard_tournament_team_during_player_claim() to service_role;

comment on table public.tournament_day_live_runs is
  'FastAPI-private authoritative tournament day activation and queue version.';
comment on table public.tournament_day_live_queue is
  'Shared multi-draw tournament day queue and current court assignment authority.';
comment on table public.tournament_day_live_participant_claims is
  'History-preserving cross-draw player claims; only rows with released_at null are active.';

notify pgrst, 'reload schema';
