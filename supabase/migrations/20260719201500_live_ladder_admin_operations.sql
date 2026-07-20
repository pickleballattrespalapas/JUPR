-- Order-24 durable Challenge Ladder, Moneyball, and one-off JUPR Live writes.
--
-- FastAPI verifies the operator's Supabase JWT, then uses only the server-side
-- service role for this private coordination ledger.  The browser never reads
-- or writes this table directly.  A stable request fingerprint plus one
-- operation key makes response-loss retries observable and non-duplicating.

create table if not exists public.live_ladder_admin_operations (
    operation_key text primary key,
    club_id text not null,
    surface text not null,
    operation_type text not null,
    entity_id text not null,
    idempotency_key text not null,
    request_fingerprint text not null,
    expected_version text not null,
    status text not null default 'intent',
    request_json jsonb not null default '{}'::jsonb,
    result_json jsonb not null default '{}'::jsonb,
    recovery_json jsonb not null default '{}'::jsonb,
    error_text text,
    attempt_count integer not null default 0,
    created_by text not null,
    updated_by text not null,
    completed_at timestamptz,
    completion_audited_at timestamptz,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint live_ladder_admin_operations_surface_check check (
        surface in ('challenge_ladder', 'moneyball', 'jupr_live_admin')
    ),
    constraint live_ladder_admin_operations_status_check check (
        status in ('intent', 'running', 'mutated', 'completed', 'recovery_required', 'failed')
    ),
    constraint live_ladder_admin_operations_attempt_check check (attempt_count >= 0),
    constraint live_ladder_admin_operations_fingerprint_check check (char_length(request_fingerprint) = 64),
    constraint live_ladder_admin_operations_key_check check (char_length(operation_key) = 64),
    constraint live_ladder_admin_operations_idempotency_unique unique (
        club_id, surface, operation_type, entity_id, idempotency_key
    )
);

create index if not exists live_ladder_admin_operations_club_surface_updated_idx
    on public.live_ladder_admin_operations (club_id, surface, updated_at desc);

create index if not exists live_ladder_admin_operations_recovery_idx
    on public.live_ladder_admin_operations (club_id, status, updated_at desc)
    where status <> 'completed';

-- Serialize one authoritative snapshot per club/surface.  A second request
-- cannot slip between the initial version check and mutation.  Uncertain
-- operations keep the lease until reconciled; known pre-mutation failures do
-- not.
create unique index if not exists live_ladder_admin_operations_active_version_unique
    on public.live_ladder_admin_operations (club_id, surface, expected_version)
    where status in ('intent', 'running', 'mutated', 'recovery_required');

-- New order-24 official publishes derive one UUIDv5 context per operation/slot.
-- The UUID-version predicate avoids imposing a constraint on legacy rows that
-- used one shared event/challenge context for several matches. The canonical
-- matches.context_id column is text so every workflow can share the column;
-- these operation-scoped values remain UUIDv5 strings to prevent collisions.
create unique index if not exists matches_live_ladder_operation_context_unique
    on public.matches (club_id, context_type, context_id)
    where context_type in ('challenge_ladder', 'moneyball', 'jupr_live')
      and context_id is not null
      and substring(context_id::text from 15 for 1) = '5';

alter table public.live_ladder_admin_operations enable row level security;
alter table public.live_ladder_admin_operations force row level security;

revoke all on table public.live_ladder_admin_operations from public, anon, authenticated;
grant select, insert, update on table public.live_ladder_admin_operations to service_role;

comment on table public.live_ladder_admin_operations is
    'FastAPI-private order-24 mutation intents/results for Challenge Ladder, Moneyball, and one-off JUPR Live.';

notify pgrst, 'reload schema';
