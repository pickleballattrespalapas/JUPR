-- Canonical forward repair for environments where the original worker ledger
-- migration was absent from the remote migration history. Keep this migration
-- idempotent so it also hardens environments where the table already exists.
create table if not exists public.worker_run_log (
  id uuid primary key default gen_random_uuid(),
  worker_name text not null,
  club_id text not null,
  status text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  summary_json jsonb not null default '{}'::jsonb,
  error_text text,
  created_at timestamptz not null default now()
);

create index if not exists worker_run_log_club_created_idx
  on public.worker_run_log (club_id, created_at desc);

-- Worker execution evidence is part of the server-only data plane. Keep the
-- table reachable by FastAPI/worker service-role clients while denying direct
-- browser Data API access.
alter table public.worker_run_log enable row level security;
revoke all on table public.worker_run_log from public, anon, authenticated;
grant all privileges on table public.worker_run_log to service_role;

comment on table public.worker_run_log is
  'Server-only durable start/completion evidence for background worker runs.';

notify pgrst, 'reload schema';
