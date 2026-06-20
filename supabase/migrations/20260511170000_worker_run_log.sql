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
