create table if not exists public.replay_jobs (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  target_reset text not null,
  status text not null default 'pending',
  actor_email text,
  actor_role text,
  started_at timestamptz,
  finished_at timestamptz,
  result_json jsonb not null default '{}'::jsonb,
  error_text text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
