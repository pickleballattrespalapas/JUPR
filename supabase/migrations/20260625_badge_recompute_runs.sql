create table if not exists public.badge_eval_runs (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    created_by text,
    mode text not null,
    scope_json jsonb not null default '{}'::jsonb,
    status text not null default 'queued',
    started_at timestamptz,
    finished_at timestamptz,
    summary_json jsonb not null default '{}'::jsonb,
    error text
);

alter table if exists public.player_badges
    add column if not exists awarded_by text not null default 'engine',
    add column if not exists rule_version text,
    add column if not exists eval_run_id uuid references public.badge_eval_runs(id);
