begin;
create table public.pcs_platform_audit (
 id bigint generated always as identity primary key,
 actor_id uuid not null, club_id text not null, action text not null,
 details jsonb not null default '{}', created_at timestamptz not null default now()
);
alter table public.pcs_platform_audit enable row level security;
revoke all on public.pcs_platform_audit from public,anon,authenticated;
grant all on public.pcs_platform_audit to service_role;
grant usage,select on sequence public.pcs_platform_audit_id_seq to service_role;
commit;
