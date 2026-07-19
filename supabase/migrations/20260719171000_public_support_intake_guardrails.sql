-- Durable anti-abuse and privacy-fulfillment state for public support intake.
-- This table remains service-role-only; public forms write through FastAPI.

alter table public.public_support_requests
  add column if not exists request_fingerprint text null,
  add column if not exists request_dedupe_key text null,
  add column if not exists identity_status text not null default 'not_required',
  add column if not exists fulfillment_status text not null default 'not_required',
  add column if not exists resolution_action text not null default 'none',
  add column if not exists resolution_evidence text null;

update public.public_support_requests
set
  identity_status = case when request_type = 'profile_privacy' then 'pending' else 'not_required' end,
  fulfillment_status = case when request_type = 'profile_privacy' then 'pending' else 'not_required' end
where identity_status = 'not_required'
  and fulfillment_status = 'not_required'
  and request_type = 'profile_privacy';

do $support_constraints$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.public_support_requests'::regclass
      and conname = 'public_support_requests_identity_status_chk'
  ) then
    alter table public.public_support_requests
      add constraint public_support_requests_identity_status_chk
      check (identity_status in ('not_required', 'pending', 'verified', 'rejected'));
  end if;

  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.public_support_requests'::regclass
      and conname = 'public_support_requests_fulfillment_status_chk'
  ) then
    alter table public.public_support_requests
      add constraint public_support_requests_fulfillment_status_chk
      check (fulfillment_status in ('not_required', 'pending', 'in_progress', 'completed', 'declined'));
  end if;

  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.public_support_requests'::regclass
      and conname = 'public_support_requests_resolution_action_chk'
  ) then
    alter table public.public_support_requests
      add constraint public_support_requests_resolution_action_chk
      check (resolution_action in ('none', 'alias', 'hide', 'anonymize', 'contact_update', 'correction', 'other'));
  end if;
end
$support_constraints$;

create index if not exists idx_public_support_requests_email_created
  on public.public_support_requests (club_id, requester_email, created_at desc);

create index if not exists idx_public_support_requests_fingerprint_created
  on public.public_support_requests (club_id, request_fingerprint, created_at desc)
  where request_fingerprint is not null;

create unique index if not exists uq_public_support_requests_daily_dedupe
  on public.public_support_requests (club_id, request_dedupe_key)
  where request_dedupe_key is not null;

drop trigger if exists public_support_requests_set_updated_at on public.public_support_requests;
create trigger public_support_requests_set_updated_at
before update on public.public_support_requests
for each row execute function public.set_updated_at_timestamp();

alter table public.public_support_requests enable row level security;
revoke all on table public.public_support_requests from public, anon, authenticated;
grant all privileges on table public.public_support_requests to service_role;
