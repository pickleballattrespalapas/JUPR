-- Legacy-root mirror of the canonical Supabase support-intake guardrails.

alter table public.public_support_requests
  add column if not exists request_fingerprint text null,
  add column if not exists request_dedupe_key text null,
  add column if not exists identity_status text not null default 'not_required',
  add column if not exists fulfillment_status text not null default 'not_required',
  add column if not exists resolution_action text not null default 'none',
  add column if not exists resolution_evidence text null;

create index if not exists idx_public_support_requests_email_created
  on public.public_support_requests (club_id, requester_email, created_at desc);

create index if not exists idx_public_support_requests_fingerprint_created
  on public.public_support_requests (club_id, request_fingerprint, created_at desc)
  where request_fingerprint is not null;

create unique index if not exists uq_public_support_requests_daily_dedupe
  on public.public_support_requests (club_id, request_dedupe_key)
  where request_dedupe_key is not null;
