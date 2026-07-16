-- Public support/data-correction/privacy intake queue.
-- Public forms only create staff-review records; they do not mutate ratings, matches, players, badges, tournaments, or privacy display state.

create table if not exists public.public_support_requests (
  id text primary key,
  club_id text not null,
  club_slug text null,
  request_type text not null default 'general_support',
  status text not null default 'new',
  requester_name text not null,
  requester_email text not null,
  player_name text null,
  player_id integer null references public.players(id) on delete set null,
  match_id text null,
  tournament_id text null,
  subject text not null,
  description text not null,
  requested_action text null,
  evidence_url text null,
  consent_to_contact boolean not null default false,
  source text null,
  admin_note text null,
  reviewed_by text null,
  reviewed_at timestamptz null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint public_support_requests_type_chk
    check (request_type in ('data_correction', 'profile_privacy', 'general_support')),
  constraint public_support_requests_status_chk
    check (status in ('new', 'in_review', 'resolved', 'dismissed'))
);

create index if not exists idx_public_support_requests_club_status
  on public.public_support_requests (club_id, status, created_at desc);

create index if not exists idx_public_support_requests_request_type
  on public.public_support_requests (request_type, created_at desc);

create index if not exists idx_public_support_requests_player_id
  on public.public_support_requests (player_id)
  where player_id is not null;
