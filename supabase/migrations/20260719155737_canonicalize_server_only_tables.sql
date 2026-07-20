-- Canonical Supabase history for server-only tables that were originally
-- introduced through legacy root migrations during the Streamlit transition.
-- The legacy files remain for fallback history; this is the deployable source.

-- Match contexts predate the canonical Supabase migration ledger. Some legacy
-- schemas added context_id as uuid, while the live application uses text for
-- deterministic league, ladder, event, and tournament context identifiers.
-- Normalize the column before any later context indexes or write RPCs exist.
alter table public.matches
  add column if not exists context_type text null,
  add column if not exists context_id text null;

do $canonical_match_context_id$
declare
  current_type text;
begin
  select format_type(attribute.atttypid, attribute.atttypmod)
    into current_type
    from pg_attribute attribute
   where attribute.attrelid = 'public.matches'::regclass
     and attribute.attname = 'context_id'
     and not attribute.attisdropped;

  if current_type is null then
    raise exception 'public.matches.context_id must exist before canonicalization';
  end if;

  if current_type <> 'text' then
    alter table public.matches
      alter column context_id type text using context_id::text;
  end if;
end
$canonical_match_context_id$;

comment on column public.matches.context_id is
  'Canonical text identifier for deterministic match source and workflow contexts.';

create table if not exists public.weekly_recaps (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  week_start date not null,
  week_end date not null,
  status text not null default 'draft',
  generated_json jsonb not null default '{}'::jsonb,
  edits_json jsonb not null default '{}'::jsonb,
  final_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  published_at timestamptz null,
  published_by text null,
  unique (club_id, week_start)
);

create index if not exists weekly_recaps_club_week_idx
  on public.weekly_recaps (club_id, week_start desc);
create index if not exists weekly_recaps_club_status_week_idx
  on public.weekly_recaps (club_id, status, week_start desc);

drop trigger if exists weekly_recaps_set_updated_at on public.weekly_recaps;
create trigger weekly_recaps_set_updated_at
before update on public.weekly_recaps
for each row execute function public.set_updated_at_timestamp();

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

create table if not exists public.league_live_sessions (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_name text not null,
  week_tag text not null default '',
  status text not null default 'setup',
  total_rounds integer not null default 1,
  current_round integer not null default 1,
  roster_json jsonb not null default '[]'::jsonb,
  current_court_state_json jsonb not null default '[]'::jsonb,
  notes text,
  created_by text,
  updated_by text,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint league_live_sessions_status_check
    check (status in ('setup', 'active', 'paused', 'complete', 'archived')),
  constraint league_live_sessions_total_rounds_check
    check (total_rounds between 1 and 50),
  constraint league_live_sessions_current_round_check
    check (current_round >= 1)
);

create index if not exists idx_league_live_sessions_club_status_updated
  on public.league_live_sessions (club_id, status, updated_at desc);
create index if not exists idx_league_live_sessions_club_league_updated
  on public.league_live_sessions (club_id, league_name, updated_at desc);

create table if not exists public.league_live_rounds (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.league_live_sessions(id) on delete cascade,
  round_number integer not null,
  round_label text,
  status text not null default 'draft',
  match_date date,
  preview_json jsonb not null default '{}'::jsonb,
  matches_json jsonb not null default '[]'::jsonb,
  movement_json jsonb not null default '{}'::jsonb,
  submitted_match_count integer not null default 0,
  submitted_match_ids jsonb not null default '[]'::jsonb,
  submitted_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint league_live_rounds_status_check
    check (status in ('draft', 'generated', 'submitted', 'voided')),
  constraint league_live_rounds_round_number_check check (round_number >= 1),
  constraint league_live_rounds_submitted_count_check check (submitted_match_count >= 0),
  constraint league_live_rounds_unique_round unique (session_id, round_number)
);

create index if not exists idx_league_live_rounds_club_session_round
  on public.league_live_rounds (club_id, session_id, round_number);

create table if not exists public.league_live_courts (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.league_live_sessions(id) on delete cascade,
  round_number integer not null,
  court_number integer not null,
  format_type text not null default '4-player',
  player_names jsonb not null default '[]'::jsonb,
  players_json jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint league_live_courts_round_check check (round_number >= 1),
  constraint league_live_courts_court_check check (court_number >= 1),
  constraint league_live_courts_unique_court unique (session_id, round_number, court_number)
);

create index if not exists idx_league_live_courts_club_session_round
  on public.league_live_courts (club_id, session_id, round_number, court_number);

do $server_table_security$
declare
  table_name text;
begin
  foreach table_name in array array[
    'weekly_recaps',
    'public_support_requests',
    'league_live_sessions',
    'league_live_rounds',
    'league_live_courts'
  ]
  loop
    execute format('alter table public.%I enable row level security', table_name);
    execute format(
      'revoke all on table public.%I from public, anon, authenticated',
      table_name
    );
    execute format(
      'grant all privileges on table public.%I to service_role',
      table_name
    );
  end loop;
end
$server_table_security$;

notify pgrst, 'reload schema';
