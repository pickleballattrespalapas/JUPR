-- Linked doubles partner requests and confirmed registration team links.
-- Keeps legacy free-text partner fields on tournament_registration_selections for admin reconciliation only.

alter table if exists public.tournament_registrations
  add column if not exists player_id integer null references public.players(id) on delete set null;

create index if not exists idx_tournament_registrations_player_id
  on public.tournament_registrations (player_id);

create unique index if not exists uq_tournament_registrations_tournament_player
  on public.tournament_registrations (tournament_id, player_id)
  where player_id is not null;

create table if not exists public.tournament_registration_partner_requests (
  id text primary key,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  requester_selection_id text not null references public.tournament_registration_selections(id) on delete cascade,
  requester_registration_id text not null references public.tournament_registrations(id) on delete cascade,
  requester_player_id integer null references public.players(id) on delete set null,
  target_selection_id text null references public.tournament_registration_selections(id) on delete set null,
  target_registration_id text null references public.tournament_registrations(id) on delete set null,
  target_player_id integer null references public.players(id) on delete set null,
  target_display_name_snapshot text null,
  status text not null default 'PENDING',
  source text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  responded_at timestamptz null,
  created_by_registration_id text null references public.tournament_registrations(id) on delete set null,
  created_by_user_id text null,
  constraint tournament_registration_partner_requests_status_chk
    check (status in ('PENDING', 'ACCEPTED', 'DECLINED', 'CANCELLED', 'ADMIN_CONFIRMED', 'EXPIRED')),
  constraint tournament_registration_partner_requests_source_chk
    check (source in ('PROFILE_SEARCH', 'NEEDS_PARTNER_LIST', 'ADMIN_RECONCILIATION', 'ADMIN_CREATED', 'INVITE_LINK', 'LEGACY_TEXT_MATCH')),
  constraint tournament_registration_partner_requests_not_self_chk
    check (
      target_selection_id is null
      or requester_selection_id <> target_selection_id
    )
);

create index if not exists idx_tournament_partner_requests_tournament
  on public.tournament_registration_partner_requests (tournament_id, created_at desc);

create index if not exists idx_tournament_partner_requests_event_status
  on public.tournament_registration_partner_requests (event_option_id, status);

create index if not exists idx_tournament_partner_requests_requester_selection
  on public.tournament_registration_partner_requests (requester_selection_id);

create index if not exists idx_tournament_partner_requests_target_selection
  on public.tournament_registration_partner_requests (target_selection_id)
  where target_selection_id is not null;

create index if not exists idx_tournament_partner_requests_requester_player
  on public.tournament_registration_partner_requests (requester_player_id)
  where requester_player_id is not null;

create index if not exists idx_tournament_partner_requests_target_player
  on public.tournament_registration_partner_requests (target_player_id)
  where target_player_id is not null;

create table if not exists public.tournament_registration_team_links (
  id text primary key,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  registration1_id text not null references public.tournament_registrations(id) on delete cascade,
  registration2_id text not null references public.tournament_registrations(id) on delete cascade,
  selection1_id text not null references public.tournament_registration_selections(id) on delete cascade,
  selection2_id text not null references public.tournament_registration_selections(id) on delete cascade,
  player1_id integer null references public.players(id) on delete set null,
  player2_id integer null references public.players(id) on delete set null,
  status text not null default 'CONFIRMED',
  accepted_request_id text null references public.tournament_registration_partner_requests(id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by_user_id text null,
  constraint tournament_registration_team_links_status_chk
    check (status in ('CONFIRMED', 'ADMIN_CONFIRMED', 'CANCELLED')),
  constraint tournament_registration_team_links_distinct_registrations_chk
    check (registration1_id <> registration2_id),
  constraint tournament_registration_team_links_distinct_selections_chk
    check (selection1_id <> selection2_id)
);

create index if not exists idx_tournament_team_links_tournament
  on public.tournament_registration_team_links (tournament_id, created_at desc);

create index if not exists idx_tournament_team_links_event_status
  on public.tournament_registration_team_links (event_option_id, status);

create index if not exists idx_tournament_team_links_accepted_request
  on public.tournament_registration_team_links (accepted_request_id)
  where accepted_request_id is not null;

create table if not exists public.tournament_registration_team_members (
  id text primary key,
  team_link_id text not null references public.tournament_registration_team_links(id) on delete cascade,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  event_option_id text not null references public.tournament_event_options(id) on delete cascade,
  selection_id text not null references public.tournament_registration_selections(id) on delete cascade,
  registration_id text not null references public.tournament_registrations(id) on delete cascade,
  player_id integer null references public.players(id) on delete set null,
  player_order integer not null,
  status text not null default 'ACTIVE',
  created_at timestamptz not null default now(),
  constraint tournament_registration_team_members_status_chk
    check (status in ('ACTIVE', 'CANCELLED')),
  constraint tournament_registration_team_members_player_order_chk
    check (player_order in (1, 2))
);

create unique index if not exists uq_tournament_team_members_team_selection
  on public.tournament_registration_team_members (team_link_id, selection_id);

create unique index if not exists uq_tournament_team_members_team_order
  on public.tournament_registration_team_members (team_link_id, player_order);

create unique index if not exists uq_tournament_team_members_active_event_selection
  on public.tournament_registration_team_members (event_option_id, selection_id)
  where status = 'ACTIVE';

create index if not exists idx_tournament_team_members_tournament
  on public.tournament_registration_team_members (tournament_id, event_option_id);
