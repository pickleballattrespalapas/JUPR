begin;

alter table if exists public.session_ladder_sessions
  add column if not exists rounds_planned integer not null default 2 check (rounds_planned in (2,3)),
  add column if not exists ratings_applied_at timestamptz null,
  add column if not exists published_at timestamptz null,
  add column if not exists recap_json jsonb null,
  add column if not exists leaderboard_json jsonb null;

create table if not exists public.session_ladder_rating_history (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  session_id uuid not null references public.session_ladder_sessions(id) on delete cascade,
  player_id bigint not null references public.players(id) on delete restrict,
  league_id text null,
  rating_before numeric(10,3) not null,
  rating_after numeric(10,3) not null,
  rating_delta numeric(10,3) not null,
  created_by text null,
  created_at timestamptz not null default now(),
  constraint session_ladder_rating_history_uidx unique (session_id, player_id)
);

create index if not exists idx_session_ladder_rating_history_session
  on public.session_ladder_rating_history (session_id, player_id);

create table if not exists public.session_ladder_attendance (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_id text not null,
  season_id text null,
  session_id uuid not null references public.session_ladder_sessions(id) on delete cascade,
  player_id bigint not null references public.players(id) on delete restrict,
  created_by text null,
  created_at timestamptz not null default now(),
  constraint session_ladder_attendance_uidx unique (session_id, player_id)
);

create index if not exists idx_session_ladder_attendance_season
  on public.session_ladder_attendance (club_id, league_id, season_id, player_id);

create table if not exists public.session_ladder_awards_attendance (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  league_id text not null,
  season_id text null,
  player_id bigint not null references public.players(id) on delete restrict,
  sessions_attended integer not null default 0,
  updated_by text null,
  updated_at timestamptz not null default now(),
  constraint session_ladder_awards_attendance_uidx unique (club_id, league_id, season_id, player_id)
);

create index if not exists idx_session_ladder_awards_attendance_lookup
  on public.session_ladder_awards_attendance (club_id, league_id, season_id, sessions_attended desc);

alter table public.session_ladder_rating_history enable row level security;
alter table public.session_ladder_attendance enable row level security;
alter table public.session_ladder_awards_attendance enable row level security;

drop policy if exists sl_rating_history_select on public.session_ladder_rating_history;
create policy sl_rating_history_select on public.session_ladder_rating_history
for select using (club_id = public.jwt_club_id());

drop policy if exists sl_rating_history_write on public.session_ladder_rating_history;
create policy sl_rating_history_write on public.session_ladder_rating_history
for all using (public.session_ladder_can_write(club_id)) with check (public.session_ladder_can_write(club_id));

drop policy if exists sl_attendance_select on public.session_ladder_attendance;
create policy sl_attendance_select on public.session_ladder_attendance
for select using (club_id = public.jwt_club_id());

drop policy if exists sl_attendance_write on public.session_ladder_attendance;
create policy sl_attendance_write on public.session_ladder_attendance
for all using (public.session_ladder_can_write(club_id)) with check (public.session_ladder_can_write(club_id));

drop policy if exists sl_awards_attendance_select on public.session_ladder_awards_attendance;
create policy sl_awards_attendance_select on public.session_ladder_awards_attendance
for select using (club_id = public.jwt_club_id());

drop policy if exists sl_awards_attendance_write on public.session_ladder_awards_attendance;
create policy sl_awards_attendance_write on public.session_ladder_awards_attendance
for all using (public.session_ladder_can_write(club_id)) with check (public.session_ladder_can_write(club_id));

commit;
