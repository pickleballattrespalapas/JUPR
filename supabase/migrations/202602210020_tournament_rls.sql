-- ============================================
-- TOURNAMENT RLS POLICIES
-- ============================================

alter table public.tournaments enable row level security;
alter table public.tournament_teams enable row level security;
alter table public.tournament_games enable row level security;
alter table public.tournament_podium enable row level security;


-- JWT club_id extractor
-- (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')


-- TOURNAMENTS
create policy tournaments_select_by_club
on public.tournaments
for select
using (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

create policy tournaments_insert_by_club
on public.tournaments
for insert
with check (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

create policy tournaments_update_by_club
on public.tournaments
for update
using (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

create policy tournaments_delete_by_club
on public.tournaments
for delete
using (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);


-- CHILD TABLES (ALL operations)
create policy tournament_teams_by_club
on public.tournament_teams
for all
using (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
)
with check (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

create policy tournament_games_by_club
on public.tournament_games
for all
using (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
)
with check (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);

create policy tournament_podium_by_club
on public.tournament_podium
for all
using (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
)
with check (
  club_id =
  (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'club_id')
);
