alter table public.players enable row level security;

drop policy if exists "Players are viewable by club" on public.players;
drop policy if exists "Players insert by club" on public.players;
drop policy if exists "Players update by club" on public.players;

create policy "players_select_by_club"
on public.players
for select
using (
  club_id = (public.jwt_claims() ->> 'club_id')
);

create policy "players_insert_by_club"
on public.players
for insert
with check (
  club_id = (public.jwt_claims() ->> 'club_id')
);

create policy "players_update_by_club"
on public.players
for update
using (
  club_id = (public.jwt_claims() ->> 'club_id')
)
with check (
  club_id = (public.jwt_claims() ->> 'club_id')
);
