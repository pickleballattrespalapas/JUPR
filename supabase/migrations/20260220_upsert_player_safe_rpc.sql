create or replace function public.upsert_player_safe(p_payload jsonb)
returns void
language sql
as $$
insert into public.players (
  club_id,
  name,
  active,
  rating,
  starting_rating,
  wins,
  losses,
  matches_played,
  last_game_at,
  inactive_at
)
select
  r.club_id,
  r.name,
  coalesce(r.active, true),
  r.rating,
  r.starting_rating,
  coalesce(r.wins, 0),
  coalesce(r.losses, 0),
  coalesce(r.matches_played, 0),
  r.last_game_at,
  r.inactive_at
from jsonb_to_record(p_payload) as r(
  club_id text,
  name text,
  active boolean,
  rating numeric,
  starting_rating numeric,
  wins integer,
  losses integer,
  matches_played integer,
  last_game_at timestamptz,
  inactive_at timestamptz
)
on conflict (club_id, normalized_name)
do update set
  name = excluded.name,
  active = excluded.active,
  rating = excluded.rating,
  starting_rating = excluded.starting_rating,
  wins = excluded.wins,
  losses = excluded.losses,
  matches_played = excluded.matches_played,
  last_game_at = excluded.last_game_at,
  inactive_at = excluded.inactive_at;
$$;
