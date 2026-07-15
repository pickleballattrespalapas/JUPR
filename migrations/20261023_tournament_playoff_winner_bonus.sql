alter table public.matches
    add column if not exists rating_bonus_elo double precision not null default 0,
    add column if not exists rating_bonus_reason text;

create index if not exists idx_matches_tournament_rating_bonus
    on public.matches (tournament_id, tournament_game_id)
    where rating_bonus_elo > 0;

notify pgrst, 'reload schema';
