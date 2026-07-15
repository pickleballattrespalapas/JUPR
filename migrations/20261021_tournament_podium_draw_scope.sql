-- Draw-scoped tournament podiums for division-based Tournament Ops

alter table public.tournament_podium
  add column if not exists draw_id uuid null references public.tournament_event_draws(id) on delete cascade;

-- The original table was tournament-wide. Next/Vercel Tournament Ops is division-draw scoped,
-- so remove tournament-wide uniqueness and replace it with uniqueness per draw while keeping
-- legacy null-draw rows unique at the tournament level.
alter table public.tournament_podium
  drop constraint if exists tournament_podium_unique_placement,
  drop constraint if exists tournament_podium_unique_team;

drop index if exists uq_tournament_podium_draw_placement;
drop index if exists uq_tournament_podium_draw_team;
drop index if exists uq_tournament_podium_legacy_placement;
drop index if exists uq_tournament_podium_legacy_team;

create unique index if not exists uq_tournament_podium_draw_placement
  on public.tournament_podium (tournament_id, draw_id, placement)
  where draw_id is not null;

create unique index if not exists uq_tournament_podium_draw_team
  on public.tournament_podium (tournament_id, draw_id, team_id)
  where draw_id is not null;

create unique index if not exists uq_tournament_podium_legacy_placement
  on public.tournament_podium (tournament_id, placement)
  where draw_id is null;

create unique index if not exists uq_tournament_podium_legacy_team
  on public.tournament_podium (tournament_id, team_id)
  where draw_id is null;

create index if not exists idx_tournament_podium_draw_id on public.tournament_podium (draw_id);

notify pgrst, 'reload schema';
