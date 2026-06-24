-- Hotfix for linked partner registration submissions failing with PGRST204.
-- The 20261019 partner-link migration introduced tournament_registrations.player_id,
-- but PostgREST can continue serving a stale schema cache after DDL changes.
-- This migration is idempotent: it creates the column/indexes if missing, then
-- explicitly requests a PostgREST schema reload.

alter table if exists public.tournament_registrations
  add column if not exists player_id integer null references public.players(id) on delete set null;

create index if not exists idx_tournament_registrations_player_id
  on public.tournament_registrations (player_id);

create unique index if not exists uq_tournament_registrations_tournament_player
  on public.tournament_registrations (tournament_id, player_id)
  where player_id is not null;

notify pgrst, 'reload schema';
