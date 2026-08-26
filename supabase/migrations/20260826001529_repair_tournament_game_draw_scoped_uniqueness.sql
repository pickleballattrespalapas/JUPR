-- Repair tournament game uniqueness after draw-scoped schedules were added.
--
-- Some staging databases retain the pre-draw objects as standalone indexes,
-- while older schema snapshots created them as table constraints.  Remove
-- both forms so separate draws in one tournament may reuse round/slot and
-- playoff game codes.  The replacement partial indexes continue to reject
-- duplicates within one draw and preserve the legacy draw-less contract.

alter table public.tournament_games
  drop constraint if exists tournament_games_rr_unique,
  drop constraint if exists tournament_games_playoff_unique;

drop index if exists public.tournament_games_rr_unique;
drop index if exists public.tournament_games_playoff_unique;

create unique index if not exists uq_tournament_games_draw_rr
  on public.tournament_games (
    tournament_id,
    draw_id,
    rr_round_number,
    rr_slot_number
  )
  where draw_id is not null and stage = 'ROUND_ROBIN';

create unique index if not exists uq_tournament_games_draw_playoff
  on public.tournament_games (
    tournament_id,
    draw_id,
    playoff_game_code
  )
  where draw_id is not null
    and stage = 'PLAYOFF'
    and playoff_game_code is not null;

create unique index if not exists uq_tournament_games_legacy_rr
  on public.tournament_games (
    tournament_id,
    rr_round_number,
    rr_slot_number
  )
  where draw_id is null and stage = 'ROUND_ROBIN';

create unique index if not exists uq_tournament_games_legacy_playoff
  on public.tournament_games (
    tournament_id,
    playoff_game_code
  )
  where draw_id is null
    and stage = 'PLAYOFF'
    and playoff_game_code is not null;
