-- Support the occupied-court reservation foreign key without full queue scans.
-- The one-reservation-per-court partial unique index is ordered by run first,
-- so this leading-column index covers direct court deletes and integrity checks.

create index if not exists ix_tournament_day_live_queue_reserved_court_id
  on public.tournament_day_live_queue (reserved_court_id)
  where reserved_court_id is not null;
