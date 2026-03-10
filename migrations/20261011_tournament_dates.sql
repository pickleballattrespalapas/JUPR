-- Add canonical tournament date window fields.
-- Backward-compatible: nullable columns, no destructive changes.

ALTER TABLE public.tournaments
  ADD COLUMN IF NOT EXISTS start_date date,
  ADD COLUMN IF NOT EXISTS end_date date;

-- Helpful for date-range filtering in admin/reporting queries.
CREATE INDEX IF NOT EXISTS idx_tournaments_start_date ON public.tournaments (start_date);
CREATE INDEX IF NOT EXISTS idx_tournaments_end_date ON public.tournaments (end_date);
