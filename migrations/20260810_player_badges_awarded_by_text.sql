-- Normalize player_badges.awarded_by to text with a stable engine default.
ALTER TABLE public.player_badges
  ALTER COLUMN awarded_by TYPE text
  USING COALESCE(awarded_by::text, 'engine');

ALTER TABLE public.player_badges
  ALTER COLUMN awarded_by SET DEFAULT 'engine';

UPDATE public.player_badges
SET awarded_by = 'engine'
WHERE awarded_by IS NULL;

ALTER TABLE public.player_badges
  ALTER COLUMN awarded_by SET NOT NULL;

NOTIFY pgrst, 'reload schema';
