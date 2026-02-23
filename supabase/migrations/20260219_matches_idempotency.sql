-- Ensure idempotency_key exists and is unique per club
ALTER TABLE public.matches
  ADD COLUMN IF NOT EXISTS idempotency_key TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS matches_club_idempotency_uidx
  ON public.matches (club_id, idempotency_key)
  WHERE idempotency_key IS NOT NULL;
