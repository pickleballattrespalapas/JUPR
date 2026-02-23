ALTER TABLE matches
ADD COLUMN IF NOT EXISTS idempotency_key text;

CREATE UNIQUE INDEX IF NOT EXISTS matches_idempotency_key_idx
ON matches (idempotency_key);
