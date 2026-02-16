CREATE TABLE IF NOT EXISTS public.processed_match_facts (
  club_id TEXT NOT NULL,
  match_id TEXT NOT NULL,
  player_id BIGINT NOT NULL,
  PRIMARY KEY (club_id, match_id, player_id)
);
