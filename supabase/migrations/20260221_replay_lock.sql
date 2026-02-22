CREATE TABLE IF NOT EXISTS replay_lock (
    club_id text PRIMARY KEY,
    started_at timestamptz,
    status text
);
