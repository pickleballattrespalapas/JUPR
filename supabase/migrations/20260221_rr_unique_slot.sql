CREATE UNIQUE INDEX IF NOT EXISTS
tournament_rr_unique_slot
ON tournament_games (
    tournament_id,
    stage,
    rr_round_number,
    rr_slot_number
)
WHERE stage = 'ROUND_ROBIN';
