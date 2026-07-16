# League Live court movement

League Manager Live now persists a resumable league-night session and applies a simple court-movement preview after each submitted round.

## Movement rule

After a scored round is submitted:

1. Each court is ranked by round wins, then point differential, then points scored.
2. The top player on every court except Court 1 moves up one court.
3. The bottom player on every court except the lowest court moves down one court.
4. Everyone else stays on the same court.

The Next UI computes and displays the movement preview before submission. When the operator types `SUBMIT LEAGUE ROUND`, official matches are submitted through the existing guarded Match Uploader endpoint. The movement payload is saved on the persisted `league_live_rounds.movement_json` row, and the next-round court snapshot is written back to the session through the guarded session snapshot endpoint.

## Safety notes

- Rating changes still run through the Python Match Uploader / match-processing path.
- Post-submit corrections still route through Match Log and Replay History.
- The movement payload is audit-visible as round state; it does not mutate player ratings by itself.
- Streamlit remains the reference fallback until the full League Live flow is validated in staging and pilot use.
