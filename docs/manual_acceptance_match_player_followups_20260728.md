# Match/player manual acceptance follow-ups — July 28, 2026

The write window was closed before this staging-only implementation began.

## Implemented fixes

1. Match Uploader submission attempts now show an inline validation message and highlight missing player and score fields instead of silently doing nothing.
2. A player selected in one slot is removed from the other player pickers within that match. Clearing or replacing the player makes them available again; other match rows are unaffected.
3. Remove All now offers **No, go back**, **Keep rows with data**, and **Yes, remove all**.
4. Doubles entry is organized into separate Team 1 and Team 2 cards, with each team’s players stacked vertically and its score kept inside the same card.
5. Match submission results display ratings on the JUPR 2.0–7.0 scale rather than raw ELO values.
6. Match Log edit confirmation remains open through **Working…**, then changes into a success summary with an **OK** button after the edit and any required replay complete.
7. Quick Replay includes a clear **Open Replay History** link for recent job status.

## Verified behaviors retained

- One blank doubles row on load.
- Accurate ready-row counting.
- Guarded individual-row removal.
- Submission success dialog and reset to one blank row.
- Compact single-match editor with working filters.
- Notes-only edits and rating-affecting edits with durable Replay ALL.
- Recent durable edit operations collapsed by default.

## Staging publication

Focused source contracts, component checks, and the full Next.js build passed before merge. This documentation-only publication commit reasserts the canonical staging branch after verification; it does not alter application behavior, database schema, or write permissions.
