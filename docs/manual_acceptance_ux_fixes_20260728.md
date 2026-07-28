# July 28 manual acceptance UX fixes

This staging-only patch addresses the Match Uploader and Match Log findings from the July 28, 2026 manual write-acceptance session.

- Match Uploader opens in doubles manual/batch with one blank row.
- Readiness reflects only complete, valid rows and stale validation clears as fields change.
- Player creation is hidden while a roster match remains available; selected names remain readable.
- Removing entered rows requires a Yes/No confirmation whenever player, score, date, week, or rating-scope data was changed.
- Successful submissions use a concise confirmation dialog and reset to one blank row after OK while preserving setup values.
- Unconfirmed API outcomes remain retry-safe and never display a false success confirmation.
- Technical runtime flag names are removed from user-facing success text.
- Match Log Edit One keeps filters and the compact editor but hides the duplicate full results table and summary cards.
- Durable operation history is collapsed by default.

Focused source checks, component checks, and the full Next.js production build passed against the final generated source before merge.

The patch does not open a write window and does not target production.
