# Badge repair and historical reconciliation

The public catalog, leaderboard, and player profile now read complete award history and exclude revoked awards. A failed award read produces an unavailable response instead of an empty cabinet. Badge cards show the earning requirement, with five categories ordered around participation, improvement, partnerships, match achievements, and trophies. Existing prestige values and badge availability remain unchanged by the presentation migration.

Calculation repairs include Elo-to-JUPR conversion for Level Up, wins only for Hall of Fame Night, the documented 15% Legendary Upset threshold, order-independent mixed timestamp parsing, ISO week 53, and signed rating changes from stored snapshots or team deltas. Most Improved (monthly), Upset Champion, and Clean Sweep Week now evaluate completed periods. These period awards are checked when the next eligible result triggers the badge worker; this change does not add a scheduled midnight job.

The worker reads complete history, computes the club once per event, and includes completed-period winners even when they are absent from the triggering match. Evaluator failures fail the job so it can be investigated. Lifetime identities recognize older context keys; existing awards and staff revocations are preserved. Conflict inserts do not rewrite an existing award's ID or earning date. New ordinary engine awards include a rule version.

## Conservative history policy

Use `scripts/preview_badge_reconciliation.py` with a complete club snapshot and an explicit UTC cutoff. It does not initialize a database client, write audit rows, or change awards. The preview includes the exact source hash and simulates a second run to require an idempotent result.

- Backfill only supported participation, first-win, partnership, 100-win, Legendary Upset, and rating milestone awards.
- Retain the earliest lifetime participation award and propose soft revocation of redundant rows.
- Propose Hall of Fame revocation only when the linked match proves the recipient lost.
- Preserve trophy awards and unsupported historical rating peaks. Keep ambiguous older rules and period results in a separate review list.
- Never recreate an already revoked semantic achievement.
- Use the recorded match date when it establishes the earning event. Aggregate-counter backfills use the verification time, explicitly labeled `earned_time_basis=eligibility_verified_at`; an exact historical date must not be invented.

Example using local exports:

```sh
python scripts/preview_badge_reconciliation.py \
  --snapshot /path/to/snapshot.json \
  --club-id tres_palapas \
  --as-of 2026-09-06T00:00:00Z \
  --output /path/to/review.json
```

The snapshot contains `players`, `league_ratings`, `leagues_metadata`, `matches`, `badges`, and `player_badges`. Include award UUIDs before an actual database repair. Keep snapshots and player-specific plans out of source control.

## Release order

1. Apply the presentation-only migration to staging and verify category/requirement values. It does not create badges or alter award history, availability, or prestige.
2. Deploy and verify this code on staging. Confirm complete counts and revocation filtering.
3. Recheck each proposed staging history change against its exact stored award and match, then apply soft revocations in one short transaction. Retain a before-image for rollback.
4. Production requires Joe's separate explicit authorization under `AGENTS.md`. Refresh the full production snapshot, including award UUIDs, after authorization. Regenerate and compare the plan; stop if the reviewed change set differs. Apply verified changes transactionally with exact row checks and save before/after evidence.
5. Review unsupported historical peaks, older High Roller awards, and period winners individually. Do not use strict delete-and-rebuild mode to resolve them.

## Validation

Regression coverage includes real award-column projection, short server-capped pages beyond 1,000 rows, revocation filtering, failure propagation, historical identity preservation, rating units, losing-side awards, upset threshold boundaries, ISO week 53, mixed timestamps, signed snapshots, completed periods, preservation of trophies and earlier earning dates, changed-snapshot rejection, and reconciliation idempotence. The web component check renders the server page to verify requirement-first copy, category order, the distinct-earner statistic, and trophy navigation. Historical seed migrations retain their original text; the new presentation migration is checked against all 55 current badge definitions.
