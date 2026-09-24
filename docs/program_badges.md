# Automatic program badges

The badge expansion adds 22 definitions with direct earning requirements.

| Series | Thresholds | Scope |
|---|---|---|
| Leagues completed | 5, 10, 25 | Player's club career; each league's minimum games at closure |
| Tournaments completed | 5, 10, 25 | Whole tournaments; actual completed competition play |
| Round robins completed | 5, 10, 25 | Completed assigned play in distinct standalone sessions |
| Matches together | 10, 25, 50 | Each distinct partnership, awarded to both players |
| Wins together | 10, 25, 50 | Each distinct partnership, awarded to both players |
| Five Winning Partners | 5 distinct winning partners | Once per completed round robin |
| Triple Crown | Medals in 3 distinct competition events | Once per completed tournament; any medal colors |
| Round-robin victories | 1, 5, 10, 25, 50 | Player's club career |

Round-robin winners use wins, point differential, total points scored, then an administrator's decision among the remaining tied leaders. Unresolved ties hold victory credit. Completion and partnership awards can still qualify. Final generator standings use these same rules, including singles and mixed sessions; ladder standings keep their existing rules.

## Processing

`program_badges.evaluate_program_badges` is a pure calculation over a complete club snapshot. The same calculation handles new results, verified historical achievements and corrections. It never guesses a player identity from a name or invents a missing completion date. The normal match-only evaluator and strict recompute do not own these awards.

Database source triggers increment a club revision when scores, rosters, moderation, exclusions or completion change. One SQL snapshot reads the revision and all source rows together, avoiding pagination cutoffs and partially assembled reads. A small staging API worker checks durable pending revisions every 15 seconds. A restart or failed request cannot discard pending work. Closing the staging badge write wave pauses the worker; its writes are explicitly disabled outside staging in this release.

`apply_program_badges_v1` locks that revision and rejects changed evidence. It inserts or reconciles only the 22 new badge types, preserving award IDs and storing the date of the qualifying result. Repeats use stable club, pair and event contexts. Corrections can soft-revoke and restore engine-owned awards; manual revocations and older award history stay untouched. Badge state/availability settings are respected.

Best-of-three competition parents count once, including when several rating games were published. Byes, walkovers, incomplete retirements and unplayed results do not create partnership wins. Tournament participation is once per whole tournament; multiple pools and playoff draws do not create extra Triple Crown events. Recorded round-robin substitutes receive their actual results.

Saved, approved social round robins and completed admin round-robin sessions qualify. A live session and its saved social event share one source identity. Public quick/private sessions and pending social submissions do not self-award. Incomplete saved scores, unknown historical finalization dates, and old publications without verifiable match links appear in **Admin → Badges → History needing review**. These holds are deliberate; source evidence must be repaired before credit is issued. Publishing new official live/generator results stores their match links for correction checks.

## Admin ties and public display

**Admin → Badges → Round-robin winners** lists unresolved ties. The server validates the current club administrator assignment, selected leader, result fingerprint and retry identity in one transaction. It records the actor, timestamp and result evidence. Saving starts a badge check immediately and leaves durable work for the background worker.

The public result records an admin decision only while its score/roster fingerprint still matches. Corrected results invalidate the old selection. The player's badge cabinet shows earning requirements plus individual partnership/event achievements, including partner names and earning dates.

## Validation and historical rollout

Focused Python tests cover thresholds, identity and club boundaries, substitutions, singles, whole-event counting, medals, best-of-three projections, pending moderation, missing history and stale decisions. Component tests exercise interrupted-save recovery for community awards and round-robin decisions. `tests/sql/program_badge_transactions.sql` runs inside `BEGIN`/`ROLLBACK` to verify inserts, retries, corrections, manual revocation preservation, source triggers, admin-only decisions and public result persistence.

The initial staging preview found 12 awards across 12 players. One legacy completed tournament lacks a lifecycle receipt/completion timestamp and is held for review. All 1,121 pre-existing staging award rows were verified unchanged after the schema changes and transaction rehearsals. The deployed worker will apply the freshly verified history; no separate production activation is included.
