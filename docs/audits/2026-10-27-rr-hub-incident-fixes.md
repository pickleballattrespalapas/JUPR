# October 27 hub incident: planned rounds and recovery

## Incident and cause

The event had 15 and 29 active players in its two divisions. Both were below the
40-entry per-division limit. A late arrival stalled the operator screen, and
future schedules repeatedly kept arrival cohorts together instead of mixing the
whole group. The operator completed the evening manually.

The original player-add path regenerated 19 future rounds before its hub caller
discarded them. A saved local cProfile attributed 141.17 of 141.27 instrumented
seconds to that regeneration. These are profiling measurements, not observed
production request latency. An unprofiled replay of the original code exceeded
12 seconds and was stopped. The initial PR1702 bypass reduced the full command
to about 2.58 seconds, with nearly all remaining time in a necessary new-round
calculation.

Separately, background check-in processing acquired the mutation lock without
showing busy state. The board waited for that processing despite having already
loaded successfully. An enabled score-save click could silently return without
sending a request. Request deadlines alone did not correct that behavior.

## Changes

1. **One prepared successor.** Standard hub round robins retain the current round
   and one complete next round. Match readiness uses each person's earlier
   obligations and court/identity claims. Partial scores reuse the plan; settling
   a round prepares one successor. Existing actual commitments are protected even
   when they are beyond this normal horizon.
2. **Whole-group rotation.** Rebuild replaceable next-round previews after roster
   changes, including late arrivals. Use the complete eligible roster and
   partnership, opponent, and bye history. Bound hub-only doubles search; the
   standalone generator's default search remains unchanged.
3. **Intentional lineup protection.** Exact team overrides lock affected games.
   Started/scored/assigned/reserved games remain protected. A departure repairs
   only affected unstarted manual lineups rather than forcing the person to stay.
   The original generator's seat-arrivals route uses the same bounded repair and
   preserves locks.
4. **Visible, recoverable writes.** Display successful board reads immediately.
   Automatic check-ins use the same visible busy state and exact saved-request
   recovery as operator commands. Reduce event-list polling while retaining
   selected-board refresh. Bound both response headers and body reads.
5. **Operator controls.** Add exact two-team edits with bye/donor swaps, batch
   departures with effective-round previews, a complete next-round view, and an
   explicit last-round choice. Reject stale requests and changed court claims.
6. **Standings.** Keep the existing completed-round totals as the default, with
   an explicit provisional view including scored partial-round games and a named
   list of missing results. Departed players retain historical results.
7. **Supplemental exhibitions.** Borrow four existing identities into a separate
   one-round game without moving source memberships. The policy is explicitly
   exhibition-only: no home-division standings, ratings, submission, or playoff
   effects. Deterministic hidden drafts recover interrupted creation; attachment,
   court state, and the complete session read set use the existing atomic RPC.
   Both hub and original generator write routes enforce the exhibition policy.
8. **Diagnostics.** Log successful command stage durations and roster/group
   counts without names, scores, credentials, or payloads.

Moneyball and playoff scheduling retain their specialized rules. No database
schema migration is needed. This change does not import or alter the evening's
manual scorebook or write to the live event.

## Local performance evidence

| Replay operation | Time | New plans |
| --- | ---: | ---: |
| Full incident-shape player add, including legacy preview migration | 0.0263 s | Bounded current/next repairs |
| Save a nonfinal current-round score | 0.0090–0.0136 s | 0 |
| Save the final current-round score | 0.0172 s | 1 successor |

The saved incident replay includes both divisions. These timings exclude HTTP,
database, and hosting latency. Production performance must be measured from the
command-stage logs after an authorized release.

Full synthetic-night checks used 15 players for 11 rounds and 29 players for 21
rounds. Both had zero repeated partnerships, zero repeated foursomes, zero
consecutive byes, and each eligible participant exactly once per round across
matches and byes. Bye counts were 2–3 and 0–1 respectively. The bounded heuristic
does not claim a globally optimal schedule for every roster/history combination.

## Regression evidence and review

- Python coverage exercises planning horizons, incremental arrival cohorts,
  score-order independence, early dispatch, fair byes, legacy migration, manual
  locks, batch departures, old generator entry points, final rounds, provisional
  standings, exact retries, publication locks, and concurrent session conflicts.
- React tests hold background requests, delay headers/body reads, replay after
  reload, leave in-flight actions, change versions, edit future lineups, and select
  a final boundary including zero.
- Loopback browser verification exercises the actual UI, FastAPI routes, domain,
  and transactional service double. It is not production/staging traffic.
- PGlite executes the real identity and lease SQL for supplemental attach,
  complete read-set enforcement, exact replay, immutable source rows, and
  overlapping guest/profile court rejection.
- Independent review found and corrected both missing round-number forwards in
  next-round exact editing, the legacy seat-arrivals lock bypass, and the
  exhibition rating-edit bypass. New regressions cover each path.

The feature targets `staging`. Production deployment remains subject to the
repository's separate explicit authorization rule.
