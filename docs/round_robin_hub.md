# Round Robin Hub

The hub shares physical courts among existing Round-Robin Generator sessions. Each
session keeps its own roster, scoring mode, rating mode, Moneyball rules, standings,
playoffs, exports, sharing, and submission/approval workflow.

## Accepted operating rules

- An event can contain multiple independent groups and named courts.
- Only matches in a group's current round enter the queue. A group must complete
  that round before generating or releasing the next. Different groups advance independently.
- More matches than courts are supported. Queued matches wait for a court; this
  waiting is distinct from a scheduled bye.
- Queue order starts with the longest-waiting player in each eligible matchup,
  then the other players' waiting times. Waiting includes time spent on bye and
  survives round changes. Completing a match starts a new waiting period.
- Pausing needs no reason. It removes a player from eligibility and cancels their
  next-court reservation. It preserves already-accrued waiting time, freezes the
  clock while away, and allows an already-playing match to finish.
- A group with dedicated courts uses only those courts, even if shared courts are
  free. Other groups cannot use those dedicated courts. Groups without dedicated
  courts use the shared pool.
- Tournament-style controls assign to the next open court, choose a court, reserve
  the next match on an occupied court, move a match, return it to the queue, and
  save a score. No player or matchup can claim two courts.
- Saving a match is durable and releases its court immediately. Its result does
  not count in standings until every match in that round is resolved. Unscored
  sessions use Match played. A completed round still offers standings, next round,
  finish, and supported playoff controls.
- Standard RR roster changes preserve completed and assigned matches and adapt
  unstarted matches. Moneyball retains its frozen inputs and substitution rules.
- Ending a session or starting playoffs must resolve pending play first. Rating
  publication continues through the existing approval workflow.

## Persistence and concurrency

The hub owns assignments and availability; existing `live_sessions` records own
match results. A service-role-only PostgreSQL RPC locks the hub and its complete
session read set, compares versions, and commits court state and session state in
one transaction. An operation receipt makes exact retries idempotent. Court and
player leases prevent overlapping assignments across active hubs in a club.

Existing admin, public organizer, and shared organizer writes for attached sessions
route through the same transaction. A database trigger rejects direct changes to
attached event data outside that authority. Public responses never contain private
Moneyball calibration data or organizer tokens.

## Verification scenarios

1. Attach two RRs with more matches than open courts; ensure only current-round
   matches enter the queue and neither RR waits for the other's round completion.
2. Save one match, refresh, confirm its court is available and its score survives,
   while that RR's standings remain unchanged until the last match finishes.
3. Preserve bye waiting time across advancement; pause and resume without a reason.
4. Dedicate a court and verify exclusivity in both directions, including reservations.
5. Attempt concurrent assignment/score operations and exact retries; no double
   booking, duplicated result, or partial court release is permitted.
6. Add/remove/substitute players with queued, assigned, and completed matches;
   protect assigned/completed lineups and keep the hub and original page synchronized.
7. Run scored, unscored, Moneyball, rated/unrated, finish, playoff, and submission
   flows using the same individual-session controls.

Deployment target for this feature is staging. Production release requires Joe's
separate authorization under the repository's AGENTS.md.

## Automated checks

- Python domain, service, API, and generator regression tests cover court eligibility,
  round sequencing, roster changes, ratings, Moneyball, and write-path integration.
- `npm run test:component` includes the hub and embedded generator contracts.
- `npm run test:round-robin-hub-browser` (from `apps/web`) starts loopback-only
  Next/FastAPI fixtures and rehearses the UI with Playwright. It needs the project's
  Python dependencies and an installed Chromium browser; `GENERATOR_BROWSER_PYTHON`
  and `GENERATOR_BROWSER_EXECUTABLE` can select them. Authentication and persistence
  are test doubles in this rehearsal.
- `node tests/sql/round_robin_hub_atomic.mjs` runs the migration and transaction
  checks on PGlite. Set `PGLITE_MODULE` if the package is installed outside this repo.
