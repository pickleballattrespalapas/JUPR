# Round Robin Hub

The hub shares physical courts among existing Round-Robin Generator sessions. Each
session keeps its own roster, scoring mode, rating mode, Moneyball rules, standings,
playoffs, exports, sharing, and submission/approval workflow.

## Accepted operating rules

- An event can contain multiple independent groups and named courts.
- Standard hub round robins prepare the current round and one complete next
  round from the whole active group. Results and court availability release
  those planned matches; they do not choose a new game from the four players
  who happened to finish first. Scores reuse the next plan until the current
  round settles. A new successor is then calculated once.
- Roster changes rebuild replaceable next-round lineups with partner, opponent,
  and bye history. Started, scored, assigned, reserved, and manually locked
  games stay protected. A departure may repair a manual future lineup containing
  that player; it does not erase results or postpone departure merely because
  an unstarted game was manually chosen. Moneyball and playoff plans retain
  their existing fixed-format rules.
- Legacy distant previews are not treated as announced games. On the first
  write they are removed if replaceable; actual commitments and explicit manual
  locks remain. Reads never migrate or rewrite a plan.
- Prepared later-round matches may enter the queue once their players have
  resolved earlier games. Results settle in round order; groups advance independently.
- More matches than courts are supported. Queued matches wait for a court; this
  waiting is distinct from a scheduled bye.
- Queue order starts with the longest-waiting player in each eligible matchup,
  then the other players' waiting times. Waiting includes time spent on bye and
  survives round changes. Completing a match starts a new waiting period.
- Pausing needs no reason. It removes a player from eligibility and cancels their
  next-court reservation. It preserves already-accrued waiting time, freezes the
  clock while away, and allows an already-playing match to finish.
- In standard round robins, a paused player's unstarted game uses an eligible
  player on bye when possible, choosing the longest waiting player first. Future
  preview games alone do not make a substitute busy. Current court assignments,
  reservations, earlier unfinished games, and unresolved identities do.
- When a direct replacement is unavailable, only the games affected by pauses
  are regrouped. Available players left over receive byes and retain their wait;
  paused absences do not earn bye credit. Unaffected games, started games, and
  scores stay intact. A group with too few available players waits for a return
  instead of generating empty rounds. Moneyball and playoff lineups remain fixed.
- Availability repairs run with hub writes, including Pause, score/release,
  Ready again, and Fill open courts. Reading the board never changes a lineup.
  Ready again restores eligibility without undoing substitutions. Original
  affected games are retained in the round's pause-adjustment audit metadata.
- Dedicated courts are exclusive to their assigned group. That group can also
  use shared courts, including manual moves and next-game reservations. Automatic
  assignment uses its available dedicated courts first, then shared courts;
  shared-court queue priority still follows player waiting time.
- Tournament-style controls assign to the next open court, choose a court, reserve
  the next match on an occupied court, move a match, return it to the queue, and
  save a score. No player or matchup can claim two courts.
- Saving a match is durable and releases its court immediately. Its result does
  not count in standings until every match in that round is resolved. Unscored
  sessions use Match played. A completed round still offers standings, next round,
  finish, and supported playoff controls.
- The admin standings view also offers explicitly labelled provisional totals
  from all scored games, including partial rounds. It names missing games and
  identifies the last completed round. Departed players retain their results.
- Exact two-team editing exchanges displaced waiting players atomically and
  updates byes. Intentional lineups are locked against later ordinary planning.
  Several players can be marked finished together; each departure displays its
  effective round, and the remaining schedule is repaired once.
- Finish can name the current or prepared next round. It cannot discard a
  started, reserved, or scored later game. No further rounds are generated after
  that boundary, and the group closes when its remaining games are resolved.
- Extra games can borrow four ready players from attached standard groups without
  moving their memberships. An extra game is a separate one-round exhibition:
  its score does not affect home-group standings, ratings, or publication. The
  same court and person-identity protections apply. An interrupted creation
  retries the same hidden draft rather than creating another game.
- Standard RR roster changes preserve completed and assigned matches and adapt
  unstarted matches. Moneyball retains its frozen inputs and substitution rules.
- Ending a session or starting playoffs must resolve pending play first. Rating
  publication continues through the existing approval workflow.

## Player identity and repeated names

Linked club profiles identify people by their profile ID, regardless of the name
shown in a group. A profile cannot occupy or reserve two courts. Two distinct
profiles may share a name without sharing availability.

Typed guest names belong to stable participants in their original sessions. When
a guest name matches another participant in the event, the Players review asks
whether they are the same person or different people. The organizer can confirm
all unresolved names as different people together. Decisions are saved with the
event and can be reviewed later; refreshing does not discard them. Batch review
preserves previously confirmed same-person links and authoritative profile links.

Confirmed same-person entries share court eligibility, waiting time, and pause
status. Confirmed different people keep those independently and receive group
labels wherever their names would otherwise be ambiguous. Original round-robin
scores and standings remain separate. Changing a display name does not itself
merge players or reset their waiting time.

Unresolved identities block only affected waiting matches. Existing matches can
finish, scores can be saved, and unrelated matches can continue. Correcting an
identity decision must not create overlapping court assignments or reservations.
Existing events receive the same review without discarding their court state.
Separating guests can preserve their current assignments. Merging people requires
releasing affected assignments first. To split a shared identity, select every
entry belonging to that person; repeated occurrences of one club profile remain
linked. A group that anchors a shared identity must stay attached until that
relationship is explicitly resolved. Completing the group can leave it attached.

Manual guests may play casually and submit scored results for review. Before
official publication, the approval workflow still requires the correct club
profile for every participant used in a result, including unrated statistics.
Initials, surnames, nicknames, and group labels help distinguish people visually;
none of them substitutes for a profile link when publishing results.

## Persistence and concurrency

The hub owns assignments and availability; existing `live_sessions` records own
match results. A service-role-only PostgreSQL RPC locks the hub and its complete
session read set, compares versions, and commits court state and session state in
one transaction. An operation receipt makes exact retries idempotent. Court and
player leases prevent overlapping assignments across active hubs in a club for
known profiles. Guest identity decisions apply within the named hub event.

Existing admin, public organizer, and shared organizer writes for attached sessions
route through the same transaction. A database trigger rejects direct changes to
attached event data outside that authority. Public responses never contain private
Moneyball calibration data or organizer tokens.

## Verification scenarios

1. Attach two RRs with more matches than open courts; ensure a prepared next-round
   game can enter the queue when its four players have finished earlier
   obligations, even while unrelated games or the other RR remain in progress.
2. Save one match, refresh, confirm its court is available and its score survives,
   while that RR's standings remain unchanged until the last match finishes.
3. Preserve bye waiting time across advancement; pause and resume without a reason.
4. Dedicate courts 1–2 to one RR and verify other RRs cannot use or reserve them.
   Verify that RR can assign, move, and reserve games on shared courts, and that
   automatic assignment prefers its available dedicated courts before overflow.
5. Attempt concurrent assignment/score operations and exact retries; no double
   booking, duplicated result, or partial court release is permitted.
6. Add/remove/substitute players with queued, assigned, and completed matches;
   protect assigned/completed lineups and keep the hub and original page synchronized.
7. Run scored, unscored, Moneyball, rated/unrated, finish, playoff, and submission
   flows using the same individual-session controls.
8. Attach two groups with the same typed names. Resolve them as different people
   and use courts concurrently; pause one guest without pausing the other.
9. Confirm a repeated guest as the same person, including a guest matching a
   linked profile, and verify shared availability and no overlapping reservation.
10. Preserve selected club profiles in a roster that also contains manual guests.
    Refresh identity decisions and retry an uncertain save without applying it twice.
11. Enroll 15 and 29 active players incrementally. Verify full-group next rounds,
    rotating byes, stable plans under different score-arrival orders, and no
    redundant distant schedule generation.
12. Hold automatic check-in processing after a successful board read. The courts
    display immediately, processing is visible, and an enabled score submission
    is never silently dropped. After a timeout, retry the exact saved action.
13. Edit a next-round team using a bye player, then add/remove another player or
    seat a late arrival through the original generator. Preserve the explicit
    lineup unless a player in it leaves.
14. Create an exhibition with borrowed guests and club profiles. Verify atomic
    attachment, exact retry, source memberships/results, and overlap rejection
    using the database's actual person-identity and lease functions.

## Timing and recovery

Hub requests bound both response headers and body reads. A timeout does not prove
that a command failed; the exact request, expected version, and idempotency key
remain available for safe replay. Background check-in writes use the same visible
pending-operation path. The event list polls less often than the selected board.

Successful commands log action, group/roster counts, and separate read, preparation,
commit, response, and total durations. Names, scores, credentials, and request
payloads are not included in these timing logs. Local CPU benchmarks must not be
reported as measured production network/database response times.

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
- `npm run test:round-robin-hub-identities-browser` rehearses duplicate-name review,
  separate and shared guests, profile protection, mixed rosters, and saved decisions
  using loopback-only browser/API fixtures.
- `node tests/sql/round_robin_hub_participant_identities.mjs` verifies database
  identity derivation, assignment eligibility, retries, legacy upgrades, and grants.
