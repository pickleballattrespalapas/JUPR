# RR Hub reliability repairs

Base: staging `27c962fa1e55b628caf8cf9a761101bd7cd90233`. Scope covers the
October 22 incident and adversarial audit. Production records are unchanged.

## Behavior

- An explicit standard-RR submission resolves two blank scores as `not_played`.
  One blank is invalid. Unplayed games have null scores and do not count as
  played results, wins, losses, points, rating entries or award evidence.
- Saving does not require fabricated 1–0 results. Moneyball still requires its
  fixed rotation. A semifinal that feeds a final requires a winner.
- Saved RR results can be cleared to a skipped round with a reason, before
  playoff/publication locks. Later rounds retain their identities and results.
- Top-four playoff mode creates one championship game. Standings retain the
  chosen ordering across Finish, public views, playoff seeding and awards.
- One person cannot occupy duplicate active seats in a group. Explicit name
  linking, transfer and rejoin preserve history and qualify court status by
  group. Atomic transfer either updates both memberships or neither.
- Unrelated arrivals preserve waiting lineups; returning a started game to the
  queue preserves its match ID and players. Pause/resume can form legal games
  from the available roster.
- Credential renewal and deferred refresh preserve typed scores. An uncertain
  save retains its exact body/key across remounts; late replies cannot erase a
  newer recovery record.
- Known validation failures release operation leases; unknown outcomes retain
  recovery. Durable receipts are committed atomically with mutations and survive
  later edits. Lease scope includes the session; fingerprints include the round.
- Response-decoding failures in public/shared saves require receipt proof before
  being reported as successful or rejected. A saved change is not classified as
  rejected merely because its response could not be decoded.
- Hub no-op checks preserve versions. Business conflicts use terminal `PT409`
  instead of retryable serialization errors. Duplicate/unknown match IDs,
  malformed scores and invalid court payloads are rejected before mutation.
- Admin controls reflect actual session capabilities. Legacy public mutation
  routes reject Generator sessions so they cannot bypass Hub/publication locks.
- Authorized staff can submit completed public sessions for approval even when
  their play controls are read-only. Pending, approved and published results
  remain locked. Mobile score inputs remain visible before the explanatory
  blank-game guidance.

## Audit coverage

| Findings | Repair and verification |
| --- | --- |
| B01–B04 and retry storm | Atomic receipts, scoped leases, terminal conflicts, no-op versions; service fault injection and SQL rollback/replay |
| B05–B07 | Draft preservation and exact retry; real React lifecycle and browser response loss |
| B08–B12 | Duplicate guards, manual identity resolution, rejoin, transfer and membership/court display |
| B13–B16 | Stable standings, unplayed games, single championship, reason-required saved-round skip |
| B17–B18 | Waiting and released-started lineup stability with availability changes |
| B19 | Explicit admin capabilities for standalone and Hub-linked public sessions |
| B20–B24 | Strict score, match and court request validation |
| Additional review | Legacy-write bypass, public standings, badge consumers, URL-round replay scope, late-response recovery race and response-decoding uncertainty |

## Release boundary

The new migration must be applied to staging before deploying the API. The
canonical staging handoff must attest the same web/API SHA before manual testing
is advertised. Production deployment remains a separate action.

New receipts are not retroactive proof of old uncertain writes. Historical
placeholder results, incomplete playoff details and old stuck operations require
individual review; this change does not guess or rewrite them.

## Verification

- The broad Python regression selection passes: 1,279 tests across 81 files,
  with no failures or skips.
- The exact adversarial CI Python selection passes: 332 tests with expected
  failures disabled, including 48 seeds of 100 scheduling actions (4,800 total).
- After the final response-decoding and concurrent replay-read review, all 148
  affected public/shared/Hub operation tests pass, including the new failure
  injection cases. The broader counts above precede these additional cases.
- The full web component suite passes, including 18 Hub operations scenarios;
  21 additional React save-recovery regressions pass.
- The optimized Next production build, TypeScript, lint (zero errors), migration
  source/version guards and whitespace checks pass. Existing dependency,
  framework and lint warnings remain.
- Actual PostgreSQL-compatible SQL tests pass for Hub transactions, identities,
  check-ins and the generator journal, plus 11 adversarial receipt, rollback,
  conflict, lease and permission scenarios.
- The complete loopback browser/API journey passes: two independent groups,
  court assignment and reservation, pause/resume, saved standings, retained
  drafts after credential renewal, one real score plus a blank game, exact retry
  after a lost response, atomic transfer, rejoin and explicit not-played outcome.
- Production-build browser acceptance also passes for organizer/staff submission,
  schedule navigation with mobile scores visible, recap editing, Hub creation,
  and 18 identity/group-isolation checkpoints. The latter retains exact
  match/bye membership and protected-lineup assertions while recognizing optional
  per-round roster overrides and current group-qualified status text.
- The deployment migration manifest requires the reviewed recovery migration;
  its missing-migration check and the remaining API safety, tournament/league,
  and parity suites pass (293 + 245 + 43 tests, plus 8 subtests).

These are synthetic/local checks. They do not attest a deployed staging build,
production load, real-account authentication or PostgREST retry timing.

## Current delivery status

The implementation is prepared on a feature branch targeting `staging`.
Applying `20261023052802_round_robin_operation_recovery.sql` through the staging
Supabase migration connector failed twice with `Invalid or expired requestState`.
A read-only check after each attempt confirmed the receipt table and migration
were still absent. Do not merge/deploy the dependent API until this migration is
successfully applied to the permitted staging project and verified.

No production deployment, historical score repair, player notification or
production database change is included. Staging is not yet ready for manual
testing of this change.
