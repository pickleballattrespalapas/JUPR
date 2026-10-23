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
- Hub no-op checks preserve versions. Business conflicts use terminal `PT409`
  instead of retryable serialization errors. Duplicate/unknown match IDs,
  malformed scores and invalid court payloads are rejected before mutation.
- Admin controls reflect actual session capabilities. Legacy public mutation
  routes reject Generator sessions so they cannot bypass Hub/publication locks.

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
| Additional review | Legacy-write bypass, public standings, badge consumers, URL-round replay scope and late-response recovery race |

## Release boundary

The new migration must be applied to staging before deploying the API. The
canonical staging handoff must attest the same web/API SHA before manual testing
is advertised. Production deployment remains a separate action.

New receipts are not retroactive proof of old uncertain writes. Historical
placeholder results, incomplete playoff details and old stuck operations require
individual review; this change does not guess or rewrite them.

Verification results and release status will be finalized after integration.
