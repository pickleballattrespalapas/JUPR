# Badge retirement

Joe retired the badge feature on November 9 UTC / November 8 Mazatlan. The
separate-worker rollout is superseded. Do not provision or restart a badge
worker, replay badge jobs, or reenable badge calculations to clear historical
queue state.

Badge calculations and public badge displays are disabled in application code.
Ratings, match history, and tournament/league podium trophies remain separate
supported features. Existing badge and queue records are retained; this change
is not a data purge or a recovery of historical processing jobs.

## Staging topology and retirement

Keep exactly two DFW API machines at shared-cpu-1x / 1 GB. There is no dedicated
badge worker and no added worker compute. `fly.staging.toml` defines only the
`app` process and its original 1 GB VM. Runtime routing is
`JUPR_BADGE_WORKER_MODE=disabled`, empty `JUPR_BADGE_WORKER_CLUB_ID`, and
`JUPR_BADGE_QUEUE_LEASES=0`. The code retirement gate also prevents embedded
processing, regardless of stale feature flags.

The staging deployment validates the existing machine inventory before any
machine mutation. It accepts exactly two unchanged API machines and at most one
known badge worker. The worker must belong to `juprleagues-api-staging`, use the
reviewed scoped command, and have no HTTP service or mounted volume. Unknown
process groups, foreign images, changed capacity, or a data volume fail closed.
Only a running identified worker is stopped; an already stopped or removed
worker makes the sequence safe to repeat. The stopped inventory and exact IDs
are rechecked before the worker is destroyed without a force flag. No volume or
database is removed. Historical in-progress queue rows remain untouched.

Only after worker retirement does the workflow deploy the `app` group. Final
inventory must contain exactly two running APIs, no worker, the original API
capacity, and one immutable image. Exact health must attest the candidate SHA,
disabled badges, disabled worker routing, and dry-run email. The existing
Fly/Vercel same-SHA handoff still gates `ready_for_manual_testing`; worker-cycle
logs and badge queue migration checks are no longer release prerequisites.

The emergency staging write-disable workflow reuses this deployment path. It
must also leave badges disabled and must never recreate the retired worker.

## Preserved migration and queue history

The additive lease migration remains in source and staging. It was applied only
to staging project `sijpxjxvdtrehmqvirfi` as hosted migration `20261109000605`,
logical name `badge_queue_worker_leases`, from source file
`20261108234006_badge_queue_worker_leases.sql`. No rollback migration is needed.
Do not apply this unused worker migration to production as part of retirement.

Unresolved jobs do not block badge retirement. They are retained evidence, not
instructions to replay matches or awards. No automatic expired-lease recovery
is enabled. Historical award data is preserved for any future explicit product
decision.

Keep `badge_program_state.revision` and the `program_badges_dirty` triggers.
Despite their historical names, the public leaderboard cache uses that revision
to invalidate data across API machines. Disabling those triggers would serve
stale standings. They do not evaluate or award badges; the retired consumers
ignore pending badge work. No schema migration is part of this retirement.

## Production scope

The staging decommissioning script is deliberately fixed to the staging Fly
app and must never be pointed at production. A production retirement candidate
must preserve its existing email, rating, league, and tournament policies and
remove the production deployment's historical badge-activation step. The prior
worker deployment proposal and API resizing proposal are superseded. Production
rollout remains subject to the repository's explicit release authorization.
