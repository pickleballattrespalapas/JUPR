# Engagement badge retirement candidate

This candidate is based on production commit
`a7a7c0bacaceef3c2b0a12bee71cdf9b535306c1`. It retires engagement badges across
public pages, admin controls, saved recaps, player update emails, legacy pages,
and background processing. Ratings, match history, and formal league and
tournament trophies remain available. Historical badge records remain stored.

## Runtime and release behavior

- The shared badge policy is disabled in code. Existing environment flags cannot
  reactivate it. Production's separate badge feature gate also returns false.
- Both embedded-worker startup and direct polling return before database access.
  Queue producers, consumers, recomputes, and notification award dependencies
  stop. Rating and match-history updates continue without waiting for badges.
- Match exclusions complete their durable badge checkpoint with an audited
  preservation result. They do not calculate, insert, revoke, or delete awards.
  The existing bounded catalog lookup remains necessary for the correction RPC
  contract; it does not load player award history.
- The production release disables `JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS` and
  removes the post-deploy historical badge activation command. The exact previous
  flag set is retained as `pre_badge_retirement` for snapshot verification and
  existing rollback handling. Other production feature and email gates are
  unchanged.
- The production web build waits for the API to attest badges disabled, worker
  mode disabled, leases disabled, and the badge diagnostics flag disabled. It
  retains all other existing readiness checks and keeps the current site active
  while the API deployment catches up.
- API machines remain shared CPU, one CPU, 1 GB. This candidate adds no worker
  process, machine, lease machinery, or capacity upgrade.

## Data preservation

There is no schema migration, ledger change, or migration-contract fingerprint
change. The twelve existing program dirty triggers and
`badge_program_state.revision` remain because they invalidate leaderboard caches
across API machines. Historical awards and formal trophy award paths remain.

## Validation and authorization

Local validation covers the retired APIs and workers without a database client,
public projection cleanup, archived badge behavior under an explicit test-only
fixture, rating/email continuity, formal trophy awarding and idempotence, and
correction progress against the existing SQL functions. Production-specific
notification links, profile privacy behavior, and unrelated frontend behavior
are preserved by applying only retirement changes.

This document records a prepared candidate, not a deployment. Production was not
read or changed while preparing it. Merging, pushing, dispatching, or deploying
production still requires separate authorization for that exact action under
`AGENTS.md`. Use the existing guarded production workflow after approval; do not
run historical badge activation or alter the preserved database triggers.
