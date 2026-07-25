# Next Replay History

This document tracks the guarded Replay History workflow for the Next.js and FastAPI stack.

## Scope

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/replay-history`.
- FastAPI run endpoint: `POST /admin/clubs/{club_id}/replay-history`.
- Next staff route: `/admin/replay-history`.
- Replay options include `ALL (Full System Reset)` plus leagues discovered from metadata or existing rating/match rows.
- Replay calls the Python `replay_history` domain function server-side.
- Replay writes are audit-attributed and flagged for review.

## Runtime flag

Replay is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_REPLAY=1
```

The status endpoint is useful while disabled: it lets the Next page render fallback instructions without requiring database configuration.

## Authorization

When enabled, replay requires a Supabase access token with a role that has `run_replay` permission.

## Operator confirmation

The operator confirms an accessible Yes/No dialog. The Next client supplies the
internal `REPLAY` API safeguard only after the operator chooses **Yes, run
replay**; there is no typed-confirmation field.

## Behavior

- League replay rewrites match snapshots and rebuilds `league_ratings` for the selected league.
- Full reset also updates overall player rating, wins, losses, and match count.
- The reviewed `20260725181500_singles_replay_recovery` migration preserves each
  player's legacy singles aggregate as a baseline. New singles rows can then be
  marked `singles_replay_managed`; full reset rebuilds their rating, W/L/match
  count, last-game timestamp, and match snapshots from that baseline. The
  migration is in the candidate repository but is not yet formally applied and
  accepted in staging.
- Match Log refuses to exclude a legacy singles row that cannot be recovered
  deterministically. Its duplicate-cleanup and bulk-exclusion subflows are
  implemented but remain `Blocked` behind
  `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0` until they have atomic,
  idempotent recovery. Ordinary atomic edits and duplicate no-issue resolution
  remain available under the normal Match Log apply gate.
- Unrated singles are retained as official history rows with zero rating or
  counter change.
- Challenge Ladder, Moneyball, and JUPR Live recovery links may filter Match Log
  by an exact context. Replay History remains a global rebuild surface; its link
  and operator copy never claim context-scoped replay.
- Replay remains in Python; TypeScript does not implement rating or replay logic.

## Next step

Apply and verify the singles replay migration in staging, then validate the
Tournament Operations compare-and-swap official-singles publisher without
claiming manual recovery acceptance while destructive Match Log exclusion is
dormant. Direct Match Uploader singles stays `Blocked` behind
`JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES=0` until its writer is atomic.
Preserve the rated/unrated submit→exact-exclude→full-replay protocol for a later
candidate after both gates are reviewed and enabled. Durable replay job history
and idempotent retry are already exposed; a stricter two-person approval flow
remains a later policy decision.
