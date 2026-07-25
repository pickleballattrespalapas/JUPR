# Next Match Log Planning

This document tracks the Match Log migration slices for the Next.js and FastAPI stack.

## Scope shipped so far

- FastAPI read endpoint: `GET /admin/clubs/{club_id}/match-log`.
- Next route: `/admin/match-log`.
- Match list filters for type, match id, league, week tag, date range, and limit.
- Duplicate scan using the existing canonical duplicate-key logic.
- Cleanup preview that identifies the oldest row to keep and the later rows to review.
- Replay-scope guidance for affected leagues and players.
- Guarded edit endpoint: `PATCH /admin/clubs/{club_id}/match-log/edits`.
- Guarded duplicate-cleanup endpoint: `POST /admin/clubs/{club_id}/match-log/duplicates/cleanup`.
- Next apply panel that requires an operator-supplied Supabase access token and confirmation text.
- Guided notes editing and bulk staging for up to 100 visible matches.
- Atomic, idempotent edit operations with complete before/after evidence.
- Mandatory tracked Replay History execution for rating-affecting edits.
- Durable `RECOVER` workflow when a post-commit replay attempt fails.

## Runtime flags

Read/scan visibility is controlled by:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1
```

Write/apply actions are separately controlled by:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1
```

Destructive cleanup/exclusion actions require a third gate:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=1
```

That gate is deliberately dormant and forced to `0` in every staging wave and
in production. This split lets operators keep Match Log visibility and ordinary
atomic edits available while duplicate cleanup and bulk exclusion remain
disabled. Duplicate no-issue resolution is non-destructive and remains
available under the normal apply gate.

## Auth and authorization

When apply mode is enabled:

- `PATCH /admin/clubs/{club_id}/match-log/edits` requires Supabase JWT auth and `manage_matches` permission.
- `POST /admin/clubs/{club_id}/match-log/duplicates/cleanup` and bulk exclusion
  require both write gates, Supabase JWT auth, and `delete_matches` permission.
- Disabled write endpoints return `403` before auth or writes.
- Edits call the service-role-only `apply_match_log_patches_atomic` RPC and then the Python Replay History domain service when ratings are affected.
- Duplicate cleanup validates requested IDs against the current duplicate scan
  before removing rows. It is implemented but `Blocked` until the destructive
  path has atomic, idempotent recovery.

## Operator confirmations

- Match edits require confirmation text: `APPLY`.
- Duplicate cleanup requires confirmation text: `DELETE`.

## Current candidate boundary

Guided ordinary field edits couple any required Replay History job to the atomic
edit operation and do not report success before replay succeeds. Duplicate
cleanup and bulk exclusion are not candidate-authorized manual actions. Any
match-producing acceptance protocol whose required cleanup depends on bulk
exclusion must also wait; its future protocol remains documented.

## Next slice

Validate one reversible ordinary edit and one legitimate duplicate no-issue
resolution with independent readback/recovery. Before enabling destructive
actions, implement and test one atomic, idempotent cleanup/exclusion recovery
contract, then perform the preserved exact-ID recovery protocols. See
`docs/next_match_durability.md` for the transaction and recovery contract.
