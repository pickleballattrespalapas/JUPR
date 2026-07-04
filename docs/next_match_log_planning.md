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

## Runtime flags

Read/scan visibility is controlled by:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1
```

Write/apply actions are separately controlled by:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1
```

This split lets operators keep Match Log visibility enabled while leaving edits and duplicate cleanup disabled.

## Auth and authorization

When apply mode is enabled:

- `PATCH /admin/clubs/{club_id}/match-log/edits` requires Supabase JWT auth and `manage_matches` permission.
- `POST /admin/clubs/{club_id}/match-log/duplicates/cleanup` requires Supabase JWT auth and `delete_matches` permission.
- Disabled write endpoints return `403` before auth or writes.
- Edits call the Python `apply_bulk_match_edits` domain service.
- Duplicate cleanup validates requested IDs against the current duplicate scan before removing rows.

## Operator confirmations

- Match edits require confirmation text: `APPLY`.
- Duplicate cleanup requires confirmation text: `DELETE`.

## Non-goals

This slice does not run rating replay from Next. It records correction actions and returns replay guidance so staff can run the approved replay path after review.

## Next slice

The next Match Log slice should add replay orchestration or a safer replay request workflow after edit/cleanup use is validated in the closed-club pilot.
