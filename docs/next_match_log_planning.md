# Next Match Log Planning

This document tracks the first Match Log migration slice for the Next.js and FastAPI stack.

## Scope shipped in this slice

- FastAPI read endpoint: `GET /admin/clubs/{club_id}/match-log`.
- Next route: `/admin/match-log`.
- Match list filters for type, match id, league, week tag, date range, and limit.
- Duplicate scan using the existing canonical duplicate-key logic.
- Cleanup preview that identifies the oldest row to keep and the later rows to review.
- Replay-scope guidance for affected leagues and players.
- Correction planning metadata for future audited edit flows.

## Non-goals

This slice does not write, edit, or remove match records. It also does not replay ratings. It is a planning and visibility step so staff can inspect issues before we enable operational mutation from the Next stack.

## Runtime flag

The route returns fallback instructions until the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1
```

The endpoint is intentionally useful when disabled: it lets the Next page render a clear Streamlit fallback state without requiring Supabase configuration.

## Next slice

The next Match Log slice should add an authenticated, audited apply flow that calls Python domain services instead of implementing write logic in TypeScript.
