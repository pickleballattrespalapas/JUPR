# Next Match Uploader

This document tracks the Match Uploader migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/match-uploader/status`.
- FastAPI submit endpoint: `POST /admin/clubs/{club_id}/match-uploader/batch`.
- Next route: `/admin/match-uploader`.
- Manual/batch score-entry rows with shared defaults for date, league, week/session, and match type.
- Official League and Pop-Up/Social contexts.
- Rating scope options: overall + league, overall only, and unrated/record-only.
- Supabase access token entry for the closed-club pilot.
- Submission feedback with inserted/skipped counts and affected player rating deltas.

## Runtime flag

The workflow is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER=1
```

This flag is separate from the legacy Score Entry MVP flag.

## Authorization

When enabled, batch submit requires a Supabase access token with `enter_scores` permission. The route resolves roles through `admin_role_assignments` and writes audit attribution through FastAPI.

## Data path

The browser submits rows to FastAPI. FastAPI normalizes the rows, loads the current player/league context, and calls the existing Python `process_matches` path through `submit_match_batch`.

No browser-side code writes directly to Supabase tables and no rating logic is implemented in TypeScript.

## Follow-up slices

- Add Streamlit-style single round-robin schedule generation.
- Add new-player creation/continue flow.
- Add court-based quick-entry helpers.
- Replace token-paste UX with real Next admin session/auth once the auth shell is ready.
