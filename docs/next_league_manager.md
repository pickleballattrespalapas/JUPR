# Next League Manager

This document tracks the League Manager migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/league-manager/status`.
- FastAPI league list endpoint: `GET /admin/clubs/{club_id}/league-manager/leagues`.
- FastAPI league detail endpoint: `GET /admin/clubs/{club_id}/league-manager/leagues/{league_name}`.
- Next route: `/admin/league-manager`.
- Read-only league status, K-factor, min-games, schedule preview, court-board/rules/awards configuration visibility, and standings snapshot.
- Supabase access token entry for the closed-club pilot.

## Runtime flag

The workflow is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1
```

## Authorization

When enabled, league list/detail reads require a Supabase access token whose resolved role has `manage_matches` permission. Role lookup uses `admin_role_assignments` through FastAPI.

## Data path

The browser sends admin reads to FastAPI. FastAPI keeps club scope, permission checks, and league read-model normalization in Python.

No browser-side code writes directly to Supabase tables and no league movement, schedule generation, score submission, award minting, or rating logic is implemented in TypeScript.

## Explicitly out of scope for this foundation slice

- League setup or metadata writes.
- Court-board roster movement.
- Live ladder round generation or movement.
- League-night score submission.
- End-of-league awards and badge minting.

Those remain Streamlit-only until Match Log, Replay History, Match Uploader, and Player Editor foundations are proven in staging and the operator recovery path is clear.

## Follow-up slices

- Add guarded league setup/edit APIs after schema and audit contracts are reviewed.
- Add roster/court-board dry-run previews before any roster movement write path.
- Add live ladder round planning as a preview-only API before score writes.
- Replace token-paste UX with real Next admin session/auth once the auth shell is ready.
