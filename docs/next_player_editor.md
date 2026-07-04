# Next Player Editor

This document tracks the Player Editor migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/players/editor/status`.
- FastAPI roster endpoint: `GET /admin/clubs/{club_id}/players/editor/players`.
- FastAPI player detail endpoint: `GET /admin/clubs/{club_id}/players/editor/players/{player_id}`.
- FastAPI create endpoint: `POST /admin/clubs/{club_id}/players/editor/players`.
- FastAPI basic update endpoint: `PATCH /admin/clubs/{club_id}/players/editor/players/{player_id}`.
- Next route: `/admin/players`.
- Supabase access token entry for the closed-club pilot.
- Roster/detail read, add-player, and basic player profile edits: name, overall JUPR, starting JUPR, and active/inactive status.
- Read-only league rating table and match-reference count visibility so operators can decide when to fall back to Streamlit or replay tools.

## Runtime flag

The workflow is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR=1
```

## Authorization

When enabled, roster/detail reads and player writes require a Supabase access token whose resolved role has `manage_players` permission. Role lookup uses `admin_role_assignments` through FastAPI.

## Data path

The browser sends all admin reads and writes to FastAPI. FastAPI keeps club scope, permission checks, player creation, player updates, and audit attribution in Python.

No browser-side code writes directly to Supabase tables and no rating, replay, merge, or league-rating logic is implemented in TypeScript.

## Explicitly out of scope for this foundation slice

- Player account merge.
- League-rating edits.
- Social identity linking.
- Bulk rewiring of historical matches.

Those remain Streamlit-only until replay/correction safety and approval flows are proven in the Next/FastAPI admin stack.

## Follow-up slices

- Add guarded league-rating edit API and UI after replay/correction recovery is proven.
- Add merge dry-run API before any merge write path.
- Add social identity linking only after club_people audit semantics are defined.
- Replace token-paste UX with real Next admin session/auth once the auth shell is ready.
