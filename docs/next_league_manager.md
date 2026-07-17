# Next League Manager

This document tracks the League Manager migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/league-manager/status`.
- FastAPI league list endpoint: `GET /admin/clubs/{club_id}/league-manager/leagues`.
- FastAPI league draft creation endpoint: `POST /admin/clubs/{club_id}/league-manager/leagues`.
- FastAPI configuration-only duplication endpoint: `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/duplicate`.
- FastAPI lifecycle endpoint: `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/lifecycle`.
- FastAPI league detail endpoint: `GET /admin/clubs/{club_id}/league-manager/leagues/{league_name}`.
- Next route: `/admin/league-manager`.
- League status, K-factor, min-games, schedule preview, court-board/rules/awards configuration visibility, and standings snapshot.
- Guarded draft creation and configuration-only duplication. Duplication never copies roster membership, standings, results, lifecycle dates, or issued awards.
- Guarded start, pause, resume, end, and archive transitions. Generic settings writes cannot change lifecycle status.
- State-aware settings locks: full configuration in draft, description-only while active/paused, and read-only after ended/archived.
- Stored Supabase admin session for the closed-club staging pilot.

## Runtime flag

The workflow is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1
```

## Authorization

When enabled, League Manager reads and writes require a stored Supabase admin session whose resolved role has `manage_matches` permission. Role lookup uses `admin_role_assignments` through FastAPI.

## Data path

The browser sends admin reads and guarded writes to FastAPI. FastAPI keeps club scope, permission checks, audit attribution, and league read-model normalization in Python.

No browser-side code writes directly to Supabase tables and no league movement, schedule generation, score submission, award minting, or rating logic is implemented in TypeScript.

## Current safety boundaries

- Create and duplicate operations always produce inactive drafts.
- Duplication copies only whitelisted league configuration and never roster, result, lifecycle-date, or issued-award data.
- Lifecycle actions enforce `draft → active`, `active → paused|ended`, `paused → active|ended`, and `ended → archived` transitions and require an action-specific confirmation phrase.
- Ending a league freezes its lifecycle state; award preview, overrides, and optional badge minting remain in the separate Awards workflow.
- Settings writes are checked against the current lifecycle state on FastAPI, not only disabled in the browser. Stale saves are rejected.
- Mutations require explicit confirmation text and an authorized club-scoped admin session.
- Staging requires successful API audit logging. Lifecycle and settings mutations are rolled back if their required audit write fails; Streamlit remains the production fallback.
- Rating, match, movement, and award calculations remain in Python services.

## Follow-up slices

- Validate draft create/duplicate, live rounds, court movement, awards close, and Match Log/Replay recovery against isolated staging data.
