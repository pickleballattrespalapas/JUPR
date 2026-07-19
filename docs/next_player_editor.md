# Next Player Editor

This document tracks the Player Editor migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/players/editor/status`.
- FastAPI roster endpoint: `GET /admin/clubs/{club_id}/players/editor/players`.
- FastAPI player detail endpoint: `GET /admin/clubs/{club_id}/players/editor/players/{player_id}`.
- FastAPI create endpoint: `POST /admin/clubs/{club_id}/players/editor/players`.
- FastAPI basic update endpoint: `PATCH /admin/clubs/{club_id}/players/editor/players/{player_id}`.
- Guarded merge preview/execute endpoints: `POST /admin/clubs/{club_id}/players/editor/merge/preview` and `POST /admin/clubs/{club_id}/players/editor/merge`.
- Merge recovery lookup: `GET /admin/clubs/{club_id}/players/editor/merge/{operation_id}`.
- Pre-replay compensation and replay-evidence endpoints under the merge operation.
- Next route: `/admin/players`.
- Supabase Auth admin session for the closed-club pilot.
- Roster/detail read, add-player, and basic player profile edits: name, overall JUPR, starting JUPR, and active/inactive status.
- Guarded league-rating edits, Club Social identity linking, and exact-name auto-linking.
- Match-reference, league-conflict, social-link, and source/target match-collision visibility before merge.

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

## Atomic merge and recovery contract

The merge preview computes a fingerprint over the exact source/target player state, affected match IDs, league move/delete plan, and social links. Execute requires that fingerprint. PostgreSQL locks and recomputes the same state before changing anything; stale previews and matches containing both source and target are rejected.

The `server_merge_player_accounts` RPC runs match rewiring, league conflict resolution, social relinking, source deactivation, recovery-ledger insertion, and audit insertion in one transaction. It serializes against tracked replay-job writes and refuses to start while a club replay is pending or running. Its execute grant is service-role only; browser, anon, and authenticated roles have no RPC or ledger access.

Every committed merge is `merged_pending_replay`. The operator must either:

1. Run a tracked `ALL (Full System Reset)` Replay History job created after the merge and attach its succeeded job UUID; or
2. Before replay, type `COMPENSATE MERGE` to restore pre-merge player, match, league business fields, and social links. Compensation refuses to overwrite newer edits; trigger-maintained timestamps may advance.

The recovery panel links the tracked Streamlit Admin Tools replay as an explicit fallback until the Next Replay History surface returns its replay-job UUID. Both recovery paths must write the same server-side `replay_jobs` ledger; the browser never manufactures evidence.

## Follow-up slices

- Run staging manual tests for stale previews, collision blocks, replay evidence, and pre-replay compensation before enabling production merge writes.
