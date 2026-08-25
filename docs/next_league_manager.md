# Next League Manager

This document tracks the League Manager migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/league-manager/status`.
- FastAPI league list endpoint: `GET /admin/clubs/{club_id}/league-manager/leagues`.
- FastAPI league draft creation endpoint: `POST /admin/clubs/{club_id}/league-manager/leagues`.
- FastAPI configuration-only duplication endpoint: `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/duplicate`.
- FastAPI lifecycle endpoint: `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/lifecycle`.
- FastAPI league detail endpoint: `GET /admin/clubs/{club_id}/league-manager/leagues/{league_name}`.
- FastAPI unsaved schedule preview endpoint: `POST /admin/clubs/{club_id}/league-manager/leagues/{league_name}/schedule/preview`.
- FastAPI read-only league print model: `GET /admin/clubs/{club_id}/league-manager/leagues/{league_name}/printout?week_num=...`.
- FastAPI read-only previous-month ranking model: `GET /admin/clubs/{club_id}/league-manager/top-players-printable?limit=...`.
- Recoverable League Awards endpoints under `.../leagues/{league_name}/awards`: state, freeze, persisted preview, documented overrides, verified mint, and archive.
- Next route: `/admin/league-manager`.
- League status, K-factor, min-games, schedule preview, court-board/rules/awards configuration visibility, and standings snapshot.
- Guarded draft creation and configuration-only duplication. Duplication never copies roster membership, standings, results, lifecycle dates, or issued awards.
- Guarded start, pause, resume, end, and archive transitions. Generic settings writes cannot change lifecycle status.
- State-aware settings locks: full configuration in draft, description-only while active/paused, and read-only after ended/archived.
- Guided draft settings for the Streamlit overview, schedule, courts, competition, ratings, and award-category fields. Compatible extension keys are preserved when the structured form saves.
- Server-normalized settings with bounded JSON shape/size, strict dates, times, numeric ranges, timezone characters, court constraints, and award-depth validation.
- Authenticated, read-only schedule preview and ICS generation for unsaved form values. Preview requests do not update league metadata or write an activity-log row.
- Authenticated league detail includes a calendar-safe ICS export that matches the schedule preview and omits blackout dates.
- Authenticated league detail includes server-derived capabilities and validation errors/warnings. The browser consumes the allowed lifecycle actions and roster lock instead of inventing them.
- League print output combines schedule, roster, standings, weekly rating-gain and win leaders, and configured season Top Performers. Stored rating snapshots are authoritative; Python rating replay is the explicit fallback when a scored row lacks complete snapshots.
- Top Active Players matches the Streamlit export policy: active players only, at least 10 scored games in the previous UTC calendar month, ranked by current JUPR with games/wins as deterministic tie-breakers.
- Stored Supabase admin session for the closed-club staging pilot.
- Python-authoritative League Live roster/bench suggestion, round planning, deterministic movement, reviewed overrides, and resumable next-round state at `/admin/league-manager/live`.
- Staging-only League Live score-first all-match publish with durable intent, deterministic official-match contexts, response-loss reconciliation, rating readback, unusual-score review acknowledgements, separately approved post-publish movement, real-player guests, CSV exports, and verified compensation.

## Runtime flag

The workflow is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN=1  # staging pilot only
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT=1  # separate staging-only write gate
JUPR_ENV=staging
```

League Awards writes have a separate staging-first gate and require the FastAPI service-role credential:

```text
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=1
SUPABASE_SERVICE_ROLE_KEY=<server-only>
```

Never put either server credential/key in Vercel or a `NEXT_PUBLIC_` variable. Production keeps the Awards write gate off until `docs/league_awards_parity_evidence.md` is signed off.

## Authorization

When enabled, League Manager reads and writes require a stored Supabase admin session whose resolved role has `manage_matches` permission. Role lookup uses `admin_role_assignments` through FastAPI.

## Data path

The browser sends admin reads and guarded writes to FastAPI. FastAPI keeps club scope, permission checks, audit attribution, and league read-model normalization in Python.

No browser-side code writes directly to Supabase tables and no league movement, schedule generation, score submission, award minting, or rating logic is implemented in TypeScript. League Live additionally requires `SUPABASE_SERVICE_ROLE_KEY` on FastAPI; its Supabase tables revoke browser roles.

Core League Manager mutations additionally require `SUPABASE_SERVICE_ROLE_KEY` on FastAPI. The publishable/anonymous key can never be used as a mutation fallback. Supabase service/secret credentials remain server-only and are never exposed through `NEXT_PUBLIC_*` variables.

## Current safety boundaries

- Create and duplicate operations always produce inactive drafts.
- Duplication copies only whitelisted league configuration and never roster, result, lifecycle-date, or issued-award data.
- Lifecycle actions enforce `draft → active`, `active → paused|ended`, `paused → active|ended`, and `ended → archived` transitions and require an action-specific confirmation phrase.
- Ending a league freezes its lifecycle state; award preview, overrides, and optional badge minting remain in the separate Awards workflow.
- League Awards stores every wizard revision in `leagues_metadata.end_awards`, rejects stale previews, persists override reasons, fails closed when any required top-performer badge definition is unavailable, and verifies every expected badge row after mint before reporting success.
- A failed/interrupted mint remains retryable with its idempotency key; archive remains blocked until mint is verified.
- Settings writes are checked against the current lifecycle state on FastAPI, not only disabled in the browser. Stale saves are rejected.
- Schedule saves derive date tags in Python while preserving existing skill tags. Unsaved previews run the same normalized Python schedule/ICS logic without mutating staging data.
- Mutations require explicit confirmation text and an authorized club-scoped admin session.
- Create, duplicate, lifecycle, settings, and roster endpoints fail closed with `503` if FastAPI has no service-role credential.
- Roster writes first verify a club-scoped league and player, reject ended/archived leagues and idempotency mistakes, preserve history during reactivation, use club/league/player filters on every update, and compensate when a required audit write fails.
- Settings, roster, and lifecycle writes reject inconsistent `status`/`is_active` pairs; lifecycle compare-and-set also matches both values before updating, so stale or corrupt state cannot silently mutate.
- Staging requires successful API audit logging. Lifecycle and settings mutations are rolled back if their required audit write fails; Streamlit remains the production fallback.
- Rating, match, movement, and award calculations remain in Python services.
- League Live snapshots use `expected_updated_at`. Official scores publish before movement; the later movement apply recomputes the Python plan from persisted official scores and requires its `expected_operation_key`. Manual movement and bench overrides require server validation and a reason.
- The canonical League Live table contract is `supabase/migrations/20260719182921_league_live_domain_contract.sql`. Apply it to staging before enabling the Live domain flag.
- The score-publish path refuses partial rounds, stale sessions, changed score contexts, unusual scores without explicit review, and reused idempotency keys with different fingerprints before its first mutation. Direct round submission is disabled while the submit gate is on.
- Every publish receives deterministic UUIDv5 match contexts. A retry first verifies those contexts: all rows reconcile without republish, some rows stop in `recovery_required`, and zero rows may retry under the same key.
- Intent and completion audits carry the durable operation ID, request fingerprint, and recognizable audit marker. A completion-audit failure reports that the operation may already have completed and instructs the operator to reload/reconcile rather than blindly publish.
- Guest creation persists an idempotency record and a real club player. Existing names are rejected; abandoned guests are recovered through Player Editor.
- Compensation never deletes or rewrites matches. After Match Log soft-exclusion and Replay History recovery, FastAPI verifies zero active deterministic contexts, requires the recovery reference/reason, then records `compensated`.
- The canonical publish coordination contract is `supabase/migrations/20260719190954_league_live_publish_reconciliation.sql`. Both migrations, canonical match context columns, and the staging-only submit flag are hard preconditions.

## Follow-up slices

- Validate draft create/duplicate, the complete League Live manual test book (including response loss, partial recovery, guests, rating readback, exports, and compensation), awards close, and Match Log/Replay recovery against isolated staging data. Keep the matrix `Partial` until that manual evidence exists.
- League Awards manual evidence remains deferred in `docs/league_awards_parity_evidence.md`.
- League Live order-21 automated evidence is recorded in `docs/league_live_submit_parity_evidence.md`; authenticated staging writes remain deferred to the consolidated manual session.
