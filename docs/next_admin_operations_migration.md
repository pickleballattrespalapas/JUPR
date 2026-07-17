# Next Admin Operations Migration

This is the control document for moving JUPR staff operations from Streamlit to the Next.js + FastAPI stack during the closed-club summer window.

## Posture

The migration is no longer read-only-only. During a closed-club or explicitly approved operational window, production-write pilots may be enabled one workflow at a time.

The goal is not to remove safety controls. The goal is to replace broad read-only restrictions with workflow-specific write controls.

## Permanent boundaries

These boundaries stay in place even when production-write pilot mode is enabled:

1. No privileged backend credentials in Vercel or browser-visible environment variables.
2. No direct browser writes to rating, match, player, league, badge, tournament, or replay tables.
3. No JavaScript rewrite of rating, match-processing, replay, badge-evaluation, or tournament-finalization logic.
4. Rating-adjacent writes must go through FastAPI and Python domain/service code.
5. Destructive operations require audit attribution and a recovery path.
6. Streamlit remains the fallback until each migrated workflow is proven.

## Pilot mode

The API status endpoint exposes the current posture:

```text
GET /admin/operations/status
```

Closed-club production-write pilot mode is controlled by the FastAPI runtime flag:

```text
JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=1
```

This flag does not enable every workflow. It only allows individual workflow flags to be considered for pilot use.

## Workflow flags

Enable only one workflow at a time unless a prior workflow is already proven.

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1
JUPR_ENABLE_NEXT_ADMIN_REPLAY=1
JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1
JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER=1
JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR=1
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1
JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER=1
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS=1
JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP=1
JUPR_ENABLE_NEXT_ADMIN_TOOLS=1
```

## Recommended sequence

1. Admin operations shell.
2. Match Log read, duplicate scan, correction preview, and replay planning.
3. Guarded Match Log apply and duplicate cleanup.
4. Replay History.
5. Score Entry MVP.
6. Match Uploader manual/batch score entry.
7. Match Uploader round-robin scheduling and new-player creation.
8. Player Editor.
9. League Manager.
10. Challenge Ladder Admin.
11. Tournament Admin/Ops.
12. Weekly Recap Admin.
13. Admin Tools, workers, and backfills.

Challenge Ladder Admin now includes the guarded core staging workflow: tier-aware challenge creation with server-side player-status eligibility and explicit override, domain-built email/SMS notice copy, accept-clock transitions, a read-only played-result preview, exact-draft browser review gating, official match publication through the Python match service, direct challenger-win rank swaps, forfeits, server-month-scoped pass resolution, roster append/reactivation, audited cross-tier moves with optional old-tier rank recompression, stale-safe whole-tier roster replacement preview/apply, vacation/reinstate overrides, and a read-only computed tier-movement review queue that hands off to the existing guarded move form. Whole-tier replacement resolves exact club names, blocks affected open challenges, preserves removed rows as inactive, recompresses affected source tiers, and records the reviewed before/after set in centralized Admin Tools activity. The Next workflow remains a staging pilot until exercised against isolated ladder data with Match Log and Replay History recovery checks.

## Enablement checklist

Before enabling a write workflow for staff use:

- Confirm the club is closed or the operational pilot window is explicitly approved.
- Confirm the workflow writes only through FastAPI.
- Confirm the FastAPI code calls Python domain/service authority rather than duplicating logic in TypeScript.
- Confirm staff auth, club scope, and role checks are in place for the endpoint.
- Confirm audit attribution is present for writes.
- Confirm the correction/replay path exists for match/rating-adjacent writes.
- Confirm Streamlit fallback remains available.
- Run API contract tests and the public/staff smoke checks against staging or the pilot deployment.

## Match Log, replay, and uploader foundation

The first serious admin migration target is Match Log, not broad score entry. Once Next can enter or mutate operational data, it must also be able to inspect, correct, and replay mistakes.

Current migrated slices include:

- match list/read endpoint with filters,
- duplicate scan using the existing duplicate-key logic,
- audited match edit apply,
- audited duplicate cleanup,
- server-side Replay History endpoint and page,
- role-gated replay execution through the Python replay domain function,
- Match Uploader manual/batch entry through FastAPI and the Python match-processing path.

## Notes

This document does not approve a full production cutover. It defines how to move from a public read-only SaaS surface to a staff-write pilot without putting the browser in charge of trusted data mutation.
