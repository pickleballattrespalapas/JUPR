# Next Match Uploader

This document tracks the Match Uploader migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/match-uploader/status`.
- FastAPI submit endpoint: `POST /admin/clubs/{club_id}/match-uploader/batch`.
- FastAPI singles endpoint: `POST /admin/clubs/{club_id}/match-uploader/singles`.
- FastAPI multi-court round-robin preview endpoint: `POST /admin/clubs/{club_id}/match-uploader/round-robin/preview`.
- FastAPI new-player creation endpoint: `POST /admin/clubs/{club_id}/match-uploader/players`.
- Next route: `/admin/match-uploader`.
- Manual/batch score-entry rows with shared defaults for date, league, week/session, and match type.
- Streamlit-style single round-robin schedule generation using the existing Python schedule templates.
- New-player create-and-continue flow with starting JUPR values before regenerating the pending schedule.
- Official League and Pop-Up/Social contexts, including server-side Pop-Up event hydration from the event name when needed.
- Rating scope options: overall + league, overall only, and unrated/record-only.
- Supabase Auth admin session for the closed-club pilot.
- Submission feedback with inserted/skipped counts and affected player rating deltas.
- Explicit post-commit player-update email handoff results (`auto_sent`, `disabled`, `skipped`, or `error`).

## Runtime flag

The workflow is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER=1
```

This flag is separate from the legacy Score Entry MVP flag.

The direct singles route has a second, fail-closed gate:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES=1
```

That gate is deliberately dormant and forced to `0` in every staging wave and
in production. The direct singles UI/API contract is implemented, but the
writer is not yet one atomic operation across match history and player singles
aggregates. Direct Match Uploader singles therefore remains `Blocked`; enabling
the broader Match Uploader wave does not authorize it.

League Live does not enable that full flag. Its domain and submit waves enable
only:

```text
JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW=1
```

That dependency authorizes the exact round-robin preview route only. Player
creation, direct singles, batch upload, and every other Match Uploader write
remain denied in both League Live waves.

## Authorization

When enabled, batch submit, round-robin preview, and new-player creation require a Supabase access token with `enter_scores` permission. The route resolves roles through `admin_role_assignments` and writes audit attribution through FastAPI for mutating operations.

## Data path

The browser submits doubles rows to FastAPI. FastAPI normalizes the rows, loads
the current player/league context, and calls the existing Python
`process_matches` path through `submit_match_batch`.

The implemented direct singles path still uses the Python singles rating
authority, and its request/response and replay-managed-row contracts have
automated coverage. The reviewed
`20260725181500_singles_replay_recovery` migration adds the preserved baseline
and managed marker needed for deterministic replay. It is applied in staging as
connector ledger entry `20260725193213_singles_replay_recovery`; formal
exact-candidate acceptance remains pending. More importantly, the direct writer
must be made atomic before its dormant gate can be reviewed for enablement.

Tournament Operations official singles publishing is a separate path. Its
compare-and-swap RPC preserves the replay-managed marker atomically and is
automated-ready; the dormant direct-uploader gate does not disable that
implementation. Candidate-bound manual publish/recovery evidence remains
pending.

The round-robin preview endpoint calls the existing Python schedule service and returns generated match rows to the browser. New-player creation calls the existing Python `safe_add_player` helper, then the browser regenerates the pending schedule with the refreshed player list.

No browser-side code writes directly to Supabase tables and no rating or schedule-generation logic is implemented in TypeScript.

## Commit and email recovery contract

Batch match submission completes before the optional player-update email handoff begins. Once the Python match service returns success, FastAPI returns `match_write_committed=true` even if email delivery later fails. The response includes Match Log and Player Updates recovery links and the UI tells the operator never to resubmit committed matches solely because email failed.

If the browser loses the response, the outcome is treated as unknown: check Match Log before retrying. This prevents duplicate matches while preserving an explicit email retry path.

## Follow-up slices

- Pilot every round-robin format against production-like staging player sets.
- Replace direct singles persistence with one atomic writer, retain the applied
  singles replay schema, and only then review a bounded enablement of
  `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES`.
- Preserve the future direct-singles protocol: submit one rated and one unrated
  disposable row, verify exact managed-history readback and zero aggregate
  change for the unrated row, then exclude only the returned IDs and require a
  full replay to restore the captured baseline. Do not run that protocol while
  the gate is dormant.
