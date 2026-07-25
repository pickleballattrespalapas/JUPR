# Match Log + Replay History closed-club pilot

This runbook is for the first Next/FastAPI admin write pilot. The current
candidate limits the pilot to ordinary atomic Match Log edits, duplicate
no-issue resolution, and Replay History inspection/recovery. Duplicate cleanup
and bulk exclusion are future protocols, not currently authorized actions.

## Runtime flags

Set these on the FastAPI runtime only:

```text
JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=1
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1
JUPR_ENABLE_NEXT_ADMIN_REPLAY=1
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0
```

Recommended for production pilot once `admin_activity_log` is confirmed available:

```text
JUPR_REQUIRE_API_AUDIT_LOG=1
```

Do **not** enable these yet for this pilot:

```text
JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY
JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER
JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES
JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE
JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR
JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER
JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS
JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP
JUPR_ENABLE_NEXT_ADMIN_TOOLS
```

## Preconditions

1. `/admin/login` works for the pilot operator.
2. The operator has an active `admin_role_assignments` row for `club_id=tres_palapas`.
3. The role has:
   - `manage_matches` for Match Log edits.
   - `delete_matches` for future duplicate cleanup and bulk exclusion.
   - `run_replay` for Replay History.
4. Streamlit admin remains available as fallback.
5. FastAPI has `SUPABASE_SERVICE_ROLE_KEY` configured.
6. FastAPI has JWT verification configured with either:
   - `SUPABASE_JWT_SECRET`, or
   - `SUPABASE_JWKS_URL` plus `JUPR_SUPABASE_JWT_MODE=jwks`.
7. `supabase/migrations/20260719172000_replay_job_idempotency.sql` is applied to staging before Match Log apply is enabled.

## Non-mutating preflight

After enabling flags, run the non-mutating pilot smoke with a short-lived Supabase access token for the pilot operator:

```bash
export STAGING_JUPR_API_BASE_URL=https://api.juprleagues.com
export JUPR_SMOKE_CLUB_ID=tres_palapas
export JUPR_ADMIN_BEARER_TOKEN='<supabase access token from /admin/login browser session>'
make admin-pilot-smoke
```

Expected result:

```text
PASS api: health
PASS pilot: operations mode
PASS pilot: match log enabled
PASS pilot: replay enabled
PASS auth: match log apply permission preflight
PASS auth: replay permission preflight
```

The last two checks intentionally send invalid/no-op write bodies. They should fail with `400` after authentication and role checks, proving that the flags and permissions are wired without mutating match, rating, or replay state.

## Manual pilot sequence

### Step 1 — Match Log read/scan

Open:

```text
/admin/match-log
```

Validate:

1. The page says Match Log is enabled.
2. Duplicate scan loads.
3. Cleanup preview is visible if duplicates exist.
4. The apply panel shows the signed-in admin session.

### Step 2 — Apply one tiny, reversible edit first

Prefer a non-rating structural edit before changing players or scores, for example:

```json
[
  {"id": 123, "week_tag": "Week 1"}
]
```

Operator steps:

1. Add a correction note.
2. Paste one patch only.
3. Type `APPLY`.
4. Confirm FastAPI response is `ok: true`.
5. Confirm the operation appears as `succeeded` in the recent durable edit operations table.
6. Confirm an audit row exists in `admin_activity_log`.
7. Review the match on public match history or Streamlit.

For a later rating-affecting smoke, confirm the response includes a replay job ID and the corresponding Replay History row is `succeeded`. If the page shows **Mandatory replay recovery required**, stop additional edits, type `RECOVER`, and complete the same job before continuing.

### Step 3 — Future duplicate cleanup protocol; do not run while dormant

Keep `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0` for the current
candidate. The steps below are retained for a future candidate only after
cleanup/exclusion has atomic, idempotent recovery and the gate has been
explicitly reviewed for a bounded staging wave.

Future operator steps:

1. Confirm the preview keeps the oldest/canonical row.
2. Type `DELETE`.
3. Confirm FastAPI response lists deleted IDs.
4. Confirm audit row exists and is flagged for review.
5. Immediately run Replay History.

### Step 3a — Resolve duplicate false positives as no issue now

If the scanner flags a legitimate repeated matchup, do **not** run cleanup. Mark the group as no issue instead.

Operator steps:

1. Confirm the rows are distinct real matches.
2. Add a reason, for example: `Legitimate repeated matchup with same players and score`.
3. Confirm the accessible Yes/No dialog. The client supplies the internal
   `NO ISSUE` safeguard.
4. Confirm FastAPI response is `ok: true` with `mode: duplicate_no_issue`.
5. Refresh Match Log and confirm the group is no longer listed as an active cleanup candidate.
6. Confirm an audit row exists in `admin_activity_log` with `action_type=match_duplicate_false_positive_resolved`.

### Step 4 — Replay History recovery

Open:

```text
/admin/replay-history
```

For the first production pilot, prefer a league-specific replay when possible. Use full reset only after confirming the expected blast radius.

Operator steps:

1. Select replay scope.
2. Type `REPLAY`.
3. Confirm response summary.
4. Verify public leaderboard/player/match pages still load.
5. Keep Streamlit open as fallback during pilot validation.

## Rollback path

1. Disable `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE` first.
2. Disable `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY`.
3. Disable `JUPR_ENABLE_NEXT_ADMIN_REPLAY` if replay should stop.
4. Leave `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1` temporarily if read-only investigation is still needed.
5. Fall back to Streamlit correction tools.
6. If a bad duplicate cleanup happened in a future authorized wave, stop and
   use the reviewed atomic/idempotent recovery path; do not improvise direct
   repair.
7. If a false-positive group was marked no issue by mistake, update the corresponding `admin_match_log_duplicate_resolutions` row to `is_active=false` and reload Match Log.

## Stop conditions

Stop the pilot immediately if any of these occur:

- Audit logging fails while `JUPR_REQUIRE_API_AUDIT_LOG=1` is enabled.
- A signed-in admin receives unexpected club access.
- Replay output differs from Streamlit expectation.
- Public leaderboard/player pages fail after replay.
- Any write cannot be traced to an actor email and role.
