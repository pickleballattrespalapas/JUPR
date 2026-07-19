# Match Log and Replay History durability

This slice closes the remaining Match Log notes/bulk-edit and Replay History durability gaps while keeping FastAPI and the Python replay service authoritative.

## Transaction boundary

`apply_match_log_patches_atomic` is a service-role-only Postgres RPC. In one transaction it:

1. serializes a stable `(club_id, idempotency_key)` request;
2. locks and validates every target match in the requested club;
3. applies at most 100 supported patches, including notes and two- or four-player rows;
4. records complete before/after evidence in `match_edit_operations` and `admin_activity_log`; and
5. creates the uniquely keyed pending `replay_jobs` row whenever ratings are affected.

Any validation, edit, operation-ledger, audit, or replay-job insert failure rolls the whole transaction back.

## Mandatory replay contract

Metadata-only edits can finish immediately. Player, score, date, league, or match-type edits cannot report success until the tracked Python `replay_history` call succeeds. The API marks an edit `recovery_required` if the committed edit is followed by a replay failure and returns the durable operation ID with HTTP 409.

The operator must use the guarded `RECOVER` action. Recovery retries the same idempotent replay job; it never creates a second edit operation or reapplies the match patches.

## Operator evidence

- Match Log shows the latest durable edit operations and any recovery-required state.
- Replay History shows recent job IDs, status, actor, source, timestamps, and errors.
- Guided and bulk editors generate stable client request IDs so a network retry cannot duplicate a write.
- The Streamlit correction and replay tools remain available until the combined staging manual suite passes.

## Deployment order

1. Apply `supabase/migrations/20260719172000_replay_job_idempotency.sql` in staging.
2. Deploy FastAPI with both Match Log and Replay History flags enabled for the pilot role.
3. Deploy the matching Next preview.
4. Run the non-mutating automated smoke.
5. During the final combined manual window, test a reversible notes-only edit first, then one rating-affecting edit and its automatic replay evidence.

Do not enable Match Log apply if the migration or Replay History flag is absent.
