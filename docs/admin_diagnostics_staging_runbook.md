# Admin Diagnostics staging runbook

This runbook covers the guarded Next/FastAPI parity slice for Badge Debug & Audit, Match Canonical Audit, Admin Tools, and the static Admin Guide. These writes are staging-only. Streamlit remains the fallback until the manual staging matrix passes.

## Release prerequisites

1. Apply `supabase/migrations/20260719204500_admin_diagnostics_guarded_operations.sql` to the isolated staging Supabase project.
2. Confirm `admin_guarded_operations` has RLS enabled, no `anon` or `authenticated` grants, and explicit `service_role` access only.
3. Confirm `apply_match_canonical_patches_atomic` is executable by `service_role` only.
4. Configure `SUPABASE_SERVICE_ROLE_KEY` on the staging FastAPI service only. Never add it to Vercel or any `NEXT_PUBLIC_*` variable.
5. Set `JUPR_ENV=staging` and keep these existing FastAPI flags enabled only in staging:
   - `JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS=1`
   - `JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT=1`
   - `JUPR_ENABLE_NEXT_ADMIN_TOOLS=1`
6. Deploy FastAPI before the Next preview. Do not enable these write flags in `fly.prod.toml`.

Every applying request requires a Supabase user JWT, the route-specific club permission, exact confirmation, a caller-retained operation key, a persisted audit intent, and a required completion audit. Dry runs do not create audit, eval-run, operation, or domain rows.

## Read-only smoke

Run these first and compare with Streamlit. No table should receive an insert, update, or delete.

- Badge options, one Badge Debug trace, and one Badge Audit (`view_audit_log`).
- Badge recompute `dry-run`; verify `read_only=true` and no `badge_eval_runs` row.
- Match Canonical options, audit, and normalize `dry_run=true`; capture the fingerprint and exact proposed IDs.
- Admin Tools overview, activity, worker status, Club Social queue, rating report, and tournament backfill preview.
- Download the server-generated rating CSV and confirm leading `=`, `+`, `-`, or `@` values are prefixed to prevent spreadsheet formula execution.
- Open Admin Guide and verify the route/permission/phrase/stop/fallback table is present.

Stop if any read-only request writes data, crosses club scope, exposes service-role material, or falls back to an unbounded query.

## Applying smoke

Use disposable staging fixtures only. Run one action at a time and verify the source row plus both audit events before continuing.

| Workflow | Permission | Exact phrase | Verification |
| --- | --- | --- | --- |
| Badge definition state | `run_replay` | `UPDATE BADGE STATE` | Expected state lock, one state change, intent + completion audit |
| Badge recompute | `run_replay` | `RECOMPUTE BADGES` | Scoped eval run/result, operation completed, Badge Audit agrees |
| Badge revoke | `run_replay` | `REVOKE BADGE` | Exact row revoked; compensation restores it if completion audit fails |
| Canonical normalize | `manage_matches` | `APPLY NORMALIZE` | Same fingerprint and exact IDs; one atomic RPC; Match Log readback agrees |
| Club Social moderation | `manage_matches` | `APPROVE SOCIAL SUBMISSION` or `REJECT SOCIAL SUBMISSION` | Current status/version changes once; raw event JSON is absent from audit |
| Staff role | `manage_roles` | `SAVE ROLE` or `REVOKE ROLE` | Optimistic version/readback; final `super_admin` remains; activity agrees |
| Badge queue | `run_replay` | `PROCESS BADGE QUEUE` or `DRAIN BADGE QUEUE` | Queue counts and completion audit agree |
| Admin Tools recompute | `run_replay` | `RUN BADGE RECOMPUTE` | Scoped result and Badge Audit agree |
| Tournament backfill | `run_replay` | `BACKFILL TOURNAMENT MATCHES` | Exact reviewed games each have one official match; Match Log and Replay History agree |

For each successful operation, submit the same operation key and payload again. It must return the saved idempotent result without another domain mutation.

## Failure and recovery drills

1. Reuse an operation key with different inputs. The request must fail before a domain write.
2. Change an expected version or canonical row after preview. The apply must reject the stale request.
3. Simulate a lost response after an apply, retain the exact operation key, and use the workflow operation-status route. Never retry with a new key while status is incomplete.
4. For canonical changes, stop and compare the exact IDs in Match Log; use Replay History before any further rating-adjacent write.
5. For an uncertain tournament backfill, inspect Match Log. Only when every reviewed game has exactly one official match may `RECOVER TOURNAMENT BACKFILL` reconcile the operation; recovery does not create or delete matches.
6. If completion audit fails, verify compensation. If compensation cannot be proven, status must be `recovery_required` and the UI must tell the operator to stop.
7. If FastAPI, strict audit, schema, JWT, permission, or recovery is unavailable, stop and use the established Streamlit workflow. Do not copy staging data into production.

## Automated evidence

Before publishing the PR, run focused admin diagnostics contracts, the complete Python suite, `python -m compileall`, the full Next build, and both parity guard commands. Manual staging writes remain deferred until all partial-page PRs are assembled and ready for the coordinated smoke pass.
