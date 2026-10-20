# PCS audit remediation — first operational batch

Prepared 2026-10-20 UTC from staging `ac0dfdd43742907573dc7beddfb98a525b12b355`.
This records implementation and verification, not completion of the entire audit.
Production promotion, production migrations, historical data repairs, queue
releases and live email delivery require their own reviewed release actions.

## Changes in this batch

| Audit finding | Implemented behavior | Remaining acceptance / recovery |
| --- | --- | --- |
| F01 — badge retry storm | Custom stale-snapshot conflicts use PT409 instead of PostgreSQL serialization code40001; program evaluation retries one fresh snapshot, then yields to durable backoff. | Existing stuck database transactions and old queue entries require targeted production inspection after approval. |
| F04 / F23 — nonexistent player activity column | Verified-update and tournament recovery queries select the real `active` column while preserving response aliases and club restrictions. | Verify against the eventual deployed release. |
| F05 — intake without operational review | Support intake requires enabled staff review, both write permissions, and readable queue schema. Public status and staff readiness report the same availability. | A named staff reviewer, controlled staging acceptance and any production review-flag enablement remain separate. Existing requests are not automatically resolved. |
| F06 — unrated participation omitted from activity | Recorded unrated doubles/singles update activity atomically in official approval without modifying rated counters or nullable singles baselines. Legacy updates reread and use bounded compare-and-swap retries. | Historical activity repair is preview-only. Legacy insert/activity writes remain separate; exhausted retries require reconciliation, not resubmission. |
| F14 — query credentials in logs, partial | Uvicorn access logs remove query strings; structured request logs include only approved fields. | Browser URLs, upstream proxy logs and other exception tracebacks are outside this mitigation. Token transport still needs a separate design. |
| F15 — support club context | Public support, correction and privacy forms require explicit club context and preserve profile context. | Depends on the staging club directory; excluded from the urgent production backport until that dependency is reviewed. |
| F17 — repeated JWKS retrieval | Reuse a bounded five-minute key-set cache per trusted configured URL; every request still verifies token signature, expiry and audience. | Unknown-key throttling follows PyJWT behavior; no indefinite per-key cache. |
| F18 — missing request logs | Docker starts Uvicorn with explicit structured request logging, sanitized request IDs and failure events. | Does not claim universal exception-message sanitization. |
| F20 — commerce publication mismatch | Public commerce uses the shared tournament visibility classifier before registration-window checks. | Existing authorized edit-link behavior is preserved. |
| F22 — draft outcomes exposed | Public results suppress score/winner/outcome/finalization fields until FINAL, with a defensive web display guard. | Does not alter staff draft editing. |

F02 (Next.js security upgrade) and F03 (session email scheduler) are isolated
candidates. The email scheduler exists only on the current production branch;
staging and production differ substantially. Do not merge all of staging into
production to obtain this batch. The Next.js upgrade also changes async route
boundaries and React tooling, so it has its own browser verification.

## Database rollout

Three forward migrations were independently reviewed and applied to the staging
project `sijpxjxvdtrehmqvirfi` before application integration:

1. `badge_conflicts_are_not_serialization_failures` patches four exact routines
   without replacing their permissions, locks, signatures or other logic.
2. `unrated_match_activity` changes the direct-match player-plan guard to allow
   activity-only unrated participants while preserving all rated projections,
   participant restrictions and existing compare-and-swap checks. Old writers
   remain valid. Install before deploying the new application writer.
3. `restrict_team_match_admin_rpc_execution` closes a newly confirmed staging
   authorization gap: three legacy SECURITY DEFINER admin routines inherited
   PUBLIC execution without checking caller authorization. Only their exact
   signatures lose PUBLIC/anon/authenticated execution; service-role access,
   bodies, owners and other grants remain unchanged. Missing legacy routines
   are safely skipped. Post-migration catalog checks confirm client access is
   false and service-role access true for all three. The corresponding Supabase
   anonymous/authenticated execution warnings are now absent.

The two body-patching migrations abort on unexpected installed-function drift;
all three can be safely reapplied. Migration application alone does not deploy
the application.

Badge verification used real installed staging routines/triggers inside one
isolated synthetic-club transaction. Both existing SQL regression suites passed;
ROLLBACK left zero fixture rows in ten checked tables. This is not a test of
PostgREST transport or concurrent database connections. The repeatable generator
is `tests/sql/build_badge_staging_verification.py`.

Activity verification exercised the public RPC as `service_role` through the
installed wrapper chain with every trigger enabled. Doubles/singles activity,
unchanged ratings/baselines, stale snapshot rejection, invalid-plan rejection,
receipt replay and old-writer compatibility passed. ROLLBACK left zero fixture
rows across nine checked tables. This was a synthetic Overall-context SQL test,
not a browser/PostgREST test or a historical backfill.

The advisor still reports three unrelated mutable-search-path functions and
disabled leaked-password protection. RLS-enabled tables without client policies
remain informational and closed to direct clients. Those residuals are not
claimed repaired by this batch.

## Verification and release boundaries

Focused component owners tested each change and another reviewer inspected the
database and authentication/logging changes. Integration checks and exact remote
deployment evidence must be recorded in the pull request before marking staging
ready. Per `AGENTS.md`, staging readiness requires a successful exact-SHA Fly
workflow and its `staging-handoff-<sha>` artifact with
`ready_for_manual_testing`; a passing local build alone is insufficient.

Keep staging email in `dry_run`. Never replay historical match submissions to
repair activity or clear/retry email rows to prove delivery. The source includes
a SELECT-only activity repair preview and separate badge/support recovery
runbooks. No historical backfill or production queue release is included.

Other audit findings and optimization items remain open unless separately
verified against the deployed version. This batch does not certify the entire
PCS system as repaired.
