# League Live submit/reconciliation parity evidence

This slice implements closure-program order 21. It does **not** mark League Manager `Done`; all authenticated staging mutations and operator-layout acceptance remain deferred to the consolidated manual session.

## Implemented contract

- `POST .../rounds/{round}/submit` is the only Next complete-round publish path. FastAPI validates every score, stale session version, Python movement key, permission, service role, and request fingerprint before mutation.
- A required audit intent and `league_live_publish_operations` row precede Match Uploader. One UUIDv5 context per session/round/slot makes response-loss recovery deterministic.
- Retry behavior is explicit: all contexts reconcile without republish; a partial set becomes `recovery_required`; no contexts may retry with the same idempotency key.
- Snapshot reconciliation, rating readback/replay warning, publish history, guest creation, and matches/ratings/roster/rounds CSV exports are available in the Next workflow.
- Verified compensation requires Match Log/Replay History cleanup, zero active deterministic contexts, a recovery reference, a reason, and exact confirmation. It records recovery; it never performs an unaudited delete.
- Intent/completion audits share the durable operation ID and request fingerprint. Post-mutation audit uncertainty tells the operator to reload/reconcile and never claims that no mutation occurred.

## Automated evidence

- `tests/test_league_live_publish_domain.py`: complete-round validation, deterministic contexts/fingerprints, publish-metadata-independent movement keys, rating review, and CSV formula neutralization.
- `tests/test_api_contract_admin_league_live_publish.py`: staging gate, schema preflight, stale/incomplete refusal before mutation, idempotent publish, response-loss reconciliation without republish, snapshot recovery, partial compensation, audit uncertainty/retry, guest recovery/idempotency, and exports.
- `tests/test_league_live_publish_schema.py`: coordination schema, uniqueness, recovery states, RLS, explicit browser-role revocation/service-role grants, canonical context uniqueness, and PostgREST reload.
- `tests/test_league_live_submit_static_contract.py`: single FastAPI submit authority, separate staging flag, deterministic/recovery endpoints, evidence links, and `Partial` matrix discipline.
- `apps/web/e2e/league-live-submit.staging.spec.ts`: non-mutating protected-preview check for durable publish/recovery copy or a fail-closed Streamlit fallback.

## Hard staging prerequisites

1. Apply `supabase/migrations/20260719182921_league_live_domain_contract.sql` and verify `league_live_rounds.operation_key` exists.
2. Apply `supabase/migrations/20260719190954_league_live_publish_reconciliation.sql`.
3. Verify canonical `matches.context_type`, `matches.context_id`, and `matches.deleted_at`; verify the deterministic-context unique index.
4. Keep `SUPABASE_SERVICE_ROLE_KEY` only on staging FastAPI. Set `JUPR_ENV=staging`, the League Manager and Live domain gates, then separately set `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT=1`.
5. Keep production submit and Live-domain gates off. Never copy the service role into Vercel or a `NEXT_PUBLIC_*` value.

A read-only implementation-time inspection found the staging League Live tables private/service-role scoped but did not find the order-20 `league_live_rounds.operation_key` column. The new preflight therefore blocks before audit/mutation. No migration or staging write was performed from this slice.

## Deferred manual staging book

Use a disposable staging league, staging-only players/guest, and a role with `manage_matches`. Record screenshots, operation IDs, audit rows, Match Log rows, replay IDs, exports, and cleanup.

1. With submit off (and once with `JUPR_ENV=production`), confirm planning can remain available but publish/guest/export controls are guarded off and Streamlit is named as fallback.
2. Apply both migrations, enable staging submit, score every generated match, preview the Python plan twice, and confirm the operation key is stable.
3. Remove one score and prove no audit, operation, official match, rating, or session mutation occurs. Repeat with stale `expected_updated_at` and a changed plan.
4. Publish the full round once. Verify deterministic contexts, one publish operation, intent/completion audit markers, round snapshot, next courts/roster, rating before/after readback, and public/admin result consistency.
5. Retry the identical browser request and confirm `idempotent_replay`, unchanged match count, and no second rating application. Reuse the key with changed values and confirm HTTP 409.
6. Simulate response loss after all inserts but before state save; retry and confirm reconcile without Match Uploader. Simulate a post-publish snapshot/audit failure; reload/reconcile and confirm the warning never says no mutation occurred.
7. Create a uniquely named guest, use it in add/substitute planning, publish the following round, retry guest creation, and verify one real player/guest operation. Confirm an existing name is rejected. Retire cleanup-only guests through Player Editor.
8. Download matches, ratings, roster, and rounds CSVs; compare counts/ratings with Supabase and confirm formula-like text is inert in a spreadsheet.
9. In a controlled partial-publish fixture, confirm blind retry is blocked. Recover through Match Log soft-exclusion and full Replay History, record the replay reference, verify compensation, and confirm zero active deterministic contexts plus a `compensated` operation.
10. Test wrong-club, insufficient-role, expired JWT, missing service role, anon/authenticated direct-table denial, and missing-schema preflight. Confirm production remains off.
11. Restore/clean disposable data and verify final ratings, standings, Match Log, replay queue, round/session state, audit trail, and public pages.

Only the final evidence PR may move the matrix row after this book is signed off.
