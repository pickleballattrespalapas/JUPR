# League Live domain parity evidence

This slice implements closure-program order 20. It makes League Live orchestration Python-authoritative, but it does **not** mark League Manager `Done`. Manual staging acceptance and the order-21 durable all-match submit/reconciliation slice remain gates.

## Automated evidence

- `tests/test_league_live_orchestration.py`: deterministic roster/court/bench suggestions, Python movement, override rules, court balance, and cross-court match rejection.
- `tests/test_api_contract_admin_league_live.py`: service-role fail-closed behavior, club player verification, create/resume, stale snapshot rejection, Python plan/save, and idempotent retry.
- `tests/test_league_live_domain_schema.py`: canonical migration, RLS, browser-role revocation, service-role-only grants, operation-key uniqueness, and PostgREST reload.
- `tests/test_league_live_domain_static_contract.py`: no browser movement implementation, status/flag guard, stale guard, operation-key contract, and Streamlit fallback copy.
- `apps/web/e2e/league-live-domain.staging.spec.ts`: protected-preview route health and explicit Python-authority or fail-closed fallback state.

## Staging prerequisites

1. Apply `supabase/migrations/20260719182921_league_live_domain_contract.sql` to the isolated staging Supabase project.
2. Set `SUPABASE_SERVICE_ROLE_KEY` only on the staging FastAPI service.
3. Set `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1` and `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN=1` only on staging FastAPI, then redeploy it.
4. Keep the production Live-domain flag off.
5. Use an isolated staging club and a role with `manage_matches`.

## Deferred manual staging book

Run these together after all parity PRs are integrated:

1. With the Live-domain flag off, confirm the Next page fails closed and directs staff to Streamlit.
2. Enable staging, load a league with roster counts covering exact courts and a bench remainder, and compare Python's deterministic suggestion after repeated refreshes.
3. Create, reload, pause/resume, and snapshot a session. In a second browser, change the session and prove the stale first browser receives HTTP 409.
4. Enter a full valid round. Preview twice and confirm the same operation key, rankings, and movement.
5. Change one target without a reason and confirm rejection; add a reason and confirm balanced-court validation.
6. Exercise a default bench, reviewed bench override, add-player, and substitute-player plan with verified club players. Confirm cross-club/unknown IDs are rejected.
7. Submit one isolated round, verify official matches, round history, next roster/courts, audit attribution, and resume behavior.
8. Retry the identical round-state request and confirm idempotent replay without a second round row.
9. Exercise the documented Match Log/Replay recovery path after a controlled post-submit state-save failure.
10. Sign out or use an insufficient role and confirm every private route refuses access. Verify no League Live table is readable with anon/authenticated Supabase roles.

Record screenshots, request IDs, audit rows, and cleanup results. Only the final evidence PR may reconcile the parity matrix after these tests pass.
