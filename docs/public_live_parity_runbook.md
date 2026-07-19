# Public JUPR Live parity runbook

This runbook covers the public one-off event product at `/clubs/{clubSlug}/live`.
It is deliberately separate from Tournament Live and from the authenticated JUPR
Live Admin workflow. Public sessions are unrated. Club Social completion creates a
pending moderation submission and does not update ratings.

## Deployment boundary

Public reads can remain available wherever the private FastAPI projection is
healthy. Public writes require all of the following on FastAPI:

- `SUPABASE_SERVICE_ROLE_KEY` (never expose it to Vercel or a browser);
- `JUPR_PUBLIC_LIVE_TOKEN_SECRET`, a stable random server-only value of at least
  32 characters;
- `JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET`, preferably a separate stable random
  server-only value of at least 32 characters;
- `JUPR_ENABLE_PUBLIC_LIVE_WRITES=1`;
- `JUPR_ENV=staging` during the pilot.

Production additionally requires `JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=1`.
The committed production Fly configuration keeps both public-write gates off.
Vercel needs none of the token, rate-limit, Supabase service-role, or database
credentials.

Optional lower application limits are:

- `JUPR_PUBLIC_LIVE_CREATE_LIMIT_PER_HOUR` (default 8, maximum 30);
- `JUPR_PUBLIC_LIVE_MUTATION_LIMIT_PER_HOUR` (default 120, maximum 500).

The database migration also installs atomic hard ceilings per requester and per
club. Those limits protect the service if forwarding metadata is spoofed or many
requests race at once.

## Migration and preflight

Apply `supabase/migrations/20260719220000_public_live_durability.sql` to the
isolated staging project before enabling writes. Apply the same additive
migration to production **before deploying this API/web commit**, while both
production write flags remain off. Legacy reads have a view-only fallback, but
the migration-first sequence avoids a mixed-version window and is the supported
rollout. The migration:

1. adds row versions, completion reservations, operation markers, and completion
   timestamps to `live_sessions`;
2. hashes any legacy plaintext edit token and removes the plaintext value from
   session JSON;
3. creates the private `public_live_operations` idempotency/recovery ledger;
4. installs version-normalizing write guards for legacy/admin writers and a
   single-executor lease for multi-table Club Social completion;
5. forces RLS and revokes `public`, `anon`, and `authenticated` access from both
   private tables;
6. grants table access only to `service_role`; and
7. installs advisory-lock-backed insert limits with matching requester/club
   indexes.

Deployment order is: migrate staging → deploy staging with the staging gate on
→ complete the consolidated smoke → migrate production with both gates off →
deploy production with both gates still off. Enabling the separate production
gate is a later, explicit decision.

After migration and secret configuration, deploy the staging API and check:

```text
GET https://juprleagues-api-staging.fly.dev/health/live-sessions
```

`ok`, `live_sessions_query_ok`, and `operation_ledger_query_ok` must be true.
Do not continue if the endpoint says the schema, service role, token secret, or
rate-limit secret is unavailable.

## Automated evidence

Run before the consolidated manual session:

```bash
python -m pytest -q \
  tests/test_public_live_service.py \
  tests/test_public_live_write_service.py \
  tests/test_api_live_sessions_contract.py \
  tests/test_public_live_security_contract.py
npm --prefix apps/web run build
npm --prefix apps/web exec playwright test e2e/public-live.staging.spec.ts -- --list
```

The gated mutating Playwright scenario runs only when
`JUPR_RUN_PUBLIC_LIVE_WRITE_E2E=1` is deliberately supplied against the exact
staging origin. Do not set that flag for production.

## Recovery semantics

- Creation and every mutation use a browser-held idempotency key. The browser
  retains the complete unresolved request body and key in `sessionStorage`, so a
  refresh retries the same expected version and payload rather than duplicating
  or conflicting with itself.
- Edit tokens travel in the URL fragment, are removed from the address bar after
  capture, and are retained only for the current browser session. FastAPI stores
  only the token hash.
- A stale version returns `409` without changing the session. Refresh durable
  state before issuing a new operation key.
- A `503` recovery response means the result may be uncertain. Retry the exact
  request with the same operation key. Do not invent a new key first.
- Club Social completion reserves the session before writing moderation rows.
  A database lease admits only one same-key executor, and table triggers make
  Streamlit/admin writes advance the version while failing closed during a
  reservation. A same-key retry reconciles the stable social event and then
  clears the reservation.
- If the reservation cannot be reconciled, leave public writes disabled, retain
  the operation and session rows for evidence, and use the Streamlit fallback.

## Deferred consolidated smoke

Manual testing stays deferred until all 45 Partial pages are ready. In that
session, use disposable staging names and complete these checks on desktop and a
mobile viewport:

1. create, refresh, score, export, and complete a Quick Round Robin;
2. create a two-round League/Ladder, score round 1, advance, confirm round 1 is
   read-only, substitute a guest in round 2, refresh, export, and complete;
3. create a Club Social event, complete every score, submit it once, and verify a
   single pending moderation event with no rating change;
4. lock/suspend the mobile browser between writes and resume from the original
   fragment edit link;
5. verify a public share link is view-only and contains no edit token;
6. submit a stale write and a wrong-token write and verify neither changes data;
7. inspect the operation ledger and confirm successful, replayed, and recovery
   outcomes have durable fingerprints without plaintext tokens.

## Rollback

First set `JUPR_ENABLE_PUBLIC_LIVE_WRITES=0` and redeploy FastAPI. This preserves
public scoreboards as view-only and restores the Streamlit creation/edit fallback
without deleting evidence. Do not drop the ledger or hash columns during an
incident. Database rollback, if later approved, should occur only after exporting
the affected session and operation rows and confirming no active completion
reservation remains.
