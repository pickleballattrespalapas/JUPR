# SaaS staging deployment guardrails

This document defines deployment guardrails for the new SaaS stack.

Operator source of truth: `README.txt`. If any conflict exists, follow `README.txt`.

## Status: staging only

The **Next.js + FastAPI** path is currently approved for **staging only**.

Production Streamlit remains the active production app path and is unchanged.

Do **not** move production traffic or the `juprleagues.com` custom domain to Next.js until the public read-only cutover is explicitly approved.

## Architecture (current)

- **Next.js staging app** (`apps/web`) for public pages and staged admin UI surface.
- **FastAPI staging API** (`services/api`) for read APIs and staged admin endpoints.
- **Supabase staging project** for all staging API reads/writes.
- **Streamlit production app** remains the production path.

## Required staging environment variables

### FastAPI staging runtime

Set these on the API host:

```bash
JUPR_ENV=staging
SUPABASE_URL=<staging project>
SUPABASE_SERVICE_ROLE_KEY=<staging service role>
SUPABASE_ANON_KEY=<staging anon key if needed>
JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0
JUPR_ALLOWED_ORIGINS=<Vercel staging URL>,http://localhost:3000
```

### Vercel/Next.js staging runtime

Set these on the Vercel project/environment for `apps/web`:

```bash
JUPR_API_BASE_URL=<staging API URL>
NEXT_PUBLIC_JUPR_API_BASE_URL=<staging API URL>
NEXT_PUBLIC_JUPR_ENV=staging
NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0
```

Never set `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET`, database URLs, or other server-only secrets in Vercel/frontend environment variables.

## Vercel staging checklist

1. Create or select a staging Vercel project.
2. Set **Root Directory** to `apps/web`.
3. Use the Next.js framework preset.
4. Use `npm install` and `npm run build`.
5. Configure only the staging web variables listed above.
6. Point `JUPR_API_BASE_URL` and `NEXT_PUBLIC_JUPR_API_BASE_URL` at the staging FastAPI deployment.
7. Keep both Next admin score-entry flags disabled.
8. Add the Vercel staging origin to `JUPR_ALLOWED_ORIGINS` on the staging API.
9. Deploy Vercel staging.
10. Run the public smoke checks before sharing the URL.

## Public smoke checklist

The public smoke harness checks read-only API routes, public Next.js pages, and the guard that the admin score-entry endpoint remains disabled.

From the repository root:

```bash
export STAGING_JUPR_API_BASE_URL=<staging API URL>
export STAGING_WEB_BASE_URL=<Vercel staging URL>
make public-web-smoke
```

If the `live_sessions` migration/grants are still being applied in staging, allow the live-session route to return its setup error while the rest of the public surface is validated:

```bash
JUPR_SMOKE_ARGS=--allow-live-unconfigured make public-web-smoke
```

The same check is available through the manual GitHub Actions workflow `Staging Smoke`. Configure `STAGING_JUPR_API_BASE_URL` as a repository secret and optionally configure `STAGING_WEB_BASE_URL`, or provide URLs as workflow inputs.

## Guardrails and explicit warnings

- **Do not point staging API at production Supabase.**
- **Do not point `SUPABASE_TEST_DATABASE_URL` at production.**
- **Do not deploy Next admin score entry for rated matches.**
- **Keep `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0`.**
- **Keep `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0`.**
- **Do not migrate production traffic to Next yet.**
- **Streamlit remains the active admin console for rated events.**

## Staging deployment checklist

1. Confirm `JUPR_ENV=staging` for API runtime.
2. Confirm `SUPABASE_URL` points to the **staging** Supabase project.
3. Confirm `SUPABASE_SERVICE_ROLE_KEY` is from staging project credentials.
4. Confirm API base URL variables point to staging API.
5. Confirm Vercel contains no service-role/JWT/database secrets.
6. Confirm `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` for API.
7. Confirm `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` for web.
8. Validate health, public API routes, and public web routes before internal staging demos.

## Not in scope yet

- Production traffic migration from Streamlit.
- Production Next.js admin rated writes.
- Self-serve onboarding or billing.

## Next admin score entry status

The Next.js admin score-entry flow is **experimental** and intentionally de-risked:

- Disabled by default unless `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY` is exactly `1`, `true`, or `yes`.
- Even when enabled, current token checks and role checks are staging-first and must satisfy the auth contract before production use.

See `docs/next_admin_auth_design.md` for the minimum real-auth contract required before enabling Next.js admin score entry.

Before this flow can be active for rated events, it must implement and prove:

- Supabase JWT validation.
- Admin role lookup.
- Club-scoped authorization.
- Audit identity attribution.
- CSRF/session strategy for the chosen auth model, if applicable.
- Staging E2E tests proving scorekeeper writes affect only the intended club.
