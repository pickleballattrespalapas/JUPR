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

The deployable staging API is the dedicated Fly app described by `fly.staging.toml`.
It must never reuse the production `juprleagues-api` app or production Supabase
credentials.

## GitHub staging environment

Create a GitHub Actions environment named `staging` with:

- secret `FLY_API_TOKEN`
- secret `STAGING_SUPABASE_URL`
- secret `STAGING_SUPABASE_SERVICE_ROLE_KEY`
- secret `STAGING_SUPABASE_ANON_KEY`
- variable `STAGING_SUPABASE_PROJECT_REF`
- secret `SUPABASE_PROD_DATABASE_URL` and `SUPABASE_TEST_DATABASE_URL` only when the guarded schema-copy workflow is needed

`STAGING_SUPABASE_PROJECT_REF` is checked against the project ref parsed from
`STAGING_SUPABASE_URL` before any staging API deploy. The workflow also refuses
to target the production Fly app name.

## Required staging environment variables

### FastAPI staging runtime

Set these on the API host:

```bash
JUPR_ENV=staging
SUPABASE_URL=<staging project>
SUPABASE_SERVICE_ROLE_KEY=<staging service role>
SUPABASE_ANON_KEY=<staging anon key if needed>
JUPR_EMAIL_MODE=dry_run
JUPR_WEB_BASE_URL=<Vercel staging URL>
JUPR_REQUIRE_API_AUDIT_LOG=1
JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=1
JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1
JUPR_ALLOWED_ORIGINS=<Vercel staging URL>,http://localhost:3000
```

`fly.staging.toml` enables every current Next admin workflow so the complete
surface can be exercised against isolated staging data. Automatic player-update
email behavior is enabled there, but `JUPR_EMAIL_MODE=dry_run` prevents external
delivery during staging tests. `JUPR_WEB_BASE_URL` is the canonical Next origin
used for tokenized `/email-preferences` links; do not point it at the Streamlit
fallback.

### Vercel/Next.js staging runtime

Set these on the Vercel project/environment for `apps/web`:

```bash
JUPR_API_BASE_URL=<staging API URL>
NEXT_PUBLIC_JUPR_API_BASE_URL=<staging API URL>
NEXT_PUBLIC_JUPR_ENV=staging
NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1
```

Never set `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET`, database URLs, or other server-only secrets in Vercel/frontend environment variables.

`apps/web/next.config.js` fails closed for Git-connected Vercel previews. When
Vercel sets `VERCEL_ENV=preview`, it forces both API base variables to the
dedicated staging Fly API (or the public `JUPR_STAGING_API_BASE_URL` override),
sets the staging label, and exposes the score-entry UI for isolated testing.
Production deployments are not overridden. Check `/api/environment` on a
preview to verify the sanitized API origin and `preview_isolation_active=true`.
Also scope `NEXT_PUBLIC_STAGING_SUPABASE_URL` and
`NEXT_PUBLIC_STAGING_SUPABASE_ANON_KEY` to Vercel Preview so staff sign-in uses
the same staging Supabase project as FastAPI. `/api/environment` must report
`preview_auth_isolation_active=true` before authenticated write-flow testing.

## Vercel staging checklist

1. Create or select a staging Vercel project.
2. Set **Root Directory** to `apps/web`.
3. Use the Next.js framework preset.
4. Use `npm install` and `npm run build`.
5. Configure only the staging web variables listed above.
6. Point `JUPR_API_BASE_URL` and `NEXT_PUBLIC_JUPR_API_BASE_URL` at the staging FastAPI deployment.
7. Enable the browser score-entry flag only in the staging/preview environment.
8. Add the Vercel staging origin to `JUPR_ALLOWED_ORIGINS` on the staging API.
9. Deploy Vercel staging.
10. Run the full staging verifier and public smoke checks before sharing the URL.

## Deploy the isolated staging API

Run `Deploy FastAPI staging to Fly` from GitHub Actions. The workflow:

1. validates all staging secrets and the expected Supabase project ref;
2. refuses the production Fly app name;
3. creates or updates only `juprleagues-api-staging`;
4. deploys `fly.staging.toml` with every current Next admin feature flag enabled;
5. waits for health; and
6. verifies the complete schema inventory and every public-safe admin status endpoint.

Default staging API URL:

```text
https://juprleagues-api-staging.fly.dev
```

The deploy fails if any required table is missing, any status route returns an
error/404, a workflow reports disabled, or the API does not report staging mode.

## Public smoke checklist

The public smoke harness checks read-only API routes and public Next.js pages.
The deploy workflow separately verifies the complete admin status surface.

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

The same check is available through the manual GitHub Actions workflow
`Staging Smoke`. It now treats the Vercel browser pass as part of the gate, so
configure the following in the GitHub `staging` environment (or provide the URL
input where supported):

The workflow form pre-fills the two canonical staging origins. Clearing either
field uses its corresponding environment secret; every resolved value still
must match the hardcoded staging allowlist before any request is made.

- `STAGING_JUPR_API_BASE_URL`: the isolated FastAPI staging origin;
- `STAGING_WEB_BASE_URL`: the protected Vercel Preview/staging origin;
- `STAGING_SUPABASE_URL`: the exact isolated Auth origin the Preview must report;
- `VERCEL_AUTOMATION_BYPASS_SECRET`: Vercel Deployment Protection automation
  bypass value used only by the public-web and browser smoke request steps. Both
  clients restrict it to the isolated HTTPS Vercel origin and prevent it from
  crossing origins on redirects. The Python client uses the direct bypass header
  without requesting a cookie redirect; Chromium owns the optional bypass-cookie
  handshake used by the browser suite.

The Chromium pass runs the exact non-mutating `public-read` parity manifest,
covers the critical public and disabled-admin shells in that manifest, refuses
known production domains, and asserts the sanitized `/api/environment` contract
reports Preview + staging API + staging Auth isolation. Retries are disabled so
an intermittent first-attempt failure cannot be accepted as green evidence.
The JSON evidence gate also rejects skips, failures, flakes, focused tests, and
any count other than the committed 56-test manifest. Screenshots, video, error
contexts, and the JSON report are retained as GitHub artifacts when the gate
fails. Traces stay disabled for protected Vercel runs so the automation bypass
credential cannot be captured. The workflow does not run real-auth, admin-read,
or write-wave specs. Before Chromium starts, the workflow also requires Vercel
and Fly to attest the exact checked-out staging SHA, the isolated Supabase/Auth
project, `write_wave: none`, and the complete all-false controlled-write flag
projection.

## Guardrails and explicit warnings

- **Do not point staging API at production Supabase.**
- **Do not point `SUPABASE_TEST_DATABASE_URL` at production.**
- **Enable Next admin writes only on the isolated staging API until each workflow passes end-to-end validation.**
- **Keep production API flags guarded independently from `fly.staging.toml`.**
- **Keep staging email delivery in `dry_run` until an intentional delivery test is approved.**
- **Do not migrate production traffic to Next yet.**
- **Streamlit remains the active admin console for rated events.**

## Staging deployment checklist

1. Confirm `JUPR_ENV=staging` for API runtime.
2. Confirm `SUPABASE_URL` points to the **staging** Supabase project.
3. Confirm `SUPABASE_SERVICE_ROLE_KEY` is from staging project credentials.
4. Confirm API base URL variables point to staging API.
5. Confirm Vercel contains no service-role/JWT/database secrets.
6. Confirm every staging admin workflow flag is enabled on the dedicated staging API.
7. Confirm `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1` only for Vercel staging/preview.
8. Confirm `JUPR_EMAIL_MODE=dry_run` before write-flow testing.
9. Validate health, full admin status routes, public API routes, public web routes, and the Chromium staging smoke before internal staging demos.

## Not in scope yet

- Production traffic migration from Streamlit.
- Production Next.js admin rated writes.
- Self-serve onboarding or billing.

## Next admin score entry status

The Next.js admin score-entry flow is **staging-only** and intentionally de-risked:

- Disabled by default outside the dedicated staging environment unless `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY` is exactly `1`, `true`, or `yes`.
- Even when enabled, current token checks and role checks are staging-first and must satisfy the auth contract before production use.

See `docs/next_admin_auth_design.md` for the minimum real-auth contract required before enabling Next.js admin score entry.

Before this flow can be active for rated events, it must implement and prove:

- Supabase JWT validation.
- Admin role lookup.
- Club-scoped authorization.
- Audit identity attribution.
- CSRF/session strategy for the chosen auth model, if applicable.
- Staging E2E tests proving scorekeeper writes affect only the intended club.
