# SaaS staging deployment guardrails

This document defines deployment guardrails for the new SaaS stack.

Operator source of truth: `README.txt`. If any conflict exists, follow `README.txt`.

## Status: staging only

The **Next.js + FastAPI** path is currently approved for **staging only**.

Production Streamlit remains the active production app path and is unchanged.

## Architecture (current)

- **Next.js staging app** (`apps/web`) for public pages and staged admin UI surface.
- **FastAPI staging API** (`services/api`) for read APIs and staged admin endpoints.
- **Supabase staging project** for all staging API reads/writes.
- **Streamlit production app** remains the production path.

## Required staging environment variables

Set these when deploying the staging stack:

- `JUPR_ENV=staging`
- `SUPABASE_URL=<staging project>`
- `SUPABASE_SERVICE_ROLE_KEY=<staging service role>`
- `JUPR_API_BASE_URL=<staging API URL>`
- `NEXT_PUBLIC_JUPR_API_BASE_URL=<staging API URL if needed>`

## Guardrails and explicit warnings

- **Do not point staging API at production Supabase.**
- **Do not point `SUPABASE_TEST_DATABASE_URL` at production.**
- **Do not deploy Next admin score entry for rated matches.**
- **Keep `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0`.**
- **Do not migrate production traffic to Next yet.**
- **Streamlit remains the active admin console for rated events.**

## Staging deployment checklist

1. Confirm `JUPR_ENV=staging` for API and web runtime.
2. Confirm `SUPABASE_URL` points to the **staging** Supabase project.
3. Confirm `SUPABASE_SERVICE_ROLE_KEY` is from **staging** project credentials.
4. Confirm API base URL variables point to staging API.
5. Confirm `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` for API and web.
6. Validate health and public read routes before any internal staging demos.

## Not in scope yet

- Production deployment config for FastAPI.
- Production Vercel/custom-domain cutover for Next.js.
- Production traffic migration from Streamlit.

## Next admin score entry status

The Next.js admin score-entry flow is **experimental** and intentionally de-risked:

- Disabled by default unless `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY` is exactly `1`, `true`, or `yes`.
- Even when enabled, current token checks are temporary and **not** production auth.


See `docs/next_admin_auth_design.md` for the minimum real-auth contract required before enabling Next.js admin score entry.

Before this flow can be active for rated events, it must implement:

- Supabase JWT validation.
- Admin role lookup.
- Club-scoped authorization.
- Audit identity attribution.
- CSRF/session strategy for the chosen auth model (if applicable).
