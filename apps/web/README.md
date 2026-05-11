# JUPR Web (Next.js public shell)

This is a minimal, read-only-first Next.js app for public pages that can eventually serve `juprleagues.com` and club pages.

## Routes

- `/` public JUPR landing page with SaaS pilot shell navigation.
- `/clubs/[clubSlug]` club landing page.
- `/clubs/[clubSlug]/leaderboards` public leaderboard page.

## Environment variables

Use one of the following:

- `JUPR_API_BASE_URL` (preferred for server-side runtime)
- `NEXT_PUBLIC_JUPR_API_BASE_URL` (fallback for browser-visible configuration)
- `NEXT_PUBLIC_JUPR_ENV` (optional; set to `staging` to show a staging badge)

Example:

```bash
export JUPR_API_BASE_URL=http://localhost:8000
```

## Local development

```bash
cd apps/web
npm install
npm run dev
```

Then open `http://localhost:3000`.

If the API is unavailable, pages render graceful empty/error states instead of crashing.


## Staging deployment guardrails

This Next.js path is currently **staging-only** for the SaaS migration.

- Point `JUPR_API_BASE_URL` / `NEXT_PUBLIC_JUPR_API_BASE_URL` to the staging API.
- Keep `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` and `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0`.
- Next admin score-entry UI remains disabled unless explicitly enabled for staging experiments.
- Do **not** migrate production traffic to Next.js yet.
- Never expose `SUPABASE_SERVICE_ROLE_KEY` (or any service-role key) in browser/client env vars.

See `docs/saas_staging_deploy.md` for required environment variables and warnings.
