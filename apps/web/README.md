# JUPR Web (Next.js public shell)

This is the Next.js app for the public JUPR Leagues web surface. It is intended to serve `juprleagues.com` and club pages after the read-only FastAPI + Next.js path is proven in staging.

## Current deployment posture

The web app is **staging/read-only first**.

- Production Streamlit remains the trusted production runtime.
- FastAPI supplies public read APIs and guarded admin endpoints.
- Next.js renders public pages against FastAPI.
- Next admin score entry remains disabled unless explicitly enabled for controlled staging experiments.
- Never put Supabase service-role keys, JWT secrets, or database credentials in Vercel/frontend environment variables.

## Routes

Current public routes include:

- `/` public JUPR landing page with SaaS pilot shell navigation.
- `/clubs/[clubSlug]` club landing page.
- `/clubs/[clubSlug]/leaderboards` public leaderboard page.
- `/clubs/[clubSlug]/league-results` public league standings, weekly results, and season summaries.
- `/clubs/[clubSlug]/badge-codex` public badge definitions, unlock paths, prestige, and recent badge earners.
- `/clubs/[clubSlug]/challenge-ladder` public ladder tiers, player status, active challenge buckets, and quick rules.
- `/clubs/[clubSlug]/weekly-recap` public published weekly recap with spotlight reel, around-the-club highlights, tournament podiums, print view, and PDF download.
- `/clubs/[clubSlug]/tournament-registration` public tournament registration intake for published events.
- `/clubs/[clubSlug]/tournament-registration/confirmation?registration_id=...` public registration confirmation page.
- `/clubs/[clubSlug]/match-explorer` public matchup odds and projected rating movement preview.
- `/clubs/[clubSlug]/players` public player directory.
- `/clubs/[clubSlug]/players/[playerId]` public player profile.
- `/clubs/[clubSlug]/matches` public match history.
- `/clubs/[clubSlug]/matches/[matchId]` public match detail.
- `/clubs/[clubSlug]/live` public JUPR Live session list.
- `/clubs/[clubSlug]/live/[sessionKey]` public JUPR Live detail.
- `/how-ratings-work` public ratings explainer.
- `/faq` public JUPR rating FAQ.
- `/privacy` first-party privacy notice placeholder pending legal review.
- `/terms` first-party terms placeholder pending legal review.
- `/support` and `/contact` support/contact shell.
- `/data-corrections` public correction intake instructions with no direct mutation.
- `/admin` admin entry/fallback page.
- `/clubs/[clubSlug]/admin/score-entry` staging-only score-entry MVP, still feature-flagged and not production-active.

## Environment variables

Use one of the following API base URL variables:

- `JUPR_API_BASE_URL` (preferred for server-side runtime)
- `NEXT_PUBLIC_JUPR_API_BASE_URL` (fallback and browser-visible runtime value)
- `NEXT_PUBLIC_JUPR_ENV` (optional; set to `staging` to show a staging badge)
- `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` (keep disabled outside controlled staging)

Example:

```bash
export JUPR_API_BASE_URL=http://localhost:8000
export NEXT_PUBLIC_JUPR_API_BASE_URL=http://localhost:8000
```

See `.env.example` for the local/staging template.

## Local development

```bash
cd apps/web
npm install
npm run dev
```

Then open `http://localhost:3000`.

If the API is unavailable, pages should render graceful empty/error states instead of crashing.

## Vercel staging deployment

Use a separate Vercel project or staging environment before any custom-domain cutover.

Recommended Vercel settings:

- **Root Directory:** `apps/web`
- **Install Command:** `npm install`
- **Build Command:** `npm run build`
- **Output Directory:** leave unset for Next.js defaults
- **Framework Preset:** Next.js

Required staging web environment variables:

```bash
JUPR_API_BASE_URL=<staging FastAPI URL>
NEXT_PUBLIC_JUPR_API_BASE_URL=<staging FastAPI URL>
NEXT_PUBLIC_JUPR_ENV=staging
NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0
```

Do **not** configure `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET`, or other server-only secrets in Vercel.

If a Vercel preview build is rate-limited, rerun the preview after quota resets and wait for a fresh green Vercel status before merging frontend PRs.

## Public smoke checks

From the repository root:

```bash
export STAGING_JUPR_API_BASE_URL=<staging FastAPI URL>
export STAGING_WEB_BASE_URL=<Vercel staging URL>
make public-web-smoke
```

While live-session migrations or grants are still being applied in staging, allow the live-session endpoint to report its setup error without failing the whole smoke run:

```bash
JUPR_SMOKE_ARGS=--allow-live-unconfigured make public-web-smoke
```

The same check is available through the manual GitHub Actions workflow `Staging Smoke`.

## Admin score-entry guard

The score-entry page at `/clubs/[clubSlug]/admin/score-entry` is an MVP surface only. Rated admin writes remain blocked unless the backend feature flag is explicitly enabled and Supabase JWT role authorization succeeds.

Keep production public launch read-only until the auth, club-scoped authorization, audit, and E2E safety criteria in `docs/next_admin_auth_design.md` are complete.
