# Pickleball Club Sandwich Web (Next.js public shell)

This is the Next.js app for the public Pickleball Club Sandwich web surface. It is intended to serve `pickleballclubsandwich.com` as the primary domain, with `juprleagues.com` kept as a transitional alias while public naming is migrated.

## Current deployment posture

The web app is **public-first with guarded admin migration surfaces**.

- Production Streamlit remains the trusted admin fallback runtime during migration.
- FastAPI supplies public read APIs, guarded public intake APIs, status-only admin migration APIs, and guarded admin endpoints.
- Next.js renders public pages and the admin operations cockpit against FastAPI.
- Next admin write workflows remain disabled unless explicitly enabled for controlled staging or closed-club production-write pilot experiments.
- Never put Supabase service-role keys, JWT secrets, or database credentials in Vercel/frontend environment variables.

## Routes

Current public and staff-facing routes include:

- `/` public product home.
- `/site-map` public click-through route map.
- `/clubs/[clubSlug]` club landing page.
- `/clubs/[clubSlug]/leaderboards` public leaderboard page.
- `/clubs/[clubSlug]/league-results` public league standings, weekly results, and season summaries.
- `/clubs/[clubSlug]/badge-codex` public badge definitions, unlock paths, prestige, and recent badge earners.
- `/clubs/[clubSlug]/challenge-ladder` public ladder tiers, player status, active challenge buckets, and quick rules.
- `/clubs/[clubSlug]/weekly-recap` public published weekly recap with spotlight reel, around-the-club highlights, tournament podiums, print view, and PDF download.
- `/clubs/[clubSlug]/tournament-registration` public tournament registration intake, including secure edit-link request.
- `/clubs/[clubSlug]/tournament-registration/confirmation?registration_id=...` public registration confirmation page.
- `/clubs/[clubSlug]/tournament-registration/edit?edit_token=...` tokenized registration edit page.
- `/clubs/[clubSlug]/tournament-roster` public-safe tournament roster.
- `/clubs/[clubSlug]/tournament-partner-board` public-safe tournament board with token-gated interest flow.
- `/clubs/[clubSlug]/match-explorer` public matchup odds and projected rating movement preview.
- `/clubs/[clubSlug]/players` public player directory.
- `/clubs/[clubSlug]/players/[playerId]` public player profile.
- `/clubs/[clubSlug]/players/[playerId]/matches` public player match history.
- `/clubs/[clubSlug]/matches` public match history.
- `/clubs/[clubSlug]/matches/[matchId]` public match detail.
- `/clubs/[clubSlug]/live` public live-session list.
- `/clubs/[clubSlug]/live/[sessionKey]` public live-session detail.
- `/how-ratings-work` public ratings explainer.
- `/faq` public rating FAQ.
- `/privacy` first-party privacy policy copy.
- `/terms` first-party terms copy.
- `/support` and `/contact` support/contact shell routed to `joe@juprleagues.com`.
- `/data-corrections` public correction intake instructions with no direct mutation.
- `/admin` staff operations cockpit for the Streamlit-to-Next migration.
- `/admin/match-log`, `/admin/replay-history`, `/admin/match-uploader`, `/admin/players`, and `/admin/league-manager` guarded staff migration surfaces.
- `/clubs/[clubSlug]/admin/score-entry` staging-only score-entry MVP, still feature-flagged and not production-active by default.

## Environment variables

Use one of the following API base URL variables:

- `JUPR_API_BASE_URL` (preferred for server-side runtime)
- `NEXT_PUBLIC_JUPR_API_BASE_URL` (fallback and browser-visible runtime value)
- `NEXT_PUBLIC_JUPR_ENV` (optional; set to `staging` to show a staging badge)
- `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` (keep disabled outside controlled staging)

Use this public web URL variable for metadata and sitemap generation:

- `NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com`

The `/admin` cockpit reads `GET /admin/operations/status` from FastAPI. Its workflow flags live on the API deployment, not in Vercel. `/admin/match-log` reads `GET /admin/clubs/{club_id}/match-log` and shows fallback instructions until `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1` is enabled on FastAPI.

Example:

```bash
export JUPR_API_BASE_URL=http://localhost:8000
export NEXT_PUBLIC_JUPR_API_BASE_URL=http://localhost:8000
export NEXT_PUBLIC_JUPR_WEB_BASE_URL=http://localhost:3000
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

## Vercel staging and production deployment

Use a separate Vercel project or staging environment before custom-domain cutover.

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
NEXT_PUBLIC_JUPR_WEB_BASE_URL=<staging Vercel URL>
NEXT_PUBLIC_JUPR_ENV=staging
NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1
```

The staging API must be the dedicated `juprleagues-api-staging` Fly app backed
by staging Supabase. Keep the score-entry browser flag disabled in production;
it is enabled in preview solely so the full workflow can be tested.

Production web environment variables should use the primary domain after Vercel verification:

```bash
NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com
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

Closed-club production-write pilot work may enable one workflow at a time through FastAPI-side flags. Keep the permanent safety boundaries from `docs/next_admin_operations_migration.md` in place: no browser secrets, no direct browser writes to rating tables, and no JavaScript rating logic.
