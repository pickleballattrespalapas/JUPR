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
- `/clubs/[clubSlug]/leaderboards` public Overall/league leaderboards with Active-by-default status, search/snapshot links, qualification and badge context, stable filters, and pagination.
- `/clubs/[clubSlug]/league-results` public league standings, weekly results, and season summaries.
- `/clubs/[clubSlug]/badge-codex` Python-authoritative badge definitions grouped by availability/timing, category and scope filters, direct anchors, paged public earners, and privacy-safe trophy-room context.
- `/clubs/[clubSlug]/challenge-ladder` public tiers/status/challenge deep links, Python-computed eligible-opponent hints, a complete rulebook, and the canonical status legend.
- `/clubs/[clubSlug]/weekly-recap` public published weekly recap with spotlight reel, around-the-club highlights, tournament podiums, print view, and PDF download.
- `/clubs/[clubSlug]/tournament-registration` four-step public tournament registration intake with exact-match profile preflight, duplicate/closed recovery, partner policy, sponsor/refund/roster content, and secure edit-link request.
- `/clubs/[clubSlug]/tournament-registration/confirmation?confirmation_token=...` signed, registrant-specific confirmation page.
- `/clubs/[clubSlug]/tournament-registration/edit?edit_token=...` tokenized registration edit page.
- `/clubs/[clubSlug]/tournament-roster` public-safe tournament roster.
- `/clubs/[clubSlug]/tournament-partner-board` public-safe tournament board with token-gated interest flow.
- `/clubs/[clubSlug]/match-explorer` public matchup odds and projected rating movement preview.
- `/clubs/[clubSlug]/players` active-by-default public player directory with visible search, status/sort controls, paging, and stable profile/row links.
- `/clubs/[clubSlug]/players/[playerId]` public-display-only profile with rating trend and format/league breakdowns, badges/trophies, partner/rival summaries, Club Social aggregates, verified-update entry, and recent/full match history.
- `/clubs/[clubSlug]/players/[playerId]/matches` public player match history.
- `/clubs/[clubSlug]/matches` public match history.
- `/clubs/[clubSlug]/matches/[matchId]` public match detail.
- `/clubs/[clubSlug]/live` public live-session list.
- `/clubs/[clubSlug]/live/[sessionKey]` public live-session detail.
- `/how-ratings-work` public ratings explainer.
- `/faq` public rating FAQ.
- `/privacy` first-party privacy policy copy.
- `/terms` first-party terms copy.
- `/support` and `/contact` durable general-support intake plus a populated email fallback.
- `/data-corrections` public correction intake with no direct data mutation.
- `/profile-privacy` identity-reviewed privacy fulfillment intake with no direct public-profile mutation.
- `/admin` staff operations cockpit for the Streamlit-to-Next migration.
- `/admin/match-log`, `/admin/replay-history`, `/admin/match-uploader`, `/admin/players`, and `/admin/league-manager` guarded staff migration surfaces.
- `/clubs/[clubSlug]/admin/score-entry` staging-only score-entry MVP, still feature-flagged and not production-active by default.

## Environment variables

Use one of the following API base URL variables:

- `JUPR_API_BASE_URL` (preferred for server-side runtime)
- `NEXT_PUBLIC_JUPR_API_BASE_URL` (fallback and browser-visible runtime value)
- `NEXT_PUBLIC_JUPR_ENV` (optional; set to `staging` to show a staging badge)
- `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` (keep disabled outside controlled staging)
- `JUPR_STAGING_API_BASE_URL` (optional preview override; defaults to the dedicated Fly staging API)
- `NEXT_PUBLIC_STAGING_SUPABASE_URL` and `NEXT_PUBLIC_STAGING_SUPABASE_ANON_KEY` (preview-only staging auth project; never use service-role credentials)
- `JUPR_SUPPORT_EMAIL` (server-rendered support/contact fallback; defaults to `joe@juprleagues.com`)

Use this public web URL variable for metadata and sitemap generation:

- `NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com`

Set the matching server-side `JUPR_WEB_BASE_URL` on FastAPI/Fly. Player update
emails use it to generate tokenized Next `/email-preferences` links. The API
fails the individual outbox send closed when no unsubscribe token is available;
it never falls back to a public subscription ID.

The `/admin` cockpit reads `GET /admin/operations/status` from FastAPI. Its workflow flags live on the API deployment, not in Vercel. `/admin/match-log` reads `GET /admin/clubs/{club_id}/match-log` and shows fallback instructions until `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1` is enabled on FastAPI. When apply mode is enabled, guided and bulk edits carry stable idempotency keys; rating-affecting edits expose their durable Replay History job and block further editing when mandatory recovery is required.

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

Vercel preview deployments are isolated in `next.config.js`: when
`VERCEL_ENV=preview`, both API base variables are forced to
`https://juprleagues-api-staging.fly.dev`, the staging badge and score-entry UI
are enabled, and `/api/environment` reports only the sanitized environment,
API origin, and isolation state. Production deployments do not receive these
overrides and retain their existing Vercel production configuration.

For functional staff login in previews, scope
`NEXT_PUBLIC_STAGING_SUPABASE_URL` and
`NEXT_PUBLIC_STAGING_SUPABASE_ANON_KEY` to Vercel Preview. The config remaps them
to the canonical browser auth variables only for preview builds. If they are
missing, staging API writes still fail closed because production-project tokens
cannot validate against the staging API project; `/api/environment` reports
`preview_auth_isolation_active=false` until the preview auth variables are set.

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

That workflow also runs a Chromium smoke over the critical public and admin
shell routes, rejects production domains, fails on browser/page errors, and
asserts `/api/environment` is an isolated Vercel Preview backed by the staging
API and staging Supabase Auth project. Configure these `staging` environment
secrets before using the full gate:

- `STAGING_WEB_BASE_URL`: the protected Vercel Preview or staging alias;
- `STAGING_SUPABASE_URL`: the exact staging Auth origin the Preview must report;
- `VERCEL_AUTOMATION_BYPASS_SECRET`: the Vercel Deployment Protection
  automation bypass value.

For a local Preview-mode browser check, install Chromium once and run:

```bash
cd apps/web
npx playwright install chromium
npm run test:e2e:staging
```

## Admin score-entry guard

The score-entry page at `/clubs/[clubSlug]/admin/score-entry` is an MVP surface only. Rated admin writes remain blocked unless the backend feature flag is explicitly enabled and Supabase JWT role authorization succeeds.

Closed-club production-write pilot work may enable one workflow at a time through FastAPI-side flags. Keep the permanent safety boundaries from `docs/next_admin_operations_migration.md` in place: no browser secrets, no direct browser writes to rating tables, and no JavaScript rating logic.

Admin login also requires `NEXT_PUBLIC_JUPR_API_BASE_URL` so the browser can exchange a Supabase user JWT for the caller's server-verified JUPR club capabilities. Optional `NEXT_PUBLIC_JUPR_ADMIN_CLUB_ID` selects the requested workspace and defaults to `tres_palapas`. Supabase Auth recovery redirects must allow the exact `/admin/reset-password` URL for each deployed origin; recovery requests use PKCE and the Next page retains legacy implicit recovery links only as a fallback.
