# JUPR API (FastAPI)

FastAPI service for the JUPR SaaS surface.

Current production scope: public club/leaderboard/league-results/badge-codex/challenge-ladder/weekly-recap/tournament-registration/tournament-roster/player/match/live/Match Explorer endpoints, status/admin planning endpoints, and guarded admin write endpoints that remain disabled by default.

## Run locally

```bash
pip install -r requirements.txt
pip install -r services/api/requirements.txt
uvicorn services.api.main:app --reload
```

## Fly production deployment

The repo includes Fly-ready files at the repository root:

- `Dockerfile.api`
- `fly.toml`
- `.dockerignore`
- `.github/workflows/fly_api_deploy.yml`
- `fly.staging.toml`
- `.github/workflows/fly_api_staging_deploy.yml`
- `docs/fly_api_deploy.md`

The preferred production deploy path is online-only through GitHub Actions:

`Actions` -> `Deploy FastAPI backend to Fly` -> `Run workflow`

The Fly service runs from the repo root and starts:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

The default Fly app name in `fly.toml` is `juprleagues-api`. The GitHub Actions workflow lets the operator override the app name and primary region at run time without editing files locally.

Full migration testing uses the separate `Deploy FastAPI staging to Fly`
workflow. It deploys `fly.staging.toml` to `juprleagues-api-staging`, validates
the staging Supabase project ref before deploy, enables all current admin
workflows, keeps email in dry-run mode, and runs the exhaustive staging verifier.
It refuses the production Fly app name.

## Environment variables

Production backend runtime:

- `JUPR_ENV=production`
- `SUPABASE_URL=<production Supabase project URL>`
- `SUPABASE_SERVICE_ROLE_KEY=<production Supabase service role key>`
- `SUPABASE_ANON_KEY=<production anon key if needed>`
- `JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com` (Next.js origin used in server-generated links)
- `JUPR_REGISTRATION_CONFIRMATION_SECRET=<server-only high-entropy signing secret>`
- `JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=0`
- `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=0`
- `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=0`
- `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0`
- `JUPR_REQUIRE_API_AUDIT_LOG=0` or `1` once strict audit is required for enabled writes
- `JUPR_STREAMLIT_FALLBACK_URL=https://juprtrespalapas.streamlit.app`
- `JUPR_ALLOWED_ORIGINS=https://juprleagues.com,https://www.juprleagues.com`

For the online-only deploy workflow, set these GitHub Actions repository secrets before running the workflow:

- `FLY_API_TOKEN`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_ANON_KEY`

Staging backend runtime should use `JUPR_ENV=staging` and staging Supabase credentials only.
Set `JUPR_WEB_BASE_URL` to the staging Vercel origin and keep
`JUPR_EMAIL_MODE=dry_run` (or an explicitly configured staging redirect) until
the registration-email smoke test is approved. The confirmation signing secret
must be stable within each environment and must never be exposed to Vercel or a
browser bundle.

Never put `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET`, or other server-only secrets in Vercel frontend environment variables.

## Admin operations migration flags

`GET /admin/operations/status` powers the Next `/admin` cockpit. It is status-only and public-safe: it returns environment, pilot mode, enabled workflow keys, safety gates, and boolean backend readiness, but not credentials.

Closed-club production-write pilot mode is controlled by:

- `JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=1`

Individual workflow flags are intentionally separate:

- `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1` enables Match Log read/scan/planning visibility.
- `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1` enables authenticated/audited Match Log edit and duplicate-cleanup routes.
- `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1`
- `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER=1`
- `JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR=1`
- `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1`
- `JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER=1`
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS=1`
- `JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP=1`
- `JUPR_ENABLE_NEXT_ADMIN_TOOLS=1`

Do not enable high-risk workflow flags broadly until the workflow has server-side FastAPI writes, staff-only auth, club scoping, audit attribution, a correction/replay path when rating-adjacent, and Streamlit fallback.

## Endpoints

- `GET /health`
- `GET /admin/operations/status`
- `GET /admin/clubs/{club_id}/match-log`
- `PATCH /admin/clubs/{club_id}/match-log/edits` guarded by `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY` plus Supabase JWT role authorization
- `POST /admin/clubs/{club_id}/match-log/duplicates/cleanup` guarded by `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY` plus Supabase JWT role authorization
- `GET /clubs/{club_slug}`
- `GET /clubs/{club_slug}/leaderboards?league_name=OVERALL&status=active&q=...&sort=rank&player_id=...&limit=50&offset=0`
- `GET /clubs/{club_slug}/leaderboards/public?...` temporary compatibility alias with the same filters and response
- `GET /clubs/{club_slug}/league-results?league_name=...&week=...&player=...&weekly_min_games=...`
- `GET /clubs/{club_slug}/badges`
- `GET /clubs/{club_slug}/badges/{badge_id}/earners?offset=...&limit=...`
- `GET /clubs/{club_slug}/challenge-ladder`

The Badge Codex response projects canonical Python badge status, timing, scope,
unlock copy, lifecycle availability buckets, and public-safe trophy summaries; raw
badge-evaluation context and inactive players are not returned. The Challenge
Ladder response computes statuses and eligible-opponent hints in Python and
returns the same structured rulebook/status policy consumed by the Next route.
- `GET /clubs/{club_slug}/weekly-recaps?week_start=...&page=1&page_size=8` (published-only, page size capped at 12)
- `GET /clubs/{club_slug}/weekly-recaps/{week_start}`
- `GET /clubs/{club_slug}/weekly-recaps/{week_start}/pdf`
- `GET /clubs/{club_slug}/tournament-registration?registration_slug=...&tournament_id=...`
- `POST /clubs/{club_slug}/tournament-registration`
- `POST /clubs/{club_slug}/tournament-registration/edit-link/request`
- `GET /clubs/{club_slug}/tournament-registration/edit?edit_token=...`
- `POST /clubs/{club_slug}/tournament-registration/edit`
- `GET /clubs/{club_slug}/tournament-registration/confirmation?confirmation_token=...`
- `GET /clubs/{club_slug}/tournament-roster?registration_slug=...&tournament_id=...`
- `POST /clubs/{club_slug}/tournament-registration/pairing-interest` with token-gated `requester_selection_id` and public `board_entry_key`
- `GET /clubs/{club_slug}/players?q=...&status=active|inactive|all&sort=rating|singles|matches|name|win_pct|recent&limit=...&offset=...` public-display-only directory; defaults to active players
- `GET /clubs/{club_slug}/players/{player_id}?recent_limit=...&history_limit=...` privacy-safe profile projection with rating trend/breakdowns, awards, relationships, Club Social aggregates, verified-update state, and explicit match-format history
- `GET /clubs/{club_slug}/matches`
- `GET /clubs/{club_slug}/matches/{match_id}`
- `GET /clubs/{club_slug}/players/{player_id}/matches`
- `GET /clubs/{club_slug}/live-sessions`
- `GET /clubs/{club_slug}/live-sessions/{session_key}`
- `GET /clubs/{club_slug}/match-explorer`
- `GET /clubs/{club_slug}/match-explorer/preview?me=...&partner=...&opp1=...&opp2=...` (Python-authoritative expected score, deltas, player projections, and impact chart)
- `POST /clubs/{club_slug}/support/intake` durable general-support, data-correction, and profile-privacy intake; exact retries are deduplicated and hourly limits are enforced server-side
- `GET /admin/clubs/{club_id}/support-requests` guarded club-scoped review queue
- `PATCH /admin/clubs/{club_id}/support-requests/{request_id}` guarded stale-safe status/privacy-fulfillment update
- `POST /admin/clubs/{club_id}/matches/batch` guarded/disabled by default

`GET /clubs/{club_slug}` is backed by `public.clubs` (slug-first lookup with id fallback) and returns a normalized public club contract (`id`, `slug`, `name`, `tagline`, `support_email`, `public_base_url`, `logo_url`, `primary_color`, `is_active`). Tres Palapas default slug is `tres-palapas`.

The leaderboard endpoint delegates to `jupr_app.services.leaderboard_service.build_public_leaderboard`. `OVERALL` is built from the club player ratings; league tabs use the selected league-rating projection. Active players are the default. The explicit public response adds starting/gain/gap values, qualification, capped badge context, highlights, a selected-player snapshot, and offset pagination without forwarding raw player or badge rows.
