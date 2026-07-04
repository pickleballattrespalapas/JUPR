# JUPR API (FastAPI)

FastAPI service for the JUPR SaaS surface.

Current production scope: public club/leaderboard/league-results/badge-codex/challenge-ladder/weekly-recap/tournament-registration/player/match/live/Match Explorer endpoints, status/admin planning endpoints, and guarded admin write endpoints that remain disabled by default.

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
- `docs/fly_api_deploy.md`

The preferred production deploy path is online-only through GitHub Actions:

`Actions` -> `Deploy FastAPI backend to Fly` -> `Run workflow`

The Fly service runs from the repo root and starts:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

The default Fly app name in `fly.toml` is `juprleagues-api`. The GitHub Actions workflow lets the operator override the app name and primary region at run time without editing files locally.

## Environment variables

Production backend runtime:

- `JUPR_ENV=production`
- `SUPABASE_URL=<production Supabase project URL>`
- `SUPABASE_SERVICE_ROLE_KEY=<production Supabase service role key>`
- `SUPABASE_ANON_KEY=<production anon key if needed>`
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
- `GET /clubs/{club_slug}/leaderboards?league_name=...`
- `GET /clubs/{club_slug}/leaderboards/public?league_name=...` temporary compatibility alias
- `GET /clubs/{club_slug}/league-results?league_name=...`
- `GET /clubs/{club_slug}/badges`
- `GET /clubs/{club_slug}/badges/{badge_id}/earners?offset=...&limit=...`
- `GET /clubs/{club_slug}/challenge-ladder`
- `GET /clubs/{club_slug}/weekly-recaps?week_start=...`
- `GET /clubs/{club_slug}/weekly-recaps/{week_start}`
- `GET /clubs/{club_slug}/weekly-recaps/{week_start}/pdf`
- `GET /clubs/{club_slug}/tournament-registration?registration_slug=...&tournament_id=...`
- `POST /clubs/{club_slug}/tournament-registration`
- `GET /clubs/{club_slug}/tournament-registration/confirmations/{registration_id}`
- `GET /clubs/{club_slug}/players`
- `GET /clubs/{club_slug}/players/{player_id}`
- `GET /clubs/{club_slug}/matches`
- `GET /clubs/{club_slug}/matches/{match_id}`
- `GET /clubs/{club_slug}/players/{player_id}/matches`
- `GET /clubs/{club_slug}/live-sessions`
- `GET /clubs/{club_slug}/live-sessions/{session_key}`
- `GET /clubs/{club_slug}/match-explorer`
- `GET /clubs/{club_slug}/match-explorer/preview?me=...&partner=...&opp1=...&opp2=...`
- `POST /admin/clubs/{club_id}/matches/batch` guarded/disabled by default

`GET /clubs/{club_slug}` is backed by `public.clubs` (slug-first lookup with id fallback) and returns a normalized public club contract (`id`, `slug`, `name`, `tagline`, `support_email`, `public_base_url`, `logo_url`, `primary_color`, `is_active`). Tres Palapas default slug is `tres-palapas`.

The leaderboard endpoint delegates to `jupr_app.services.leaderboard_service.get_public_leaderboard` and only returns public-safe fields.

The League Results endpoint delegates to `jupr_app.services.public_league_results_service` and returns public-safe league options, standings, weekly summaries, cumulative player results, and highlights. It does not write matches.

The Badge Codex endpoints delegate to `jupr_app.services.public_badge_codex_service` and return public-safe badge definitions, unlock paths, prestige, grouped sections, recent earners, and paginated earner rows. They do not expose badge evaluator internals or private player fields.

The Challenge Ladder endpoint delegates to `jupr_app.services.public_challenge_ladder_service` and returns public-safe ladder tiers, player statuses, active challenge buckets, quick rules, and summarized challenge rows. It does not create challenges, write scores, expose ledger contacts, or mutate ranks.

The Weekly Recap endpoints delegate to `jupr_app.services.public_weekly_recap_service` and return published-only recap summaries, sanitized `final_json` detail, and dependency-free PDF bytes. They do not expose drafts, generated/edit JSON, private notes, or admin publishing controls.

The Tournament Registration endpoints delegate to `jupr_app.services.public_tournament_registration_service` and expose published tournament registration settings, selectable divisions, public confirmation summaries, and guarded registration intake. They do not create draws, seed brackets, enter scores, update ratings, expose admin notes, or expose service-role credentials to Vercel/browser code.

The admin operations status endpoint delegates to `jupr_app.services.admin_operations_service` and returns workflow readiness plus pilot guardrails for the Next admin operations cockpit. It is status-only and does not mutate data.

The admin Match Log endpoints delegate to `jupr_app.services.admin_match_log_service`. Read/scan visibility is enabled by `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG`. Edits and duplicate cleanup require `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY`, Supabase JWT auth, club-scoped role authorization, and Python domain services.

The Match Explorer endpoints delegate to `jupr_app.services.public_match_explorer_service` and return public-safe player names/ratings plus projected matchup odds and rating movement. They do not write matches and do not move rating logic to JavaScript.

## Admin score-entry guard

Admin score-entry remains disabled by default.

`POST /admin/clubs/{club_id}/matches/batch` returns 403 unless `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1` / `true` / `yes`.

Keep `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` unless the specific staff pilot has auth, club-scoped authorization, audit review, and a correction path.

When enabled in controlled environments, this route requires Supabase JWT Bearer auth and role-based authorization for `enter_scores`.

Auth design reference for future replacement of the temporary guard:

- `docs/next_admin_auth_design.md`

## CORS

Allowed browser origins are configured by `JUPR_ALLOWED_ORIGINS` as a comma-separated list. If unset, the API allows local Next.js development plus:

- `https://juprleagues.com`
- `https://www.juprleagues.com`

For Vercel preview testing, add the preview deployment origin to `JUPR_ALLOWED_ORIGINS` in the backend runtime.

## Admin JWT auth (Supabase)

- `PATCH /admin/clubs/{club_id}/match-log/edits` requires `Authorization: Bearer <access_token>` when the apply flag is enabled and requires `manage_matches` permission.
- `POST /admin/clubs/{club_id}/match-log/duplicates/cleanup` requires `Authorization: Bearer <access_token>` when the apply flag is enabled and requires `delete_matches` permission.
- `POST /admin/clubs/{club_id}/matches/batch` requires `Authorization: Bearer <access_token>` when the feature flag is enabled.
- Feature flags remain disabled by default.
- Disabled write endpoints return `403` before auth and writes.
- JWT verification config (server-side only):
  - Secret mode (default): `SUPABASE_JWT_SECRET` (+ optional `SUPABASE_JWT_AUDIENCE`, default `authenticated`).
  - JWKS mode: `JUPR_SUPABASE_JWT_MODE=jwks` + `SUPABASE_JWKS_URL` (+ optional audience).
- Do not use service-role or JWT secrets in browser/client code.

## Admin audit logging

- Successful admin match-batch and Match Log writes attempt to write `admin_activity_log` records with actor + club attribution.
- Denied authenticated writes are flagged for review in audit logs when safe to do so.
- `JUPR_REQUIRE_API_AUDIT_LOG=1` enables strict mode: write requests fail if audit logging cannot be recorded where strict mode is supported.
- Default behavior degrades gracefully when audit table migration is missing.
- Audit payloads include summary metadata (`source_page`, `source_client`) and do not include raw bearer tokens or secrets.
