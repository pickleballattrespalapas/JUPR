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

`GET /admin/operations/status?club_id=...` powers the Next `/admin` cockpit.
It requires a verified Supabase bearer token and at least one matching,
club-scoped `admin_role_assignments` row before building or returning
operational posture. Existing email-only assignments remain compatible; when a
row has `user_id`, it must exactly match the JWT subject. Anonymous,
wrong-club, and mismatched-user requests fail closed.

Closed-club production-write pilot mode is controlled by:

- `JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=1`

Individual workflow flags are intentionally separate:

- `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG=1` enables Match Log read/scan/planning visibility.
- `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1` enables authenticated/audited Match Log edit and duplicate-cleanup routes.
- `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1`
- `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER=1`
- `JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR=1`
- `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1`
- `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=1` enables only the persisted League Awards mutations and additionally requires `SUPABASE_SERVICE_ROLE_KEY` on FastAPI. Mint still fails closed until all four top-performer badge definitions from `supabase/migrations/20260720014744_seed_top_performer_badges.sql` are readable; the seed also aligns present legacy `_v2` compatibility columns on newly inserted rows. Keep it off in production until the manual staging gate passes.
- `JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER=1`
- `JUPR_ENABLE_STAGING_NEXT_ADMIN_CHALLENGE_LADDER_WRITES=1` permits Challenge Ladder mutations only when `JUPR_ENV=staging`.
  Played-result publish additionally requires `supabase/migrations/20260725231000_challenge_ladder_public_results.sql`; its service-role-only operation-bound RPC atomically inserts both matches, applies the exact reviewed player/league rating plan, performs the collision-safe rank change, completes the challenge, records two exact public match IDs, and writes the response-loss receipt. Recoverable badge/update post-processors never replay that core. Legacy/imported and forfeit rows remain nullable and are never assigned inferred match or rating relations.
- `JUPR_ENABLE_NEXT_ADMIN_MONEYBALL=1` enables the Python-authoritative Moneyball preview/settlement surface.
- `JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES=1` permits Moneyball official publish only when `JUPR_ENV=staging`.
- `JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE=1` enables one-off JUPR Live administration; Tournament Live remains separate.
- `JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES=1` permits one-off session mutations only when `JUPR_ENV=staging`.
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS=1`
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS=1`, `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS=1`, and `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS=1` independently open the order-26 staging-only mutation surfaces. They require `JUPR_ENV=staging`, `SUPABASE_SERVICE_ROLE_KEY`, the private operation migration, reviewed row/state versions, and strict audit intent/completion. Production refuses them.
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF=1` documents the separate staging Operations handoff gate; Registration Admin itself remains no-write for draw teams.
- `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS=1` enables Order-27 draw/import/scoring/playoff/podium/award mutations in staging only. Official publishing additionally requires `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH=1`; automatic player-update handoff additionally requires `JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF=1` and `JUPR_EMAIL_MODE=dry_run|staging_redirect`.
- `JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES=1` opens only the draw-scoped Order-28 in-play runner in `JUPR_ENV=staging`. It requires the order-26 and order-28 private operation migrations, a reviewed draw fingerprint, an exact idempotency UUID, required audit writes, and the FastAPI-only service role. Production and local environments remain read-only.
- `JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP=1` and `JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES=1` expose the authenticated communications read surfaces.
- `JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS=1` permits their POST/PATCH actions only with `JUPR_ENV=staging` and the exact `communications` write wave. Keep it `0` at rest and in production.
- `JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0` keeps Next live delivery blocked independently of the admin UI flag
- `JUPR_ENABLE_NEXT_ADMIN_TOOLS=1`

Do not enable high-risk workflow flags broadly until the workflow has server-side FastAPI writes, staff-only auth, club scoping, audit attribution, a correction/replay path when rating-adjacent, and Streamlit fallback.

## Endpoints

- `GET /health`
- `GET /admin/operations/status?club_id=...` guarded by Supabase JWT plus a bound admin assignment
- `GET /admin/clubs/{club_id}/match-log`
- `PATCH /admin/clubs/{club_id}/match-log/edits` guarded by `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY` plus Supabase JWT role authorization
- `POST /admin/clubs/{club_id}/match-log/edits/{operation_id}/recover` guarded replay recovery for an atomically committed edit
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
- `GET /clubs/{club_slug}/tournament-registration/pairing-requests?edit_token=...`
- `POST /clubs/{club_slug}/tournament-registration/pairing-requests/{request_id}/accept`
- `POST /clubs/{club_slug}/tournament-registration/pairing-requests/{request_id}/decline`
- `POST /clubs/{club_slug}/tournament-registration/pairing-requests/{request_id}/cancel`
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
- `GET /admin/clubs/{club_id}/league-manager/leagues/{league_name}/printout?week_num=...` authenticated, read-only weekly/season leader print model
- `GET /admin/clubs/{club_id}/league-manager/top-players-printable?limit=...` authenticated, read-only previous-calendar-month ranking model

League Manager create, duplicate, lifecycle, settings, and roster mutations fail closed unless FastAPI has `SUPABASE_SERVICE_ROLE_KEY`; the anonymous/publishable key is never accepted as their mutation credential.

Challenge Ladder, Moneyball, and JUPR Live mutations additionally require the private `live_ladder_admin_operations` migration, a stable idempotency key, the current Python version/fingerprint, strict intent/completion/failure audit writes, and the surface-specific staging write flag. Operation status/reconcile routes live below each surface at `/operations/{operation_key}`. See `docs/live_ladder_parity_runbook.md` for exact phrases, response-loss handling, Match Log/Replay recovery, and disposable staging evidence.
- `GET /admin/clubs/{club_id}/weekly-recap/recaps` and guarded generate/save/publish routes
- `GET /admin/clubs/{club_id}/player-updates/workspace`
- `POST /admin/clubs/{club_id}/player-updates/digests/preview` and `/digests/queue`
- `POST /admin/clubs/{club_id}/player-updates/outbox/send`, `/retry`, and `/delete`
- `POST /admin/clubs/{club_id}/player-updates/subscriptions/{subscription_id}/replace` and `/deactivate`

The recap and communications admin routes require `SUPABASE_SERVICE_ROLE_KEY` on FastAPI and reject stale versions. Keep email acceptance in `dry_run` or `staging_redirect`; see `docs/communications_parity_runbook.md`.

Tournament Setup/Admin mutations also require the FastAPI-only service role and separate staging flags. Exact retries use deterministic operation keys and stored results; any exception after intent is recovery-required, never blindly retryable. See `docs/tournament_admin_parity_evidence.md`.

Tournament Operations uses route-specific Next surfaces under `/admin/tournaments/ops/{draws,import,results,publish}` and Python-authoritative endpoints under `/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}`. Official-publish response loss is reconciled only from an exact club/tournament-scoped set of all expected `tournament_game_id` matches; zero or partial evidence stays recovery-required and never re-runs the publisher. See `docs/tournament_operations_parity_evidence.md`.

`GET /clubs/{club_slug}` is backed by `public.clubs` (slug-first lookup with id fallback) and returns a normalized public club contract (`id`, `slug`, `name`, `tagline`, `support_email`, `public_base_url`, `logo_url`, `primary_color`, `is_active`). Tres Palapas default slug is `tres-palapas`.

The leaderboard endpoint delegates to `jupr_app.services.leaderboard_service.build_public_leaderboard`. `OVERALL` is built from the club player ratings; league tabs use the selected league-rating projection. Active players are the default. The explicit public response adds starting/gain/gap values, qualification, capped badge context, highlights, a selected-player snapshot, and offset pagination without forwarding raw player or badge rows.
