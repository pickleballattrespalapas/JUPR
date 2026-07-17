# Next/Vercel Streamlit Parity Matrix

This document is the migration control board for making the Next.js/Vercel + FastAPI version of JUPR reach functional parity with the Streamlit app.

## Purpose

Streamlit remains the reference implementation for current production behavior. The Next/Vercel app should become the durable product UI, but rating math, match processing, replay, badge evaluation, and other domain operations stay in Python behind FastAPI/service/worker boundaries.

Every Streamlit page registered in `jupr_app/ui/page_registry.py` must be represented in the matrix below. When a new Streamlit page is added, update this document in the same PR.

## Status legend

- `Done`: route/workflow is implemented and validated for the target runtime.
- `Partial`: route/workflow exists but does not yet match the Streamlit behavior or release gate.
- `API needed`: FastAPI/service/read model work is needed before a useful Next page can ship.
- `Auth needed`: production-grade Next auth/session/role/audit work is needed first.
- `Not started`: no Next/FastAPI parity work has started.
- `Deferred`: intentionally not required for the first public read-only cutover.

## Cutover gates

### Public read-only cutover

Public traffic may move to Next/Vercel only when:

1. `make public-web-smoke` passes against staging FastAPI and Vercel.
2. Public read APIs return only public-safe fields.
3. Public pages render graceful empty/error states.
4. The custom-domain rollback path is documented.
5. Streamlit production admin remains unchanged.

### Closed-club admin write pilot

Because the club is closed for the summer, a production-write pilot may begin before full admin cutover. Pilot mode must still be workflow-specific:

1. `JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT=1` is set only on the FastAPI runtime.
2. Exactly one new write workflow is enabled at a time unless previous workflows are already proven.
3. The workflow writes only through FastAPI/Python domain services.
4. Browser/Vercel code never receives privileged backend credentials.
5. Staff auth, club scope, audit attribution, and Streamlit fallback are preserved.
6. Match/rating-adjacent workflows have a correction/replay path before broad use.

### Admin cutover

Any admin workflow may fully move off Streamlit only when:

1. Supabase Auth login/session is implemented in Next.
2. FastAPI verifies Supabase JWTs server-side.
3. Role lookup uses `admin_role_assignments`.
4. Every write is club-scoped.
5. Every write is audit-attributed.
6. Denied/missing/wrong-club auth cases have contract tests.
7. The workflow has staging E2E coverage.
8. The equivalent Streamlit recovery/correction workflow remains available until the replacement is proven.

## Parity matrix

| Streamlit key | Streamlit label | Access | Current Next/FastAPI coverage | Writes? | Required gate | Next action |
|---|---|---|---|---|---|---|
| `home` | Home | Public | `Done` — Next `/` product shell exists. | No | Public smoke | Keep aligned with product positioning and club-aware routing. |
| `leaderboards` | 🏆 Leaderboards | Public | `Partial` — Next `/clubs/[clubSlug]/leaderboards` and FastAPI leaderboard API exist with league/status/sort filters, chart cards, player deep links, status chips, and inactive visibility. | No | Public smoke + contract tests | Validate leaderboard display parity in staging, then close final styling/copy gaps. |
| `rating_rules` | Rating Rules | Public | `Partial` — Next `/how-ratings-work` exists as public explainer. | No | Static content review | Reconcile route naming with `/faq` and final product copy. |
| `league_results` | 📊 League Results | Public | `Partial` — public FastAPI summary API and Next `/clubs/[clubSlug]/league-results` route exist with league tabs, section/week/player deep links, standings/weekly/player charts, and print-friendly view. | No | Public smoke + contract tests | Validate chart/deep-link/printout parity in staging, then close any final styling/copy gaps. |
| `league_printout` | 🖨️ League Night Printout | Admin | `Partial` — Next `/admin/league-manager/print` provides browser-printable roster, schedule, standings, and attendance checklist output. | Export only | Auth/export gate | Validate print layout in staging; keep browser-print strategy unless true server PDFs are required. |
| `match_explorer` | 🎯 Match Explorer | Public | `Partial` — public FastAPI context/preview API and Next `/clubs/[clubSlug]/match-explorer` route exist with Streamlit-compatible deep links, share-link hydration, beat-expectation metrics, player-level projected deltas, rating-impact predictor chart, and preview proxy. | No | Public smoke + contract tests | Validate chart/deep-link/preview parity in staging, then close styling/copy gaps. |
| `players` | 🔍 Player Search | Public | `Partial` — Next player directory/profile/match history exists and exposes singles rating/record fields and match format. | No | Public smoke + contract tests | Validate public player directory/profile parity in staging, then close final styling/copy gaps. |
| `badge_codex` | 📼 Badge Codex | Public | `Partial` — public FastAPI badge/earner APIs and Next `/clubs/[clubSlug]/badge-codex` route exist with section filters, load-more links, direct badge anchors, and recent-earner trophy room. | No | Public smoke + contract tests | Validate load-more/deep-link/trophy-room parity in staging. |
| `badge_debug` | 🧪 Badge Debug | Admin | `Partial` — Next `/admin/badges` has guarded read-only Badge Debug reports with player/badge/league/date filters. | Diagnostic only | Admin auth + `view_audit_log` | Validate diagnostics against Streamlit, then scope recompute/revoke as separate super-admin workflows. |
| `badge_audit` | 🧾 Badge Audit | Admin | `Partial` — Next `/admin/badges` has guarded read-only Badge Audit expected-vs-actual summaries using Python badge audit logic. | Diagnostic only | Admin auth + `view_audit_log` | Validate audit tables in staging; keep repair/recompute Streamlit-only until separately approved. |
| `match_canonical_audit` | 🧩 Match Canonical Audit | Admin | `Partial` — Next `/admin/match-canonical-audit` adds guarded audit, dry-run normalize, and explicit apply-normalize controls backed by Python canonical audit/migration logic. | Yes/high-risk normalization | Admin auth + `view_audit_log`/`manage_matches` + audit + replay gate | Validate diagnostics and normalization dry-runs before production apply; use Badge Diagnostics and Replay History after applies. |
| `challenge_ladder` | 🪜 Challenge Ladder | Public | `Partial` — public FastAPI ladder API and Next `/clubs/[clubSlug]/challenge-ladder` route exist with section/tier/player/challenge deep links, ladder search links, status legend, public rulebook copy, and eligible-opponent hints. | No | Public smoke + contract tests | Validate deep-link/public-rule parity in staging. |
| `faqs` | ❓ FAQs | Public | `Partial` — Next `/faq` exists with public JUPR FAQ content. | No | Static content review | Validate copy against Streamlit FAQ and decide whether `/faq` or `/how-ratings-work` is canonical. |
| `privacy_policy` | Privacy Policy | Public hidden | `Partial` — first-party Next `/privacy` exists with product privacy copy and links to correction/privacy requests. | No | Static/legal review | Replace or approve final legal policy before broad production launch. |
| `terms_of_use` | Terms of Use | Public hidden | `Partial` — first-party Next `/terms` placeholder exists. | No | Static/legal review | Replace placeholder with approved legal terms before broad production launch. |
| `contact_support` | Contact Support | Public hidden | `Partial` — Next `/support` and `/contact` support shell exists. | No | Static/support review | Confirm support address/routing and link intake flows. |
| `data_corrections` | Data Corrections | Public hidden | `Partial` — Next `/data-corrections` submits persisted staff-review records through a public FastAPI support-intake endpoint; no direct data mutation occurs. | Request intake only | Support/intake + anti-abuse gate | Validate intake queue and staff review handling. |
| `email_preferences` | Email Preferences | Public hidden | `Partial` — Next `/email-preferences` and public FastAPI endpoints support token/sid lookup plus category/global unsubscribe for player-update emails. | Preference write | Tokenized preference gate | Validate unsubscribe links from real/staging emails and expand categories only as new email types are added. |
| `profile_privacy` | Profile Privacy | Public hidden | `Partial` — Next `/profile-privacy` submits admin-reviewed profile privacy requests through the public support-intake queue; no immediate hiding/anonymization occurs. | Request intake only | Privacy workflow gate | Validate request handling before any automatic public-display changes. |
| `league_manager` | 🏟️ League Manager | Admin | `Partial` — Next `/admin/league-manager` has guarded draft creation and configuration-only duplication, explicit start/pause/resume/end/archive lifecycle actions, a structured Streamlit-parity draft editor for overview/schedule/courts/competition/ratings/awards, server validation and state locks, read-only unsaved schedule/ICS preview, roster membership, printout, persisted `/admin/league-manager/live` sessions with court movement, and `/admin/league-manager/awards` top-performer close/award workflow. | Yes | Admin auth + match-write + audit gate | Validate structured draft saves and preview, lifecycle/settings locks, live sessions, court movement, awards close, and recovery through Match Log/Replay History in staging. |
| `match_uploader` | 📝 Match Uploader | Admin | `Partial` — Next `/admin/match-uploader` manual/batch, round-robin preview, single-match input, singles match input, and new-player create-and-continue routes use the stored Next admin session. Player update emails can auto-send after batch completion when enabled. | Yes | Admin auth + club scope + audit + E2E | Validate manual/batch, singles, round-robin, and post-batch player update emails. |
| `match_log` | 📝 Match Log | Admin | `Partial` — Next Match Log has read/filters, duplicate scan, duplicate no-issue resolution, guided audited edits with roster picker, duplicate cleanup, Club Social edit/delete parity, bulk rated-match soft-exclude, and Quick Replay. | Yes | Admin auth + replay/correction audit gate | Validate expanded closed-club pilot in staging and production. |
| `player_editor` | 👥 Player Editor | Admin | `Partial` — Next `/admin/players` has guarded list/detail/create/basic-update, league-rating edits, social identity linking, exact-name social auto-link, and guarded merge preview/execute. | Yes/high-risk for merge | Admin auth + club scope + audit + replay gate | Validate create/update/linking in staging; use merge only with immediate Replay History ALL until proven. |
| `admin_tools` | ⚙️ Admin Tools | Admin | `Partial` — Replay History moved to `/admin/replay-history`; Badge Diagnostics, Match Canonical Audit, and multiple admin work queues now use stored Next admin sessions. Worker/backfill tooling remains Streamlit-only. | Yes/high-risk | Super-admin-only + worker/backfill gate | Keep remaining worker/backfill operations Streamlit-only until job contracts are hardened. |
| `admin_guide` | 📘 Admin Guide | Admin | `Partial` — `/admin` operations cockpit provides migration guidance and workflow flags; full guide not ported. | No | Admin shell | Port detailed operational playbook once admin shell exists. |
| `challenge_ladder_admin` | 🛠️ Challenge Ladder Admin | Admin | `Partial` — Next `/admin/challenge-ladder` has a guarded dashboard, tier-aware challenge creation, acceptance/clock actions, club-scoped partner selection, read-only played-result preview, exact-draft review gating, official match publishing, challenger-win rank swaps, cancel/forfeit actions, monthly-pass resolution, roster append/reactivation, cross-tier moves with optional old-tier rank recompression, and vacation/reinstate overrides. Whole-tier replacement and the computed tier-movement review queue remain in Streamlit. | Yes/high-risk result and roster writes | Admin auth + ladder write/audit + match/replay gate | Validate creation, preview, publish, rank movement, pass resolution, roster/override operations, forfeit, and recovery with isolated staging ladder data before staff use. |
| `moneyball` | 💰 Moneyball | Admin | `Partial` — Next `/admin/moneyball` has 8-player selection, rating context, expected-score preview, score entry, settlement, and official match save through Python Match Uploader. | Yes | Admin auth + event/match-write gate | Validate Moneyball scheduling/settlement and correction path in staging. |
| `jupr_live` | 🔴 JUPR Live | Public | `Partial` — Next live session list/detail and FastAPI live-session APIs exist. | Public live writes exist but guarded by edit token/service role. | Public smoke + live-session contract tests | Keep as the simple one-off event workflow; do not merge it with Tournament Live. |
| `jupr_live_admin` | 🔴 JUPR Live Admin | Admin | `Partial` — Next `/admin/jupr-live` has guarded durable session create/list/detail/update controls and public-session links. | Yes | Admin auth + live-session audit gate | Validate one-off live sessions separately from Tournament Live. |
| `theme_qa` | 🎨 Theme QA | Admin | `Deferred`. | No | Design-system gate | Replace with Next design-system/storybook-style QA if needed. |
| `tournaments` | 🏆 Tournaments | Admin | `Partial` — Next Tournament Admin has guarded read, registration edits, selection/division edits, bulk registration actions, archive/unarchive, and empty draft deletion. | Yes | Admin auth + tournament write/audit gate | Validate end-to-end tournament admin in staging and keep Streamlit as fallback. |
| `tournament_manager` | 🛠️ Tournament Setup | Admin | `Partial` — Tournament setup is represented through Next Tournament Admin/Tournament Ops draw creation and registration configuration views; full separate setup manager remains consolidated. | Yes | Admin auth + tournament write/audit gate | Decide whether to keep setup consolidated or add a dedicated Next setup route. |
| `tournament_ops` | 📋 Tournament Operations | Admin | `Partial` — Next Tournament Ops covers draw creation, manual/registration/bulk team import, game generation/scoring, playoff generation/scoring, podiums, awards, official match publishing, singles tournament publishing, medal-playoff winner bonus, and automatic player update email handoff. | Yes/high-risk | Admin auth + tournament ops/audit + replay gate | Validate complete tournament ops pilot in staging before replacing Streamlit. |
| `tournament_live` | 🔴 Tournament Live | Admin | `Partial` — Next `/admin/tournament-live` provides a tournament-specific in-play runner for selected draws: live score entry, round-robin/playoff progression, draw-scoped podium/awards, and official match publishing through existing guarded Tournament Ops endpoints. | Yes/high-risk | Admin auth + tournament ops/audit + replay/email gate | Validate the live runner separately from Tournament Ops setup; keep JUPR Live as the one-off event product. |
| `tournament_registration` | 📝 Tournament Registration | Public | `Partial` — public FastAPI registration page/submit APIs and Next `/clubs/[clubSlug]/tournament-registration` route exist with secure edit-link request entry. | Yes/intake | Public form + validation + anti-abuse + staging smoke gate | Validate staging, including duplicate-email, closed-registration, and edit-link cases. |
| `tournament_registration_admin` | 🧾 Registration Management | Admin | `Partial` — Admin registration management is covered by Next Tournament Admin list/detail, single registration edits, bulk registration edits, selection edits, and ops import. | Yes | Admin auth + registration audit gate | Validate registration admin flow in staging. |
| `tournament_registration_confirmation` | ✅ Registration Confirmation | Public hidden | `Partial` — public FastAPI confirmation API and Next confirmation route exist. | No | Public form flow | Validate staging and email confirmation handoff. |
| `tournament_registration_edit` | ✏️ Edit Registration | Public hidden | `Partial` — tokenized edit-link request, edit page, and submit APIs/routes exist. | Yes/intake | Tokenized edit/auth gate | Validate token lifetime, edit locks, and imported-draw restrictions in staging. |
| `tournament_roster` | 📋 Tournament Roster | Public hidden | `Partial` — public FastAPI roster API and Next `/clubs/[clubSlug]/tournament-roster` route exist with public-safe roster, summary, and needs-partner visibility. | No | Public tournament read model + smoke | Validate staging. |
| `tournament_partner_board` | 🤝 Partner Board | Public | `Partial` — Next `/clubs/[clubSlug]/tournament-partner-board` supports partner-needed visibility, token-verified pairing interest, request review, and automatic pairing on accepted requests without exposing contact details. | Yes/intake pairing | Public token + anti-abuse + staging smoke gate | Validate request/accept/cancel-competing behavior in staging. |
| `weekly_recap` | 🗞️ Weekly Recap | Public | `Partial` — public FastAPI published-recap/PDF APIs and Next `/clubs/[clubSlug]/weekly-recap` route exists with week/section deep links, section-filtered recap rendering, PDF handoff, and print-friendly layout. | No | Public smoke + contract tests | Validate final styling/print-view parity in staging. |
| `top_players_printable` | 🧾 Top Active Players PDF | Admin | `Partial` — Next `/admin/top-players-printable` provides browser-printable top active players output. | Export only | Admin/export gate | Validate print layout in staging; server PDFs remain deferred unless needed. |
| `weekly_recap_admin` | 🗞️ Weekly Recap Admin | Admin | `Partial` — Next `/admin/weekly-recap` has guarded draft generation, Looking Ahead edits, spotlight candidate selectors, preview, publish, and unpublish. | Yes | Admin auth + recap write/audit gate | Validate generated/edit/publish parity against Streamlit. |
| `player_updates_admin` | 📬 Player Updates Admin | Admin | `Partial` — Next `/admin/player-updates` and `/admin/player-updates/verified-requests` support date-range reports, selected exact-range sends, SMTP/JUPR_EMAIL_MODE safety, and verified request review. | Yes/email | Admin auth + email safety gate | Validate SMTP dry-run/staging redirect before live email and automatic post-batch sends. |
| `admin_login` | 🔐 Admin Login | Hidden/auth | `Partial` — Next `/admin/login` Supabase Auth shell exists, `/admin` shows the browser admin session, guarded write forms consume the stored session token, and login links to password reset. | No | Supabase Auth/session + auth E2E gate | Add staging auth E2E for sign-in/sign-out, expired token, reset-password, wrong-club, and permission-denied cases. |
| `reset_password` | 🔐 Reset Password | Hidden/auth | `Partial` — Next `/admin/reset-password` can request a Supabase recovery email and update a password from the recovery session. | Auth write | Supabase Auth/session + staging auth E2E gate | Validate against staging Supabase redirect settings and password policy. |
| `verified_updates_request` | 📬 Verified Updates Request | Public hidden | `Partial` — Next `/verified-updates` and public FastAPI routes let users request verified player update subscriptions for admin review. | Email/preference request | Tokenized email safety gate | Validate public request and admin approval/reject/unsubscribe queue in staging. |

## Recommended implementation sequence

1. Keep public read-only staging smoke and Vercel guardrails green.
2. Validate public Match Explorer, League Results, Badge Codex, Challenge Ladder, Weekly Recap, Leaderboards, and Player Search parity in staging.
3. Review/approve static FAQ/legal/support copy and smoke static routes.
4. Validate public tournament registration, confirmation, edit-link, roster, and Partner Board request/accept flows in staging.
5. Validate public data correction, profile privacy, verified updates request, and email preference/unsubscribe flows.
6. Use `/admin` as the migration cockpit and enable closed-club production-write pilot mode only on the FastAPI runtime.
7. Validate Match Log guided edits, duplicate no-issue resolution, Club Social edits, bulk exclude, Replay History, Match Canonical Audit, and Badge Diagnostics in the closed-club pilot.
8. Validate Match Uploader manual/batch, singles input, round-robin scheduling, new-player creation, Moneyball, and post-batch player update emails.
9. Validate League Manager settings, roster membership, live sessions, court movement, browser-print exports, and awards close workflow.
10. Validate Player Updates Admin date-range reports and verified request review with `JUPR_EMAIL_MODE=dry_run` or `staging_redirect` before live email.
11. Validate Player Editor create/update, league-rating edits, social identity linking, and merge preview/execute with immediate Replay History recovery.
12. Validate Tournament Admin/Ops setup/import/scoring/podium/award/official-publish workflows in staging.
13. Validate Tournament Live runner separately as the in-play tournament draw surface, and JUPR Live Admin separately as the one-off event surface.
14. Validate the guarded Challenge Ladder Admin creation/result/rank-movement, monthly-pass, roster-operation, and vacation/reinstate override workflows against isolated staging data, including Match Log and Replay History recovery after official match publishing.
15. Validate admin login, password reset, sign-out, expired-token, wrong-club, and permission-denied cases in staging.
16. Port remaining super-admin worker/backfill operations only after dry-run job contracts and background status reporting are hardened.

## Maintenance rule

Run this before opening or merging migration PRs that touch Streamlit pages or Next parity docs:

```bash
make check-next-parity-matrix
```
