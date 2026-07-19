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
| `leaderboards` | 🏆 Leaderboards | Public | `Partial` — Next `/clubs/[clubSlug]/leaderboards` and FastAPI leaderboard API exist with league/status/sort filters, chart cards, player deep links, status chips, and inactive visibility. | No | Public smoke + contract tests | Add Active-by-default behavior, visible player search/snapshot deep links, gain/gap/qualification/badge context, pagination, and populated/empty/error E2E. |
| `rating_rules` | Rating Rules | Public | `Partial` — Next `/how-ratings-work` exists as public explainer. | No | Static content review | Reconcile route naming with `/faq` and final product copy. |
| `league_results` | 📊 League Results | Public | `Partial` — public FastAPI summary API and Next `/clubs/[clubSlug]/league-results` route exist with league tabs, section/week/player deep links, standings/weekly/player charts, and print-friendly view. | No | Public smoke + contract tests | Add rank delta/trend, complete player selection and recent-match detail, correct weekly-versus-season highlight semantics, qualification rules, and deterministic deep-link/print E2E. |
| `league_printout` | 🖨️ League Night Printout | Admin | `Partial` — Next `/admin/league-manager/print` provides browser-printable roster, schedule, standings, and attendance checklist output. | Export only | Auth/export gate | Validate print layout in staging; keep browser-print strategy unless true server PDFs are required. |
| `match_explorer` | 🎯 Match Explorer | Public | `Partial` — public FastAPI context/preview API and Next `/clubs/[clubSlug]/match-explorer` route exist with Streamlit-compatible deep links, share-link hydration, beat-expectation metrics, player-level projected deltas, rating-impact predictor chart, and preview proxy. | No | Public smoke + contract tests | Remove duplicated client rating math, enforce Python-authoritative contexts and four distinct players, restore reactive preview behavior, and add golden-vector/deep-link E2E. |
| `players` | 🔍 Player Search | Public | `Partial` — Next player directory/profile/match history exists and exposes singles rating/record fields and match format. | No | Public smoke + contract tests | Add visible search and Active defaults plus rating trends/breakdowns, full/recent history, trophies/badges, partner/rival/social projections, verified state, and privacy-safe E2E. |
| `badge_codex` | 📼 Badge Codex | Public | `Partial` — FastAPI now projects canonical Python badge status/timing/scope definitions, lifecycle-aware buckets, public-safe earners and trophy context; Next `/clubs/[clubSlug]/badge-codex` consumes those buckets with category/scope filters, direct anchors, badge and per-badge earner load-more/paging paths, empty states, and route-specific staging assertions. | No | Public smoke + contract tests | Compare every bucket, definition, representative earner page, direct anchor, empty/error state, and trophy-room layout with Streamlit in the consolidated staging session. |
| `badge_debug` | 🧪 Badge Debug | Admin | `Partial` — Next `/admin/badges` has guarded read-only Badge Debug reports with player/badge/league/date filters. | Diagnostic only | Admin auth + `view_audit_log` | Validate diagnostics against Streamlit, then scope recompute/revoke as separate super-admin workflows. |
| `badge_audit` | 🧾 Badge Audit | Admin | `Partial` — Next `/admin/badges` has guarded read-only Badge Audit expected-vs-actual summaries using Python badge audit logic. | Diagnostic only | Admin auth + `view_audit_log` | Validate audit tables in staging; keep repair/recompute Streamlit-only until separately approved. |
| `match_canonical_audit` | 🧩 Match Canonical Audit | Admin | `Partial` — Next `/admin/match-canonical-audit` adds guarded audit, dry-run normalize, and explicit apply-normalize controls backed by Python canonical audit/migration logic. | Yes/high-risk normalization | Admin auth + `view_audit_log`/`manage_matches` + audit + replay gate | Validate diagnostics and normalization dry-runs before production apply; use Badge Diagnostics and Replay History after applies. |
| `challenge_ladder` | 🪜 Challenge Ladder | Public | `Partial` — FastAPI computes public status and eligible-opponent projections through the canonical Python ladder policy, strips operator-only status notes, and returns the complete structured rulebook/status legend; Next `/clubs/[clubSlug]/challenge-ladder` consumes that projection with section/tier/player/challenge deep links and route-specific staging assertions. | No | Public smoke + contract tests | Compare representative ranks, statuses, opponent hints, active challenge buckets, every rulebook section, and deep links with Streamlit in the consolidated staging session. |
| `faqs` | ❓ FAQs | Public | `Partial` — Next `/faq` exists with public JUPR FAQ content. | No | Static content review | Validate copy against Streamlit FAQ and decide whether `/faq` or `/how-ratings-work` is canonical. |
| `privacy_policy` | Privacy Policy | Public hidden | `Partial` — first-party Next `/privacy` exists with product privacy copy and links to correction/privacy requests. | No | Static/legal review | Replace or approve final legal policy before broad production launch. |
| `terms_of_use` | Terms of Use | Public hidden | `Partial` — first-party Next `/terms` contains the complete product terms draft and exceeds the former Streamlit placeholder. | No | Static/legal review | Regression-guard required sections, effective date, correction link, and contact; retain only owner/legal approval as the manual gate. |
| `contact_support` | Contact Support | Public hidden | `Partial` — Next `/support` and `/contact` support shell exists. | No | Static/support review | Confirm support address/routing and link intake flows. |
| `data_corrections` | Data Corrections | Public hidden | `Partial` — Next `/data-corrections` submits persisted staff-review records through a public FastAPI support-intake endpoint; no direct data mutation occurs. | Request intake only | Support/intake + anti-abuse gate | Validate intake queue and staff review handling. |
| `email_preferences` | Email Preferences | Public hidden | `Partial` — Next `/email-preferences` and public FastAPI endpoints support token/sid lookup plus category/global unsubscribe for player-update emails. | Preference write | Tokenized preference gate | Validate unsubscribe links from real/staging emails and expand categories only as new email types are added. |
| `profile_privacy` | Profile Privacy | Public hidden | `Partial` — Next `/profile-privacy` submits admin-reviewed profile privacy requests through the public support-intake queue; no immediate hiding/anonymization occurs. | Request intake only | Privacy workflow gate | Validate request handling before any automatic public-display changes. |
| `league_manager` | 🏟️ League Manager | Admin | `Partial` — Next `/admin/league-manager` has guarded draft creation and configuration-only duplication, explicit start/pause/resume/end/archive lifecycle actions, a structured Streamlit-parity draft editor for overview/schedule/courts/competition/ratings/awards, server validation and state locks, read-only unsaved schedule/ICS preview, roster membership, printout, persisted `/admin/league-manager/live` sessions with Python court movement plus durable all-match publish/reconcile/rating-readback/guest/export/compensation workflows, and `/admin/league-manager/awards` top-performer close/award workflow. | Yes | Admin auth + match-write + audit gate | Validate structured draft saves and preview, lifecycle/settings locks, complete-round publish and response-loss/partial recovery, guest/substitution, rating readback/exports, court movement, awards close, and Match Log/Replay compensation in staging. |
| `match_uploader` | 📝 Match Uploader | Admin | `Partial` — Next `/admin/match-uploader` manual/batch, round-robin preview, single-match input, singles match input, and new-player create-and-continue routes use the stored Next admin session. Player update emails can auto-send after batch completion when enabled. | Yes | Admin auth + club scope + audit + E2E | Validate manual/batch, singles, round-robin, and post-batch player update emails. |
| `match_log` | 📝 Match Log | Admin | `Partial` — Next Match Log has read/filters, duplicate scan, duplicate no-issue resolution, guided audited edits with roster picker, duplicate cleanup, Club Social edit/delete parity, bulk rated-match soft-exclude, and Quick Replay. | Yes | Admin auth + replay/correction audit gate | Validate expanded closed-club pilot in staging and production. |
| `player_editor` | 👥 Player Editor | Admin | `Partial` — Next `/admin/players` has guarded list/detail/create/basic-update, league-rating edits, social identity linking, exact-name social auto-link, and guarded merge preview/execute. | Yes/high-risk for merge | Admin auth + club scope + audit + replay gate | Validate create/update/linking in staging; use merge only with immediate Replay History ALL until proven. |
| `admin_tools` | ⚙️ Admin Tools | Admin | `Partial` — Replay History moved to `/admin/replay-history`; Badge Diagnostics, Match Canonical Audit, activity/role administration, read-only overall/league rating reports with browser CSV export, Club Social submission review/moderation, badge queue status/processing, scoped badge recompute with dry-run, guarded badge-definition lifecycle transitions, and reviewed missing tournament-match preview/apply use stored Next admin sessions and Python/FastAPI services. Social moderation is club/result-mode scoped and requires `manage_matches`, exact confirmation, rejection reasons, stale-status protection, and centralized audit records. Badge lifecycle changes require `manage_roles`, the Python state machine, a reason, exact confirmation, expected-state locking, and explicit force for nonstandard paths. Backfill apply requires selected ready IDs from a current fingerprint, exact confirmation, capped scope, audit intent/completion, duplicate/player rechecks, and persisted-ID verification. Remaining one-time maintenance migrations stay Streamlit-only. | Yes/high-risk | Super-admin lifecycle + club-owner moderation + worker/backfill/recovery gate | Validate badge lifecycle transitions without changing live definitions, Club Social pending/saved/rejected review and guarded moderation, rating report/CSV parity, badge worker/recompute, and tournament backfill preview/apply with isolated staging data; keep remaining migrations on Streamlit until reviewed dry-run/apply/recovery contracts exist. |
| `admin_guide` | 📘 Admin Guide | Admin | `Partial` — Next `/admin/guide` now provides the Streamlit operating guidance plus route-linked day-of runbooks, workflow-specific completion/stop criteria, global stop conditions, recovery sequencing, validation phases, and emergency fallback paths. | No | Admin shell + static content review | Validate staff terminology and keep runbooks synchronized as guarded workflow contracts change. |
| `challenge_ladder_admin` | 🛠️ Challenge Ladder Admin | Admin | `Partial` — Next `/admin/challenge-ladder` has a guarded dashboard, tier-aware challenge creation with server-side status eligibility and explicit override, domain-built copyable email/SMS notices, acceptance/clock actions, club-scoped partner selection, read-only played-result preview, exact-draft review gating, official match publishing, challenger-win rank swaps, cancel/forfeit actions, monthly-pass resolution, roster append/reactivation, cross-tier moves with optional old-tier rank recompression, reviewed whole-tier replacement with stale-preview and open-challenge guards, vacation/reinstate overrides, and a computed tier-movement review queue that prefills the guarded move form. Ladder writes are visible through centralized Admin Tools activity. | Yes/high-risk result and roster writes | Admin auth + ladder write/audit + match/replay gate | Validate creation/eligibility/notices, preview, publish, rank movement/review, pass resolution, roster/whole-tier replacement/override operations, forfeit, and recovery with isolated staging ladder data before staff use. |
| `moneyball` | 💰 Moneyball | Admin | `Partial` — Next `/admin/moneyball` has 8-player selection, rating context, expected-score preview, score entry, settlement, and official match save through Python Match Uploader. | Yes | Admin auth + event/match-write gate | Validate Moneyball scheduling/settlement and correction path in staging. |
| `jupr_live` | 🔴 JUPR Live | Public | `Partial` — Next live session list/detail and FastAPI live-session APIs exist; the public creator currently supports only the basic Quick Round Robin path. | Public live writes exist but guarded by edit token/service role. | Public smoke + live-session contract tests | Add League/Ladder and Club Social flows, Python-authoritative movement, substitutions, continuation/completion, exports, token/rate limits, and deterministic mobile E2E; keep separate from Tournament Live. |
| `jupr_live_admin` | 🔴 JUPR Live Admin | Admin | `Partial` — Next `/admin/jupr-live` has guarded durable session create/list/detail/update controls and public-session links. | Yes | Admin auth + live-session audit gate | Validate one-off live sessions separately from Tournament Live. |
| `theme_qa` | 🎨 Theme QA | Admin | `Deferred`. | No | Design-system gate | Replace with Next design-system/storybook-style QA if needed. |
| `tournaments` | 🏆 Tournaments | Admin | `Partial` — Next Tournament Admin has guarded read, registration edits, selection/division edits, bulk registration actions, archive/unarchive, and empty draft deletion. | Yes | Admin auth + tournament write/audit gate | Validate end-to-end tournament admin in staging and keep Streamlit as fallback. |
| `tournament_manager` | 🛠️ Tournament Setup | Admin | `Partial` — Tournament setup is represented through Next Tournament Admin/Tournament Ops draw creation and registration configuration views; full separate setup manager remains consolidated. | Yes | Admin auth + tournament write/audit gate | Decide whether to keep setup consolidated or add a dedicated Next setup route. |
| `tournament_ops` | 📋 Tournament Operations | Admin | `Partial` — Next Tournament Ops covers draw creation, manual/registration/bulk team import, game generation/scoring, playoff generation/scoring, podiums, awards, official match publishing, singles tournament publishing, medal-playoff winner bonus, and automatic player update email handoff. | Yes/high-risk | Admin auth + tournament ops/audit + replay gate | Validate complete tournament ops pilot in staging before replacing Streamlit. |
| `tournament_live` | 🔴 Tournament Live | Admin | `Partial` — Next `/admin/tournament-live` provides a tournament-specific in-play runner for selected draws: live score entry, round-robin/playoff progression, draw-scoped podium/awards, and official match publishing through existing guarded Tournament Ops endpoints. | Yes/high-risk | Admin auth + tournament ops/audit + replay/email gate | Validate the live runner separately from Tournament Ops setup; keep JUPR Live as the one-off event product. |
| `tournament_registration` | 📝 Tournament Registration | Public | `Partial` — public FastAPI registration page/submit APIs and Next `/clubs/[clubSlug]/tournament-registration` route exist with secure edit-link request entry. | Yes/intake | Public form + validation + anti-abuse + staging smoke gate | Add Start New/Edit Existing wizard parity, required age/gender and profile/partner policy, sponsor/refund/roster content, then test duplicate/closed/recovery and family/skill/partner validation. |
| `tournament_registration_admin` | 🧾 Registration Management | Admin | `Partial` — Admin registration management is covered by Next Tournament Admin list/detail, single registration edits, bulk registration edits, selection edits, and ops import. | Yes | Admin auth + registration audit gate | Validate registration admin flow in staging. |
| `tournament_registration_confirmation` | ✅ Registration Confirmation | Public hidden | `Partial` — public FastAPI confirmation API and Next confirmation route exist, but confirmation delivery and bearer-safe lookup are not yet complete. | No | Public form flow | Remove private fields, use non-enumerating access, add delivery status and complete event/payment/roster content, and prove registration persists when email delivery fails. |
| `tournament_registration_edit` | ✏️ Edit Registration | Public hidden | `Partial` — tokenized edit-link request, edit page, and submit APIs/routes exist; selection replacement is not yet atomic/versioned and imported-draw protection is incomplete. | Yes/intake | Tokenized edit/auth gate | Block imported selections, make edits atomic and stale-safe, add token preflight/rate limits and post-edit delivery, then prove rollback and unchanged-draw behavior. |
| `tournament_roster` | 📋 Tournament Roster | Public hidden | `Partial` — public FastAPI roster API and Next `/clubs/[clubSlug]/tournament-roster` route exist with public-safe roster, summary, and needs-partner visibility; unresolved null statuses are currently mislabeled as confirmed. | No | Public tournament read model + smoke | Correct status mapping, add event/division/status filters and partner detail/deep links, enforce the public-field denylist, and validate all representative states. |
| `tournament_partner_board` | 🤝 Partner Board | Public | `Partial` — Next `/clubs/[clubSlug]/tournament-partner-board` supports partner-needed visibility, token-verified pairing interest, request review, and automatic pairing on accepted requests without exposing contact details. | Yes/intake pairing | Public token + anti-abuse + staging smoke gate | Validate request/accept/cancel-competing behavior in staging. |
| `weekly_recap` | 🗞️ Weekly Recap | Public | `Partial` — public FastAPI published-recap/PDF APIs and Next `/clubs/[clubSlug]/weekly-recap` route exists with week/section deep links, section-filtered recap rendering, PDF handoff, and print-friendly layout. | No | Public smoke + contract tests | Validate final styling/print-view parity in staging. |
| `top_players_printable` | 🧾 Top Active Players PDF | Admin | `Partial` — Next `/admin/top-players-printable` provides browser-printable top active players output. | Export only | Admin/export gate | Validate print layout in staging; server PDFs remain deferred unless needed. |
| `weekly_recap_admin` | 🗞️ Weekly Recap Admin | Admin | `Partial` — Next `/admin/weekly-recap` has guarded draft generation, Looking Ahead edits, spotlight candidate selectors, preview, publish, and unpublish. | Yes | Admin auth + recap write/audit gate | Validate generated/edit/publish parity against Streamlit. |
| `player_updates_admin` | 📬 Player Updates Admin | Admin | `Partial` — Next `/admin/player-updates` and `/admin/player-updates/verified-requests` support date-range reports, selected exact-range sends, SMTP/JUPR_EMAIL_MODE safety, and verified request review. | Yes/email | Admin auth + email safety gate | Validate SMTP dry-run/staging redirect before live email and automatic post-batch sends. |
| `admin_login` | 🔐 Admin Login | Hidden/auth | `Partial` — Next `/admin/login` now capability-checks Supabase JWTs through FastAPI before persistence, restores/refreshes authorized sessions, uses local-scope sign-out, keeps redirects on allowlisted admin paths, and has mocked browser/API denial contracts. | No | Supabase Auth/session + auth E2E gate | Run the dedicated staging-role E2E for sign-in/sign-out, expired token, reset-password, wrong-club, and permission-denied cases. |
| `reset_password` | 🔐 Reset Password | Hidden/auth | `Partial` — Next `/admin/reset-password` now provides non-enumerating request/resend, PKCE/code/token-hash plus legacy recovery consumption, capability validation, password-policy guidance, and recovery-session cleanup. | Auth write | Supabase Auth/session + staging auth E2E gate | Complete one staging Supabase recovery email, post-reset login, and configured password-policy test. |
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
14. Validate the guarded Challenge Ladder Admin creation eligibility/notice/clock, result/rank-movement review, monthly-pass, roster-operation, and vacation/reinstate override workflows against isolated staging data, including Match Log and Replay History recovery after official match publishing.
15. Validate admin login, password reset, sign-out, expired-token, wrong-club, and permission-denied cases in staging.
16. Validate badge-definition lifecycle controls without changing a live definition, Club Social pending/saved/rejected review and guarded moderation, overall/league rating reports and CSV downloads, badge queue/recompute status and dry-run behavior, plus tournament-match backfill preview/apply recovery; then port remaining super-admin migrations only after reviewed dry-run/apply contracts and recovery runbooks are hardened.

## Maintenance rule

Run this before opening or merging migration PRs that touch Streamlit pages or Next parity docs:

```bash
make check-next-parity-matrix
```
