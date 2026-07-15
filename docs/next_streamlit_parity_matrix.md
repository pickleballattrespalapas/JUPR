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
| `leaderboards` | 🏆 Leaderboards | Public | `Partial` — Next `/clubs/[clubSlug]/leaderboards` and FastAPI leaderboard API exist with league/status/sort filters, top ratings/most active/best win % chart cards, player deep links, status chips, and inactive visibility. | No | Public smoke + contract tests | Validate leaderboard display parity in staging, then close final styling/copy gaps. |
| `rating_rules` | Rating Rules | Public | `Partial` — Next `/how-ratings-work` exists as public explainer. | No | Static content review | Reconcile route naming with `/faq` and final product copy. |
| `league_results` | 📊 League Results | Public | `Partial` — public FastAPI summary API and Next `/clubs/[clubSlug]/league-results` route exist with league tabs, section/week/player deep links, standings/weekly/player charts, and print-friendly view. | No | Public smoke + contract tests | Validate chart/deep-link/printout parity in staging, then close any final styling/copy gaps. |
| `league_printout` | 🖨️ League Night Printout | Admin | `Not started`. | Export only | Auth/PDF/export gate | Defer until admin auth and export strategy are in place. |
| `match_explorer` | 🎯 Match Explorer | Public | `Partial` — public FastAPI context/preview API and Next `/clubs/[clubSlug]/match-explorer` route exist with Streamlit-compatible deep links, share-link hydration, beat-expectation metrics, player-level projected deltas, rating-impact predictor chart, and a same-origin preview proxy. | No | Public smoke + contract tests | Validate chart/deep-link/preview parity in staging, then close any styling/copy gaps. |
| `players` | 🔍 Player Search | Public | `Partial` — Next player directory, profile, match history, and match detail exist; directory has search/status/sort filters, stat cards, last-played visibility, leaderboard row links, and row deep links; profile has section deep links, league filters, linked match rows, and trophy/leaderboard handoff. | No | Public smoke + contract tests | Validate player directory/profile parity in staging, then close final styling/copy gaps. |
| `badge_codex` | 📼 Badge Codex | Public | `Partial` — public FastAPI badge/earner APIs and Next `/clubs/[clubSlug]/badge-codex` route exist with section filters, load-more links, direct badge anchors, and a recent-earner trophy room. | No | Public smoke + contract tests | Validate load-more/deep-link/trophy-room parity in staging, then close final styling/copy gaps. |
| `badge_debug` | 🧪 Badge Debug | Admin | `Not started`. | Diagnostic only | Admin auth + role gate | Keep Streamlit-only until admin diagnostics/audit shell exists. |
| `badge_audit` | 🧾 Badge Audit | Admin | `Not started`. | Diagnostic/write-adjacent | Admin auth + role gate | Define admin badge audit API after badge public read model is stable. |
| `match_canonical_audit` | 🧩 Match Canonical Audit | Admin | `Not started`. | Diagnostic/write-adjacent | Admin auth + role gate | Keep Streamlit-only until replay/correction APIs exist. |
| `challenge_ladder` | 🪜 Challenge Ladder | Public | `Partial` — public FastAPI ladder API and Next `/clubs/[clubSlug]/challenge-ladder` route exist with section/tier/player/challenge deep links, ladder search links, status legend, public rulebook copy, and eligible-opponent hints. | No | Public smoke + contract tests | Validate deep-link/public-rule parity in staging, then close final styling/copy gaps. |
| `faqs` | ❓ FAQs | Public | `Partial` — Next `/faq` exists with public JUPR FAQ content. | No | Static content review | Validate copy against Streamlit FAQ and decide whether `/faq` or `/how-ratings-work` is canonical. |
| `privacy_policy` | Privacy Policy | Public hidden | `Partial` — first-party Next `/privacy` placeholder exists. | No | Static/legal review | Replace placeholder with approved legal policy before broad production launch. |
| `terms_of_use` | Terms of Use | Public hidden | `Partial` — first-party Next `/terms` placeholder exists. | No | Static/legal review | Replace placeholder with approved legal terms before broad production launch. |
| `contact_support` | Contact Support | Public hidden | `Partial` — Next `/support` and `/contact` support shell exists. | No | Static/support review | Confirm support address/routing and add real intake workflow if needed. |
| `data_corrections` | Data Corrections | Public hidden | `Partial` — Next `/data-corrections` intake instructions exist with no direct data mutation. | Request intake shell only | Support/intake gate | Define durable ticket/intake backend; keep rating corrections staff-reviewed. |
| `email_preferences` | Email Preferences | Public hidden | `Not started` in Next app. | Preference write | Auth/tokenized preference gate | Port only after safe email preference tokens and unsubscribe rules are defined. |
| `profile_privacy` | Profile Privacy | Public hidden | `Not started` in Next app. | Preference/request write TBD | Privacy workflow gate | Define privacy request flow before porting. |
| `league_manager` | 🏟️ League Manager | Admin | `Partial` — Next `/admin/league-manager` and guarded FastAPI league-manager status/list/detail read APIs exist with schedule-preview, config, standings visibility, and stored admin session auth. Setup, roster movement, scoring, and awards remain Streamlit-only. | Read-only in this slice | Admin auth + match-write + audit gate before writes | Validate read foundation in staging, then add setup/edit and roster dry-run APIs before any League Manager writes. |
| `match_uploader` | 📝 Match Uploader | Admin | `Partial` — Next Score Entry MVP and `/admin/match-uploader` manual/batch, single round-robin preview, and new-player create-and-continue routes exist and now use the stored Next admin Supabase session instead of token paste. | Yes | Admin auth + club scope + audit + E2E | Validate manual/batch plus round-robin closed-club pilot, then add court quick-entry helpers. |
| `match_log` | 📝 Match Log | Admin | `Partial` — Next Match Log now has read/filters, duplicate scan, duplicate false-positive no-issue resolution, guided audited edits with roster picker, duplicate cleanup, Club Social row edit/delete parity, bulk rated-match soft-exclude, and Quick Replay. | Yes | Admin auth + replay/correction audit gate | Validate the expanded closed-club pilot in staging and production, then add replay job history/approval if needed. |
| `player_editor` | 👥 Player Editor | Admin | `Partial` — Next `/admin/players` and guarded FastAPI player-editor status/list/detail/create/basic-update APIs exist and use stored admin session auth; merge, league-rating edits, and social identity linking remain Streamlit-only. | Yes | Admin auth + club scope + audit gate | Validate create/update foundation, then add league-rating edit and merge dry-run APIs only after replay recovery is proven. |
| `admin_tools` | ⚙️ Admin Tools | Admin | `Partial` — Replay History moved to `/admin/replay-history` and uses the stored Next admin session; worker/backfill tooling remains Streamlit-only. | Yes/high-risk | Super-admin-only + worker/backfill gate | Keep remaining worker/backfill operations Streamlit-only until hardened. |
| `admin_guide` | 📘 Admin Guide | Admin | `Partial` — `/admin` operations cockpit now provides migration guidance; full guide not ported. | No | Admin shell | Port detailed operational playbook once admin shell exists. |
| `challenge_ladder_admin` | 🛠️ Challenge Ladder Admin | Admin | `Not started`; listed in `/admin` operations cockpit. | Yes | Admin auth + ladder write/audit gate | Port after core score entry, match log, and player editor unless scoped as a closed-club pilot. |
| `moneyball` | 💰 Moneyball | Admin | `Not started`. | Yes | Admin auth + event/match-write gate | Port after score entry and event submission APIs are stable. |
| `jupr_live` | 🔴 JUPR Live | Public | `Partial` — Next live session list/detail and FastAPI live-session APIs exist. | Public live writes exist but guarded by edit token/service role. | Public smoke + live-session contract tests | Validate staging migrations/grants and decide whether live creation/scoring is public, admin, or organizer-gated. |
| `jupr_live_admin` | 🔴 JUPR Live Admin | Admin | `Not started`. | Yes | Admin auth + live-session audit gate | Port after public Live read side is stable. |
| `theme_qa` | 🎨 Theme QA | Admin | `Deferred`. | No | Design-system gate | Replace with Next design-system/storybook-style QA if needed. |
| `tournaments` | 🏆 Tournaments | Admin | `Not started`; listed in `/admin` operations cockpit. | Yes | Admin auth + tournament write/audit gate | Port after score entry/player editor foundation. |
| `tournament_manager` | 🛠️ Tournament Setup | Admin | `Not started`; listed under tournament admin in `/admin` operations cockpit. | Yes | Admin auth + tournament write/audit gate | Merge with tournament admin product plan; avoid duplicate tournament managers. |
| `tournament_ops` | 📋 Tournament Operations | Admin | `Not started`; listed under tournament admin in `/admin` operations cockpit. | Yes | Admin auth + tournament ops/audit gate | Port after tournament setup APIs. |
| `tournament_live` | 🔴 Tournament Live | Admin | `Not started`. | Yes | Admin auth + live/tournament gate | Define relation to public Live before porting. |
| `tournament_registration` | 📝 Tournament Registration | Public | `Partial` — public FastAPI registration page/submit APIs and Next `/clubs/[clubSlug]/tournament-registration` route exist. | Yes/intake | Public form + validation + anti-abuse + staging smoke gate | Validate staging, then close edit-link/email-confirmation parity gaps. |
| `tournament_registration_admin` | 🧾 Registration Management | Admin | `Not started`; listed under tournament admin in `/admin` operations cockpit. | Yes | Admin auth + registration audit gate | Port after public registration write path is proven. |
| `tournament_registration_confirmation` | ✅ Registration Confirmation | Public hidden | `Partial` — public FastAPI confirmation API and Next confirmation route exist. | No | Public form flow | Validate staging and add secure email/edit-link handoff. |
| `tournament_registration_edit` | ✏️ Edit Registration | Public hidden | `Not started`. | Yes/intake | Tokenized edit/auth gate | Add secure edit links before porting. |
| `tournament_roster` | 📋 Tournament Roster | Public hidden | `Partial` — public FastAPI roster API and Next `/clubs/[clubSlug]/tournament-roster` route exist with public-safe roster, summary, and needs-partner visibility. | No | Public tournament read model + smoke | Validate staging and then connect edit-link handoff. |
| `tournament_partner_board` | 🤝 Partner Board | Public | `Partial` — Next `/clubs/[clubSlug]/tournament-partner-board` exists as a read-only partner-needed board backed by the public roster projection; request/accept/moderation actions are not started. | No in this slice | Public/organizer moderation gate before writes | Validate read-only board in staging, then add request flow with anti-abuse and moderation. |
| `weekly_recap` | 🗞️ Weekly Recap | Public | `Partial` — public FastAPI published-recap/PDF APIs and Next `/clubs/[clubSlug]/weekly-recap` route exists with week/section deep links, section-filtered recap rendering, PDF handoff, and print-friendly layout. | No | Public smoke + contract tests | Validate final styling/print-view parity in staging, then close final copy gaps. |
| `top_players_printable` | 🧾 Top Active Players PDF | Admin | `Not started`. | Export only | Admin/PDF export gate | Defer until PDF/export strategy is standardized. |
| `weekly_recap_admin` | 🗞️ Weekly Recap Admin | Admin | `Not started`; listed in `/admin` operations cockpit. | Yes | Admin auth + recap write/audit gate | Port after public weekly recap read side and admin auth shell. |
| `player_updates_admin` | 📬 Player Updates Admin | Admin | `Not started`. | Yes/email | Admin auth + email safety gate | Keep blocked until staging email safety is proven. |
| `admin_login` | 🔐 Admin Login | Hidden/auth | `Partial` — Next `/admin/login` Supabase Auth shell exists, `/admin` shows the browser admin session, guarded write forms consume the stored session token, and login links to the password-reset flow. | No | Supabase Auth/session + auth E2E gate | Add staging auth E2E for sign-in/sign-out, expired token, reset-password, wrong-club, and permission-denied cases before broad admin cutover. |
| `reset_password` | 🔐 Reset Password | Hidden/auth | `Partial` — Next `/admin/reset-password` can request a Supabase recovery email and update a password from the recovery session. | Auth write | Supabase Auth/session + staging auth E2E gate | Validate against staging Supabase redirect settings and password policy before broad admin cutover. |
| `verified_updates_request` | 📬 Verified Updates Request | Public hidden | `Not started`. | Email/preference write | Tokenized email safety gate | Port only after email preference/update flows are safe in staging. |

## Recommended implementation sequence

1. Keep public read-only staging smoke and Vercel guardrails green.
2. Validate public Match Explorer chart/deep-link/preview parity in staging and close any final styling/copy gaps.
3. Validate public League Results chart/deep-link/printout parity in staging and close any final styling/copy gaps.
4. Validate public Badge Codex load-more/deep-link/trophy-room parity in staging and close final styling/copy gaps.
5. Validate public Challenge Ladder deep-link/public-rule parity in staging and close final styling/copy gaps.
6. Validate public Weekly Recap final styling/print-view parity in staging and close final copy gaps.
7. Validate public Leaderboards filters/charts/inactive visibility in staging and close final styling/copy gaps.
8. Validate public Player Search/profile filters and deep links in staging, then close final styling/copy gaps.
9. Review/approve static FAQ/legal/support copy and smoke the static routes.
10. Validate public tournament registration in staging, including duplicate-email and closed-registration cases.
11. Use `/admin` as the migration cockpit and enable closed-club production-write pilot mode only on the FastAPI runtime.
12. Validate Match Log guided edits, duplicate no-issue resolution, Club Social edits, bulk exclude, and Replay History in the closed-club pilot.
13. Validate Match Uploader manual/batch entry.
14. Validate Match Uploader round-robin scheduling and new-player creation parity in staging.
15. Validate Player Editor create/update foundation in staging.
16. Validate League Manager read foundation in staging.
17. Validate admin login, password reset, sign-out, expired-token, wrong-club, and permission-denied cases in staging.
18. Validate public Tournament Roster and read-only Partner Board in staging.
19. Port public tournament edit-link handoff and partner request flow.
20. Port tournament/challenge/admin tools after the write foundation is proven.

## Maintenance rule

Run this before opening or merging migration PRs that touch Streamlit pages or Next parity docs:

```bash
make check-next-parity-matrix
```
