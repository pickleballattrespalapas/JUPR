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
| `leaderboards` | 🏆 Leaderboards | Public | `Partial` — Next `/clubs/[clubSlug]/leaderboards` and FastAPI leaderboard API exist. | No | Public smoke + contract tests | Close display parity gaps: league filters, top performers, badge chips, inactive rules. |
| `rating_rules` | Rating Rules | Public | `Partial` — Next `/how-ratings-work` exists as public explainer. | No | Static content review | Reconcile route naming with `/faq` and final product copy. |
| `league_results` | 📊 League Results | Public | `Partial` — public FastAPI summary API and Next `/clubs/[clubSlug]/league-results` route exist. | No | Public smoke + contract tests | Validate staging, then close chart/deep-link/printout parity gaps. |
| `league_printout` | 🖨️ League Night Printout | Admin | `Not started`. | Export only | Auth/PDF/export gate | Defer until admin auth and export strategy are in place. |
| `match_explorer` | 🎯 Match Explorer | Public | `Partial` — public FastAPI context/preview API and Next `/clubs/[clubSlug]/match-explorer` route exist. | No | Public smoke + contract tests | Validate staging, then close Streamlit parity gaps for deep links and rating-impact chart. |
| `players` | 🔍 Player Search | Public | `Partial` — Next player directory, profile, match history, and match detail exist. | No | Public smoke + contract tests | Add trophy room/badge/profile parity and improve search/deep links. |
| `badge_codex` | 📼 Badge Codex | Public | `Partial` — public FastAPI badge/earner APIs and Next `/clubs/[clubSlug]/badge-codex` route exist. | No | Public smoke + contract tests | Validate staging, then close load-more/deep-link/trophy-room parity gaps. |
| `badge_debug` | 🧪 Badge Debug | Admin | `Not started`. | Diagnostic only | Admin auth + role gate | Keep Streamlit-only until admin diagnostics/audit shell exists. |
| `badge_audit` | 🧾 Badge Audit | Admin | `Not started`. | Diagnostic/write-adjacent | Admin auth + role gate | Define admin badge audit API after badge public read model is stable. |
| `match_canonical_audit` | 🧩 Match Canonical Audit | Admin | `Not started`. | Diagnostic/write-adjacent | Admin auth + role gate | Keep Streamlit-only until replay/correction APIs exist. |
| `challenge_ladder` | 🪜 Challenge Ladder | Public | `Partial` — public FastAPI ladder API and Next `/clubs/[clubSlug]/challenge-ladder` route exist. | No | Public smoke + contract tests | Validate staging, then close deep-link and public challenge-rule parity gaps. |
| `faqs` | ❓ FAQs | Public | `Partial` — Next `/faq` exists with public JUPR FAQ content. | No | Static content review | Validate copy against Streamlit FAQ and decide whether `/faq` or `/how-ratings-work` is canonical. |
| `privacy_policy` | Privacy Policy | Public hidden | `Partial` — first-party Next `/privacy` placeholder exists. | No | Static/legal review | Replace placeholder with approved legal policy before broad production launch. |
| `terms_of_use` | Terms of Use | Public hidden | `Partial` — first-party Next `/terms` placeholder exists. | No | Static/legal review | Replace placeholder with approved legal terms before broad production launch. |
| `contact_support` | Contact Support | Public hidden | `Partial` — Next `/support` and `/contact` support shell exists. | No | Static/support review | Confirm support address/routing and add real intake workflow if needed. |
| `data_corrections` | Data Corrections | Public hidden | `Partial` — Next `/data-corrections` intake instructions exist with no direct data mutation. | Request intake shell only | Support/intake gate | Define durable ticket/intake backend; keep rating corrections staff-reviewed. |
| `email_preferences` | Email Preferences | Public hidden | `Not started` in Next app. | Preference write | Auth/tokenized preference gate | Port only after safe email preference tokens and unsubscribe rules are defined. |
| `profile_privacy` | Profile Privacy | Public hidden | `Not started` in Next app. | Preference/request write TBD | Privacy workflow gate | Define privacy request flow before porting. |
| `league_manager` | 🏟️ League Manager | Admin | `Not started` for full parity; listed in `/admin` operations cockpit. | Yes | Admin auth + match-write + audit gate | Port after score entry/corrections foundation; this is a major workflow. |
| `match_uploader` | 📝 Match Uploader | Admin | `Partial` — Next score-entry MVP exists for one-match submission only and is listed in `/admin` operations cockpit. | Yes | Admin auth + club scope + audit + E2E | Replace token-paste MVP with real admin score entry, then add batch/RR/new-player parity. |
| `match_log` | 📝 Match Log | Admin | `Not started` for UI/API; listed as first operational target in `/admin` operations cockpit. | Yes | Admin auth + replay/correction audit gate | Port read, duplicate scan, correction preview, and replay-planning before broad score-entry writes. |
| `player_editor` | 👥 Player Editor | Admin | `Not started`; listed in `/admin` operations cockpit. | Yes | Admin auth + club scope + audit gate | Add player CRUD/merge APIs after match correction safety exists. |
| `admin_tools` | ⚙️ Admin Tools | Admin | `Not started`; listed as critical-risk in `/admin` operations cockpit. | Yes/high-risk | Super-admin-only + worker/replay gate | Keep Streamlit-only until jobs, replay, and diagnostics have hardened APIs. |
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
| `tournament_registration` | 📝 Tournament Registration | Public | `Partial` — public FastAPI registration page/submit APIs and Next `/clubs/[clubSlug]/tournament-registration` route exist. | Yes/intake | Public form + validation + anti-abuse + staging smoke gate | Validate staging, then close partner-board/edit-link/email-confirmation parity gaps. |
| `tournament_registration_admin` | 🧾 Registration Management | Admin | `Not started`; listed under tournament admin in `/admin` operations cockpit. | Yes | Admin auth + registration audit gate | Port after public registration write path is proven in staging. |
| `tournament_registration_confirmation` | ✅ Registration Confirmation | Public hidden | `Partial` — public FastAPI confirmation API and Next confirmation route exist. | No | Public form flow | Validate staging and add secure email/edit-link handoff. |
| `tournament_registration_edit` | ✏️ Edit Registration | Public hidden | `Not started`. | Yes/intake | Tokenized edit/auth gate | Add secure edit links before porting. |
| `tournament_roster` | 📋 Tournament Roster | Public hidden | `Not started`. | No | Public tournament read model | Add Next roster route using public tournament roster state. |
| `tournament_partner_board` | 🤝 Partner Board | Public | `Not started`. | Possibly yes | Public/organizer moderation gate | Define safe public interaction rules before porting. |
| `weekly_recap` | 🗞️ Weekly Recap | Public | `Partial` — public FastAPI published-recap/PDF APIs and Next `/clubs/[clubSlug]/weekly-recap` route exist. | No | Public smoke + contract tests | Validate staging, then close final styling/print-view parity gaps. |
| `top_players_printable` | 🧾 Top Active Players PDF | Admin | `Not started`. | Export only | Admin/PDF export gate | Defer until PDF/export strategy is standardized. |
| `weekly_recap_admin` | 🗞️ Weekly Recap Admin | Admin | `Not started`; listed in `/admin` operations cockpit. | Yes | Admin auth + recap write/audit gate | Port after public weekly recap read side and admin auth shell. |
| `player_updates_admin` | 📬 Player Updates Admin | Admin | `Not started`. | Yes/email | Admin auth + email safety gate | Keep blocked until staging email safety is proven. |
| `admin_login` | 🔐 Admin Login | Hidden/auth | `Partial` — Next `/admin` operations cockpit and FastAPI `/admin/operations/status` exist; real auth shell still needed. | No | Supabase Auth/session gate | Build real Next login/session flow before full admin cutover. |
| `reset_password` | 🔐 Reset Password | Hidden/auth | `Not started`. | Auth write | Supabase Auth/session gate | Add as part of real Next auth implementation. |
| `verified_updates_request` | 📬 Verified Updates Request | Public hidden | `Not started`. | Email/preference write | Tokenized email safety gate | Port only after email preference/update flows are safe in staging. |

## Recommended implementation sequence

1. Keep public read-only staging smoke and Vercel guardrails green.
2. Validate public Match Explorer in staging and close chart/deep-link parity gaps.
3. Validate public League Results in staging and close chart/deep-link/printout parity gaps.
4. Validate public Badge Codex in staging and close load-more/deep-link/trophy-room parity gaps.
5. Validate public Challenge Ladder in staging and close deep-link/public-rule parity gaps.
6. Validate public Weekly Recap in staging and close final styling/print-view parity gaps.
7. Review/approve static FAQ/legal/support copy and smoke the static routes.
8. Validate public tournament registration in staging, including duplicate-email and closed-registration cases.
9. Use `/admin` as the migration cockpit and enable closed-club production-write pilot mode only on the FastAPI runtime.
10. Port Match Log read, duplicate scan, correction preview, and replay planning.
11. Re-evaluate the guarded Score Entry MVP after Match Log correction visibility exists.
12. Replace the score-entry MVP with real Match Uploader parity.
13. Port Player Editor.
14. Port League Manager.
15. Port public tournament roster and partner board.
16. Port tournament/challenge/admin tools after the write foundation is proven.

## Maintenance rule

Run this before opening or merging migration PRs that touch Streamlit pages or Next parity docs:

```bash
make check-next-parity-matrix
```
