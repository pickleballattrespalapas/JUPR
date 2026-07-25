# Next/Streamlit Parity Closure Program

This is the execution plan for moving every page currently marked `Partial` in
`docs/next_streamlit_parity_matrix.md` to proven functional parity. It complements
the matrix: the matrix remains the release truth, while this document defines the
automated and manual evidence required to change a row to `Done`.

## Status discipline

A page may move through these operational states without changing the matrix legend:

1. **Implemented** — the Next/FastAPI surface exists.
2. **Automated-ready** — contract, build, and page-specific browser checks exist.
3. **Manual-ready** — fixtures, rollback steps, and an operator checklist exist.
4. **Done** — automated checks pass on staging and the manual acceptance checklist is signed off.

No row is promoted to `Done` merely because its route renders. Rated, destructive,
email, role, badge, ladder, league, and tournament writes must remain club-scoped,
audit-attributed, feature-flagged, and recoverable through Python/FastAPI authority.
Streamlit remains the operational fallback until the corresponding manual acceptance
is complete.

## Delivered implementation stack

The implementation was delivered as reviewable, merge-ordered PRs, with P0
security and data-integrity work preceding feature expansion. All 29 orders were
present at the reconciled July 24 checkpoint. Their history remains useful for
scope and dependency tracing, but it is no longer a future work plan. No feature
PR changed a matrix row to `Done`; the final evidence PR may do that only after
the consolidated manual session.

| Order | Delivered branch | Track | Scope | Dependency |
|---|---|---|---|---|
| 1 | `agent/parity-closure-plan` | `foundation` | This control plan, corrected matrix actions, shared staging helper, closure guard, deterministic fixture conventions, and page-spec layout | `staging` |
| 2 | `agent/parity-admin-data-security` | `foundation` | Canonical Supabase migrations, RLS/grant lockdown for server-only admin tables, security-advisor evidence, and anonymous/authenticated denial tests | order 1 |
| 3 | `agent/parity-tournament-edit-integrity` | `public-tournaments` | Imported-draw refusal, atomic/versioned selection edits, stable-token preflight, throttling, and post-edit notification status | order 2 |
| 4 | `agent/parity-tournament-confirmation` | `public-tournaments` | Non-enumerating confirmation access, private-field-safe projection, saved-on-mail-failure behavior, delivery status, and complete confirmation content | order 3 |
| 5 | `agent/parity-tournament-roster` | `public-tournaments` | Correct unresolved-status mapping, filters, partner metadata, public safety, and partner-board handoff | order 4 |
| 6 | `agent/parity-subscription-links` | `static-support` | Next unsubscribe URLs, scoped/idempotent preferences, legacy credential decision, and club-aware verified-request status | order 5 |
| 7 | `agent/parity-auth` | `auth` | Capability-checked login, safe `next` redirects, PKCE recovery, recovery cleanup, and wrong-club/expired-session E2E | order 6 |
| 8 | `agent/parity-public-leaderboards` | `public-data` | Active defaults, search/snapshot, gain/gap/qualification/badges, pagination, deep links, and route-specific E2E | order 7 |
| 9 | `agent/parity-public-league-results` | `public-data` | Rank delta/trend, recent matches, correct highlight semantics, full selectors, print/deep-link E2E | order 8 |
| 10 | `agent/parity-public-players` | `public-data` | Search, active defaults, trends/breakdowns/history, badges/trophies, partner/rival/social projections, privacy assertions | order 9 |
| 11 | `agent/parity-public-gamification` | `public-data` | Badge status/timing buckets and earner paging; Python-authoritative challenge eligibility and complete rulebook | order 10 |
| 12 | `agent/parity-public-explorer-recap` | `public-data` | Python-authoritative Match Explorer charts/contexts and reactive validation; Weekly Recap complete cards, paging policy, PDF/print E2E | order 11 |
| 13 | `agent/parity-static-support` | `static-support` | Rating/FAQ completeness, legal-link corrections, durable general-support form, anti-abuse/evidence/club validation, privacy fulfillment SOP | order 12 |
| 14 | `agent/parity-tournament-registration` | `public-tournaments` | Start/edit wizard, profile/partner policy, required fields, sponsor/refund/roster content, validation, and recovery | order 13 |
| 15 | `agent/parity-tournament-partner-flow` | `public-tournaments` | Deterministic request/email/review/accept/cancel-competing browser flow and notification safety | order 14 |
| 16 | `agent/parity-match-durability` | `match-player` | Match Log notes/bulk parity, atomic edit/replay requirement, tracked Replay History jobs, idempotency, audit/recovery semantics | order 15 |
| 17 | `agent/parity-match-player-flows` | `match-player` | Guarded Score Entry fallback contract, Match Uploader email handoff visibility, Player Editor transactional merge/compensation, E2E | order 16 |
| 18 | `agent/parity-league-core` | `league-comms` | League settings/roster/lifecycle validation, true leaders/top-performer print exports, and browser print evidence | order 17 |
| 19 | `agent/parity-league-awards` | `league-comms` | Persisted freeze/preview/override-reason/mint/archive wizard with idempotent retry and no false-success mint result | order 18 |
| 20 | `agent/parity-league-live-domain` | `league-comms` | Move court movement/orchestration from TypeScript to Python/FastAPI; roster, suggestion, bench, and override contracts | order 19 |
| 21 | `agent/parity-league-live-submit` | `league-comms` | Idempotent all-match submission, snapshot/publish reconciliation, rating refresh, substitutions/guests, exports, recovery | order 20 |
| 22 | `agent/parity-communications` | `league-comms` | Full unpublished recap preview/print plus subscription/digest/outbox queue/history/retry/delete/replace workflows and stale-state guards | order 21 |
| 23 | `agent/parity-admin-diagnostics` | `badges-tools` | Badge Debug/Audit, canonical audit, Admin Tools and Guide route-specific permission, confirmation, recovery, and no-write evidence | order 22 |
| 24 | `agent/parity-live-ladder` | `live-ladder` | Ladder and Moneyball deterministic lifecycle/recovery E2E; admin/public one-off runner Python-domain alignment | order 23 |
| 25 | `agent/parity-public-live` | `public-data` | Durable public Round Robin, League/Ladder and Club Social sessions, substitutions, continuation, completion, exports, and token/rate limits | order 24 |
| 26 | `agent/parity-tournament-setup-admin` | `tournament-admin` | Dedicated setup decision/route plus tournament and registration management stale-state/audit E2E | order 25 |
| 27 | `agent/parity-tournament-operations` | `tournament-admin` | Draw/import/scoring/playoffs/podium/awards/publish/singles/email workflow and recovery E2E | order 26 |
| 28 | `agent/parity-tournament-live` | `tournament-admin` | Draw-scoped in-play runner progression/publish/recovery evidence, explicitly separate from JUPR Live | order 27 |
| 29 | `agent/parity-final-evidence` | `foundation` | Consolidated CI, route-specific staging suites, manual test book, rollback ledger, readiness report, and post-manual matrix reconciliation | order 28 |

Order 21 is automated-ready: durable submit/reconcile/compensation, rating readback, guest and export contracts are implemented behind a separate staging-only gate. The `league_manager` matrix row intentionally remains `Partial` until the consolidated staging book is executed.

Order 24 is automated-ready and manual-ready: Challenge Ladder, Moneyball, and one-off JUPR Live use Python/FastAPI authority, separate staging-only write gates, version leases, durable idempotent result recovery, strict audit evidence, and Match Log/Replay handoffs. The three matrix rows intentionally remain `Partial` until `docs/live_ladder_parity_runbook.md` is executed and signed off.

Order 25 is automated-ready: public Quick Round Robin, League/Ladder, and Club
Social sessions now use versioned, idempotent FastAPI writes, hashed edit
credentials, durable recovery records, atomic completion reservations, private
database access, exports, and a truthful view-only fallback. The `jupr_live`
matrix row intentionally remains `Partial` until the consolidated staging book is
executed.

Order 26 is automated-ready/manual-ready: the dedicated setup route, Python
templates and impact review, guided structured day/event-family/division/options
builder, advanced JSON import/export, stale-safe tournament/registration edits,
durable audit/replay semantics, imported-draw refusal, and the read-only
Operations import handoff are covered. The three tournament rows intentionally
remain `Partial` until the consolidated staging book is executed.

Order 27 has automated implementation and focused recovery evidence:
route-specific draw/import/results/publish workflows, Python-authoritative
lifecycle checks, reviewed DUPR import, atomic result/player persistence, game
score CAS, and deterministic exact-set official-publish reconciliation are
covered. The tournament rows intentionally remain `Partial`; no `Done` marker is
allowed until the disposable staging acceptance and cumulative Order-29 evidence
are recorded.

Order 28 is automated-ready/manual-ready: the in-play runner is draw-scoped and
Python-authoritative, uses a separate staging-only write gate, retains exact
requests for deterministic replay, exposes durable audit/recovery evidence, and
refuses to unlock ambiguous official publishing. Tournament Live remains
`Partial` until the documented desktop/mobile staging book and Order-29 sign-off
are complete.

All 29 implementation orders were present at the reconciled
`eab384545c493f145af383c8e26d8bf97686ab21` checkpoint. Subsequent UX and
deployment-policy hardening can advance an implemented row to automated-ready or
manual-ready, but never to `Done` without exact-candidate evidence.

## Page-level closure contracts

The automated column defines evidence that can be built before the final manual test
session. The manual column is deliberately retained for real staging data, layout,
email, recovery, and operator-language checks.

| Streamlit key | Track | Automated closure | Manual acceptance |
|---|---|---|---|
| `leaderboards` | `public-data` | Active-default/search, filters, sort, snapshot/gain/gap/qualification fields, badge context, paging, chart cards, and player links are browser-asserted against staging. | Compare ordering, calculations, labels, empty states, and responsive layout with Streamlit. |
| `rating_rules` | `public-data` | Canonical explainer route and FAQ cross-links are asserted; rating copy is regression-guarded. | Approve wording and canonical navigation label. |
| `league_results` | `public-data` | League/section/week/player deep links, rank delta/trend, recent-match table, correctly scoped weekly/season highlights, chart data, and print mode are browser-asserted. | Compare representative standings, deltas, weekly results, recent matches, charts, and print layout. |
| `league_printout` | `league-comms` | Authenticated print route renders roster, schedule, standings, and attendance sections with print CSS. | Print or save a representative league night and approve pagination. |
| `match_explorer` | `public-data` | Share-link hydration, selections, preview, expected score, projected deltas, and impact chart are asserted. | Compare a representative matchup and shared URL with Streamlit. |
| `players` | `public-data` | Directory search plus profile rating trend, trophies/badges, social identity, partner/rival summaries, verified-update entry, match history, singles fields, and format labels are asserted. | Compare representative active/inactive/singles player profiles, partner/rival facts, awards, and mobile layout. |
| `badge_codex` | `public-data` | Status/timing buckets, filters, direct anchors, per-badge earner pagination, load-more behavior, earners, and trophy-room content are asserted. | Compare badge definitions, timing/status groupings, and representative earners with Streamlit. |
| `badge_debug` | `badges-tools` | Auth/permission gates and player/badge/league/date diagnostic filters are contract/browser tested. | Compare a representative diagnostic result set with Streamlit. |
| `badge_audit` | `badges-tools` | Expected-vs-actual summaries and safe no-write behavior are contract/browser tested. | Review mismatches and confirm no repair occurs from audit-only controls. |
| `match_canonical_audit` | `badges-tools` | Audit and dry-run normalization are asserted; apply requires exact confirmation, permission, and audit intent. | Run a staging dry-run and verify recovery links; do not normalize production. |
| `challenge_ladder` | `public-data` | Section/tier/player/challenge links, rulebook, status legend, and eligible-opponent hints are asserted. | Compare ranks, active challenges, and public rules with Streamlit. |
| `faqs` | `static-support` | FAQ content, anchors, and rating-explainer cross-links are regression-guarded. | Approve final wording and duplicate-topic treatment. |
| `privacy_policy` | `static-support` | First-party privacy sections, operator/alias disclosure decision, and direct correction/privacy/email-preference links are asserted. | Owner/legal approval is recorded before public launch. |
| `terms_of_use` | `static-support` | The complete first-party scope/ratings/registration/disputes/availability/limitation/contact draft is regression-guarded and stale placeholder claims are removed from trackers. | Owner/legal approval is recorded before public launch. |
| `contact_support` | `static-support` | Support/contact alias, address, populated mail link, FAQ/correction links, and durable `general_support` intake are asserted. | Test mail handling on real devices and confirm one queued staging request reaches staff. |
| `data_corrections` | `static-support` | Email/evidence validation, safe URL schemes, rate/deduplication guard, club-owned player validation, persisted intake, wrong-club admin denial, audit, and no-direct-mutation boundaries are tested. | Submit a staging request, move it through review/resolution, and verify source data remains unchanged. |
| `email_preferences` | `static-support` | Senders emit the Next token URL; invalid/expired/idempotent unsubscribe, category/global semantics, and deliberate legacy-sid handling are contract tested. | Open a staging-redirect email link, confirm masked identity, unsubscribe, and verify future queueing skips it. |
| `profile_privacy` | `static-support` | Persisted review request, identity/fulfillment status contract, permission/club/audit guards, and no-immediate-public-mutation behavior are tested. | Verify identity, perform the approved alias/hide SOP, and check every public projection before resolving. |
| `league_manager` | `league-comms` | Draft/save/preview/lifecycle locks, roster, live movement, awards, and recovery handoffs are tested. | Run a disposable staging league through create, live round, close, and recovery. |
| `match_uploader` | `match-player` | Manual, batch, singles, round-robin, new-player continuation, audit, and email handoff contracts are tested. | Submit disposable rows, verify ratings/audit, then correct through Match Log/Replay. |
| `match_log` | `match-player` | Filters, duplicate/no-issue, guided edit, social edit/delete, exclude, cleanup, audit, and replay handoffs are tested. | Exercise reversible staging edits/deletes and verify restored final state. |
| `player_editor` | `match-player` | Create/update/rating/link/auto-link/merge preview and permission/audit guards are tested. | Use staging-only players; execute merge only with immediate Replay History recovery. |
| `admin_tools` | `badges-tools` | Roles/activity, reports/CSV, social moderation, badge worker/recompute/lifecycle, and tournament backfill guards are tested. | Exercise isolated staging fixtures, inspect audit rows, and verify recovery runbooks. |
| `admin_guide` | `badges-tools` | Every migrated admin route, accessible confirmation dialog and internal API safeguard, stop condition, and fallback link is statically guarded. | Staff review terminology and day-of operating sequence. |
| `challenge_ladder_admin` | `live-ladder` | Eligibility/notices/clock/result preview/publish/rank/pass/roster/tier/override/audit guards are tested. | Run isolated ladder fixtures and recover published matches through Match Log/Replay. |
| `moneyball` | `live-ladder` | Eight-player selection, expected score, settlement, official publish, and correction handoff are tested. | Run and reverse a disposable staging event. |
| `jupr_live` | `public-data` | Round-robin, League/Ladder, and Club Social session views; refresh persistence; substitutions; exports; public score state; and edit-token boundaries are asserted. | Operate each one-off event type on mobile, substitute a player, lock the phone, resume, export, and approve persistence. |
| `jupr_live_admin` | `live-ladder` | Session create/update, score save, round advance, official publish, audit, and tournament rejection are tested. | Run a disposable one-off session and verify public view plus recovery. |
| `tournaments` | `tournament-admin` | Read/edit/selection/division/bulk/archive/delete guards and stale-state handling are tested. | Operate a staging draft and confirm Streamlit fallback consistency. |
| `tournament_manager` | `tournament-admin` | Dedicated setup route, settings/draft/templates/impact review, structured add/remove/reorder controls, payload-equivalent advanced JSON import/export, validation, and publish confirmation are tested. | Build a disposable setup with the guided controls on desktop/mobile, review its summary, then publish it only in the bounded tournament write session. |
| `tournament_ops` | `tournament-admin` | Draw/import/games/scoring/playoffs/podium/awards/publish/singles/email handoff are tested. | Run a disposable tournament end to end and verify replay recovery. |
| `tournament_live` | `tournament-admin` | Draw-scoped live scoring/progression/podium/awards/publish and separation from JUPR Live are tested. | Run the in-play staging runner on desktop/mobile and verify recovery. |
| `tournament_registration` | `public-tournaments` | Profile resolution, partner-selection wizard, submit validation, duplicate email, closed state, selection integrity, sponsor/refund fields, roster handoff, and edit-link request are tested. | Submit representative doubles/singles registrations, including partner-needed and sponsor/refund cases, in staging. |
| `tournament_registration_admin` | `tournament-admin` | List/detail/single/bulk/selection edits and ops import handoff are tested with audit guards. | Review and modify disposable registrations, then import them into a draw. |
| `tournament_registration_confirmation` | `public-tournaments` | Save-survives-mail-failure, dry-run/redirect delivery status, token-safe lookup, private-field exclusion, event/payment/roster content, and email handoff are tested. | Submit in staging, receive the redirected email, open its link, and simulate delivery failure while confirming the registration remains saved. |
| `tournament_registration_edit` | `public-tournaments` | Stable token secret, lifetime, non-enumerating/rate-limited request, locked email, atomic/versioned selection edits, imported-draw refusal, and post-edit confirmation are tested. | Use normal/expired/closed links and verify imported selections and draw teams cannot change. |
| `tournament_roster` | `public-tournaments` | Null/legacy review-state mapping, filters/counts, public-safe projection, partner metadata, summaries, needs-partner links, and private-field exclusion are asserted. | Compare every representative status/filter with admin registration data on desktop/mobile. |
| `tournament_partner_board` | `public-tournaments` | Interest/request/review/accept/cancel-competing/token/privacy behavior is tested. | Complete a disposable pairing flow and verify notifications plus contact privacy. |
| `weekly_recap` | `public-data` | Week/section links, recap rendering, PDF handoff, and print view are asserted. | Compare a published recap and printed/PDF output with Streamlit. |
| `top_players_printable` | `league-comms` | Authenticated ranking output and print CSS are browser-asserted. | Print/save representative output and approve pagination. |
| `weekly_recap_admin` | `league-comms` | Generate/edit/spotlight/preview/publish/unpublish permission and audit contracts are tested. | Publish and unpublish a disposable staging recap. |
| `player_updates_admin` | `league-comms` | Exact-range reports, selected sends, verified-request review, dry-run/redirect safety, and audit are tested. | Send only through staging redirect and inspect the delivered message. |
| `admin_login` | `auth` | Password-form behavior, session restore/refresh, invalid/expired token, wrong-club, and permission denial have component/browser contracts. The exact-candidate workflow separately proves a server-minted, live capability-checked staging session and local-scope sign-out; it does not claim operator password entry. | Manually sign in/out with the required staging roles and approve operator messaging; retain password entry and inbox-backed recovery as manual evidence. |
| `reset_password` | `auth` | Recovery request, callback consumption, invalid/expired recovery, password policy, and post-reset login are tested. | Complete one staging recovery email flow. |
| `verified_updates_request` | `static-support` | Club-aware options/status/request, player prefill, server email validation, rate/deduplication, pending/active display, wrong-club admin denial, approve/reject/audit, safe delivery, and unsubscribe handoff are tested. | Submit for an isolated staging player, approve/reject, inspect redirected delivery, unsubscribe, and verify no future sends. |

## Global acceptance gates

Before any page moves to `Done`:

- `make check-next-parity-matrix` and `make check-parity-closure-program` pass.
- API contract, production-deployment verifier, authenticated auto-load, and
  focused domain/static safety tests pass in GitHub Actions.
- The full Next production build passes.
- The Next build runs the Tournament Setup payload-builder contract on its pinned
  Node 20 runtime.
- The isolated staging smoke passes all 56 strict public-read tests and all five
  guided Tournament Setup browser tests with zero skips, failures, or flakes.
- Authenticated tests use only the staging Supabase project and staging-only users.
- Mutating tests use isolated fixtures and include a verified cleanup or recovery step.
- Email tests use `dry_run` or `staging_redirect`, never an unrestricted live mode.
- The page's manual acceptance row above is signed off in the consolidated test book.
