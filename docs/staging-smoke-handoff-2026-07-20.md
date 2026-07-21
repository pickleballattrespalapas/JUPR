# Staging smoke handoff — 2026-07-20/21

This handoff preserves the staging session state without relying on chat history.
It contains no secret values. Continue from canonical branch `staging`; `Test` is
legacy/deprecated and must not be used for a new candidate or acceptance run.

## Current deployed baseline identity

| Surface | Bound value |
|---|---|
| Git SHA | `e695365ce508e03a094f528ff9c1179c7f7947de` |
| Merged change | PR `#1016`, “Stabilize staging player search smoke” |
| Vercel project / deployment | `jupr` / `dpl_6zDSfrXKV4PC2M5MX3K5q4UQuvzf` |
| Vercel staging alias | `https://jupr-git-staging-pickleballattrespalapas1.vercel.app` |
| Vercel immutable origin | `https://jupr-84z27cfwn-pickleballattrespalapas1.vercel.app` |
| Fly staging API | `https://juprleagues-api-staging.fly.dev` |
| Fly image / machine | `registry.fly.io/juprleagues-api-staging:deployment-01KY180GGE2V9HC13E8N9VFE3Y` / `01KY1821CAK5RC7GE2BWMBQYYM` |
| Supabase project ref | `sijpxjxvdtrehmqvirfi` |
| Final restore evidence | GitHub run `29795882496`, job `88527006272` |

That is the last deployed baseline. The local hardening patch is a new candidate.
After it is merged and deployed, replace the SHA, PR, Vercel deployment ID/origin,
Fly image, and identity artifact as one unit. Never combine baseline evidence with
the replacement candidate.

## Final runtime state already restored

- `write_wave=none`
- `business_data_write_wave_active=false`
- every controlled write flag false
- external email delivery unavailable; `JUPR_EMAIL_MODE=dry_run`
- production data and production Fly/Vercel targets untouched

`none` is the steady state. The only valid write transition is
`none -> one named least-privilege wave -> none`. Do not enable all flags and do
not move directly from one active write wave to another.

## Completed read and guard sweep

- The authenticated Operations Cockpit route sweep completed.
- Challenge Ladder Admin loaded the authenticated club state with 8 players,
  0 active challenges, and an explicit read-only guard.
- Admin Tools loaded its disabled state without worker or backfill controls.
- The Site Map loaded the public, tournament, support/privacy/account, and staff
  route inventory.
- Public club, leaderboards, players, matches, Match Explorer, League Results,
  Badge Codex, Challenge Ladder, Weekly Recap, JUPR Live, tournament registration,
  roster, and partner-board navigation had already been covered in the route sweep.
- Ratings explainer and FAQ rendered complete informational content.
- `/support` and `/contact` rendered the blank durable support form.
- `/data-corrections` rendered a staff-review-only correction form; it does not
  directly edit ratings, match history, players, tournaments, or badges.
- `/privacy`, `/profile-privacy`, and `/terms` rendered their expected content and
  safe review-only forms/links.
- Tokenless `/email-preferences` rendered the expected safe state
  `Preference link not found`; no preference mutation occurred.
- A malformed/direct `/faq/support` route produced the safe Next fallback page;
  the canonical support route is `/support`.

## Completed support intake/deduplication/dismissal evidence

The disposable fixture was intentionally retained as staging evidence:

| Field | Value |
|---|---|
| Request ID | `req_2baca74d135646e6be38` |
| Type | `general_support` |
| Subject | `STAGING SMOKE - support intake 2026-07-20` |
| Description | `Disposable staging-only support intake fixture. No customer action is requested.` |
| Requester | `PCS Staging Smoke 2026-07-20` |
| Email | `pcs-staging-smoke-20260720@example.invalid` |
| Requested action | `Staff response requested.` |
| Final status | `dismissed` |
| Admin note | `staging smoke only. no customer action. retained as test evidence.` |
| Reviewer | authenticated staging admin (identity retained in private evidence) |

Observed behavior:

1. The first public submission created one `new` queue item.
2. An exact repeat was deduplicated and reported that the existing request remained
   in the staff queue; the queue count remained one.
3. The manually deployed `support-requests` wave exposed the authenticated queue.
4. The operator loaded, selected, and dismissed the fixture using confirmation
   `SAVE REQUEST STATUS`.
5. Readback showed reviewer attribution and one audit event with action
   `update_public_support_request_admin`, entity `public_support_request`, before
   status `new`, and after status `dismissed`.
6. A separate Fly deployment restored `none`; run `29795882496` verified the
   all-false controlled-write projection.

The historical Fly image/run references for the two active write-wave deployments
were not retained in the chat record. Do not invent them or treat the final `none`
image as their identity. The final `none` release is fully identified above.

## Automated checks already completed locally

- Supabase migration version check: pass.
- Next parity matrix: 47 entries, pass.
- Parity closure program: 45 entries, pass.
- Manual staging book structure: 45 entries, pass.
- Integrated manifest and migration-source guard against `origin/staging`: pass.
- Python compile: pass.
- Production readiness report for the baseline: pass; no risky migration.
- Next dependency tree, TypeScript check, and production build: pass; 54 pages.
- Browser manifest collection: exactly 56 tests across 5 specs.
- Staging GET-only API sweep: 31/31 pass after one bounded transient `/health`
  502 retry.
- Focused API contract suite: 405 passed.
- Full Python suite: 1,677 passed with 48 deprecation/future warnings.

Local Chromium could not reach the protected Vercel candidate reliably in the
sandbox, so local browser navigation is not acceptance evidence. The canonical
GitHub `Staging Smoke` workflow remains the evidence-grade browser gate.

## Current approved next work

The owner approved reconciliation/commit/push/deploy of the tested hardening patch
and approved the next staging writes discussed in this session. The bounded next
packet deliberately authorizes exactly:

- A: one short data-correction intake plus one exact retry for deduplication;
- B: dismissal of that same disposable request through the authenticated admin
  support queue.

See `docs/staging-next-approval-packet-2026-07-20.md`. Approval of these two actions
does not authorize registration, pairing, email-preference, privacy fulfillment,
rating, match, player, tournament, live, worker, or production mutations, even if
the selected infrastructure wave technically contains additional routes.

The owner explicitly prefers short, repeatable fixture values for future intake
and deduplication tests. Preserve that preference in all later packets.

## Manual boundaries and where the owner is next needed

1. A connector/agent may reconcile code, run local/CI checks, open the PR to
   `staging`, and inspect deployments, but it cannot manually dispatch the
   canonical GitHub workflows in this environment.
2. After the final hardening candidate is deployed, the owner must manually run
   `Deploy FastAPI staging to Fly` from ref `staging` for each named wave in the
   approval packet and later for `none`.
3. The public intake itself does not require an admin token. The dismissal does:
   the operator must use the existing browser admin session or a bearer token
   kept outside chat/logs. Never paste an admin token or secret into the packet.
4. After the last same-SHA `none` release attests all writes off, the owner must
   manually launch `Staging Smoke`. Do not launch it during a write wave.
5. Legal-copy approval, safe staging email accounts/inbox, and any production
   cutover remain deferred and require separate approval later.

## Operator and witness for formal acceptance

- **Operator:** the named person who performs the manual action under their own
  authenticated identity and records the run/readback. The prior dismissal was
  performed through an authenticated staging-admin session; personal identity
  belongs in private evidence, not this public repository.
- **Witness/reviewer:** a different person who independently checks that the
  candidate identity, approved scope, before/after state, audit evidence, and final
  `none` restoration all match the record. They do not share the operator's login
  or token and do not need to perform the mutation.

The separation is an acceptance-control requirement, not another application
role. One person may run an exploratory smoke, but formal acceptance remains
incomplete until a distinct witness is named. If no witness is currently
available, keep the relevant manual-book results `Pending`; do not fabricate one.

## Stop conditions

Stop immediately on candidate/Supabase mismatch, active wrong wave, unexpected
write route, changed/stale request, ambiguous response, missing audit, wrong-club
visibility, external email, private-field exposure, non-idempotent exact retry, or
failure to restore `none`. Preserve evidence and reconcile before any retry.
