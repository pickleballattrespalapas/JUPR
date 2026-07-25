# Exact-candidate staging acceptance packets — 2026-07-25

This guide prepares the remaining operator and witness work for the next
Pickleball Club Sandwich staging candidate. It is a preparation artifact, not a
formal acceptance record. The authoritative results remain in
`docs/next_parity_manual_staging_book.md`; no row becomes `Pass` until its exact
candidate evidence and required human sign-off exist.

Merge this guide with the executable repair candidate before that candidate is
frozen. Executable waves require the application candidate to equal the current
`staging` head, so changing this guide after the freeze would unnecessarily
change the candidate. The completed formal evidence book still has a separate,
deliberate book-only completion path.

## Final candidate identity

| Field | Exact value |
|---|---|
| Application candidate | `Pending — record the repair-merge staging SHA` |
| Vercel deployment | `Pending — record the exact-SHA deployment` |
| Vercel immutable origin | `Pending — record the immutable deployment origin` |
| Fly app | `juprleagues-api-staging` |
| Final none-wave Fly image | `Pending — record the exact-SHA none release` |
| Supabase project | `sijpxjxvdtrehmqvirfi` |
| Migration inventory | `Pending — require the reviewed 38-name inventory, including singles_replay_recovery, and record the connector ledger head` |
| Email policy | `dry_run`; live player-update email disabled |
| Streamlit fallback | `https://juprtrespalapas.streamlit.app` |

The Fly image changes at every write-wave deployment. Record the image returned
by each deployment and never reuse one wave's image for a later wave.

### Superseded 2026-07-25 checkpoint

The following identity was reconciled successfully for read evidence, then a
fixture/recovery audit found defects in managed singles replay, League Live
preview gating, and recovery-context filtering. It is preserved for diagnosis
only and is not the final candidate.

| Wave | Result | Evidence |
|---|---|---|
| Identity | SHA `7cedb81ca251023806b0953db996a6e7b80c381a`; Vercel `dpl_3SJQdBYoUtPyeMmdEFCgmNMPmcX3`; immutable origin `https://jupr-b79zliik8-pickleballattrespalapas1.vercel.app`; Fly none image `registry.fly.io/juprleagues-api-staging:deployment-01KYD3NVQR5KF9GGMKG34JQ2EF`; 37 migrations through `20260720123402_baseline_worker_run_log` | Superseded after the repair audit |
| Fly exact-candidate deployment, `none` | Passed health, OpenAPI, schema, identity, migration, and feature-policy gates | Run `30166837522` |
| `public-read` | 56/56 passed; zero skipped, unexpected, flaky, or retried | Run `30167694296`; artifact `8622060773`; digest `sha256:0d8704007ff0a2f41db307ac1abd16a871ce19e25eea9eac0a7a0abdcfd18d1b` |
| `admin-read-export` | 16/16 passed; zero skipped, unexpected, or flaky; refresh session ended and exported credentials cleared | Run `30167701340`; artifact `8622073617`; digest `sha256:1ae4f3c4cac382d60e43e56891b2237754a024a96abf9f6a669fb02329ef9e48` |

Both read waves began and ended with `write_wave=none`,
`business_data_write_wave_active=false`, all 29 then-controlled business-write
flags false, public live writes false, and email mode `dry_run`. Preserve that
historical count. The repair adds the isolated preview flag plus dormant direct
singles and Match Log destructive flags, so the final all-false projection
contains 32 controlled flags. These results must be rerun for the final repair
SHA and cannot be copied into the formal book.

## Reviewed debt and production blockers

- At the superseded checkpoint Supabase was `ACTIVE_HEALTHY`, but
  leaked-password protection still had one security warning and eight
  duplicate-index groups remained for reviewed remediation. Reread the advisors
  after the final migration. These findings do not invalidate the historical
  staging read runs, but leaked-password protection remains a production-launch
  blocker.
- The sampled PostgreSQL log is not clean. It contains handled compatibility
  probes for optional legacy columns plus the deliberate nonexistent-player
  browser test. The API fallback succeeded and no Vercel 5xx or severe runtime
  error resulted, but this should be reduced after candidate acceptance.
- GitHub runner logs contain Node-action and npm-package deprecation warnings.
  They did not affect the evidence result.

These items are not legal acceptance, production monitoring, or production
cutover approval.

## Remaining automated sequence

Run one transition at a time. GitHub permits only one running and one pending run
in the shared staging concurrency group, so do not queue the whole sequence.

1. Merge the repair PR, record the exact `staging` SHA, apply and verify the new
   staging-only singles replay migration, and record the connector-assigned
   38-name migration ledger head. Do not infer that head from the repository
   filename.
2. Confirm Vercel serves that exact SHA, then deploy Fly from the same SHA with
   `write_wave=none`. Capture its image and re-attest health, OpenAPI,
   environment, staging Supabase identity, migration inventory, and the all-false
   controlled-write projection.
3. Run `Parity Final Evidence` first in `public-read` mode and then in
   `admin-read-export` mode against that exact Vercel/Fly identity. Neither
   superseded artifact can be carried forward.
4. Deploy Fly from exact `staging` with `write_wave=public-intake-auth`.
5. Capture its new Fly image and run `Parity Final Evidence` in
   `public-intake-auth` mode with mutation confirmation blank.
6. Deploy Fly from exact `staging` with `write_wave=none`; attest every controlled
   write flag false.
7. After the manual packets that precede Tournament Live are reconciled, deploy
   `write_wave=tournament-live`.
8. Run `Parity Final Evidence` in `match-rating-writes` mode with exact internal
   confirmation `RUN DISPOSABLE STAGING WRITES`. The workflow owns its disposable
   score fixture and must restore and reread the original score in `finally`.
9. Deploy `write_wave=none` and attest the all-false projection.
10. Only after every remaining manual write/recovery packet is complete, perform
   the final same-candidate `none` deployment and canonical `Staging Smoke`.

The `public-intake-auth` automated mode proves authenticated readiness only. It
does not submit a support request, registration, or other application
business-data write.

## Universal operator rules

- Operate only on the immutable candidate origin supplied for the active packet.
- The evidence runner records Git SHAs, deployment IDs, Fly images, database IDs,
  fingerprints, and audit IDs. The operator and witness do not transcribe them.
- Every manual write uses `none -> one named wave -> authoritative
  readback/recovery -> none`.
- Never move directly between active write waves.
- Use an accessible Yes/No dialog. Never type an internal confirmation phrase.
- Stop on wrong-club data, private-field exposure, external email, stale-state
  conflict, uncertain response, duplicate durable rows, missing audit
  attribution, or unavailable recovery.
- A distinct human witness observes every manual-only write and visible result.
  Automated-only workflow waves do not require a witness.

## Short fixture vocabulary

| Field | Value |
|---|---|
| Name | `Test` |
| Email | `test@x.invalid` |
| Subject/title | `smoke` |
| Details | `staging only` |
| Admin note | `test only` |
| Requested action | `none` |
| Player display name, only when required | `Test Player` |

Do not append dates or long random strings. Resource IDs, edit tokens,
idempotency keys, version timestamps, scores, and before-state values come from
the prepared packet.

## Session 1 — non-mutating visual and operator acceptance

Use the existing staging admin account in a normal browser. This session makes no
business-data writes. Open each group on desktop and one narrow/mobile viewport,
then record only visible discrepancies. Legal approval, real inbox delivery, and
password recovery remain separately owner-bound.

This session supplies visual/operator evidence only for the exact rows exercised.
It does not convert the entire 20-row non-mutating group to `Pass` by association.
Record each page key separately in the formal book.

### Public discovery and print

- Leaderboards: verify default ordering, one alternate sort, search, empty state,
  labels, and mobile layout.
- League Results and printout: select a representative league/week, verify
  standings and recent matches, then print/save and inspect pagination.
- Match Explorer: load a representative matchup and its share URL; compare
  expected score, deltas, and chart.
- Players: search active, inactive, and singles examples; inspect mobile profile,
  partner/rival facts, trophies, badges, and history.
- Badge Codex and Challenge Ladder: inspect representative definitions, earners,
  ranks, active challenges, rules, and deep links.
- Weekly Recap and Top Players Printable: inspect one published recap and both
  print/PDF layouts.
- Rating Rules and FAQ: approve navigation, labels, canonical cross-links, and
  duplicate-topic treatment.
- Privacy and Terms: verify the deployed draft and links, but record
  `Blocked` rather than `Pass` until owner/legal approval exists.
- Support/contact: verify the mail link on a real device and the blank form
  state. Staff-mail handling remains owner/inbox-bound; the durable intake is a
  separate witnessed write in Session 2.
- Tournament roster, confirmation, edit, and partner-board routes: verify
  public-safe empty/invalid-token states, representative filters, and the private
  field denylist. Valid delivered-token flows remain inbox-bound.

### Authenticated read, diagnostics, and guided setup

- Sign in normally, refresh once, sign out, and approve the visible session and
  permission messaging. Do not run password recovery during automated
  authenticated evidence.
- League Manager, Awards, Tournament lists/Ops/Live/Status/Registration/Bulk,
  support queues, Match Log/Replay, Player Editor/Updates, Weekly Recap, Badge
  Diagnostics, Challenge Ladder, JUPR Live, and Admin Tools must populate
  automatically after authentication and on selector change. A Reload/Retry
  control is recovery only.
- For one failed/empty state, verify the selection is not silently cleared and no
  stale selected record remains visible.
- Build a Tournament Setup draft with guided rows on desktop and mobile. Add,
  remove, and reorder one row; trigger one inline validation error; review the
  readable summary; inspect Advanced import/export; do not save or publish.
- Verify all destructive controls use keyboard-accessible Yes/No dialogs with
  focus contained and restored. No user-facing typed confirmation is allowed.
- Print League Awards and Top Players output and approve narrow-screen and print
  pagination.
- Compare Badge Debug and Badge Audit read results and review the Admin Guide's
  terminology, stop conditions, recovery order, and Streamlit fallback links.
  A Match Canonical Audit dry-run is a separately gated action and is not implied
  by viewing the page.
- Exercise the required staging roles, wrong-club denial, expired/invalid session
  messaging, and password-form behavior. The exact-candidate workflow proves one
  capability-checked session; password recovery still requires a delivered
  staging recovery email and remains blocked until that inbox session.

## Session 2 — bounded write/recovery packets

The required route, prestate, positive readback, truthful inverse or retained
evidence, and witness instruction for each packet are listed below. Several
packets still have unresolved fixtures or owner/inbox gates and are therefore
preparation records, not Joe-ready prompts. Do not begin a packet until its exact
fixture and recovery owner are resolved.

Two additional fail-closed boundaries narrow Joe's next active session:
`JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES=0` and
`JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0` in every current wave. Direct
uploader singles, duplicate cleanup, and bulk exclusion are implemented but
`Blocked` pending atomic writers and idempotent recovery. Defer every manual
match-producing case whose required cleanup uses bulk exclusion. The sections
below retain those future protocols so they are not lost; a retained protocol
is not an instruction to run it on this candidate.

All 25 manual mutation rows below are still `Pending` or `Blocked`. This guide
describes how to collect evidence; it does not supply evidence and does not turn
a bounded subset of a row into a full page-row pass. The page-level closure
contracts in `docs/next_parity_closure_program.md` remain authoritative when
they require several cases for one page.

For every row:

- Joe is the operator and a second human is the witness. The witness only needs
  to observe the named browser action and visible result; they do not need to
  understand infrastructure identity.
- Start from `write_wave=none`, deploy exactly the one wave named below, perform
  the write and authoritative readback, then return to `none`. If cleanup needs
  another wave, return to `none` before enabling it.
- Capture the exact prestate before the write. Use the UI Yes/No dialog and the
  shortest fixture values from this guide.
- In the formal book, replace every route parameter below with its real value
  and record only the root-relative path, without a query string. A JSON-bearing
  2xx response is required.
- Record a real inverse only when one exists. A dismiss, reject, archive,
  complete, publish, unsubscribe, exclusion, replay, or compensation action is
  a forward/finalizing action unless the API truly restores the captured
  prestate. Otherwise record `RETAIN; path=N/A; status=N/A` and preserve the
  durable row plus its audit evidence.
- Stop immediately on a stale-version conflict, duplicate durable record,
  external email attempt, unexpected player/rating change, missing audit
  attribution, or unavailable recovery.

The route notation uses `<value>` only to show where the prepared packet inserts
an exact ID, slug, key, or version. Never copy a bracketed value into the formal
book.

### Packet 1 — public intake and registration

Enable only `public-intake-auth`. The witness observes each submission,
confirmation, idempotent retry when named, and visible result. These writes may
share one wave only after all exact fixtures and tokens are prepared.

| Ledger row | Joe-ready action and shortest fixture | Authoritative readback | Recovery or retention |
|---|---|---|---|
| `manual:support-intake` | At `/support`, submit General support with `Test`, `test@x.invalid`, `smoke`, `staging only`, consent Yes, and optional IDs blank. Route: `POST /clubs/<club-slug>/support/intake`. Precheck that the same fingerprint has not existed for 24 hours and the address is below the hourly rate limit. | Expect `accepted=true`, `deduplicated=false`, one request ID, type `general_support`, status `new`. In a later `support-requests` wave, `GET /admin/clubs/<club-id>/support-requests` must show that exact row. | No delete exists. In Packet 2, dismiss the exact row with note `test only`; then `RETAIN` the dismissed request and audit. |
| `manual:data-corrections` | At `/data-corrections`, submit `Test`, `test@x.invalid`, `smoke`, `staging only`, requested action `none`, consent Yes, and optional player/match/tournament IDs blank. Route: `POST /clubs/<club-slug>/support/intake`. Retry the identical request once. | First response: one new correction request. Retry: `deduplicated=true` and the same request ID. Packet 2 must show exactly one admin row. | No delete exists. Resolve the no-op request in Packet 2, prove no player/match/rating/tournament source projection changed, and `RETAIN` the request and audit. Do not evade deduplication by changing text. |
| `manual:profile-privacy` | At `/profile-privacy`, submit requester `Test`, `test@x.invalid`, player `Test Player`, request kind Review display name, details `staging only`, consent Yes. Route: `POST /clubs/<club-slug>/support/intake`. | Packet 2 admin readback must show type `profile_privacy`, status `new`, and identity/fulfillment still `pending`. | Dismiss only the disposable intake row, with note `test only`; leave identity/fulfillment pending and resolution action unset. `RETAIN`. Real fulfillment remains owner/identity-verification blocked. |
| `manual:verified-updates` | At `/clubs/tres-palapas/verified-updates`, use one isolated real staging-only player, `test@x.invalid`, note `staging only`. Route: `POST /clubs/<club-slug>/verified-updates/request`. Retry the identical request once. | `GET /clubs/<club-slug>/verified-updates/status` for that player must show one pending request; retry must deduplicate to the same subscription/request. | Review only in Packet 3. Rejecting the disposable request is the safest finalizer and is retained; it is not an inverse. Approval and delivered-link evidence remain inbox blocked. |
| `manual:tournament-registration` | At `/clubs/tres-palapas/tournament-registration`, register `Test A`, `a@x.invalid`, age 18, a compatible gender, note `staging only`, terms Yes, and one prepared open event. Route: `POST /clubs/<club-slug>/tournament-registration`. | Response supplies the registration ID, selected event, and a confirmation token. `GET /clubs/<club-slug>/tournament-registration/confirmation` with that confirmation token must show the same saved registration. It does **not** supply an edit token. | No delete exists: `RETAIN`. Do not start until an open tournament/day/event and unused staging-only email are named. Delivered confirmation and edit-link testing remains inbox blocked. |
| `manual:tournament-partner-pairing` | At `/clubs/tres-palapas/tournament-partner-board`, prepare Test A/B/C with `a@x.invalid`, `b@x.invalid`, `c@x.invalid`, the same doubles event, `NEEDS_PARTNER`, and consent. Only after each disposable registration has a securely delivered or prepared edit token, submit A→B and B→C pairing interest, then accept A→B from B's review flow. Routes: `POST /clubs/<club-slug>/tournament-registration/pairing-interest` and `POST /clubs/<club-slug>/tournament-registration/pairing-requests/<request-id>/accept`. | Token-scoped `GET /clubs/<club-slug>/tournament-registration/pairing-requests` must show A→B accepted, one team link, and B→C cancelled without exposing private contact fields. | Accepted pairing locks edits and has no truthful inverse. `RETAIN` the registrations, team link, and audit. Do not start without exact edit tokens and pairing-board keys; notification/inbox evidence remains blocked until staging redirect is approved. |

Return to `none` and attest the all-false write projection before Packet 2.

### Packet 2 — support queue finalization

Enable only `support-requests`. At the support queue, load the three exact IDs
created in Packet 1. Joe enters `test only`, confirms Yes, and the witness
observes both the action and resulting status. Support and privacy fixtures may
be dismissed; the no-op data correction must be resolved so its full review
contract can be exercised without changing source data.

| Ledger row | Route and expected result | Final evidence |
|---|---|---|
| `manual:support-intake` | `PATCH /admin/clubs/<club-id>/support-requests/<request-id>`; reread the list and require the same request to be `dismissed`. | `RETAIN`; dismissal is a finalizer, not an inverse. |
| `manual:data-corrections` | The same PATCH route for the correction ID; set status `resolved`, reread exactly one deduplicated row, and verify the before/after source projections remain unchanged because no source ID or mutation was requested. | `RETAIN`; resolution is a forward finalizer, not an inverse. Preserve request and audit IDs plus the unchanged-source evidence. |
| `manual:profile-privacy` | The same PATCH route for the privacy ID; require status `dismissed`, identity/fulfillment `pending`, and no resolution action. | `RETAIN`; never describe this as fulfilled or identity verified. |

Return to `none` and attest the all-false write projection before Packet 3.

### Packet 3 — communications and recap

Use only the `communications` wave for every row in this packet; there is no
separate `weekly-recap` wave. The wave contains the guarded Weekly Recap and
Player Updates routes. Keep Fly in `dry_run`; do not turn on real customer
delivery. Return to `none` before any other domain wave.

| Ledger row | Joe-ready action | Authoritative readback | Recovery, retention, or blocker |
|---|---|---|---|
| `manual:weekly-recap` | At `/admin/weekly-recap`, reload the prepared draft for week `2099-01-05` and capture its current row version. Generate/preview, change one disposable spotlight or `staging only` field, save, then publish with `POST /admin/clubs/<club-id>/weekly-recap/recaps/2099-01-05/publish`, action `publish`, using the fresh version. | Admin GET for the same recap and public `GET /clubs/<club-slug>/weekly-recaps/2099-01-05` must show the published content. A second stale browser must be refused rather than overwrite the fresh row. | Truthful inverse: repeat the publish route with action `unpublish` and the fresh row version; require draft state and no publisher. Generated/edit history is retained. |
| `manual:verified-updates` | At `/admin/player-updates/verified-requests`, review the exact Packet 1 request. Route: `PATCH /admin/clubs/<club-id>/verified-updates/requests/<subscription-id>`. For this disposable request, choose Reject and confirm Yes. | The status endpoint and admin workspace must show the exact request rejected, with operator attribution. | `RETAIN`; rejection is final. Full approve→email→unsubscribe acceptance is blocked until a reviewed staging-redirect delivery path and Joe-controlled inbox exist. |
| `manual:player-updates` | At `/admin/player-updates`, select exactly one prepared pending outbox row and Send. Route: `POST /admin/clubs/<club-id>/player-updates/outbox/send`, with its exact ID, current version, and operation key. | `GET /admin/clubs/<club-id>/player-updates/workspace` must show that row sent or intentionally skipped, delivery mode `dry_run`, provider result, and one attempt. | A sent/skipped row has no inverse: `RETAIN`. This proves dry-run API/audit behavior only, not inbox delivery. A full delivered-link row remains blocked. |
| `manual:subscription-outbox` | Queue one digest for an isolated active player over a collision-free one-day range, `only_matches=false`, with one UUID. Route: `POST /admin/clubs/<club-id>/player-updates/digests/queue`. Retry the exact operation once. | Require `saved=1`, `queued=1`; the retry is idempotent. Workspace must show exactly one pending outbox row. | True cleanup: `POST /admin/clubs/<club-id>/player-updates/outbox/delete` with the fresh row ID/version; require `deleted=1`, `stale=0`, then zero matching workspace rows. Digest operation/audit rows remain. Error/sending retry paths need a separately prepared fixture. |
| `manual:email-preferences` | Only with a secret token from an isolated active verified subscription, open `/email-preferences` and unsubscribe scope `player_updates`. Route: `POST /email-preferences/unsubscribe`. Retry once. | Token-scoped `GET /email-preferences` must show the same subscription, unsubscribed, with the relevant preference disabled; retry must be idempotent. Queue the same future range and prove that subscription is skipped. | No resubscribe inverse exists: `RETAIN`. A subscription ID alone is not acceptable. This row remains blocked until Joe can use a delivered staging-only link/inbox. |

Return to `none` after each named wave and before Packet 4.

Staging currently has no prepared active subscription or outbox row for this
packet. `manual:player-updates`, `manual:subscription-outbox`, and
`manual:email-preferences` remain blocked until that isolated fixture, its
versions, and its delivery/recovery owner are recorded. A `dry_run` send proves
API and audit behavior only; the Player Updates page cannot receive a full-row
pass until a `staging_redirect` message, content, preference link, provider
evidence, error/retry path, and future-queue skip are inspected as specified in
`docs/communications_parity_runbook.md`.

### Packet 4 — league configuration and awards

These are separate irreversible-risk domains: use `league-manager`, return to
`none`, then use `league-awards`.

| Ledger row | Joe-ready action | Authoritative readback | Recovery or retention |
|---|---|---|---|
| `manual:league-manager` | At `/admin/league-manager`, use only the exact disposable league named in the fixture ledger. Capture its full configuration and version, then change one short reversible field from its captured value to `smoke` through `PATCH /admin/clubs/tres_palapas/league-manager/leagues/<league-name>`. | GET the exact league route and require the new value, stable identity, current version, and no selection clearing. Also exercise the required draft save/preview and lifecycle-lock cases against the prepared fixture; a description edit alone is not full-row evidence. | PATCH the fresh record back to the captured value and reread the complete configuration. If fixture preparation instead creates a draft, it has no delete and must be explicitly `RETAIN`ed; Joe must not improvise that choice during the session. |
| `manual:league-awards` | At `/admin/league-manager/awards`, use the exact prepared ended league with at least three eligible disposable players. Exercise freeze, preview, one override reason `test only`, mint, and archive. Terminal route: `POST /admin/clubs/tres_palapas/league-manager/leagues/<league-name>/awards/archive`. | GET the exact league awards route and require the archived state, minted badge evidence, and audit attribution. Confirm all four top-performer badge definitions are available before starting. | Freeze, mint, and archive are finalizers. `RETAIN` the archived league, badge, and audit; do not claim an inverse. |

Return to `none` and attest the all-false write projection before Packet 5.

The readable `Summer Social` and `Spring League` rows are not automatically
disposable. This packet remains blocked until the fixture ledger names the exact
league, baseline export, eligible roster, versions, and retention owner.

### Packet 5 — match and player operations

Enable only `match-player`. Prepare four clearly disposable players and capture
their ratings plus every edited match/player field before the first write.

| Ledger row | Current action or preserved future protocol | Authoritative readback | Recovery / blocker |
|---|---|---|---|
| `manual:score-entry` | `Blocked` for this candidate. Preserve the future one-row doubles protocol: submit `11–7`, tag `test`, through `POST /admin/clubs/tres_palapas/matches/batch`. | Future evidence requires the returned match ID exactly once plus expected rating deltas. | Do not submit while bulk exclusion is dormant; its required exact-ID exclusion and replay recovery are unavailable. |
| `manual:match-uploader` | `Blocked` for match-producing acceptance. Joe may inspect automatic loading and run round-robin preview in Session 1, but must not submit doubles or direct singles. | Future evidence requires every returned ID exactly once, committed-write attribution, and dry-run/disabled email handoff. | Direct singles also requires its atomic writer and dedicated gate. All uploads that require exact-ID exclusion wait for destructive recovery. |
| `manual:match-log` | Joe-ready only for one prepared, independently reversible ordinary edit and one legitimate duplicate no-issue resolution. Change a captured non-rating field from its exact prestate to `smoke`, note `test only`, through `PATCH /admin/clubs/tres_palapas/match-log/edits`. | Reload that exact match ID and require the new value, current version, and audit attribution. For duplicate no-issue, require the reviewed group to disappear from active cleanup candidates with its audit. | PATCH the fresh ordinary edit back to its captured value and reread it. `RETAIN` a truthful duplicate no-issue resolution with owner/audit. Duplicate cleanup and bulk exclusion remain `Blocked`; do not attempt them. |
| `manual:player-editor` | At `/admin/players`, toggle one disposable `Test Player` from active to inactive through `PATCH /admin/clubs/tres_palapas/players/editor/players/<player-id>`. | Reload the exact player detail and require inactive state without losing the selection. | PATCH the fresh record back to active and reread it. Merge acceptance is a separate destructive case requiring two disposable players plus preview/execute and a prepared pre-replay compensation; this toggle alone cannot pass merge coverage. |

Return to `none`, confirm replay completion and captured ratings restored, then
attest the all-false write projection.

The fixture audit identified synthetic player IDs `990001` through `990008`, but
their current ratings, singles baselines, versions, and ownership still must be
reread and recorded immediately before this packet. The basic Player Editor
toggle does not satisfy merge coverage; merge preview/execute/compensation needs
two separately prepared disposable players or remains `Blocked`.

#### Future direct Match Uploader singles protocol — do not run now

Keep this protocol for the later candidate in which the direct writer is one
atomic operation, the singles replay migration is formally accepted in staging,
`JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES` is reviewed for an isolated
wave, and destructive exclusion has atomic idempotent recovery:

1. Capture four disposable-player singles baselines.
2. Submit one rated `11–7` row and one unrated `11–7` row.
3. Require exact replay-managed IDs; require the rated aggregate change and zero
   aggregate/counter change for the unrated row.
4. Exclude only those returned IDs, run one full Replay, and require the exact
   captured baselines plus durable succeeded recovery evidence.
5. Return to `none` and attest both dormant gates disabled. A witness observes
   the named browser actions and visible results.

Tournament Operations official singles publishing is a separate automated-ready
CAS path. It does not clear the direct-uploader blocker, and its manual recovery
protocol also waits for destructive exclusion.

### Packet 6 — League Live future protocol; do not run now

This packet deliberately crosses three waves, always through `none`.

1. Enable `league-live-domain`. At `/admin/league-manager/live`, create one
   session with the exact prepared league, Week 1, four disposable players, one
   court, one round, and score `11–7`. Confirm that the player and round-robin
   options auto-load, then run the round-robin preview before saving. Route:
   `POST /admin/clubs/tres_palapas/league-manager/live-sessions`.
2. Return to `none`; enable `league-live-submit`. Publish round 1 through
   `POST /admin/clubs/tres_palapas/league-manager/live-sessions/<session-id>/rounds/1/submit`.
   For `manual:league-live`, the witness observes the publish and visible
   result. Session GET and Match Log must show the exact session, returned match
   contexts, and one committed round.
3. Return to `none`; enable `match-player`, exclude only those returned match
   contexts, run Replay, and require restored ratings. Return to `none`.
4. Re-enable `league-live-submit`, compensate round 1 with reason `test only`
   and the prepared recovery reference through
   `POST /admin/clubs/tres_palapas/league-manager/live-sessions/<session-id>/rounds/1/compensate`.
   Require compensated state and audit, then return to `none`.

Do not start until the league, roster, session version, operation key, baseline
ratings, and expected contexts are named. Exclusion/replay plus compensation is
the recovery chain; neither step alone is an inverse.

Both League Live waves intentionally enable only
`JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW` for the exact round-robin preview
route. Full `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER` stays off; the waves do not
authorize player creation, singles, batch upload, or other Match Uploader
writes. Verify preview success in both waves and fail closed if any non-preview
uploader route is available.

The complete League Manager page contract also requires lifecycle/settings
locks, court movement, guest/substitution, rating readback, export, and
response-loss/partial-recovery cases. This representative publish/compensation
chain does not by itself pass that page row.

### Packet 7 — ladder, Moneyball, and JUPR Live future protocols; do not run now

Run each row in its own named wave and return to `none` between rows. When a row
creates rated matches, separately enter `match-player` only for exact-ID
exclusion and replay, then return to `none`.

Every row in this packet is `Blocked` for the current candidate because that
cleanup transition depends on dormant bulk exclusion. Preserve the steps below
for the later gate-enabled candidate.

| Ledger row and wave | Joe-ready action and expected readback | Recovery or retention |
|---|---|---|
| `manual:challenge-ladder` — `challenge-ladder` | At `/admin/challenge-ladder`, use four prepared ladder players, persist the result preview, and complete the challenge through `POST /admin/clubs/tres_palapas/challenge-ladder/challenges/<challenge-id>/result` with the current preview fingerprint. Dashboard and the context-filtered Match Log recovery link must show the completed challenge, exact returned contexts, and new ranks. | Under `match-player`, exclude the returned matches and replay. Under `challenge-ladder`, call `POST /admin/clubs/tres_palapas/challenge-ladder/roster/replace-tier/preview`, then restore the complete captured tier/rank roster through `POST /admin/clubs/tres_palapas/challenge-ladder/roster/replace-tier` with the fresh preview/state fingerprint. `RETAIN` the completed challenge and audits. |
| `manual:moneyball` — `moneyball` | At `/admin/moneyball`, use exactly eight disposable players, preview settlement, and submit non-tied `11–7` results through `POST /admin/clubs/tres_palapas/moneyball/submit`. GET the exact operation key and Match Log; require the expected committed matches and settlement. | Exclude returned match IDs under `match-player`, replay captured ratings, and `RETAIN` operation/audit evidence. |
| `manual:jupr-live` — `jupr-live` | At `/admin/jupr-live`, create title `smoke` with four disposable players and round-robin play, PATCH scores, then publish. Routes: `POST /admin/clubs/tres_palapas/jupr-live/sessions`, `PATCH /admin/clubs/tres_palapas/jupr-live/sessions/<session-key>/scores`, and `POST /admin/clubs/tres_palapas/jupr-live/sessions/<session-key>/publish`. Admin session, public `/clubs/tres-palapas/live/<session-key>`, and Match Log must agree. | Exclude/replay returned match IDs under `match-player`. A published/completed session has no true inverse: `RETAIN` it and its audit. |

For Challenge Ladder, Moneyball, and JUPR Live, open the supplied recovery link
and require Match Log to be filtered by the exact `context_type` and
`context_id`, with no unrelated selected record left visible. Replay History is
deliberately global; its link and copy must not claim context filtering.
Challenge Ladder full-row evidence additionally requires the prepared
creation/eligibility/notices, clock, pass, roster/tier/override, forfeit, stale,
and response-loss cases. Public JUPR Live acceptance additionally requires Quick
Round Robin, League/Ladder, and Club Social mobile resume, substitution, export,
wrong-token, and persistence cases.

### Packet 8 — public live and social moderation

Enable `public-live` for the first row, return to `none`, then enable
`admin-tools` for the second.

| Ledger row | Joe-ready action | Authoritative readback | Recovery or retention |
|---|---|---|---|
| `manual:public-live` | At `/clubs/tres-palapas/live`, create session `smoke`, host `Test`, four disposable players; score, advance, and complete it. Routes: `POST /clubs/tres-palapas/live-sessions`, score PATCHes, and `POST /clubs/tres-palapas/live-sessions/<session-key>/complete`. | GET the exact live session and require completed state, persisted scores, and stable resume/share behavior on a narrow viewport. | Completion is final; `RETAIN` session and operation. Full-row evidence also needs prepared Quick RR, two-round league/ladder, Club Social, mobile resume, export, stale-state, and wrong-token cases. Do not pass the full row from one session. |
| `manual:social-moderation` | At `/admin/tools`, reject the exact disposable Club Social submission, reason `test only`, through `POST /admin/clubs/tres_palapas/tools/social-submissions/<event-id>/moderate`. | `GET /admin/clubs/tres_palapas/tools/social-submissions` must show the same submission rejected with moderator and timestamp. | No reset exists: `RETAIN` the rejected submission. This row depends on a prepared Club Social submission from the public-live coverage. |

Return to `none` and attest the all-false write projection before Packet 9.

Social moderation is only one Admin Tools case. Full page-row evidence also
requires the read-only rating reports/CSV and activity/role views plus separately
prepared badge queue/recompute/lifecycle, tournament-backfill preview/apply, and
Match Canonical Audit dry-run/recovery cases. Use only their named
`admin-tools`, `badge-diagnostics`, or `match-player` wave, through `none`, and
never change a live badge definition or production data.

### Packet 9 — tournament status and draw operations

Use `tournament-mutations` for the first row, return to `none`, then use
`tournament-operations` for the second. The known read fixture
`93000000-0000-4000-8000-000000000002` is not automatically a disposable write
fixture; verify ownership and prestate before using it.

| Ledger row | Joe-ready action | Authoritative readback | Recovery or remaining scope |
|---|---|---|---|
| `manual:tournament-admin` | At `/admin/tournaments/status`, archive one isolated DRAFT tournament through `PATCH /admin/clubs/tres_palapas/tournaments/admin/tournaments/<tournament-id>/status-action`, using its current version. | Tournament detail must show the same identity and `ARCHIVED` status. | Truthful inverse: invoke the same route with `unarchive` and the fresh version; require `DRAFT`. |
| `manual:tournament-operations` | At `/admin/tournaments/ops/draws`, capture one disposable game's score and versions; change `11–7` to `11–8` through `PATCH /admin/clubs/tres_palapas/tournaments/admin/tournaments/<tournament-id>/games/<game-id>/score`. | Ops GET must show `11–8`, the expected winner/progression, fresh versions, and audit attribution. | PATCH the fresh game back to `11–7` and reread the original progression. One score edit cannot pass the full row: singles/doubles setup, import, generation, podium, awards, and publish still need their prepared cases. Official publish uses `tournament-official-publish`; email handoff uses `tournament-email-handoff`, each through `none`. |

The Tournament Admin page row also requires a prepared `tournament-setup` wave
for guided settings/draft/impact-review/publish, a
`tournament-registration` wave for single/bulk/selection edits and
stale/imported locks, and `tournament-operations` for registration-to-draw
import. The Session 1 no-save builder check is not publish evidence. These cases
remain blocked until a disposable DRAFT tournament, registrations, selections,
draw, versions, and recovery/retention owner are recorded.

Official publish must not run until the final migration and atomic publish
contract preserve `singles_replay_managed` on every newly published singles
match. After publish, retain the tournament operation but recover its rated
effect separately: record the exact returned match IDs, verify their
context-filtered Match Log rows, exclude only those IDs under `match-player`,
run full Replay, and require exact captured player baselines. Retention of the
tournament audit is not rating recovery. The publisher is automated-ready
through its atomic CAS RPC, but this manual protocol remains `Blocked` while
destructive exclusion is dormant.

The automated `match-rating-writes` workflow is not one of the 25 witnessed
manual rows. Run it only in the exact sequence described under Remaining
automated sequence; the workflow owns and restores its disposable tournament
score in `finally`.

### Packet 10 — Tournament Live non-score operations

Enable only `tournament-live`. For `manual:tournament-live-non-score`, Joe opens
`/admin/tournament-live` and uses a newly prepared empty draw with a supported
team count of 4, 5, 6, 7, or 8. Generate round robin through:

`POST /admin/clubs/tres_palapas/tournament-live/tournaments/<tournament-id>/draws/<draw-id>/commands`

The witness observes the Yes confirmation and generated schedule. The command
response must report success and the expected game count; the exact Tournament
Live snapshot must show matching games, progression, operation key, and audit.
Schedule generation has no structural delete, so `RETAIN` the disposable draw
and tournament.

This single command does not pass the full row. Playoff generation, podium,
awards, and terminal publish/recovery require prepared fixtures. Terminal
official publish must run separately under
`tournament-live-official-publish`, through `none`, and its operation/audit is
retained. Its rated matches still require the exact Match Log/exclude/full
Replay restoration described in Packet 9. Do not use an existing draw that
already contains games merely because it is readable.

This packet remains blocked until an empty disposable draw exists and the
complete round-robin scoring, playoff, podium, awards, publish,
ambiguous-response reconciliation, mobile, and restoration cases are prepared.

Return to `none` and attest the all-false write projection.

### Unresolved fixture and owner gates

These are real blockers, not instructions for Joe to improvise during the
session.

| Gate | Affected rows | Required resolution before execution |
|---|---|---|
| Staging-only delivered links and inbox | `manual:email-preferences`, full `manual:verified-updates`, full `manual:player-updates`, `manual:tournament-registration`, `manual:tournament-partner-pairing` | The staging Fly workflow currently enforces `JUPR_EMAIL_MODE=dry_run` and disables automatic player-update email. Prepare and review a staging-redirect-only delivery path plus a Joe-controlled inbox. Never enable real customer delivery. |
| Identity and fulfillment authority | `manual:profile-privacy` | An authorized owner must verify identity and approve the requested fulfillment. Until then, test intake/dismissal only and keep fulfillment pending. |
| Password recovery and role accounts | `reset_password`, full `admin_login` | Prepare an approved staging recovery inbox plus the required allowed, denied, wrong-club, expired, and invalid-session account cases. The automated no-email session proves only one capability-checked login/sign-out path. |
| Communications fixture | `manual:player-updates`, `manual:subscription-outbox`, `manual:email-preferences` | No isolated active subscription or pending outbox row was found. Prepare one with captured versions, redirect-only recipient, provider/retry plan, and cleanup/retention owner before execution. |
| Exact isolated fixtures and captured prestate | League Manager/Awards, all match/player rows, League Live, Challenge Ladder, Moneyball, JUPR Live, public live/social, and every tournament row | Record exact IDs, edit tokens, current versions, operation keys, before-state values, baseline ratings, expected match contexts, and named recovery owner. Readable production-like fixtures are not disposable by default. |
| Known read fixtures are not write fixtures | Leagues and tournaments | `Summer Social`, `Spring League`, tournaments `93000000-0000-4000-8000-000000000001` / `...0002`, draw `94000000-0000-4000-8000-000000000001`, and game `96000000-0000-4000-8000-000000000001` are readable staging records only. Do not mutate one until ownership, exact prestate, recovery, and retention are approved. No empty Tournament Live draw is currently prepared. |
| Synthetic player baselines | Match/player, League Live, ladder, Moneyball, and JUPR Live | Player IDs `990001`–`990008` and the existing synthetic ladder roster were identified, but current ratings, singles baselines, ranks, flags, versions, and cross-packet ownership must be reread immediately before use. |
| Dormant direct-singles and destructive-recovery gates | `manual:score-entry`, `manual:match-uploader`, destructive `manual:match-log`, `manual:league-live`, `manual:challenge-ladder`, `manual:moneyball`, `manual:jupr-live`, and official-publish recovery | Keep both flags `0`. Prepare atomic writer/recovery implementation, idempotent retry and unknown-outcome tests, migration acceptance, exact fixtures, and a later reviewed wave before asking Joe to run these protocols. |
| Full multi-case row contracts | `manual:league-manager`, `manual:match-uploader`, `manual:match-log`, `manual:player-editor`, `manual:challenge-ladder`, `manual:public-live`, `manual:tournament-admin`, `manual:tournament-operations`, `manual:tournament-live-non-score`, and Admin Tools | Prepare the remaining lifecycle, merge, duplicate/social, live-session variants, setup/registration/import/podium/award/publish, playoff, stale/response-loss, and recovery cases. A successful representative mutation is useful evidence but is not a full-row pass. |

The next active witnessed order is: Packet 1 intake/registration, Packet 2 queue
finalization, Packet 3 communications, Packet 4, only the independently
reversible Match Log/Player Editor parts of Packet 5, Packet 8, the reversible
Tournament Admin/Ops score-edit parts of Packet 9, then any independently
recoverable Packet 10 case. Skip Packets 6–7 and every named blocked subflow.
After the last completed packet, proceed directly to Session 3; do not leave an
active wave overnight. The preserved future protocols become a later session
only after both blocker rows are cleared.

## Separate release finish lines

This 45-row book is the full Next/FastAPI administrative-replacement and
Streamlit-retirement gate. It must not be used to make deferred billing,
self-serve onboarding, or a second-club pilot block the earlier Tres Palapas
public launch.

The public launch has its own checklist: accept the public/read candidate,
complete legal, domain, minimum monitoring, approved-inbox email/recovery, and
rollback prerequisites, and retain Streamlit as the admin fallback. Remaining
guarded admin write rows may stay `Partial` for that launch, but they cannot be
called `Done`, and Streamlit cannot be retired, until this complete book passes.

## Session 3 — restoration and final proof

1. Reconcile every packet resource and every uncertain operation.
2. Confirm no unexpected delivery/provider event and no unresolved replay,
   outbox, or compensation row.
3. Deploy the exact candidate to Fly with `write_wave=none`.
4. Verify exact SHA/image/project identity, `business_data_write_wave_active=false`,
   all 32 controlled write flags false, public live writes false, and email
   `dry_run`.
5. Recheck Supabase health, migration head, Auth/API/PostgreSQL logs, and
   security/performance advisors. Record handled compatibility probes accurately.
6. Run canonical `Staging Smoke`; require all 56 public-read and five guided
   Tournament Setup checks with zero skips, failures, or flakes.
7. If any row remains blocked or incomplete, leave it `Pending` or `Blocked` and
   stop after recording the safe restored state. Only after all 45 page rows and
   every supporting ledger are complete may the exact-candidate book be merged
   as the sole book-only change and `Parity Final Evidence` mode
   `complete-book` run against the original immutable Vercel candidate and final
   none-wave Fly image.
