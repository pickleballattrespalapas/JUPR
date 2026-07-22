# Final staging bounded-write closeout — ancestor A / exact-candidate B

This record closes the two narrow staging exercises approved after the
2026-07-20/21 smoke:

- **A:** create one disposable data-correction intake and submit one exact retry
  to exercise deduplication;
- **B:** dismiss one disposable request through the authenticated support-request
  queue while proving the revised automatic-load, optional-note, and Yes/No-dialog
  behavior.

No production system, live email, player, match, rating, league, or tournament
business data was changed. Email remained `dry_run` throughout.

## Truth boundary

A and B were not completed against one candidate or one request:

- A ran on direct ancestor `6c27f5de5c04b1d565ba051efb322fff2804ff10`.
- B ran on final application candidate
  `eab384545c493f145af383c8e26d8bf97686ab21` using a separately seeded,
  disposable staging queue fixture.

The direct Supabase insert used to prepare B is fixture setup, not proof that the
final candidate's public intake route works. A remains useful ancestor evidence;
B is exact-final-candidate support-queue evidence. They must not be combined into
a formal same-candidate parity `Pass`. The full parity ledgers and witness-bound
manual rows remain `Pending` in `docs/next_parity_manual_staging_book.md`.

## Final candidate and infrastructure

| Field | Recorded value |
|---|---|
| Canonical branch/ref | `staging` |
| Final application candidate SHA | `eab384545c493f145af383c8e26d8bf97686ab21` |
| Runtime PR | `#1023` (`Streamline support request review`) |
| Vercel deployment ID | `dpl_FTmDVFSdftMmAj4sMwAHrvQpVYLE` |
| Vercel immutable origin | `https://jupr-bifdisfdg-pickleballattrespalapas1.vercel.app` |
| Vercel staging alias | `https://jupr-git-staging-pickleballattrespalapas1.vercel.app` |
| Fly API | `https://juprleagues-api-staging.fly.dev` |
| Initial final-candidate `none` | Run `29955849970`; image `registry.fly.io/juprleagues-api-staging:deployment-01KY5RQNAW9QHTHRHNSW6NG8VA`; machine version `01KY5RS9HHMJ8C2AV7AG5EH5PZ` |
| Final-candidate support wave | Run `29956323772`; image `registry.fly.io/juprleagues-api-staging:deployment-01KY5S453MDYZZSKXKV9MZEYY8`; machine version `01KY5S55879ZZRXR0RK7TEFEKA` |
| Final resting `none` | Run `29957218074`; image `registry.fly.io/juprleagues-api-staging:deployment-01KY5SVX80SHEZMN7GX37NASRP`; machine version `01KY5SXBQDWYKHBGFTQATZSVW3` |
| Supabase project ref | `sijpxjxvdtrehmqvirfi` |
| Email mode | `dry_run` |
| Operator | Joe Baumann |
| Witness | Pending — no formal witness-bound manual parity row is claimed |

### Recorded final-candidate preflight

- Vercel deployment `dpl_FTmDVFSdftMmAj4sMwAHrvQpVYLE` was `READY`, sourced
  from `staging`, and attested SHA
  `eab384545c493f145af383c8e26d8bf97686ab21` at the immutable origin above.
- Initial Fly run `29955849970` deployed the same SHA with
  `write_wave=none`, business-data writes false, every controlled flag false,
  and email `dry_run`.
- The remote targeted read confirmed the support-intake guardrails and RLS posture:
  no `anon`/`authenticated` DML and the required service-role access. This is not a
  full migration inventory or full cross-surface security sign-off.

## A — ancestor-only data-correction intake and exact retry

The operator used the requested short, repeatable values:

| Form field | Value |
|---|---|
| Your name | `Test` |
| Your email | `test@x.invalid` |
| Player/match/tournament fields | blank |
| Short subject | `smoke` |
| What looks wrong? | `staging only` |
| What should staff change? | `nothing` |
| Evidence link | blank |
| Contact confirmation | checked |

### A evidence

| Field | Recorded evidence |
|---|---|
| Candidate | `6c27f5de5c04b1d565ba051efb322fff2804ff10` (direct ancestor only) |
| Fly run/image | Run `29952977995`; image `registry.fly.io/juprleagues-api-staging:deployment-01KY5PBDKQNYJ777ZE94JA33RE`; machine version `01KY5PD43W85H4D7G0TCTR0XTC` |
| First browser result | “Request received. Staff will review it before any data changes are made.” |
| Exact retry result | “This request was already received and remains in the staff review queue.” |
| Request ID | `req_0eae0a691fe94e72b88e` |
| Fingerprint | `414153a37567f07bc5a1be6aedf4ce00197bda52c3e230a856a869699229eab7` |
| Deduplication key | Same fingerprint with day suffix `:20260722` |
| Readback | Exactly one matching `public_support_requests` row |
| Raw JSON 2xx artifact | Not captured; do not promote this targeted browser evidence to formal parity Pass |
| Operator | Joe Baumann |
| Witness | Pending / not recorded for this targeted ancestor exercise |
| Separate restoration | Run `29953685135`; image `registry.fly.io/juprleagues-api-staging:deployment-01KY5PY423HPAHZXFRDAREPD0E`; machine version `01KY5PZE4YC3S79ZD7CW7REZSY`; restored `none` |

## B — exact-final-candidate support-queue dismissal

The earlier short request was already dismissed, so a fresh disposable staging
fixture was seeded directly for the final-candidate UX check:

| Fixture field | Value |
|---|---|
| Request ID | `req_staging_ux_20260722_2048` |
| Type/status | `data_correction` / `new` |
| Requester/email | `T` / `ux@x.invalid` |
| Subject/description | `ux` / `blank-note test` |
| Requested action | `dismiss` |
| Source | `staging_support_queue_fixture` |
| Initial review state | no reviewer; SQL `NULL` admin note |

### Browser action and acceptance

1. With only the `support-requests` wave active, the operator hard-refreshed
   `/admin/support-requests` and did **not** click **Refresh requests**.
2. The `ux` fixture automatically appeared. The operator selected the exact row,
   set status to `dismissed`, and left the optional admin note blank.
3. The operator clicked **Save request status**, reviewed the dialog, and selected
   **Yes, dismiss request**. No confirmation phrase was typed.
4. The UI reported success. Supabase then provided the authoritative row and audit
   readback below.

### B evidence

| Field | Recorded evidence |
|---|---|
| Candidate | `eab384545c493f145af383c8e26d8bf97686ab21` |
| Fly run/image | Run `29956323772`; image `registry.fly.io/juprleagues-api-staging:deployment-01KY5S453MDYZZSKXKV9MZEYY8`; machine version `01KY5S55879ZZRXR0RK7TEFEKA` |
| Active flags | Only support requests plus its required admin-write pilot; all unrelated controlled flags false |
| Admin actor | Joe Baumann, authenticated staging super-admin; login identity retained outside this public record |
| Request | `req_staging_ux_20260722_2048`; prestate `new` |
| Authoritative result | `dismissed`; `admin_note` SQL `NULL` |
| Review timestamps | reviewed `2026-07-22T20:55:01.736928+00:00`; updated `2026-07-22T20:55:01.781893+00:00` |
| Audit | Exactly one event, ID `6`; action `update_public_support_request_admin`; entity `public_support_request`; before `new`; after `dismissed` |
| Audit attribution | authenticated staging super-admin; source `next_admin_support_requests`; audit note SQL `NULL`; `flagged_for_review=true` |
| Retained-evidence owner | Joe Baumann; disposable dismissed row and audit event retained |
| Witness | Pending / no formal witness-bound parity row claimed |

## Final restoration and canonical smoke

1. Final Fly restore run `29957218074` completed successfully from the same
   application SHA. `/health` reported `write_wave=none`,
   `business_data_write_wave_active=false`, email `dry_run`, and every controlled
   write flag false. The resting image is
   `registry.fly.io/juprleagues-api-staging:deployment-01KY5SVX80SHEZMN7GX37NASRP`
   with machine version `01KY5SXBQDWYKHBGFTQATZSVW3`.
2. Canonical Staging Smoke run `29957623653` completed successfully against SHA
   `eab384545c493f145af383c8e26d8bf97686ab21`. It passed public checks, exact
   read-only Vercel/Fly deployment identity attestation, incomplete-evidence
   rejection checks, and all 56 strict browser tests.

Run links:

- <https://github.com/pickleballattrespalapas/JUPR/actions/runs/29957218074>
- <https://github.com/pickleballattrespalapas/JUPR/actions/runs/29957623653>

## Remaining boundaries

- This packet is targeted hardening evidence, not the completed 45-page parity
  acceptance book.
- Witness identity remains pending until a formal manual-only parity row is
  executed with a separate human witness.
- Legal copy, staging email-account/inbox acceptance, and production cutover remain
  separately deferred.
- The successful dismissal proves the normal row-plus-audit path. It does not prove
  transactionality if the audit insert itself fails; atomic row/audit handling is a
  separate hardening follow-up.

## Stop conditions for any later wave

Stop without further writes on wrong SHA/project/origin, a wave other than the
specifically approved one, more than one controlled wave enabled, unexpected
external email, response ambiguity, stale-state conflict, a second durable intake
row, missing reviewer/audit attribution, wrong-club visibility, or inability to
restore and attest `none`.
