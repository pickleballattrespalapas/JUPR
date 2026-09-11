# Partner invitation production release — 2026-09-11

Joe authorized production deployment with “ok, push to production.”
The accepted staging feature is PR #1399, staging SHA
`5e2ce521aaf415c80741034afe30e1b27c4d8553`.

## Completed

- PR #1401 promoted the feature onto production `880363b7`, preserving current
  registration navigation, SMTP Message-ID/Date handling, live email/write
  settings, and the production migration inventory. Merge: `ff3bc004`.
- 194 focused Python tests, all production component tests, TypeScript, the
  Next production build, migration guards, and release PR CI passed.
- The exact `20260909203736_partner_email_invitations.sql` migration was applied
  to production project `dnoockbwfenunhcibwfn`. Its connector ledger version is
  `20260911035442`, name `partner_email_invitations`. Do not apply it again.
- Production migration verification passed: 120 ledger rows, no pending required
  names, reviewed head `20261022000000`, contract fingerprint
  `d9ca8fa3272b09553b8418beb6194742a6abd462d46591f00d0bd035f27aff37`.
- New tables have RLS and no anon/authenticated access. New functions are
  invoker-only with an empty search path and service-role-only execution.
  Security Advisor added only the two intentional private-table INFO findings;
  the existing Auth leaked-password warning was unchanged.

## Initial deployment hold

The first API release candidate, `ecf20444`, stopped before runtime mutation
because the deployment's Supabase Database Read token returned HTTP 401.
[Run 34560310024](https://github.com/pickleballattrespalapas/JUPR/actions/runs/34560310024)
captured the prior API image (`39645c15`) for rollback.
PR #1402 restored the previous partner-board page while the credential was renewed.
No test invitations or customer emails were sent.

## Resumed production release

Joe confirmed the replacement credential was saved. The new API candidate is
`e26ff6376e7f15e157ba3ec87875c1f2183e2718`.
[Run 34563209898](https://github.com/pickleballattrespalapas/JUPR/actions/runs/34563209898)
passed production migration attestation, SMTP authentication without sending,
and registration signing checks with the renewed credential.

This release deploys the prepared API first, while retaining the existing working
partner page. The page activation restores these files to their accepted
`ecf20444` content only after the API run succeeds:

- `apps/web/app/clubs/[clubSlug]/tournament-partner-board/page.tsx`
- `tests/test_next_tournament_partner_board_source.py`
- `tests/test_api_contract_public_workspace_followup.py`

The website activation changes only those three files and this release record;
its later web SHA therefore differs from the API SHA without changing API code.
Production migration, email and enabled write settings are preserved.

Acceptance requires a successful exact-candidate API run, the completed Vercel
website deployment, the visible private request form, and rejection of invalid
signed links. The pairing lifecycle was tested on isolated staging. Production
verification must not send unsolicited test requests to players.

For a future credential replacement, renew the existing protected GitHub
`production` environment secret `SUPABASE_PROD_DATABASE_READ_TOKEN` with a
short-lived `sbp_fc` credential scoped to production project
`dnoockbwfenunhcibwfn`, Database: Read. Keep values out of source, logs, chat and
artifacts. Use a fresh single-parent release trigger after production HEAD moves;
do not bypass provenance checks or reapply the completed schema migration.

## Direct sending update

Joe reported that sender email verification plus a second website confirmation
made the original Send request action unnecessarily difficult. The follow-up
sends the player their message immediately. It retains recipient acceptance,
private email links, rate limits, retry identity and automatic pairing.

The additive `20260911135628_direct_partner_request_delivery.sql` migration
supports direct delivery without marking a typed email as verified. Existing
verification links still work; applying it sends no stored requests. The real
rollback-only staging SQL fixture passed for registered pairing, guest
reservation/completion, name matching, stale state, duplicate actions and grants.
104 staging and 134 production focused Python tests passed, along with the full
component suite, TypeScript and the Next build. Advisor categories are unchanged.

Release the API and migration first while the current form still supports both
API responses. After the exact API deployment succeeds, activate the updated
form wording and its component test. Keep production email live and writes
enabled. Verification must not send unsolicited player test messages.

Production migration ledger version: `20260911140741`. It is already applied;
do not reapply it. Both updated functions remain service-only invokers, and
production Security Advisor findings are unchanged.

The API release candidate is `02f98dc497cdca138ce17f9f7b58d555c173027e`.
Activate the two form files from reviewed staging PR #1404 only after its exact
production API deployment succeeds. Website and API SHAs intentionally differ;
this activation changes no API code, database schema or runtime settings.
