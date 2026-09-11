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

## Deployment blocker and temporary page hold

The production API release trigger created candidate `ecf20444`.
[Workflow run 34560310024](https://github.com/pickleballattrespalapas/JUPR/actions/runs/34560310024)
captured the rollback identity, then stopped before any Fly runtime mutation.
The Supabase read-only migration attestation endpoint rejected the protected
`SUPABASE_PROD_DATABASE_READ_TOKEN` credential with HTTP 401.
The live API remains `39645c15`; SMTP and runtime activation steps were skipped.
No test invitations or customer emails were sent.

The production web deployed successfully ahead of the blocked API. Until the
credential is renewed, the partner-board page and its two source assertions are
restored to their exact pre-release `880363b7` content. Existing registration-link
partner contact remains available. The new API implementation, private response
page, registration handoff, and additive migration are retained for completion.

## Resume

1. The production owner must replace `SUPABASE_PROD_DATABASE_READ_TOKEN` in the
   GitHub `production` environment with a valid, short-lived `sbp_fc` credential
   scoped only to production project `dnoockbwfenunhcibwfn`, Database: Read.
   Keep credential values out of chat, source, logs, and artifacts.
2. Restore these three files from `ecf20444509eaad577191e1985fec75d2a407ec0`
   onto the current production branch in a reviewed PR:
   - `apps/web/app/clubs/[clubSlug]/tournament-partner-board/page.tsx`
   - `tests/test_next_tournament_partner_board_source.py`
   - `tests/test_api_contract_public_workspace_followup.py`
3. Recheck the current migration ledger and release tests, then merge. Create a
   fresh, single-parent production release-trigger commit using the existing
   workflow contract. An old run will reject a moved production HEAD; do not
   weaken or bypass that provenance check or the migration attestation gate.
4. Verify the exact live API/web identities, SMTP authentication, current enabled
   production write/email profiles, private form, and signed-link error handling.
   Use the previously tested staging lifecycle for pairing verification; do not
   send unsolicited production test requests to players.
