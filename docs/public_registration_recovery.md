# Public registration recovery

Removing the last event opens an explicit withdrawal confirmation. The same action is available when registration changes have closed, unless the registration has already entered a draw. Withdrawal removes this registration and releases partnerships, preserves other players' registrations and payment history, and never issues a refund.

The API verifies the existing edit-link signature and expiry. A service-role-only database function binds the action to its club, tournament, registration and email hash, locks the registration graph, checks every selection version, then invokes the existing cancellation transaction. Its private cancellation audit supports safe retries after a lost response. Imported draws, four-player captains and fulfilled extras continue to require organizer intervention.

The edit page lists its verified owner's active partner requests with explicit continuation and review/cancel links. A reserved-partner conflict remains enforced and now explains recovery. Continuation uses a full navigation to ensure the invitation fragment is consumed when the editor remounts. No request is cancelled automatically.

Partner Board contact consent is controlled alongside public visibility in the event dialog. Saving checks consent before submitting. The form never opts a player in automatically.

Support history records successful edit-page loads without exposing link tokens. A page load alone is labelled as no save recorded. A later rejection no longer disappears behind an earlier successful save, and a successful withdrawal is labelled Withdrawn. Browser activity remains diagnostic evidence, not identity proof.

## Validation and rollout

- Focused Python API, edit, invitation, withdrawal, cancellation and attempt-history tests.
- Web edit-usability, registration recovery, partner invitation, registration save and eligibility tests; TypeScript and Next build.
- `tests/sql/public_registration_withdrawal.sql` and `tests/sql/complete_registration_cancellation.sql` executed on staging. Both use disposable fixtures in rolled-back transactions.
- Migration `20261030224214_public_registration_withdrawal.sql` applied only to staging project `sijpxjxvdtrehmqvirfi`. Public/anonymous execution is revoked and the search path is fixed.

Production promotion must be separately authorized under AGENTS.md. Promote only this change, preserving the production-only tournament write guard; apply that guard to the new withdrawal route too. Do not merge unrelated staging changes. No player registrations or live email deliveries are part of the code rollout.
