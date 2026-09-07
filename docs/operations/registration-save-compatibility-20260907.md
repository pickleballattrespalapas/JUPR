# Registration save compatibility repair

The production Baja Classic 2026 registration reported in the incident still
has the legacy `pending` status. The June 2026 migration retired pending admin
confirmation, but the current edit payload builders still round-tripped that
value into a validator that accepts only confirmed, waitlist and cancelled.
The admin select had no pending option and its generic confirmation dialog
hid the server rejection.

This repair interprets stored pending values under the existing auto-confirm
policy. Admin reads and the editor agree on confirmed; an already-open editor
may echo pending only when the stored row is still pending. Explicit attempts
to assign pending to another registration remain invalid. An authorized admin
save persists confirmed through the existing versioned and audited path.

Public edits use the compatible status for validation. Their atomic RPC patch
still excludes status, payment and identity linkage, so player edits preserve
those server-managed values. No schema migration or bulk data cleanup is needed.
Admin validation and conflict errors remain visible inside the confirmation
dialog; internal server errors stay generic and failed saves preserve drafts.

Verification: the new regressions fail on the baseline for public pending saves
and the admin status round-trip. The repaired code passes 133 focused Python
tests plus the registration editor component tests, including unchanged
payment/status on public edits, guarded admin saves, rejected pending assignments,
version conflicts, draft retention and error sanitization.

Production diagnosis was read-only. No registration data or emails were changed.
This is a separate release from the completed email signing-key repair and
requires specific production deployment authorization under AGENTS.md.
