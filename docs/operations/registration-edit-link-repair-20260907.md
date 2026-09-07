# Registration edit-link repair — 7 September 2026

## Confirmed production blocker

The production API at `4415cab4bb50eef0b87c80efcc0137caab1bb7ff`
reported `registration_edit_secret_configured=false`,
`registration_confirmation_secret_configured=false`, and configured SMTP in
live mode. The public edit-link service calls `_stable_edit_secret()` before
registration lookup or delivery. Its missing-secret exception becomes HTTP 503
and the generic error shown in Joe's screenshot. SMTP configuration alone did
not establish that registration edit links were ready.

The public page also rendered both the wizard's edit form and a separate
always-visible recovery card. The prepared UI uses the wizard's form only,
including after registration closes. Existing `#manage-registration` links
open that same form.

## Prepared production operation

Production mutation remains subject to the exact approval required by
`AGENTS.md`. This preparation does not change production or send an email.

Run `scripts/initialize_registration_edit_secret.py` in the existing protected
production deployment environment, serialized with the production deploy job.
It requires the current `FLY_API_TOKEN`, SSH-capable `FLY_SSH_TOKEN` when needed,
`SUPABASE_URL`, and `SUPABASE_SERVICE_ROLE_KEY`, plus `JUPR_ENV=production` and
`FLY_APP_NAME=juprleagues-api`. Reuse the protected environment credentials;
do not ask the user to paste credentials into chat or put them in command text.

The default invocation only checks readiness and fails if setup is missing:

```bash
python scripts/initialize_registration_edit_secret.py
```

The exact approved configuration repair is:

```bash
python scripts/initialize_registration_edit_secret.py --initialize-missing
```

It performs these steps:

1. Check the exact production app identity and reject pending/partial/unknown
   Fly secret deployments. Inspect only names and safe readiness evidence.
2. Refuse to replace any existing edit signing key, even an invalid one.
3. Preserve the effective confirmation signer. Production currently falls back
   to a server-only service-role-derived signer. Adding an edit secret without
   pinning that existing value would invalidate confirmation URLs already sent.
   Compare a one-way fingerprint on the running machine with the protected
   credential's derived value, and stop on any mismatch. If an explicit
   confirmation key already exists, preserve it unchanged.
4. Generate 48 cryptographically random bytes for the missing edit secret.
   Import it and, only if absent, the preserved confirmation signing value in
   one operation. Values travel in process memory and stdin, never argv, files,
   logs, artifacts, or source. This uses Fly's documented
   [secret import command](https://fly.io/docs/flyctl/secrets-import/).
5. Verify fully deployed secret status, edit signing readiness, and an unchanged
   confirmation fingerprint. Re-running after success makes no mutation.
   An ambiguous failure must be investigated; do not delete or regenerate keys.

The operation updates secrets on the existing API image. It does not change
SMTP credentials, email modes, feature flags, registration records or schema.
The confirmation value remains the existing signer; a later independent key
rotation requires a separately planned compatibility transition.

## Release and verification

- First verify the targeted feature on staging with email still in dry-run.
  No production operation may be called by staging automation.
- Promote only the reviewed repair changes onto the current production branch;
  staging contains unrelated changes and must not be merged wholesale.
- After approval, run the configuration operation through the protected
  production environment. Run the read-only default check before subsequent
  production releases so missing signing prerequisites stop the release.
- Deploy the consolidated web form using the established production release
  procedure. Keep exact baseline and candidate identity checks in place.
- Verify both signing readiness booleans in `/health`, unchanged live SMTP
  posture, and one recovery form with working confirmation-page anchors.
- Once Joe explicitly approves a test recipient, request one edit email and
  verify provider acceptance and receipt. No inbox delivery has been verified
  during preparation. Do not use a registration mutation or bulk send as a test.

## Local checks

Focused tests cover absent/invalid/existing keys, exact production targeting,
confirmation-token continuity, concurrent configuration changes, pending
secrets, stdin-only import, safe failures, and a non-mutating rerun. Existing
public edit-link API/service and token tests are included. The legacy Streamlit
UI import test is excluded because this API test environment lacks Streamlit.
The React test renders the full registration page and verifies one recovery
form, old hash links, and recovery while new registration is closed.
