# Finish the invitation email test

The new-account flow is ready in the application: verify email, choose a password,
review the invitation, and accept club administrator access. La Ribera's pending
invitation now uses the test inbox Joe approved in chat. No staging Auth account
exists for that inbox yet, and no email has been sent.

The approved test window ends **September 15, 2026 at 19:22:20 UTC**. Only the
approved mailbox can receive invitation verification mail during this window.
General staging email remains `JUPR_EMAIL_MODE=dry_run`.

## Remaining step: configure staging mail

The API currently reports `smtp_ready=false`. Its SMTP settings are not available
in the workspace, and the connected tools cannot manage Fly secrets. The mail
provider credentials must be added through Fly by someone with access.

Use only the Fly app **`juprleagues-api-staging`**. Enter the mail provider's
settings as app secrets; do not paste passwords into chat or commit them to this
repository. Use an SMTP credential authorized for this staging test.

| Setting | Value to enter |
| --- | --- |
| `SMTP_HOST` | SMTP server supplied by the mail provider |
| `SMTP_PORT` | Provider's STARTTLS port |
| `SMTP_USERNAME` | Provider's SMTP username |
| `SMTP_PASSWORD` | Provider's SMTP password or SMTP API credential |
| `SMTP_FROM_EMAIL` | Sender address verified with that provider |
| `SMTP_USE_TLS` | `true` |
| `SMTP_FROM_NAME` | `PCS Staging` |

The provider supplies the sending credentials; the recipient's Gmail password is
not used. Leave the existing general email and player-update flags unchanged.

Fly stores these as encrypted app secrets and injects them when the app starts.
See [Fly's app secrets documentation](https://fly.io/docs/apps/secrets/).

If using Fly's CLI, import a local file containing the settings as `NAME=VALUE`
pairs. Keep the file outside this repository and use its actual path:

```sh
fly secrets import --app juprleagues-api-staging < /path/to/staging-mail.env
```

The import applies secrets and updates the app. If secrets were added using
`--stage`, apply them with:

```sh
fly secrets deploy --app juprleagues-api-staging
```

See [Fly's secret import reference](https://fly.io/docs/flyctl/secrets-import/).
Do not print secret values or use debug logging to verify them.

## Run the account test

1. Refresh the recipient's existing La Ribera invitation in a private browser
   window, separate from the organizer's sign-in.
2. Choose **Create account** and enter the approved test address.
3. Select **Email me a verification link**. The email's subject begins with
   **[PCS staging test]**. Allow a minute before retrying; each invitation permits
   at most five requests.
4. Open the verification link from the inbox. Choose and confirm a new password.
5. Review La Ribera's administrator invitation and explicitly accept it.
6. Sign out and sign in with the new email/password. This account should open
   La Ribera directly; Tres remains associated with the organizer's account.
7. Continue the interclub setup and select players for an upcoming meet.

If the page still says email is unavailable, the mail settings, approved window,
or staging environment check have not passed. Staging health reports only safe
status: `invitation_email_test.smtp_ready`, `active`, `reason`, and `expires_at`.
`smtp_ready=true` confirms configuration is present; receiving the message is the
delivery test. No role is granted before verified, explicit acceptance.

## Configuration maintenance

The approved inbox is stored as a lowercase SHA-256 digest in
`config/staging_invitation_email_test.json`, not as plaintext in this public
repository. The runtime normalizes the entered address and compares its digest.
This preserves exact-address matching and rejects unusable mailbox syntax before
Auth token creation. Digests are identifiers, not passwords or encryption.

To close the pilot early, deploy the same config with `enabled=false`. It stops
automatically at the recorded expiry even if SMTP remains configured.
