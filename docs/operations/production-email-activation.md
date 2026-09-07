# Production email activation

Prepared on 6 September 2026 for Joe's request to make production email operational.

## Observed production state

The live API at commit `22b7ea6bcf58798ecd336ec589c42eace82e2f99` reports
`email_mode=dry_run`, `smtp_configured=false`, and disabled player-update controls.
The production deployment workflow explicitly restored dry-run mode on each release.
The communications guard also rejected production independently of its feature flag.

The production player-update outbox contains 71 historical sent rows, three error
rows, and one skipped row, all from April–May 2026. There are no pending or sending
rows. None of these rows was changed or retried during preparation.

## Prepared behavior

- Application emails use live delivery: registration confirmations and edit links,
  partner requests and invitations, pairing notifications, tournament broadcasts,
  and verified-subscriber player updates.
- Player Updates Admin can preview, queue, send, and manage subscriptions under
  its existing club authorization, audit, row-version, and confirmation controls.
- A task on the existing production API machine polls automatic match-update
  rows every minute and delivers at most 25 selected messages per club per pass.
  No additional machine or scheduled GitHub workflow is required.
- The automatic sender only selects the deterministic match-generated operation
  keys with empty digest snapshots. Manually prepared messages wait for an
  operator to send them. Error and uncertain rows are never reset automatically.
- Delivery reuses atomic row claims. Multiple API replicas, manual sends, and
  the worker cannot claim the same pending row. A missing worker-run marker stops
  delivery. A process interruption leaves uncertain delivery for operator review.
- Tournament publishing hands queued updates to the worker, keeping mail-server
  latency out of result publishing. Normal staging dry-run behavior is unchanged.
- Player profile links use the current club/player route. Messages include Date
  and Message-ID headers. Port 465 uses implicit TLS; other TLS ports use STARTTLS
  with certificate verification.

## Required sender connection

Production has no configured SMTP connection. Before activation, configure these
server-only Fly secrets on the existing `juprleagues-api` app:

| Secret | Value supplied by the email provider |
| --- | --- |
| `SMTP_HOST` | SMTP server hostname |
| `SMTP_PORT` | TLS port, usually 465 or 587 |
| `SMTP_USERNAME` | SMTP login |
| `SMTP_PASSWORD` | SMTP credential or provider-issued app password |
| `SMTP_FROM_EMAIL` | Sender address authorized by that provider |
| `SMTP_FROM_NAME` | Pickleball Club Sandwich |
| `SMTP_REPLY_TO` | Joe's chosen reply address |
| `SMTP_USE_TLS` | `1` |

Enter credentials through the hosting provider's secret controls. Never place
them in source, chat, issue text, or browser-visible application configuration.
Complete any sender/domain verification required by the selected provider.
The operator must identify the intended provider and From address; the application
cannot supply missing provider credentials.

Supabase Auth account-confirmation and password-reset emails have a separate
SMTP configuration. The application SMTP settings do not configure that service.
Its sender configuration and delivery still need verification if those account
emails are part of the activation request.

## Deployment and verification

1. Configure the sender securely while the currently deployed API remains in
   dry-run mode.
2. Run the prepared production release through the existing deployment workflow.
   Before its first mutation, it executes `scripts/production_email_probe.py`
   inside the current Fly machine. This verifies the exact production identity,
   required settings, encrypted connection, and SMTP authentication, sends zero
   messages, and outputs no credentials or raw provider error details.
3. The workflow preserves the captured baseline mail mode while deploying. It
   switches the candidate to the reviewed live profile and verifies configured
   SMTP, the live gate, and a running player-update worker through health checks.
   The exact previous `pre_email` profile remains available for rollback.
4. Once Joe names an approved test recipient, send one controlled application
   email and verify provider acceptance, inbox receipt, From/Reply-To, and links.
   Authentication alone is not proof of inbox delivery. Do not replay historical
   messages or use a real registration mutation just to test delivery.
5. Check one newly triggered automatic update and its outbox/worker ledger.
   Staging must continue to report `dry_run` with live delivery disabled.

No production configuration was changed and no messages were sent while preparing
this release. Activation is blocked on the missing sender connection.

## Preparation checks

184 focused tests passed across production activation/rollback, SMTP transports,
worker startup/shutdown, automatic-versus-manual outbox selection, required worker
markers, subscriber preferences, registration/invitation/broadcast senders,
communications authorization, and tournament publishing. All 20 production
workflow shell blocks passed syntax checking; `git diff --check` passed.
