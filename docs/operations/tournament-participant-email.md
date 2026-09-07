# Tournament participant email

The registration communications page can send to an explicitly selected person
or group. Preview remains read-only. Sending requires tournament-management
permission for the club, the communications write gate, and the existing email
configuration. Staging stays in `dry_run`; production uses its configured SMTP
sender and Reply-To. No new secrets or database migration are needed.

## Operator flow

1. Select participants, enter a subject and message, and preview.
2. Check the recipient list, sender, Reply-To and message; confirm Send email.
3. Follow each recipient's result. Shared email addresses receive one copy.
4. After an interruption or reload, open Recent emails, review saved results,
   and continue with recipients marked Not sent.

Sent means SMTP accepted the message; it does not prove inbox delivery. An
uncertain attempt is marked Check delivery and cannot be retried automatically.
Check the recipient's inbox before intentionally composing another message.

## Delivery and recovery

- A fingerprint binds the exact selected registrations, deduplicated audience,
  content, sender and delivery mode to the reviewed preview. The server resolves
  recipients from the tournament; callers cannot provide an arbitrary address.
- The existing private `communications_admin_operations` table stores a UUID
  broadcast and deterministic UUID recipient attempts. A unique INSERT owns each
  SMTP attempt. Replays return saved results, and concurrent losers never send.
- A campaign confirmation and each recipient intent require an audit record.
  Recipient outcomes are persisted before returning success. If outcome storage
  fails after SMTP, the durable claim prevents a second send.
- One HTTP request handles one recipient. The page drives sequential progress;
  closing it leaves unattempted recipients available for an explicit continuation.
- Contact or delivery-setting changes stop remaining sends and require a new
  review. Existing registrations, payment status and partner links are untouched.
- Rollback may remove the UI/routes without deleting the saved ledger. Restoring
  this release restores the saved results; no background process sends these rows.

## Verification

Focused service/API and component tests cover exact individual/bulk selection,
shared addresses, stale previews, permissions, audit failure, claim races, SMTP
ambiguity, result persistence failure, duplicate clicks, reload and session changes.
Tests use fixtures and dry-run delivery; no participant messages are sent.
