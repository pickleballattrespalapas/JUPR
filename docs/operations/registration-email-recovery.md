# Registration email delivery recovery

Edit-link delivery records distinguish provider acceptance, a confirmed rejection,
and an unknown outcome. Private receipts include SMTP phase, exception type,
numeric response codes, enhanced status code and attempt count. Raw provider text,
recipient addresses, edit tokens, credentials and message contents are excluded
from diagnostic errors and logs. Accepted receipts and transport failure details
retain the Message-ID for provider tracing, including uncertain attempts;
acceptance alone does not confirm inbox placement.

The shared SMTP sender makes at most two attempts, with a one-second pause,
only for temporary explicit rejections or connection failures before submission.
The second attempt starts only if the first failed within ten seconds, avoiding
an additional retry after an already slow connection timeout.
It reuses the exact MIME payload and Message-ID. A lost connection or timeout
during message submission stays uncertain and is never automatically repeated.
Permanent rejections and certificate errors do not trigger automatic retries.
QUIT errors cannot undo an accepted message or replace the original failure.

Within a 15-minute recipient/tournament bucket, a fresh user request can claim
one additional durable recovery slot only after a confirmed rejection has been
persisted. Exact request replays, concurrent requests, accepted sends, unknown
outcomes and failed receipt writes cannot create another send in that slot.
Old uncertain attempts are never reset or replayed by this change. A later
15-minute bucket continues to permit a new user-requested link as before.

The public response remains identical for matched and unmatched addresses. The
form acknowledges a request rather than claiming delivery and gives spam,
address and tournament-staff recovery instructions. Failed network requests
retain their request identity and release the Sending state.

No schema migration or sender credentials are changed. Staging remains dry-run;
all transport tests use local fakes. This improves delivery recovery and evidence
for the next provider incident; it does not establish the cause of historical
SMTPResponseException records, which lack the provider response code.
