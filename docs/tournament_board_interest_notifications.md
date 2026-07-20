# Tournament Board Interest Notifications

When a player uses a valid tournament registration edit link to send interest from the public tournament board, FastAPI now creates a pending pairing-interest row and sends notification email to both:

- the player whose public board entry was selected; and
- the tournament organizer inbox.

Accept, decline, and requester-cancel transitions send the minimum relevant
lifecycle message after the transactional write. Repeating the same request or
transition does not repeat delivery.

## Organizer inbox

The organizer notification uses this resolution order:

```text
JUPR_TOURNAMENT_ORGANIZER_EMAIL
JUPR_SUPPORT_EMAIL
joe@juprleagues.com
```

For Tres Palapas / first public draft operations, use:

```text
JUPR_TOURNAMENT_ORGANIZER_EMAIL=joe@juprleagues.com
JUPR_SUPPORT_EMAIL=joe@juprleagues.com
```

## Email safety modes

The notification helper uses the existing email safety controls:

```text
JUPR_EMAIL_MODE=dry_run|staging_redirect|live
JUPR_STAGING_EMAIL_REDIRECT_TO=<safe test inbox>
JUPR_TOURNAMENT_PARTNER_CONTACT_DENYLIST=<email-or-@domain,...>
```

In staging, use `dry_run` or `staging_redirect`. Do not use `live` until the tournament board behavior has been reviewed with staging data.

## Safety model

- The browser does not send emails directly.
- The browser does not read or write Supabase directly.
- The requester must have a valid tokenized registration edit link.
- The selected board entry must still be public, in the same tournament, in the same division, and marked as looking for a pairing.
- The email does not auto-confirm a team; it only communicates pending interest.
- Email/provider failure never rolls back or disguises a completed database write.
- Public API responses contain reduced delivery status only and never expose an
  address, edit token, or provider error body.
- Exact addresses and whole domains on the contact denylist are skipped.
- Public roster and board pages continue to hide phone and email fields.
