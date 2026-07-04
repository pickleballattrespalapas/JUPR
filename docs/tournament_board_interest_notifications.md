# Tournament Board Interest Notifications

When a player uses a valid tournament registration edit link to send interest from the public tournament board, FastAPI now creates a pending pairing-interest row and sends notification email to both:

- the player whose public board entry was selected; and
- the tournament organizer inbox.

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
```

In staging, use `dry_run` or `staging_redirect`. Do not use `live` until the tournament board behavior has been reviewed with staging data.

## Safety model

- The browser does not send emails directly.
- The browser does not read or write Supabase directly.
- The requester must have a valid tokenized registration edit link.
- The selected board entry must still be public, in the same tournament, in the same division, and marked as looking for a pairing.
- The email does not auto-confirm a team; it only communicates pending interest.
- Public roster and board pages continue to hide phone and email fields.
