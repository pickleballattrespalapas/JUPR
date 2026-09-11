# Tournament Partner Board and email invitations

## Player flow

The public `/clubs/[clubSlug]/tournament-partner-board` lists consenting players
by division. Each listing has a visible **Request to partner** button. Anyone may
open the form, enter their name, email and personal message, and submit it without
first finding a registration edit link. The listed player's email stays private.

The original **Send request** button immediately emails the message to the listed
player. Senders do not confirm their email or return to the website to send it.
The recipient receives the message with a large **Accept
partnership** email button. That opens a private, simple confirmation page; the
player clicks **Confirm partnership** to accept. Opening a link alone never sends
mail or changes registration data, including when email software scans links.

When both players are registered in that division and eligible, acceptance:

- creates the canonical confirmed team;
- updates both selections to `HAS_PARTNER`;
- removes both players from the Partner Board for that division;
- cancels competing invitations and legacy partner requests;
- sends a confirmation to both players.

If the sender still needs that division's registration, acceptance reserves the
partnership and emails them a registration link. Their own name, email and division
are prefilled. Saving the linked new or edited registration completes pairing
automatically. The private request page also offers **I've registered — finish
pairing** if registration was completed separately or a connection interrupted
completion. Registration saves survive email or pairing-service failure.

Reservations and request links expire 14 days after the initial request. Either
player may cancel a reservation. A reserved target cannot be claimed by a second
request or by the legacy pairing flow. Expired reservations no longer hide the
player from the board. Existing registration-edit-token request review remains
available for older requests.

## API and transaction boundary

All reads/writes are FastAPI-mediated. Browsers never connect to Supabase or send
mail directly. New endpoints under
`/clubs/{club_slug}/tournament-registration/partner-invitations` are:

| Endpoint | Purpose |
| --- | --- |
| `POST /partner-invitations` | Save an idempotent form submission and immediately email the request |
| `POST /partner-invitations/review` | Read the scoped private request; no mutation |
| `POST /partner-invitations/respond` | Verify, accept, decline, cancel, complete registration pairing or retry failed email |

The route supports the existing tournament ID/registration-slug context.
`20260909203736_partner_email_invitations.sql` adds private invitation and delivery
tables, invoker RPCs, expiry/idempotency controls and reservation protection on
canonical team links. New tables have RLS enabled and no browser role access.
All new RPCs are callable by `service_role` only, with an empty search path.

Acceptance uses the existing universal registration/selection lock order and
canonical `create_tournament_partner_request` /
`transition_tournament_partner_request` functions. Registration, selection and
event versions are checked inside the transaction against the profiles validated
by Python. Concurrent or stale requests cannot produce two partners for one
selection. Gender, age, skill and imported-draw restrictions use the same rules
as normal registration.

## Privacy and delivery

- Public listings require global/event board enablement, selection display consent
  and registration contact consent. Availability and consent are checked again
  on acceptance. Reserved and paired selections are omitted from public lists.
- Public entries contain only allowlisted player/division information and an
  opaque board reference. Recipient emails, phone numbers, exact ages, DUPR IDs,
  database row IDs and private registration tokens are not returned on the board.
- Email capabilities are signed, expire, and are bound to the invitation, club,
  tournament, participant role and email hash. They cannot serve as registration
  edit tokens. URLs carry them in fragments; requests use POST bodies and
  `no-store`/`no-referrer`. An email-address change invalidates an old target link.
- The sender explicitly sees that their own email will be shared privately with
  the recipient for replies. The target email is never shown in the request form.
  Target links cannot fetch sender registration edit capabilities or prefill data.
- Anonymous requests are sent directly; this does not mark the typed address as
  verified. Automatic pairing requires matching registration name and email.
  Private sender links are delivered only to that mailbox, and only those links
  can expose their registration edit capability. Existing verification links
  remain usable for requests created before direct sending was introduced.
  A honeypot and
  an atomic limit of five new requests per sender email per hour reduce abuse.
  Exact retries retain the same request identity and do not consume another slot.
- Email messages escape personal text and honor
  `JUPR_TOURNAMENT_PARTNER_CONTACT_DENYLIST`. A durable per-message delivery claim
  avoids repeated notifications; failed/stuck attempts can be retried. SMTP
  response loss cannot guarantee exactly-once delivery, but never repeats pairing.
- Staging remains `JUPR_EMAIL_MODE=dry_run`. No real-player email is part of
  automated validation. Production deployment requires separate authorization.

## Validation

- `tests/test_partner_email_invitations.py`: signed/scoped links, privacy, sender
  identity, API actions, registration handoff, delivery safety and save recovery.
- `tests/sql/partner_email_invitation_lifecycle.sql`: real PostgreSQL transaction
  checks for verification, repeated requests/acceptance, registered pairing,
  reservation and registration completion, competing requests, stale versions,
  private grants and deduplicated delivery claims. The disposable fixture runs
  inside a transaction and rolls back completely.
- `apps/web/tests/tournament-partner-invitations.cjs`: form submission and retry,
  read-only email landing, explicit confirmation, roster confirmation and
  registration handoff. Included in `npm run test:component`.
- Existing registration create/edit, legacy partner lifecycle, API contract,
  component, type/build and migration/parity guards remain required.

The migration and rollback-only SQL fixture were verified on the isolated staging
project `sijpxjxvdtrehmqvirfi`. Staging readiness additionally requires the matching
Fly/Vercel commit and successful `staging-handoff-<sha>` artifact as described in
`AGENTS.md`. No production migration or deployment is included.
