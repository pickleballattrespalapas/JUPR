# Tournament sponsors

Manage sponsors under tournament Setup → Basics → Sponsors. Each sponsor can have
a name, optional website, uploaded logo, tier, public tier label, visibility, and
private staff notes. Changes save to the tournament draft; publish from Review to
update public pages. Preview placement shows the draft at desktop or phone width.

Premier / Presenting sponsors appear immediately below the tournament title as
“Presented by Sponsor Name” followed by the logo. Supporting and Community
sponsors appear in separate groups at the bottom of tournament pages. Sponsor
names remain visible with logos, without logos, and when an image fails to load.
Reordering operates within each tier. Legacy sponsors default to Community.

Uploads accept static PNG, JPEG, or WebP up to 5 MB and 4096 pixels per dimension.
The API checks tournament management permission and tournament ownership, then
strips image metadata and stores a resized WebP in private storage. Public pages
receive short-lived image URLs for visible, published sponsors only; private notes
and storage paths are excluded. Removing a sponsor or logo updates the draft;
previously uploaded assets are retained to preserve published and draft references.

The migration `20260906052216_tournament_sponsor_logo_storage.sql` creates the
private bucket and restrictive client-access policies. It was applied to staging
project `sijpxjxvdtrehmqvirfi` before the staging merge.

## Tournament emails

Registration confirmations, registration edit links, participant broadcasts,
partner requests and updates, and four-player team invitations include published,
visible sponsors automatically. A compact “Presented by” credit and small logo
appear below the email title. The presenting sponsor's name and full-size logo
repeat after the message, registration details, and action buttons, followed by
their public tier label, description, and website link. Supporting and Community
sponsors follow that feature. Plain text uses the same order, with only a short
credit at the top. Private notes and unpublished setup changes are excluded.

Sponsored emails use a readable content width and line spacing. Communications
gives the organizer's message larger text, while sponsor details sit below a
divider. Repeated logos reference the same embedded image, so the extra placement
does not add another attachment.

Uploaded logos are embedded as PNG MIME parts, so delivered emails do not depend
on expiring storage URLs. Images are resized and limited to 64 KB each / 256 KB
total. If a logo cannot be loaded, the sponsor's text still appears and the
transactional message can be delivered. The logo bucket stays private.

Communications displays the complete email in a sandboxed preview. Sponsor
details and embedded images are bound to the confirmation and recorded with the
broadcast. Changed published sponsors require a new preview. Each recipient
uses the saved logo copies, including after continuing an interrupted broadcast;
previously attempted recipients are never sent another copy automatically.

No database migration, new sender credentials, or email-mode change is required.

Validation includes sponsor rendering checks, the existing Next component suite,
TypeScript and Next production build checks, migration guards, and focused Python
tests covering uploads, authorization, draft/published separation, validation, and
public response privacy. Hosted staging readiness is determined by the exact-SHA
deployment handoff described in `AGENTS.md`.

## Optional registration events in Communications

The Communications composer offers **Include registration events**, off by default.
It appends the selected player's event names, days/dates, and partner names or
partner-needed status below the organizer's message and above the full sponsor details.
Each recipient receives only the selected registrations associated with their
email address. Shared inboxes receive one copy with separately named player
sections; other registrations sharing that address are not added automatically.
A participant selected using an event filter still receives all their registered
events when this option is enabled. Empty and explicitly selected cancelled
registrations are labelled clearly. Partner contact details and private notes
are never included.

**Preview for** switches the displayed recipient without changing the audience.
Event details and the option are bound to the reviewed message; editing an event,
day, partner, or selection requires a new review before remaining emails can be
sent. Confirmed recipient event snapshots are retained with the existing
communications ledger; no migration is required.


## Optional registration edit buttons in Communications

**Include Edit Registration button** is off by default and works independently
of **Include registration events**. Each selected registration gets a named button
below the message/event details. A shared inbox receives one email containing only
the selected players' buttons. HTML and plain-text versions both support the links.

The preview shows inactive buttons. No bearer tokens are issued during preview,
confirmation, or dry-run delivery, and no bearer URLs are stored in the audit or
communications ledger. The sender generates each personal link immediately before
its claimed SMTP attempt, using the existing explicit edit signing secret and
48-hour expiry. Resuming a batch generates fresh links only for remaining recipients.
The existing public edit endpoint continues to enforce tournament, registration,
email binding, expiry, and draw/relationship editing restrictions.

The option, selected registration identities, recipient email, club slug, and web
origin are included in the preview fingerprint. Changes require a fresh review.
The saved results retain the inclusion choice without exposing the links. No new
database migration, credentials, email provider, or live staging send is required.
