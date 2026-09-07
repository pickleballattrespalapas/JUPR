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
visible sponsors automatically. Presenting sponsors appear below the email title;
Supporting and Community sponsors follow the message. Names, public tier labels,
descriptions and website links appear in both HTML and plain text. Private notes
and unpublished setup changes are excluded.

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
