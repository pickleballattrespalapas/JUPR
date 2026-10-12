# Tournament registration support history

Tournament Manager → Registration → Registrants now includes **Registration attempt history**.
Search by name or full email to see steps, selected events, client validation messages,
server rejections and the server-confirmed save outcome. The reporting page has the same panel.

The new-registration and secure edit forms send debounced, best-effort support breadcrumbs.
Errors are sent immediately; pagehide/visibilitychange also flush the latest safe snapshot.
Incomplete attempts older than 30 minutes are shown as having no recent activity, not
as proof that the player deliberately abandoned the form. An interrupted connection,
disabled JavaScript or blocked request can prevent capture. Past attempts cannot be recovered.

Each browser form has a random attempt UUID. Server profile/submit/edit requests use
the same UUID, or generate one for older clients. Only server results can mark a save.
Browser-reported fields and all contact identities are unverified and cannot create
registrations, pair players, approve ratings, send mail or appear on public rosters.

The allowlisted snapshot contains player name/email, selected event IDs, partner mode/name,
a same-email boolean, form step, error code/message and server save ID. It excludes
phone numbers, partner contact details, private notes, payment fields, passwords and
edit/invitation/confirmation tokens. Errors omit URLs and recognizable secret/token strings;
unknown backend exceptions use a generic support message. Contact breadcrumbs expire
after 30 days through `registration-support-retention` in pg_cron.

Tables/RPCs are service-role-only with RLS and revoked public/authenticated grants.
Public POST responses are non-enumerating; no public read endpoint exists. Staff reads
require the existing club-scoped tournament-management permission and are `no-store`.
Search uses separate exact email/escaped name filters, never concatenated PostgREST OR syntax.
RPC quotas are serialized across workers: 80 browser breadcrumbs per attempt, 100 total
non-save breadcrumbs per attempt, 12,000 browser / 24,000 total daily per club. One server
save remains recordable after those quotas. Storage failures never change registration outcomes.

The migration is staging-first. Production needs separately authorized deployment/migration.
