# PCS interclub implementation plan

Status: design foundation; no interclub rating writes enabled.

## First season
Southern BCS (La Paz and south), January–March. Eight participating clubs,
with four clubs at each meet and two simultaneous meets at different hosts.
Primary divisions 3.5 and 4.0; optional 4.5/Open. Four-player teams. Meets fit
within three hours. Preserve the agreed host-club encounters and multiple
opponents per trip. Match format: three games to 11, win by two, no cap.
These are starting season settings, not hardcoded platform limits.

## Account boundaries
One login can have separate club roles. Club player records and ratings remain
islands. A league is owned by an organizing club; participating clubs retain
control of their own rosters. Free participation includes one administrator and
no operators. Either organizer or host administrators can assign meet access
within their club's staff allowance. Host access never authorizes league rules,
other clubs' directories, published-result corrections, or rating approval.

## Data model to implement
- interclub_seasons: organizing club, name, dates, timezone, state, version,
  eligibility rules, scoring rules, divisions and roster deadline.
- interclub_participations: season, participating club, acceptance state.
- interclub_teams: participation, division, team name.
- interclub_entries: team, represented club, source club player ID, immutable
  starting league rating, submitted name and eligibility facts.
- interclub_meets: season, host participation, scheduled time, court inventory,
  state, version and rating approval state.
- interclub_meet_teams: meet/team association and submitted roster revision.
- interclub_meet_staff: explicit staff assignment referencing an eligible club
  staff membership. Effective access expires/revokes with either assignment.
- interclub_games: meet, division, teams, player entry IDs, individual game
  scores and non-play result. Store represented club at time of play.
- interclub_rating_batches: meet, approved result revision, organizer actor,
  calculation version, idempotency key and processing status.
- interclub_rating_effects: batch, source game, entry, destination club/player,
  rating stream and before/after values. Unique on game/player/stream/revision.
- interclub_audit: actor, action, version and before/after changes.

Do not copy email, phone or whole club directories into league records. Resolve
submitted source player IDs under the submitting club's authorization, never
from user-supplied destination club IDs. Enforce source player/club ownership
with database constraints as well as API checks.

## Operator workflow
1. Organizer creates a draft season, divisions, eligibility and scoring rules.
2. Organizer invites clubs; each club administrator accepts and submits teams.
3. Organizer schedules meets and hosts. Host or organizer assigns eligible staff.
4. Club administrators may update their own roster after the deadline, but
   eligibility violations require organizer-administrator approval. Never
   rewrite participants in already played games when changing a roster.
5. Meet staff run check-in, courts and scores using established league rules.
6. Results can be published separately from rating approval.
7. Organizer administrator reviews the meet and approves it for ratings.
8. Approved corrections automatically recalculate affected rating histories.

## Rating pipeline
Seed each league entry from its represented club's rating at entry time.
Reuse the existing separate league and overall calculation behavior. League
rating changes are not simply copied as deltas into overall club ratings.
The overall destination is only the represented club; other islands do not
change. Participants from Free clubs still receive the approved rating update.

Approval snapshots the complete meet revision. Reject stale approval requests.
Process each game once per player and rating stream; duplicate submissions or
worker retries cannot double-apply ratings. Preserve chronological ordering,
including local club games between interclub games. Later corrections replay
all dependent effects consistently across both streams. Use a durable job with
visible pending/failed/completed status and retry-safe checkpoints. No partially
updated meet may be labeled fully rated. Retain audit of original and corrected
results. Existing club rating algorithms and replay semantics must be inspected
and characterized with fixtures before enabling this pipeline.

## Delivery sequence and gates
1. Club staff management and enforced scope/expiry boundaries (current work).
2. Draft season editor, club participation, teams/rosters and host scheduling.
3. Day-scoped meet operations using shared tournament/live primitives.
4. Organizer review and dual-stream rating pipeline; correction/retry tests.
5. Public schedule/results, privacy audit and pilot acceptance in staging.

Required tests include cross-club roster injection, host escalation to league
rules, expired staff, invalid eligibility, post-deadline valid substitutions,
unauthorized publication corrections, duplicate approval, simultaneous approval
and correction, failures between rating streams, and historical replay with
interleaved club games. Production release requires separate authorization.

## Agreed plans and billing (future enforcement)
Free: participation only, 1 admin / 0 operators.
Basic: $19/month or $190/year; generators/live play; 2 admins / 3 operators.
Middle: $39/month or $390/year; Basic plus leagues, challenge ladders and club
rating reports/history; 3 admins / 5 operators.
Top: $69/month or $690/year; Middle plus tournaments/interclub organizing;
unlimited staff. Basic/Middle extra slots: $3/month or $30/year for either role.
Personal rating access stays free. No staff add-ons on Free.

One month trial per club starts on activation of chosen tier; tier changes keep
original trial end. Card upfront once billing launches; automatic billing unless
canceled. Upgrades/add-ons immediate and prorated. Downgrades/cancellations and
slot removals at period end, no partial refund. Select retained staff before
confirming downgrade. Unsupported programs remain viewable but pause. Failed
renewal: three days grace, then paid operations pause while admins retain billing
and records. Pricing/usage review reminder: November 30, 2026.

## Platform onboarding increment
The `/admin/platform` dashboard lists club accounts, creates a draft club plus
its first administrator atomically, and tracks draft/in-progress/ready-for-review
onboarding. Only identities in `pcs_platform_admins` can access these endpoints.
The migration bootstraps the existing Tres Super Admin through its Auth user ID;
subsequent club role grants do not grant platform authority. There is no public
endpoint to appoint platform administrators. Every onboarding write is audited.

This increment does not launch a club, activate trials, charge cards, send email,
or implement public club signup. Draft status is onboarding/public-visibility
metadata, not a complete entitlement gate. Plan enforcement and invitations are
required before opening self-service onboarding. The dashboard displays existing
plan status rather than implying subscription enforcement is already complete.

## Saved season planning increment
Organizer administrators can now create and reopen season drafts at
`/admin/interclub`: dates, divisions, proposed participating clubs, host meets,
local meet times, courts and duration. Saves are revision checked and audited.
Meet validation rejects unselected clubs, hosts outside the meet, more than four
clubs, durations above three hours, dates outside the season and overlapping
meet assignments for the same club. Disjoint groups can play simultaneously.
The club picker exposes only club identity, never player directories.

Super Admin can open a club at `/admin/platform`, edit its name/contact email,
and inspect active administrator assignments. An assignment does not prove that
the person has signed in. Profile updates require platform authority and audit
the previous and new values atomically.

These are planning drafts, not invitations or confirmed participation. Teams,
eligibility checks, meet scoring, organizer approval and rating replay remain
future increments. Plan entitlements and billing remain unenforced. Staging
verification includes API authorization/scheduling tests, database rollback
checks for revision conflicts/audit/privileges, and the web production build.

## Multi-club workspace increment (September 8, 2026)

The earlier account/staff and season-draft work did not yet make daily admin
operations usable for a second club. Login requested Tres capabilities by
default, and 46 admin entry points supplied Tres IDs or slugs directly.

This increment removes that prerequisite gap:
- Login discovers all current club assignments for the verified identity.
- `/admin/select-club` lists only assigned club names and roles, including
  draft club accounts that still need onboarding.
- The admin workspace names the current club and offers Switch club.
- Players, Match Uploader/Log, leagues, tournament phases/live operations,
  generators, communications, staff and interclub planning use the same scope.
- Public links follow the current club; public navigation follows its URL.
- A club switch starts at `/admin` with a full reload. Prior record IDs, forms
  and client/router caches cannot carry across to the newly selected club.
- The selection cookie is navigation context, never authorization. The shell
  checks both current capabilities and assigned-club identity before mounting
  client tools; protected APIs still independently authorize every operation.
- Other tabs detect a changed selection and block clicks/submissions until
  reloaded. A late response cannot replace another signed-in identity's club
  list, and refreshing the same user's token preserves the selected workspace.

No schema, production, billing, role grant, invitation, email delivery or rating
mutation is included. Existing club data remains separate. Cookie-free sessions
must select a club before opening a club operation. The platform dashboard is
still independently authorized and available without a selected club.

### Next delivery stages

1. **Second-club acceptance and onboarding completion.** Exercise a second-club
   administrator, a multi-club administrator and a scoped operator against
   staging. Complete account invitation/acceptance and administrator onboarding
   readiness before opening public self-service signup. Expand the central
   public club directory/site map, which still starts with Tres.
2. **Interclub participation and rosters.** Convert proposed clubs into audited
   invitations and club-admin acceptance. Each club submits its own four-player
   teams using its own directory. Add eligibility facts, roster versions and
   organizer exceptions; never expose whole directories or contact details.
3. **Meet operations.** Confirm host/court schedules, grant eligible meet staff,
   then add check-in, individual game scoring, standings and results review.
   Host authority remains separate from organizer rule/approval authority.
4. **Rating approval and corrections.** Snapshot organizer-approved meet
   revisions; apply league ratings and represented-home-club ratings through
   retry-safe jobs. Verify duplicate approvals, partial failures and replay of
   corrections interleaved with local club games before enabling rating writes.
5. **Plan enforcement and commercial onboarding.** Apply the agreed staff and
   feature limits, trials, proration, downgrades and three-day payment grace
   period. Keep records accessible when paid operations pause. Launch public
   signup only after these rules and onboarding are enforced.

Validation: 71 focused Python tests; executable React tests for all-club login,
selection/cookie validation, cross-tab action blocking, revoked access, scoped
server data and mutation-panel props, stale identity responses and token refresh;
existing component suite; Next production build with type/lint checks; migration
source guard and whitespace checks. The local cloud-browser URL is blocked by
its network policy, so authenticated real-browser acceptance remains a staging
pilot task. Do not represent these local checks as that acceptance.

## Club administrator setup increment (September 8, 2026)

`/admin/club-settings` lets an assigned club administrator update the club name,
short description, and public contact email. The sidebar refreshes the saved
club name. Operators cannot read or save settings; the API checks the verified
identity and the route's club assignment on every request.

Draft clubs can save their progress and submit setup for Super Admin review.
Submission requires a contact email and records an audit with the verified
actor. The existing platform dashboard shows the resulting Ready for review
status and the club description. Editing a pending submission returns setup to
In progress until it is submitted again. Active clubs keep their existing status
when saving details and cannot accidentally submit themselves as new clubs.

The service-only database function serializes saves with staff changes, repeats
authorization, rejects outdated timestamps, and writes the audit atomically.
The form preserves drafts after a conflict, requires reload after uncertain saves,
blocks duplicate clicks, and discards late responses from a different account
or club. Token refresh preserves unsaved edits. Club URLs and commercial settings
are not editable here. This does not activate a club or send messages.

Validation: 57 focused API/permission/navigation checks, executable component scenarios,
full component suite, Next build, migration guards, and a staging transaction
test (`tests/sql/club_settings_transaction.sql`) covering permissions, save,
submission, stale writes, audit, cross-club denial, revoked access and active-club
status. Test records were rolled back. No new database security advisor findings.
Authenticated browser acceptance remains a staging pilot task.

## Staff invitation increment (September 8, 2026)

New staff additions in `/admin/staff` now create email-bound invitations instead
of immediately granting access. The administrator selects the role, operator
scopes and optional access end date, then copies a seven-day invitation link.
The list shows pending, accepted, cancelled and expired invitations. Existing
active assignments remain editable; removed or expired staff can be invited again.
Creating an invitation does not send email or create an Auth account.

The recipient opens `/admin/accept-invitation`, signs in with the invited email,
reviews the club and permissions, and explicitly accepts. This page can authenticate
an account that has no staff assignments yet. Only successful acceptance refreshes
admin capabilities and opens the invited club. Other club assignments remain intact.
Single-club accounts continue to sign in directly to their club; accounts with
multiple assignments retain the club selector.

The service-only invitation transaction shares the staff advisory lock and checks
the inviter's current authority, the recipient's confirmed Auth email, invitation
expiry, and the original staff assignment snapshot. A narrowly scoped identity
helper checks Auth without granting the API database role access to the user table.
Creation and acceptance are retry-safe. Replays cannot restore revoked access or
overwrite a newer assignment. Club access and invitation audit records commit
together. The table has RLS, no browser policies and no anon/authenticated grants.

Recipients can request an email sign-in link, including for a new account. The
server claims a matching valid invitation before generating a Supabase magic-link
token; the credential goes only to the invited mailbox, never to the inviting
administrator or an API response. Callback origins come from server configuration,
and credential fragments are removed before verification. Requests are limited
to one per minute and five per invitation. Non-live email modes, including staging
redirect mode, stop before creating an Auth user/token or sending any mail.
Staging stays `dry_run`: use an existing confirmed test account to accept.

Validation: 135 focused API, permissions, auth and deployment checks; executable
staff/recipient component scenarios; full component suite; Next build; and
`tests/sql/staff_invitation_transaction.sql` on staging under the actual API role.
The transaction test covers no grant before acceptance, verified/wrong emails,
expiry, cancellation, revoked inviter, newer assignments, retry/replay, preserved
scopes/expiry, email throttling, audits and cross-club isolation. All database
fixtures were rolled back. No new security warnings; the intentional private
invitation table adds one informational RLS-without-policy advisor entry.
Authenticated browser acceptance and real email delivery remain unclaimed.

## Interclub participation and rosters increment (September 8, 2026)

An organizer can now open registration from a saved season at `/admin/interclub`.
The registration captures that draft's exact revision and invites all proposed
clubs in their PCS workspace. Later planning edits do not alter the registration
that clubs accepted. Each division has explicit optional minimum/maximum player
ratings and an optional four-player gender composition. A division label does
not silently impose a rating limit. Rules and season details cannot be edited
after registration opens in this increment.

`/admin/interclub/registrations` lists the current club's organized seasons and
invitations. Each invited club's administrator accepts or declines for that club;
organizers can cancel an unused invitation or invite a declined/cancelled club
again. Organizers cannot accept on another club's behalf. Accepted participation
does not grant staff roles, directory access, or access to another club's tools.
Invitations are in-app records only; no messages are sent.

Accepted clubs submit named four-player teams using active players from their
own directory. Team division is fixed after first submission. A player can belong
to only one current team per represented club/division; another division remains
possible. Unrated, inactive, missing or cross-club players cannot be submitted.
The player's first season entry captures `players.rating / 400` from the represented
club as an immutable starting league rating. That seed is reused for eligibility
and subsequent rosters, even when the home-club rating later changes. This does
not write ratings or transfer players between clubs.

Every submission creates a roster version with names, starting ratings, gender
facts and eligibility issues. Rating/composition violations save as Needs organizer
exception and remain ineligible until an organizer administrator approves that
exact version with a reason. A denial is also recorded. Replacing players creates
a new version and never carries forward an old exception. Valid updates after
the roster deadline are allowed and labelled late; newly entered late teams need
an organizer exception. Withdrawing a team releases its current player places
and preserves the roster history; it can be restored through a new roster version.

Only the represented club can edit or withdraw a team. Organizer administrators
can review submitted teams and their eligibility history, but receive no other
club's player directory, player contact data, or source player IDs. Participants
see only their own teams. A composite player/club foreign key enforces ownership
below the API, and current lineup uniqueness is constrained in the database.
Registration writes share the staff and season locks, repeat role/expiry checks,
reject stale revisions and commit their audit together. The current-roster view
joins each team to its exact current version in one database read. All new tables
and the view are private to the service role; the view uses security-invoker RLS.

The UI blocks duplicate submits, preserves unsaved rosters during token refresh,
discards responses after account/club changes, and requires reload after stale or
uncertain saves. Players and teams have explicit pagination; history shows the
latest 50 versions while the full history remains stored. The season selector
loads the latest 100 organized seasons and latest 100 club invitations.

Validation: 127 focused API, role, planning and deployment checks; full component
suite including rule opening, acceptance, four-player selection, revisions,
privacy/role boundaries and stale account responses; Next build/type/lint checks;
and `tests/sql/interclub_registration_transaction.sql` on staging. The rollback
test exercised real service-role submissions, source ownership constraints,
invitation cancellation/retry, frozen seeds, late changes, organizer decisions,
version history, withdrawn places and atomic audits. No fixtures were retained.
No new reported security advisor findings. Authenticated browser pilot acceptance
remains pending.

Next: validate organizer and second-club accounts with the pilot checklist, then
build meet staffing, check-in, courts and individual game scoring. Meet lineups
must bind to roster versions before play; registration eligibility alone must
not be treated as an approved result or permission to write ratings. Scheduling
changes after clubs accept need an explicit future change/notification workflow.
Player merges involving an interclub entry must preserve its represented-club
and historical source identity; the FK deliberately prevents silent deletion.


## Meet rosters (2026-09-08 correction)

Joe clarified that travel makes a fixed season roster unsuitable. Clubs enroll
for the season, but submit four available players separately for each scheduled
meet. The roster workspace now selects a meet, defaults to the next upcoming
one, and shows only that meet's teams, revisions and eligibility decisions.
Nothing is automatically copied from the previous meet. The same player and
team name can be used at multiple meets; duplicate player places are blocked
only within the same club, meet and division.

Opening season invitations no longer asks for a roster deadline. Each scheduled
meet receives its own deadline, initially its start time. Before the first team
submits, the organizer can set an earlier future deadline for that meet. Valid
substitutions remain allowed after the deadline until the meet starts. A newly
submitted late team requires an exception for that meet. Once the meet starts,
lineups and decisions are read-only history in this registration workspace;
future live-operations corrections need their own explicit workflow.

Existing season submissions are preserved as reference-only history, without
assigning them to a meet or reserving player places at future meets. The former
season write endpoints and RPCs require reload. Starting league ratings remain
captured once per represented club/player/season when that player first enters
a meet; players can join at any later meet. This does not lock season membership
or change home-club ratings. Season eligibility rules and accepted schedule
snapshots remain unchanged by roster edits.

Meet IDs and season ownership are checked by the API, transactions and composite
foreign keys. Writes carry both meet and roster revisions and share the existing
staff/season locks. Meet switches abort pending responses and discard the old
editor. Organizer views retain only submitted eligibility facts, and a club's
player picker remains private to that club.

Validation passed: 132 focused Python checks, including API meet/season/club
isolation and old-client rejection; the full component suite;
component checks for separate lineups/deadlines and switching during a save; and
an isolated, rolled-back SQL scenario with two meets. The SQL scenario proves
reusing players in the next meet, independent deadlines, unchanged history and
next-meet lineups, stale revision protection, late substitutions, scope foreign
keys, and refusal to change a started meet. Browser pilot steps are below in
`docs/second_club_staging_pilot.md`.

## Guided interclub setup (2026-09-08)

The former long planning and invitation forms are now a five-step wizard:
season details, participating clubs, divisions and eligibility, meet schedule,
then review and invitations. `/admin/interclub` separates unfinished drafts from
opened season workspaces. Existing drafts resume through the wizard, while
opened seasons lead to club responses and individual meet rosters.

Each Continue action saves the draft and next step. Save and exit also retains
incomplete dates and meet details. Division eligibility rules now live in the
saved draft so leaving setup does not reset them. The final review explicitly
lists the clubs, rules and meets that will be fixed when invitations open.
It requires acknowledgement, saves the reviewed draft, then opens invitations
against the exact returned revision. Only the final action opens invitations;
there is no email or player selection during setup. Missing second-club account
guidance, timezone-aware meet inputs, season-date checks and shared-club overlap
checks explain corrections at the relevant step.

The API separates partial planning validation from complete invitation
validation. Both use the same eligibility rule model. The invitation boundary
checks complete dates/meets, at least two clubs, an upcoming meet, and the saved
rules before calling the existing revision-locked transaction. The staging
migration `interclub_setup_wizard` prevents further draft saves once invitations
are open under the same season lock, preserving accepted terms. It changes no
tables, grants, player records or ratings. Existing season rosters remain
reference-only; each upcoming meet still gets its own lineup.

Verification: 114 focused API/setup/deployment/navigation checks and the full
component suite passed, including wizard resume, incomplete meets, saved rules,
invalid/overlapping times, explicit review, duplicate clicks and stale context.
Two rolled-back staging SQL scenarios verify draft/audit round trips, revoked
access, exact revisions, opened-season protection and independent meet rosters.
The Next build passed. Local browser verification could not run because this
workspace blocks browser process sockets; desktop/mobile visual acceptance
remains in the staging pilot checklist. No new database advisory findings.


## Invite missing clubs during setup (September 8, 2026)

Step 2 keeps club selection in place and now includes **Invite a new club**.
Organizers enter a club name and its first administrator's email. Creating the
invitation atomically creates an inactive free club, selects it in the season,
and saves the current draft. It does not grant the organizer access to the new
club. A duplicate name/address directs the organizer to select the existing club.
At least two clubs are still required before continuing from this step.

The organizer copies a seven-day link from this same step. No email is sent by
creating it. The recipient signs in with the invited, verified email, reviews
the club and explicitly accepts administrator access. Other club assignments
remain intact. Recipients can request an email sign-in link in live mode; staging
stays dry_run and requires an existing confirmed test account. The organizer can
correct the email, renew or cancel the account invitation. After opening the
season, these controls remain available through **Club account invitations** on
the season card. Season invitations still open after reviewing rules and meets;
club onboarding does not accept a season or choose any players.

The service-only transaction checks current organizer authority, season ownership,
revisions and existing club/staff records. RLS and revoked browser grants protect
invitation contact details. Creation retries cannot duplicate clubs; verified
acceptance retries cannot restore revoked grants. Existing clubs cannot be claimed
through this flow. Email claims are capped and throttled, and non-live modes stop
before any Auth user/token creation or mail delivery. No production changes.

Validation: focused API and executable component scenarios, web build and type
checks, and `tests/sql/club_join_invitation_transaction.sql` against staging under
the API database role. Transaction fixtures are rolled back. Browser acceptance
remains a manual staging check because this workspace cannot launch a browser.

## Invitation account setup (September 8, 2026)

Invitation recipients now choose **Create account** or **Already have an account**.
The new account path explains and follows three steps: verify the invited email,
choose and confirm a password, then review and explicitly accept club access.
It no longer starts by asking a new recipient for a password they have never set.
Existing accounts can sign in with their current password or use an email link.
Recipients using an older email link can also choose **Set a password** during
review. Setting a password applies to their PCS sign-in across all assigned clubs.

Verification uses the existing invitation-bound, throttled email claim. The link
retains the invitation ID and club-invitation kind and carries `setup=password`
to return to password setup after authentication. The token hash stays in the
fragment and is removed before verification. The password goes only to Supabase
Auth using the recipient's session; there is no administrator credential or
unauthenticated password change. Password setup does not save an admin session
or grant a role. Club permissions still require the existing verified acceptance
transaction. See Supabase's [email link creation](https://supabase.com/docs/reference/python/auth-admin-generatelink)
and [authenticated password updates](https://supabase.com/docs/reference/javascript/auth-updateuser).

Public GET requests to either invitation's `/sign-in` resource report only whether
email is enabled, without looking up or revealing an invitation or account. The
page explains the staging limitation immediately and disables email actions.
Staging remains `dry_run`: no email, Auth account or token is created by these
requests, so a new-account email journey cannot be manually completed there.
For the staging club pilot, update the pending invitation in step 2 to an existing
confirmed test account's email and use its existing sign-in. An invented email
address and an invitation alone are not a sign-in account.

Regression coverage includes both invitation kinds, callback intent, password
confirmation and failure recovery, no role grant before acceptance, wrong accounts,
duplicate submissions, email capability failures and non-live email restrictions.
No schema migration or change to season setup, club selection or meet rosters.

## Controlled invitation email pilot (prepared September 8, 2026)

The full new-account journey needs a real mailbox. An existing account can test
acceptance, but it cannot establish that a new recipient receives verification,
sets a first password and signs into the newly assigned club successfully.

The restricted test path is configured in
`config/staging_invitation_email_test.json`. Joe approved one actual test inbox
on September 8; the seven-day window ends September 15 at 19:22:20 UTC. The public
repository holds only the SHA-256 digest of the normalized address. The invitation
record retains the actual email. Configuration permits at most three mailbox
digests, an approval time and an expiry no more than seven days later. No wildcard,
redirected mailbox or `.invalid` address is accepted. Configuration changes go
through the normal staging PR, build and deployment process.

The runtime requires the isolated staging Fly app, Supabase project and canonical
web origin, `JUPR_EMAIL_MODE=dry_run`, and configured SMTP with TLS. The invitation
email endpoints are the only callers of this exception. Other staging mail
remains in its existing dry-run mode. Both the request and final sender check the
approved address before any Auth user/token creation; the normal invitation
claim still checks validity, email binding and rate limits. Messages carry
`[PCS staging test]` in the subject and go only to the invited, approved address.
No token is exposed in the API response, list, handoff or organizer screen.

Staging health and its handoff report activation, recipient count, expiry and
SMTP configuration readiness, without revealing addresses or credentials. A
missing/invalid configuration, expired window, incorrect environment or missing
TLS/SMTP keeps delivery disabled. Delivery activation does not send any mail by
itself: the recipient must request the verification link from their invitation.

After activation, the manual pilot is:

1. Update La Ribera's pending invitation to the approved real test address.
2. Open its link in a separate browser session and choose **Create account**.
3. Request and receive the verification email; follow the link.
4. Set and confirm a password, review La Ribera, then accept the invitation.
5. Sign out and sign in with the new email/password. Confirm only La Ribera is
   available to this single-club account; Tres remains in the organizer's account.
6. Continue the season invitation and choose a lineup for an upcoming meet.

La Ribera's existing pending invitation has been renewed for the approved inbox.
The staging SMTP configuration is still missing, so delivery remains blocked even
with the approved recipient configuration. See [staging mail setup](staging_invitation_email_setup.md)
for the remaining secure configuration and manual acceptance steps. No email
delivery or mailbox ownership test has been completed yet.
