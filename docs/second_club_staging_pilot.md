# Second Club Staging Pilot Checklist

## Purpose

Validate JUPR multi-club behavior in staging using a non-production second club before any production multi-club rollout.

## Prerequisites

- Staging Supabase is verified and currently used for `staging` branch validation.
- `clubs` table migration is applied in staging.
- Club-scoped admin roles are implemented and enforceable.
- Public leaderboards are working for club-scoped reads.
- Staging email safety is enabled (non-production routing/allowlist guardrails).
- Staging follows the persistent-open policy in `AGENTS.md`; email stays `dry_run`.

## Setup checklist

- [ ] Create a second club row in staging (non-production tenant only).
- [ ] Create a small set of staging-only players for the second club.
- [ ] Create league metadata for the second club.
- [ ] Assign `administrator` and scoped `operator` roles scoped to the second club.
- [ ] Confirm Tres Palapas staging data remains separate after setup.

## Validation checklist

### Public

- [ ] Validate `/clubs/{second-club}` resolves correct club public profile in staging.
- [ ] Validate `/clubs/{second-club}/leaderboards` returns only second-club data.
- [ ] Confirm no private fields are exposed in public responses.

### Admin

- [ ] A second-club-only account signs in directly to its club, with no switcher.
- [ ] A multi-club account can switch clubs and lands on the chosen admin home.
- [ ] Players, leagues, tournaments and communications use the selected club.
- [ ] Switching in a second tab blocks actions in an older tab until reload.
- [ ] Reload/back navigation cannot restore actionable controls from another club.
- [ ] Staff and interclub pages follow the same selected club.
- [ ] Expired/revoked assignments cannot open a workspace.
- [ ] An administrator can save its name, description and contact email in Club settings.
- [ ] Draft setup can be submitted and appears Ready for review in PCS administration.
- [ ] Editing a pending submission returns it to In progress until resubmitted.
- [ ] Two administrators editing the same club receive a conflict instead of overwriting newer details.
- [ ] Saving details for an active club preserves its active/setup status.
- [ ] Operators and another club's administrators cannot read or write club settings.

### Staff invitations

- [ ] In Club staff, create an operator invitation with a program scope and access end date.
- [ ] Verify the invitation appears Pending and grants no staff access yet.
- [ ] Copy the link and open it in a separate browser profile with an existing confirmed staging test account.
- [ ] Verify a different email cannot view or accept the invitation.
- [ ] Sign in with the invited email, review the correct club/role/scopes, then accept.
- [ ] Verify acceptance opens that club and the administrator sees Accepted after refreshing.
- [ ] Verify a single-club account has no switcher; an account assigned to two clubs retains both.
- [ ] Cancel a pending invitation and verify its link cannot grant access.
- [ ] Remove accepted staff access and verify reopening the old invitation cannot restore it.
- [ ] Verify operator access stops on its end date and that expired staff can receive a new invitation.
- [ ] Verify a removed inviter or a newer staff assignment prevents stale invitation acceptance.
- [ ] Verify Email me a sign-in link explains that email is disabled in staging; no email or Auth account is created.

Staging email remains `dry_run`. Creating links does not send invitations. Real
new-account email delivery is a separate future acceptance check when explicitly
authorized in an appropriate environment.

- [ ] Verify second-club operator cannot access Tres Palapas write operations.
- [ ] Verify Tres Palapas operator cannot write second-club matches.
- [ ] Verify admin activity log records the correct `club_id` for every write attempt.
- [ ] Verify worker runs are scoped to the selected club.

### Interclub participation and teams

- [ ] As organizer, save a season with two or more proposed clubs and the required divisions.
- [ ] Set a future roster deadline, explicit rating limits and any required team composition, then open invitations.
- [ ] Confirm later planning edits do not silently change the open registration's rules or dates.
- [ ] Open Club invitations and team rosters from the second club's workspace.
- [ ] Accept for the second club; verify the organizer cannot accept on its behalf.
- [ ] Cancel an unused invitation, verify acceptance is blocked, then invite that club again.
- [ ] Submit four distinct active players from the represented club; verify other clubs' players are unavailable.
- [ ] Verify a player cannot join two of that club's teams in the same division.
- [ ] As organizer, verify only submitted roster names, starting ratings and eligibility facts are visible.
- [ ] Submit a roster outside a configured limit; verify it needs an organizer exception.
- [ ] Confirm the represented/host club cannot approve the exception; approve or decline with an organizer reason.
- [ ] Edit that roster; verify the old lineup and decision stay in history and a new violation needs a new decision.
- [ ] Verify a valid substitution after the deadline can be submitted; a late new team needs organizer review.
- [ ] Withdraw a team, then verify its players can join another team and its history remains available.
- [ ] Verify stale roster saves and stale organizer approvals require reload without overwriting newer work.
- [ ] Verify switching accounts or clubs while a save is pending cannot show an old success in the new workspace.
- [ ] Confirm entering or approving a roster changes no home-club or league ratings.

Meet operations and rating approval remain separate future increments.

### Data isolation

- [ ] Confirm matches are club-scoped.
- [ ] Confirm leaderboards are club-scoped.
- [ ] Confirm roles are club-scoped.
- [ ] Confirm email subscriptions/outbox are club-scoped.

## Rollback

- Disable second club in staging.
- Remove staging-only test data if needed.
- Do not touch production systems, data, or configuration.

## Non-goals

- No billing.
- No public signup.
- No production multi-club launch.
