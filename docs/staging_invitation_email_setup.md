# Test multiple clubs with the existing staging sign-in

Updated September 15, 2026. Joe chose to test club operations without configuring
SMTP or creating another account. The invitation email pilot is disabled and its
recipient approval has been cleared.

La Ribera's club account invitation has been reassigned to Joe's existing verified
staging identity and accepted through the normal guarded invitation transaction.
That identity now has its existing Tres Palapas role and an administrator
assignment for La Ribera. Both are ordinary club assignments checked on each API
request. No password, Auth user, verification flag, or authentication rule changed.

## Start testing

1. Open [staging club selection](https://jupr-git-staging-pickleballattrespalapas1.vercel.app/admin/select-club)
   using the existing staff sign-in. If La Ribera is missing from a session opened
   before the assignment was added, sign out and sign in again.
2. Open **Tres Palapas** to continue organizing the **Southern BCS** season under
   **Interclub seasons**. Select La Ribera in the participating-clubs step.
3. Finish the divisions, meet schedule, and review steps, then open season
   invitations when ready.
4. Use **Switch club** to open **La Ribera**. Use the interclub invitations and
   meet-roster workspace to respond as the participating club.
5. Add test players through each club's own **Players** page. Choose a lineup for
   each upcoming meet; a season does not lock one roster for every meet.

The club account invitation is already accepted. Accepting a season invitation
and choosing lineups remain explicit actions for manual testing.

This route tests a single signed-in person administering multiple clubs. A later
pilot with a separate existing account can test the experience of an administrator
assigned to only one club. New-account email delivery is outside this test.

## Environment

All data changes apply only to the isolated staging project and La Ribera test
club. General mail remains `JUPR_EMAIL_MODE=dry_run`; the optional invitation mail
configuration has `enabled=false` and no recipients. No SMTP setup is needed for
this test. The club and staff audit records preserve the invitation changes.
