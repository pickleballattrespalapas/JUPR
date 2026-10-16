# Share a meet signup link

In the league workspace, choose **Get signup link for the next meet** (or choose
a meet under **Meet signup**), then **Open meet signup** and **Copy link**.
The link belongs to that club and meet. Players do not need an account.

Players choose an approved profile from the season pool and one division. The
signup reads the profile’s league rating and gender; players cannot supply or
override these values. Each club/division has two women’s and two men’s spots.
The database assigns registration order under the same season lock used by
roster updates, including simultaneous submissions.

- Numeric bands start at the division and end just before the next half-point.
  Thus 3.00–3.499… receive 3.0 priority; 2.93 is a play-up waitlist request.
  Open and 4.5/Open prioritize ratings of at least 4.5.
- The first two in-band players of each gender reserve spots. Later in-band
  players form the substitute queue, followed by play-up requests. Each group
  keeps registration order. Unknown gender goes to club review.
- A complete group of two women and two men automatically creates/updates the
  actual meet team through the existing roster validation. An incomplete group
  keeps its reservations but does not present a complete team as ready to play.
- Before the signup cutoff, withdrawing promotes the next in-band player of the
  same gender. An admin can approve a play-up player into a vacancy. Later
  in-band registrations still have priority over that player.
- A private link lets the player check their place or withdraw. The club can
  retrieve it; it is never exposed in the shared board. No email is sent by this
  workflow. Players should check their link for promotions.
- Exact network retries recover the same private link and position. A new
  duplicate request does not reveal it. Rejoining after withdrawal receives a
  new order and invalidates the old private link.

The signup cutoff cannot be later than the roster deadline. Eligibility is
rechecked during signup changes and **Refresh signups and lineups**; normal
roster-deadline snapshots and competition eligibility remain authoritative.
After signup closes, contact the club for changes. The signup board is the
registration record; **Lineups** is the official team after manual changes.

Admins can add verbal commitments through the same queue. Close signup before
editing a lineup manually. Opening signup cannot adopt existing manual teams,
and reopening cannot overwrite a manually edited automatic team. A schedule
change closes player intake until the club reopens/reconfirms it; existing
versioned lineups may require continued manual management.

Database tables and RPCs are service-role only, with RLS enabled and no browser
grants. The delegated club admin is revalidated before every public mutation.
Private tokens use a separate signing purpose and URL fragments. Public boards
contain names, ratings and queue positions, never email or management links.

Verification: `tests/sql/interclub_meet_signup_transaction.sql` rolls back all
fixtures; API capability tests cover privacy/scope; the staging browser rehearsal
tests anonymous signup, concurrent capacity, ordered substitutes, retries,
withdrawal into the real roster, and the mobile page.


Admins can click a division opening or Add substitute to choose an approved player. Candidates load across all pages, exclude existing meet registrations and assigned lineups, and show eligible league ratings highest first. Playing-up choices are marked and remain waitlisted behind in-band signups.

Both the player and admin registration forms require a gender selection: Woman, Man, Non-binary, or Prefer not to say. The latter two keep a place in the registration order but enter admin review with a clear message. Admins choose a women’s or men’s lineup place after review, retaining the player’s declared gender privately. The declaration is meet-specific and does not overwrite the club profile. Rating priority, capacity, deadlines and registration order still apply after approval.
