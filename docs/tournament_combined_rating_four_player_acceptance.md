# Combined-rating and four-player team tournament acceptance

## Combined-rating divisions

- An organizer can configure a registration division with a strict combined-rating
  cap, including under 7.0, under 8.0, and under 9.0.
- A linked JUPR rating is authoritative. An organizer-verified rating is used only
  when no linked rating exists, and the source is retained for review and audit.
- Eligibility is strictly below the cap. For example, 3.47 + 4.33 is eligible for
  under 8.0, while 8.00 is not.
- A team may choose a higher-cap division even when eligible for a lower-cap
  division. Choosing a lower-cap division without meeting its cap is refused.
- A registration without a partner is provisional. A paired registration with a
  missing rating is held for organizer review and cannot enter a draw until
  resolved.
- Closing registration records an immutable eligibility snapshot. Later rating
  changes do not silently rewrite the closed field.

## Four-player team registration

- A captain registers exactly four distinct confirmed players: two men and two
  women. Invitations are bound to one team, member slot, email, version, and
  expiry.
- A recipient can accept or decline only with the matching confirmed
  registration identity. Tokens are removed from the browser address before use,
  never appear in a query string, and are never returned by public APIs or audit
  data.
- Invitation email honors staging dry-run and recipient-redirect controls.
- An organizer can configure whether substitutions are allowed. When disabled,
  the roster locks once match activity starts. When enabled, a replacement still
  requires a reason, a complete eligible roster, optimistic version checks, and
  no in-progress matchup using that lineup.

## Match format and standings

- Each matchup creates games in this fixed order: women's doubles, men's
  doubles, mixed doubles 1, mixed doubles 2.
- Each player appears in the gender doubles game and exactly one mixed doubles
  game. Both teams submit their mixed pairing independently; neither lineup is
  revealed until both are locked.
- A tiebreak is created only after a 2-2 regulation result. The configured mode
  is singles or skinny-singles relay. Singles contributes to ratings; the
  skinny-singles relay does not.
- Round robin schedules every team against every other team once and handle an
  odd number of teams with byes.
- Standings rank match record first, then head-to-head among tied teams, game
  differential, game wins, and a stable name/id fallback.
- Optional playoffs support top-two final, top-four semifinals and final, or
  top-four semifinals, final, and bronze match. Playoffs cannot be created until
  round robin is complete.
- Podium placements are calculated from completed results. Caller-supplied
  winners cannot override the recorded bracket or standings.

## Ratings, permissions, recovery, and public results

- The four-game parent matchup is result-only and cannot be published to ratings.
  Each rated child game has exactly one protected canonical tournament game with
  the source players, format, and score.
- Deleted, excluded, corrected, or side-swapped canonical matches reconcile back
  to the child result with explicit status and downstream playoff dependency
  handling. The same game cannot be published a second time.
- Every mutation is club- and tournament-scoped, manage-tournaments permission
  only, service-role only at the database boundary, idempotent, version checked,
  serialized per draw, and durably audited. Reconciliation and public corrections
  require a reason.
- Public pages expose only published active teams, accepted member display data,
  standings, brackets, and podiums. They never expose emails, invitation evidence,
  delivery state, operation fingerprints, private eligibility evidence, or
  recovery internals.
- Admin and public experiences remain usable at normal laptop and mobile widths,
  and public write routes remain closed unless the staging write gate is
  explicitly opened.
