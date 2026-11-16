# Public provisional ratings

Public player ratings read **Not rated** until the player has rated match history in that discipline. Doubles and singles are independent; an unrated session, a starting estimate, or an administrator confirming a seed does not establish a public rating.

The public projection uses the existing rated `matches_played` counter (or wins plus losses for legacy rows without that counter). Zero or unknown history masks the current and starting ratings in API responses. The profile's high/low summary therefore cannot reveal a hidden seed. Explicitly unrated matches do not contribute rating history or expose rating snapshots.

Player directories, profile cards, league ratings, search suggestions, challenge ladders, and team-league player choices display the missing public rating as **Not rated**. Hidden estimates do not determine directory rating order. Match Explorer requires established doubles ratings rather than substituting an assumed value for an unrated player. Rating leaderboards already exclude players without rated history.

This is a public presentation policy. Stored estimates, administrator review, match calculation, player records, and unrated participation are unchanged. No migration or rating replay is required. Tournament entry skill declarations remain registration data, not a published player rating.

Regression coverage: `tests/test_public_provisional_ratings.py`, `apps/web/tests/public-provisional-ratings.cjs`, and the player-search component tests exercise hidden raw/normalized values, summaries, unrated snapshots, directory sorting, the first rated game, and independent doubles/singles status.
