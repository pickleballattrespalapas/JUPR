# Player creation from completed generator results

In the generator submissions review, an administrator can select an existing club profile or choose **Create new player**. Creation pre-fills the entered participant name, allows a spelling correction, and does not ask for a starting rating. Existing-profile selection and the pre-event Moneyball setup remain supported.

The starting estimate uses the average of the known player ratings in the submitted group. Saved scores then adjust that estimate for teammates and opponents. New players are fitted jointly, with a conservative prior equivalent to four doubles appearances, so a single result cannot dominate the estimate. Singles and doubles evidence are separate. With no connection to a known player, the group average is retained; with no known ratings at all, the existing 3.5 default is explicitly shown as a default estimate.

Creation does not publish matches or increment rated-history counters. Unrated approval keeps new players provisional and leaves existing player ratings unchanged. Rated approval uses the estimate as a starting baseline and publishes the scored matches once through the normal rating pipeline. A starting estimate is not an established rating.

Creation uses the existing guarded-operation receipts, including duplicate checks and recovery after a lost response. The browser retains the exact request until creation is confirmed; a definitive stale-session conflict allows refresh and a new request. The approval flow finalizes eligible new doubles estimates against the complete identity mapping and saved scores.

## Acceptance checks

- Open a completed submission with unmatched players; create one with the prefilled name and no rating field.
- Correct its name before creation, or choose an existing profile without creating a duplicate.
- Reload the review: the created profile stays selected and its estimate remains visible.
- Create multiple new players, then approve an unrated session: their estimates stay provisional and existing ratings remain unchanged.
- Approve a rated session: newly estimated players do not require manual starting-rating entry, and results publish once.
- Interrupt creation or approval and retry: the same receipt is recovered without duplicate profiles, matches, or rating application.
