# Badge reactivation

Joe requested restoration of the 13 unavailable public badge types and the six hidden definitions. Work targets staging. The earlier production approval covered the completed badge repair, not this new activation release.

## Existing criteria

- Mr. Reliable: at least 30 matches and a final win rate of at least 70% in the same season.
- League Third Place: third place in final league standings, awarded on league close.
- Podium: any top-three finish on league close.
- Good Sport, Community Builder, and Mentor were defined as manual recognition awards. Joe has now supplied the qualifying actions below.

The 11 tracked public badges have written criteria in badge_requirements.md. Four have evaluators but are explicitly disabled; seven have placeholder evaluators. League Champion and League Runner-Up are inactive definitions with a legacy award path.

## Confirmed community award criteria — September 6, 2026

Joe selected every action below and confirmed these three badges are awarded manually by a club admin. They must not be inferred from match statistics or issued by automatic badge evaluators.

Any one of a badge's listed actions can qualify; recipients do not need to complete the whole list.

- **Good Sport:** honest calls even when they cost a point; respectful handling of a difficult situation; consistently encouraging and supporting other players.
- **Community Builder:** welcoming newcomers and helping them find games; volunteering at club events; organizing inclusive social play; introducing new participants to the club.
- **Mentor:** helping a newer player improve over several visits; leading beginner lessons or practice sessions; helping players learn rules and court etiquette.

The admin award form should record the player, qualifying action, a short recognition note, and the issuing admin. Use the existing club admin authorization and activity log patterns; operator access is not implied by this decision. The currently available badge diagnostics actions do not include manual issuance, so a working admin award path is still required before activation.

These criteria are now reflected in both the canonical requirement documentation and the badge catalog copy. The three definitions remain inactive while the award flow and cadence are completed.

## Decisions still needed

1. Award cadence for the three manual badges: once per player, once per season, or repeat recognition for separate contributions.
2. Meaning of season: the current match-fact season key is a calendar year, while Tres operations run across calendar years. Confirm before implementing final-season awards.
3. League podium ranking, eligibility and tie handling; generic Podium overlaps placement-specific trophies and needs a display/award policy.

## Prepared implementation

Clutch Performer now has an evaluator for the first five distinct eligible wins by one or two points. It ignores losses, invalid margins, duplicate match facts, and results after the cutoff. Lifetime identity recognizes historical award contexts, including revocations. Activation remains disabled until the complete release is ready.

No new historical backfill or production mutation has been authorized for this activation work. Preview any historical impact separately before applying it.
