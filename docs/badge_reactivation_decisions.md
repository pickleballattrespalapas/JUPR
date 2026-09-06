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

Joe confirmed that all three badges can be awarded repeatedly for separate qualifying contributions, with no lifetime or season cap. The catalog marks them stackable; each contribution gets a stable recognition ID so a retry cannot create a second award for that contribution. Criteria, contribution date, and a recognition note are included in the prepared award payload. Club membership, issuing admin authorization, and durable request verification belong in the write service, which is still to be built.

These criteria and the repeat-award rule are reflected in both the canonical requirement documentation and the badge catalog copy. The three definitions remain inactive while the admin award flow is completed.

## Confirmed season policy — September 6, 2026

Joe chose admin-defined seasons. Do not hard-code a calendar year or the Tres membership year as the badge season.

The prepared calculation layer accepts a stable season ID, club ID, display name, inclusive start and end dates, and a timezone. Seasons for one club cannot overlap; different clubs have independent seasons. An absent season configuration produces no candidates for the restored seasonal badges. Renaming a season retains the same award identity. Dates and names are copied into award evidence so historical records retain the criteria used at issuance.

- Battle Tested: first 50 eligible matches in the configured season, once per season.
- Consistency: first run of six consecutive ISO weeks with eligible play in that season, once per season. ISO weeks retain the existing UTC boundaries.
- Mr. Reliable: at least 30 eligible matches with a final win rate of at least 70%, checked only after the configured end date has fully elapsed in the season timezone.

The season storage, admin controls, and worker loading are still required. Existing live calendar-year calculations have not been switched in this preparation commit. The release must wire the new configuration into all applicable season-based rules while preserving historical award IDs and separately previewing any historical effects. Swiss Army Knife explicitly uses a calendar year and is not silently redefined by season settings.

## Decisions still needed

1. League podium selection: use an admin-configured rule and reviewed final winners, or one fixed ranking rule. The existing league awards flow already has per-category minimum participation, ranking depth, tied winners, and reviewed overrides that should be reused where appropriate.
2. General Podium badge: whether it marks the first top-three league finish, every finish, or a repeated-finishes milestone alongside the placement-specific trophies.

## Prepared implementation

Clutch Performer now has an evaluator for the first five distinct eligible wins by one or two points. It ignores losses, invalid margins, duplicate match facts, and results after the cutoff. Lifetime identity recognizes historical award contexts, including revocations. Activation remains disabled until the complete release is ready.

Battle Tested, Consistency, and Mr. Reliable now have evaluators using the configured-season calculation layer. Community awards have a pure validated award builder and repeatable identity; automatic evaluators explicitly return no candidates. None of these prepared changes enables a new award path yet.

Validation: 34 focused tests pass, covering duplicate matches, first qualifying events, cross-year ISO week 53, date boundaries and DST, final-season win percentage, club separation, overlapping-season rejection, repeat recognition identity, invalid community criteria, schema and requirement contracts, and existing repair regressions.

No new historical backfill or production mutation has been authorized for this activation work. Preview any historical impact separately before applying it.
