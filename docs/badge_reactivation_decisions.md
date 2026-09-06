# Badge reactivation

Joe requested restoration of the 13 unavailable public badge types and the six hidden definitions. Work targets staging. The earlier production approval covered the completed badge repair, not this new activation release.

## Existing criteria

- Mr. Reliable: at least 30 matches and a final win rate of at least 70% in the same season.
- League Third Place: third place in final league standings, awarded on league close.
- Podium: any top-three finish on league close.
- Good Sport: a manual sportsmanship award; no specific qualifying actions recorded.
- Community Builder: a manual community-impact award; no specific qualifying actions recorded.
- Mentor: a manual mentorship award; no specific qualifying actions recorded.

The 11 tracked public badges have written criteria in badge_requirements.md. Four have evaluators but are explicitly disabled; seven have placeholder evaluators. League Champion and League Runner-Up are inactive definitions with a legacy award path.

## Decisions still needed

1. Qualifying actions for Good Sport, Community Builder, and Mentor.
2. Award cadence and who can issue the three manual badges.
3. Meaning of season: the current match-fact season key is a calendar year, while Tres operations run across calendar years. Confirm before implementing final-season awards.
4. League podium ranking, eligibility and tie handling; generic Podium intentionally overlaps placement-specific trophies and needs a display/award policy.

## Prepared implementation

Clutch Performer now has an evaluator for the first five distinct eligible wins by one or two points. It ignores losses, invalid margins, duplicate match facts, and results after the cutoff. Lifetime identity recognizes historical award contexts, including revocations. Activation remains disabled until the complete release is ready.

No new historical backfill or production mutation has been authorized for this activation work. Preview any historical impact separately before applying it.
