# Badge reactivation decisions

Joe requested restoration of unavailable badges and plain earning requirements. This release targets staging. The earlier production approval covered the completed badge repair; it did not cover this new activation release.

## Community recognition

Good Sport, Community Builder, and Mentor are awarded manually by club administrators. Any one listed action qualifies. Players may receive each badge repeatedly for separate contributions, with no lifetime or season cap.

| Badge | Qualifying actions selected by Joe |
| --- | --- |
| Good Sport | Honest calls even when they cost a point; respectful handling of a difficult situation; consistently encouraging and supporting other players. |
| Community Builder | Welcoming newcomers and helping them find games; volunteering at club events; organizing inclusive social play; introducing new participants to the club. |
| Mentor | Helping a newer player improve over several visits; leading beginner lessons or practice sessions; helping players learn rules and court etiquette. |

The admin form records the player, selected actions, contribution date, and recognition note. The authenticated issuing admin is recorded. The database rechecks current club authorization; operators cannot issue awards. A stable request ID makes retries safe, while separate contributions use separate IDs. Award, audit record, and retry receipt commit together.

## Admin-defined seasons

Joe chose seasons configured by club admins: name, inclusive start/end dates, and timezone. Each season has a stable UUID. Seasons in one club cannot overlap. No configured season means no new season-based awards. Existing historical awards remain intact.

| Badge | Requirement |
| --- | --- |
| Battle Tested | Complete 50 eligible matches in the configured season. |
| Consistency | Play in six consecutive ISO weeks within the season. ISO weeks retain UTC Monday–Sunday boundaries. |
| Steady Hand | Reach at least 20 matches and a win rate of at least 60% in the season. Later losses do not remove the award. |
| Mr. Reliable | Finish the season with at least 30 matches and a win rate of at least 70%. Checked after the end date fully elapses in the season timezone. |

Each badge is earned once per configured season. Saving a season starts a badge check. Later match processing and staff badge checks also pick up completed seasons. There is no independent midnight scheduler. Season dates cannot change after awards reference them; renaming remains available. Concurrent date changes and award inserts are checked against the saved earning evidence.

Swiss Army Knife explicitly uses a calendar year and retains that requirement. League Top Performer awards and tournament podiums retain their existing event rules.

## Retired duplicate trophies — confirmed September 7, 2026

Retire League Champion, League Runner-Up, League Third Place, and generic Podium. Joe prefers the existing league awards and tournament podium flows. These four definitions are removed from the public catalog and goals and cannot produce new candidates. Existing award rows, earning dates, and profile trophies are preserved.

## Restored automatic badges

Clutch Performer, Breakthrough, Above Expectations, Dominant Run, High Output, Battle Tested, Consistency, Nemesis Found, Rivalry Win, Rivalry Streak, Settled the Score, and Mr. Reliable are enabled. Their requirements are documented in badge_requirements.md and displayed directly in the catalog.

Clutch Performer counts five distinct wins by one or two points. Rivalry calculations follow chronological opponent records: six meetings with a win rate of 40% or less identify a nemesis; that opponent stays tracked as the record improves. Rivalry wins are per nemesis and match; Rivalry Streak is earned on the third consecutive win after identification, with a loss resetting the streak. Settled the Score requires a win that exactly levels the historical record.

Breakthrough handles tied ratings without inventing movement. Above Expectations rejects missing/nonfinite metrics when the criterion requires them. Catalog scopes and repeatability match the restored evaluator identities.

## Historical handling and release boundary

The activation migration changes definitions and adds admin/season infrastructure. It does not update, delete, revoke, or backfill historical award rows. Revoked award identities continue to prevent automatic resurrection. No community recognition is inferred from old match data, and no season dates are invented.

Automatic processing can discover previously unissued achievements when it evaluates a player's history. Preview that impact before a separately authorized production activation. Use a fresh production snapshot for that later review; an older export is useful only as an estimate.

Staging readiness requires the focused Python and frontend checks, the transactional SQL rehearsal, migration verification, and the exact merged SHA's successful Fly/Vercel handoff. Production activation remains a separate release.
