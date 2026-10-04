# Recorded-game activity and merged-profile visibility

Joe clarified that merged identities should disappear from player lists and
leaderboards, and tournament registration must not establish leaderboard activity.

The prior daily job used account creation as a fallback when last_game_at was
null. All 15 remaining active production profiles were recent zero-game profiles.
The Players directory excluded merged sources, but the leaderboard projection
and the admin player list did not.

This change:

- Requires an actual recorded game within the existing 14-day window for active
  leaderboard status, including reads before the next nightly sweep.
- Excludes never-played and merged profiles before leaderboard ranking, counts,
  snapshots, and highlights. Existing players keep their ratings when a new
  season begins; a season's zero games does not erase their lifetime history.
- Excludes merged sources from the admin player list and rejects their public
  profile URL. The retained database row remains available for merge recovery.
- Keeps non-merged inactive and never-played profiles available in the Players
  directory and registration/roster selectors.
- Makes historical score imports and removal of the last game update activity
  correctly. Database guards cover older clients that still clear inactive_at.

The migration player_game_activity_only preserves game dates, ratings, results,
registrations, explicit active flags, maintenance ACLs, and the existing cron
schedule. It sets inactive_at on profiles that have no recent recorded game.

Staging database checks passed transactionally for new signup, non-game update,
historical game, recent game, final-game removal, merged identity, unchanged
results, maintenance idempotency, the 14-day boundary, ACLs, and cron schedule.
No active-without-games or overdue-active rows remained after migration.
Security advisor results contain no finding for either changed function.

The broader rating-integrity check has an existing failure in
test_tracked_replay_fences_every_projection_write_batch (five heartbeats versus
four RPC calls). It reproduces on the unchanged baseline and is unrelated to
these corrections.

Production has not been changed by this correction. Its migration and deployment
require the repository's separate production release authorization.
