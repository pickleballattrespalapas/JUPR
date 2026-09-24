# Interclub competition document contract

`schema_version: 1` is the new paper-first competition document. The legacy
publication editor's single three-game encounter is not equivalent to this
document and must never be silently converted into a two-pairing matchup.

The canonical Pydantic model is `services/api/interclub_competition_models.py`.
Domain functions live in `jupr_app/domain/interclub_competition.py` and consume
and return JSON-compatible dictionaries. They raise `ValueError` with actionable
messages on invalid input. Database revision, permissions, source roster checks,
entry ownership, gender, rating cutoff, and late approvals are API concerns.

## Upcoming meet schedule API

Base `/admin/clubs/{club_id}/interclub/competition/{season_id}`. Scheduling
requires the commissioner's administrator assignment and closed season
registration. The season workspace exposes these actions under **Meet schedule**.

- `POST {base}/meets` accepts `{request_id?:UUID,host_club_id,club_ids,starts_at,
  roster_deadline,courts,duration_minutes,competition_phase}` and returns
  `{meet,publication_review_required:true}`. Regular meets have two to four
  accepted clubs. Identical request IDs and payloads return the original meet;
  reusing an ID with different details returns 409.
- `PUT {base}/meets/{meet_id}/schedule` accepts `{expected_revision,starts_at,
  roster_deadline,courts,duration_minutes}` and returns
  `{meet,availability_reset_count,rosters_refreshed,publication_review_required:true}`.
  Host, clubs and competition phase remain bound to this meet. Changed revisions
  return 409. The whole meet must fit the season's local dates, follow registration,
  and avoid overlapping host/club commitments.
- Meet context supplies `schedule_editable`, `schedule_locked_reason`,
  `schedule_deadline_editable`, and `courts_editable`. Started, scored, replayed or
  approved meets use the existing weather/results workflow. Once ratings are
  frozen or a competition batch exists, the eligibility cutoff stays fixed.

Changing a future unfrozen cutoff creates new roster revisions preserving player
choices; existing exception decisions require review again. Changing the meet
time or duration resets RSVP answers, rotates reply links and closes availability
collection. Clubs must reopen collection and invite players to reconfirm. Previous
responses and roster revisions remain audited. No invitations are sent by schedule
edits. The season's canonical calendar projection updates immediately; a public
publication remains unchanged until explicitly reviewed and republished.

```json
{
  "schema_version": 1,
  "meet_id": "meet-id",
  "phase": "regular",
  "format": "gender", "schedule_mode": "staggered",
  "weather": "normal",
  "encounters": [{
    "id": "encounter-id", "division": "3.5",
    "club_a": "club-a", "club_b": "club-b", "rotation": 1,
    "pairings": [{
      "id": "pairing-id", "kind": "women", "court": 1,
      "eligibility_deadline": "2027-01-15T19:00:00Z",
      "players_a": ["entry-a1", "entry-a2"],
      "players_b": ["entry-b1", "entry-b2"],
      "games": [{
        "id": "game-id", "status": "pending", "a": null, "b": null,
        "winner": null, "players_a": [], "players_b": [],
        "played_at": "2027-01-16T17:00:00Z", "injury_reason": null
      }]
    }],
    "tiebreak": null
  }]
}
```

This abbreviated example shows one pairing/game; a real regular encounter has
exactly two pairings (`women`, `men` or `mixed_a`, `mixed_b`) with exactly three
games each. Finals/qualification playoffs use `format: mlp`, four pairings in
women/men/mixed-A/mixed-B order, and one game in each. `phase` is `regular`,
`final`, or `qualifier`. `weather` is `normal`, `delay`, `rescheduled`, or
`finalized_partial`. Pairing `eligibility_deadline` is set and protected by the
server; completed original pairings keep their old cutoff after a reschedule.

Game statuses are `pending`, `completed`, `retired`, `forfeit`, `double_forfeit`,
and `unplayed`.
Completed games require valid side-out scores to 11, win by two with no cap.
Retired games require actual stoppage scores and an explicit winner. Forfeits
require an explicit winner and have no scores. Unplayed games have no winner or
scores and are official only in a regular meet finalized without a reschedule.
Double forfeits have no score or winner and count one game loss for both clubs;
they count no game win, point differential, appearance, or rating effect. A
whole double-forfeited pairing is neither a pairing win nor a weather draw.
Operational default: the other pairing's winner earns 3 points and its loser 0;
two double-forfeited pairings earn neither club a point.
Empty per-game player arrays inherit that pairing's lineup. Changed actual
players require an injury reason and must be eligible replacements verified by
the API. Completed/retired games require an actual `played_at` timestamp.

An MLP tiebreak has `{status, a, b, order_a, order_b}`. A completed 2–2 encounter
requires a completed tiebreak to 21, win by two, no cap. Each order lists its four
active players exactly once. Both clubs rotate every four rallies, cycling
through that fixed order. The court is skinny below 4.0 and full at 4.0 or higher;
it is derived from the skill level. This game never affects individual ratings.

## Function interface

- `generate_round_robin(meet_id, entries, format="gender", played_at=None,
  courts=None, schedule_mode="simultaneous")`: entries are a list of `{division, club_id, team_id?, revision?,
  roster: [{entry_id, name?, gender, starting_rating?}]}`. Each complete roster
  contains two women and two men. A declared partial roster can contain two
  real players: two of one gender at a gender meet, or one woman and one man at
  a mixed meet. Its missing pairing receives three scoreless forfeits. When
  both clubs lack the same pairing it receives three double forfeits. Only
  pairings that can play use courts; no phantom player or score is generated.
  The result is one full regular document. Generated IDs
  are stable for the meet, phase, clubs, division, pairing, and game. Court
  assignments do not overlap within a rotation. Simultaneous scheduling requires
  enough courts for every division in that opponent round. Staggered scheduling
  splits each opponent round into ordered waves using the meet's available
  courts. Both playable doubles pairings in a matchup start together (two courts
  for a full matchup); missing pairings consume no court. A wave starts after
  the preceding wave finishes its three-game pairings; no estimated timestamp
  is substituted for the actual game time. Three-club entries retain their byes.
  Matchup, pairing and game IDs are unchanged by scheduling mode.

The regular `/generate` request accepts `schedule_mode: "simultaneous" |
"staggered"` (legacy callers default to simultaneous). **Run meet → Court
schedule** defaults new UI draws to **Staggered — fit N courts**. The saved
document records that mode; old documents default to simultaneous. The screen
and paper packet show the same ordered wave/court assignments. Score edits
cannot change the mode, waves or courts. Before play, refreshing approved
lineups reallocates courts within those waves if a missing pairing becomes
available. Weather replays retain the existing historical schedule.
- `generate_championship(meet_id, division, club_a, club_b, phase="final",
  played_at=None)`: clubs use the same entry objects; returns an MLP document.
- `validate_document(document, official=False)`: returns a normalized document.
  Official validation rejects partial/unresolved batches. The API separately
  checks that the persisted schedule, clubs, IDs, and roster snapshots match.
- `prepare_reschedule(document, played_at=..., eligibility_deadline=...)`:
  preserves complete three-game pairings; replaces every game of an unfinished
  pairing with a new pending game and updates only that pairing's cutoff.
  Persist the previous revision in the audit history.
- `summarize_document(document)`: results per encounter and game/stat totals.
- `rating_games(document)`: only fully completed doubles games, flattened with
  source IDs, club/entry identities, score, actual play timestamp, and integer
  `sequence`. Equal actual timestamps use schedule order (rotation, game number,
  pairing order, matchup ID), never random game UUID order. Call only
  on the organizer-approved exact document revision.
- `league_standings(documents, clubs=None)`: `{divisions: {division: [rows]},
  qualification: {division: {qualifiers, playoff_required, eligible, status}}}`.
  Documents supplied here must be official accepted revisions only.
- `qualifying_clubs(documents, clubs=None)`: just the qualification mapping.
- `club_cup(documents, clubs=None)`: `{standings: [rows], champions: [club_ids],
  status}`. Cup status remains provisional until eligible division finals are
  complete. Qualifying playoff games never add Cup points or Cup statistics.

Standings rows contain `club_id`, `points`, `pairings_won`, `games_won`, `games_lost`,
`point_differential`, `head_to_head_points`, `meets_played`, `played_meet_ids`,
`position`, and `tied`. Cup rows additionally have `regular_points` and
`championship_points`. Equal rows share a position; club name/ID ordering is only
display ordering and never a sporting tiebreak. Qualification output explicitly
blocks a residual tie crossing second place. A completed two-club MLP qualifying
playoff resolves that tie. For a multiway residual tie, the operational default
is a complete MLP round robin among the tied clubs. Every club pair must finish
its matchup before that playoff round can rank clubs by matchup wins, doubles
games won, then doubles point differential. A remaining cutoff tie requires a
further round robin (or one matchup for two clubs) among only the still-tied
clubs. Singles decides matchup wins but does not add doubles statistics. There
is no random bracket, alphabetical advancement, or Cup credit for these games.

Head-to-head compares standings points earned among statistically tied clubs.
For the Cup it sums those regular points and final bonuses against tied clubs.
This is an explicit operational interpretation of the agreed head-to-head rule.
