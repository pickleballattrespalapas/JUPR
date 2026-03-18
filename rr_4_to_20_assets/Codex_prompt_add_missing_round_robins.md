Implement missing doubles round-robin counts in the JUPR repo using the supplied reference generator and templates.

Repo context
------------
The current JUPR app already supports these doubles schedule counts:
4, 5, 6, 8, 9, 12, 14

We need to add the missing counts from 4..20:
7, 10, 11, 13, 15, 16, 17, 18, 19, 20

Reference files to read first
-----------------------------
- prototype/live-brackets-mvp/PRODUCT_SPEC.md
- prototype/live-brackets-mvp/README.md
- prototype/live-brackets-mvp/index.html
- prototype/live-brackets-mvp/styles.css
- prototype/live-brackets-mvp/engine.js
- prototype/live-brackets-mvp/app.js
- rr_4_to_20_assets/doubles_round_robin_reference.py
- rr_4_to_20_assets/jupr_missing_round_robin_templates.json
- rr_4_to_20_assets/jupr_missing_round_robin_summary.md

Files in the JUPR repo to inspect first
---------------------------------------
- jupr_app/domain/schedule.py
- jupr_app/ui/pages/match_uploader.py
- jupr_app/ui/pages/moneyball.py
- jupr_app/ui/pages/league_manager.py
- jupr_app/ui/pages/jupr_live.py   # if already present on branch Test

Primary goal
------------
Add support for all missing doubles round-robin counts from 4..20 into the current JUPR app, while preserving all existing supported counts and existing behavior.

Implementation requirements
---------------------------
1. Keep existing hand-authored schedule formats unchanged for:
   4, 5, 6, 8, 9, 12, 14

2. Add support for these additional formats:
   7-Player
   10-Player
   11-Player
   13-Player
   15-Player
   16-Player
   17-Player
   18-Player
   19-Player
   20-Player

3. Use the supplied generated templates as the source of truth for the new counts.
   The template rows use 1-based player positions and need to be mapped to the actual
   player IDs / names the same way the current schedule code maps existing formats.

4. Do not redesign current schedule semantics.
   Keep the current return shape from get_match_schedule exactly consistent with the rest of the app.

5. Update all format selectors and validation logic anywhere the app currently hardcodes
   the supported doubles counts. This likely includes:
   - match_uploader format dropdowns / expected game counts
   - JUPR Live Beta setup help text and validation
   - any count checks in schedule helpers
   - any UI labels that list supported counts

6. Add tests for the new counts.
   At minimum verify:
   - get_match_schedule("7-Player", ids) through get_match_schedule("20-Player", ids) all return non-empty schedules
   - every match has exactly 4 distinct players
   - each team has exactly 2 players
   - no partner pair repeats within a generated format
   - play counts are balanced within 1 game across players
   - bye counts are balanced within 1 across players when byes exist

Preferred implementation approach
---------------------------------
Option A:
- Add the new templates directly into jupr_app/domain/schedule.py as static constants
- Reuse the same mapping logic used by the existing formats

Option B:
- Add a new helper module for generated doubles templates
- Import it into schedule.py
- Keep the runtime lookup path simple and deterministic

Do not do this
--------------
- Do not replace existing 4, 5, 6, 8, 9, 12, 14 schedules
- Do not change official match-processing behavior
- Do not change return payload fields used by match_uploader / moneyball / league_manager
- Do not introduce a database dependency for schedule generation

Acceptance checklist
--------------------
- Missing counts 7, 10, 11, 13, 15, 16, 17, 18, 19, 20 all work in get_match_schedule
- The JUPR Live Beta setup no longer rejects those counts
- UI copy listing supported doubles counts is updated to reflect 4..20 support
- Existing supported counts still work exactly as before
- Tests pass

After coding
------------
1. Show the exact files changed
2. Summarize the schedule integration approach
3. Call out any places where the branch Test UI still needs manual UX polish
