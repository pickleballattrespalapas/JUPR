# JUPR tournament UI/UX rewrite

This package contains rewritten versions of the four tournament-facing Streamlit pages:

- `tournaments.py`
- `tournament_manager.py`
- `tournament_registration.py`
- `tournament_partner_board.py`

## What changed

### Tournament Manager
- Reworked around an operator workflow:
  1. Tournament Info
  2. Days
  3. Events
  4. Divisions
  5. Schedule Preview
  6. Publish & QA
- Removed the confusing raw `day_key` / `event_key` admin experience.
- Removed the `max_events_per_day` concept.
- Added event defaults for format/scoring.
- Added division-level scheduling so each division is assigned to exactly one day.
- Added structured capture for age rules, auto age split thresholds, and split-age thresholds.
- Persists the builder output as the canonical registration contract (`days` + rich `event_options`) consumed by public registration.

### Tournaments
- Creation is now a tournament shell flow instead of a bracket-first flow.
- Division operations are now tied to selected event/division draws when available.
- Improved manual team editing and bulk upload flow.
- Fixed bulk upload append logic so append no longer overwrites slot 1.
- Added registration-to-draw import support.
- Team sizing is treated as an operations-stage setting rather than a tournament-creation setting.
- Singles divisions now allow one-player slots in readiness validation and labeling.

### Tournament Registration
- Public registration is grouped by day -> event -> division.
- Added clearer division labels and event detail captions.
- Partner-board opt-in now happens per selected doubles division.
- Added validation for at least one selected division.

### Partner Board
- Replaced a flat table-only experience with grouping by day and event.
- Kept filters for day and event.

## Assumptions
- `tournament_event_draws` exists and is already part of the branch work.
- If `team_count` or `playoff_advance_count` do not exist on `tournament_event_draws`, the rewritten `tournaments.py` falls back to the legacy `tournaments` table.
- Existing registration tables accept the extended event-option fields already used in your branch work, including fields such as `event_family_label`, `division_name`, `event_format_default`, `scoring_default`, `age_mode`, and `age_rules`.
- Current round-robin / playoff generation still depends on `SUPPORTED_TEAM_COUNTS` from the existing domain logic.

## Next recommended follow-up
- Add draw-scoped podium storage so podium/completion is no longer tournament-wide for multi-division tournaments.
- Add formal parsing/enforcement of `age_rules` inside the registration compiler.
